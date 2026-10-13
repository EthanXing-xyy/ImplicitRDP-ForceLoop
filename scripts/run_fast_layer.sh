#!/bin/bash
# C49 第二轮：快层参照（文献里的 C 类，低维力 × 快层那个空格）。
# 策略 = m3（历史 64 + 预测损失，部署时不多输入）。快层：GRU 状态 h 的线性探针读压力——now = 现在的压力，ahead = 4 步后的压力（控制环死区），
# 残差 d = clip(gain · (f̂ − 10 N), ±3 mm) 加在估计的板法向上，只在接触时生效。
# 问题：同一条力环，用 4 步提前量（ahead）比用当前压力（now）好吗？两者比不加（m3）好吗？
# 探针在 seed 1 的 m3 评测状态上拟合（runs/probe_m3_force.npz），评测只用 seed 2、3；基线同种子：m3 74.0 / 62.5，m0 67.7 / 71.9。
# 每条 96 回合约 8 min，8 条约 1.1 h。
#   nohup bash run_c49_fast.sh >> runs/pipeline.log 2>&1 &
# 第二批（第一批 now 0.2 > 0.1 > ahead，再往上试增益，并用真实压力代替探针看读数的上限）：
#   GAINS="0.4" ORACLE_GAINS="0.2 0.4" nohup bash run_c49_fast.sh >> runs/pipeline.log 2>&1 &
# 第三批（成功率 对 力读数白噪声 的曲线，给融合指尖定精度指标；oracle 读数加噪声 0.5/1/2/4 N，增益 0.2/0.4，种子 2/3，16 条约 2.1 h）：
#   GAINS="" ORACLE_GAINS="" NOISE_N="0.5 1 2 4" nohup bash run_c49_fast.sh >> runs/pipeline.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/../sim"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tacsl
export WIPE_TASK=v2
EP=${EPISODES:-96}
OUT=../results/l1
PROBE=../results/probe_m3_force.npz
CKPT=${CKPT:-../checkpoints/wipe_m3/policy.pt}     # from the GitHub release
step() { echo "=== $(date '+%F %T') $*"; }
while pgrep -f "implicit_rdp_lite.py --task|eval_irdp_wipe" >/dev/null; do sleep 60; done
[ -f $PROBE ] || { echo "!!! no probe file $PROBE (run fit_force_probe.py)"; exit 1; }
# evf <name> <seed> <residual> <gain> [extra args]
evf() {
    local name=$1 seed=$2 res=$3 gain=$4; local out=$OUT/$name; shift 4
    [ -f $out.npz ] && return 0
    step $name
    python eval_irdp_wipe_fast.py --ckpt $CKPT --perturb 1 --episodes $EP --seed $seed --no-h --residual $res --gain $gain \
        --probe $PROBE --stats "$PWD/$out.npz" "$@" 2>&1 | grep -av "~!~" > $out.txt
    [ -f $out.npz ] || echo "!!! $name failed, see $out.txt"
}
for s in ${SEEDS:-2 3}; do
    for g in ${GAINS:-0.1 0.2}; do
        evf wipe_m3_fast_now_g${g}_s$s   $s now   $g
        evf wipe_m3_fast_ahead_g${g}_s$s $s ahead $g
    done
    for g in ${ORACLE_GAINS:-}; do
        evf wipe_m3_fast_oracle_g${g}_s$s $s oracle $g
    done
    for nz in ${NOISE_N:-}; do
        for g in ${NOISE_GAINS:-0.2 0.4}; do
            evf wipe_m3_fast_oracle_n${nz}_g${g}_s$s $s oracle $g --noise-n $nz
        done
    done
done
python - <<'EOF'
import glob, re, numpy as np
rows = {}
for f in sorted(glob.glob("../results/l1/wipe_m3_fast_*_s*.npz")) + sorted(glob.glob("../results/l1/wipe_m[03]_p1*.npz")):
    m = re.match(r"../results/l1/(wipe_.*?)(?:_p1)?(?:_s(\d))?\.npz", f)
    d = np.load(f)
    rows.setdefault(m.group(1), {})[int(m.group(2) or 1)] = (d["success"].mean(), d["wiped"].mean())
print("FAST success (wiped) by evaluation seed, 96 episodes each:")
for k, v in rows.items():
    print("   %-28s " % k + "  ".join("s%d %.1f%% (%.0f)" % (s, 100 * r[0], 100 * r[1]) for s, r in sorted(v.items()))
          + "   mean %.1f%%" % (100 * np.mean([r[0] for r in v.values()])))
EOF
step "fast layer done"
