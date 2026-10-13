**中文** | [English](README_EN.md)

# ImplicitRDP + 力环：触觉"预测"到底有没有用？一个受控的仿真实验

![擦板：只有策略（左）压得忽轻忽重、线没擦完；加一条法向力环（右）压力稳在 5–15 N 窗口内、线擦完](media/wipe.gif)

*同一回合（种子 8）。左：ImplicitRDP 单独跑。右：同一个策略外面套一条写死的法向力 P 环。下方是两条实时压力曲线，绿带是任务要求的 5–15 N 窗口。*

## 一句话

在 Isaac Gym / TacSL 的擦板任务里（板会晃、相机估计有误差），给 ImplicitRDP 外面套一条**写死的法向力比例环**，板晃动成功率从 **70% 提到 84%**（压力从策略隐状态线性读出）或 **96%**（用仿真真实压力，相当于真机上力传感器直接给）。把环里的"当前压力"换成"预测 4 步后的压力"，**没有更好**。力读数的每帧白噪声几乎不伤（4 N 才掉 6–9 个点），**持续几步的相关误差和随压力变的偏差才伤**。

这不是一个新方法。它是一组对照实验，回答一个文献里没有对照的问题：**触觉预测类方法的收益，来自"预测"，还是来自"有一条每步看压力的快环"？** 在这个任务上，答案是后者。

## 问题

OmniVTA、TacPAC 这类工作把预测出来的触觉当快层的参照，报大幅提升；但消融里没有"只用实测、不预测"这一行（TacPAC 有，但它的预测版和基线差的不止预测一处）。这里把两件事拆开，各自单独测。

## 设置

| | |
|---|---|
| 任务 | 擦板 v2：UR5 + Robotiq 2F-85 + GelSight Mini 仿真触觉（TacSL），工具压着板沿线擦；板在法向抖动 1–2 mm，相机对板位姿的估计有误差、有延迟。成功 = 线擦够 **且** 压力规则（5–15 N 窗口，超 15 N 不能连续 0.5 s，不能超 40 N）。脚本专家 100%。 |
| 控制 | 24 Hz 位置命令，Isaac Gym 120 Hz 物理。 |
| 策略 | ImplicitRDP（arXiv:2512.10946）最小复现 `sim/implicit_rdp_lite.py`：Transformer 去噪器 + GRU 快 token，慢 token 每块（32 步）一次，快 token 每步一个，每步 DDIM 5 步重去噪。**m0** = 原版；**m3** = 加 64 帧历史和一个触觉预测辅助损失（部署时不多任何输入；单独跑和 m0 一样，见下）。 |
| 力环 | 每步：`d = clip(K · (f̂ − 10 N), ±3 mm)`，沿**估计的**板法向加到这一步的位置命令上；只在 f̂ > 2 N（接触）时生效；不累积；策略网络看不到它。 |
| f̂ 三种来源 | **now**：线性探针从 GRU 状态 h_t 读当前压力（留出 R² 0.85）；**ahead**：同一个 h_t 读 4 步后的压力（R² 0.67，"假设压力不变"是 0.64）；**oracle**：仿真真实压力（= 真机上力传感器直接给）。 |
| 评测 | 板晃动，评测种子 2、3 各 96 回合（探针在种子 1 的评测状态上拟合，不碰评测种子）。同种子逐回合条件相同，用配对 McNemar。 |

## 结果

### 1. 力环有用，收益全来自"每步看压力"，不来自"预测"

成功率，种子 2 + 3 合并（192 回合）：

| 快层读数 | K = 0.1 mm/N | 0.2 | 0.4 |
|---|---|---|---|
| now（探针，当前压力） | 76.0% | **83.9%** | 50.5% |
| ahead（探针，4 步后） | 74.5% | 77.6% | 70.8% |
| oracle（真实，当前压力） | — | 93.2% | **96.4%** |
| 不加（m3 单独） | 68.2% | | |
| ImplicitRDP 原版（m0） | 69.8% | | |

配对 McNemar：now 0.2 对 m3 +41 / −11，p < 0.001；对 m0 +42 / −15，p < 0.001。oracle 0.4 对 m3 +55 / −1。now 0.2 对 ahead 0.2 +25 / −13，p = 0.07（预测没有赢过实测；K = 0.4 下 ahead 反超只是因为 ahead 探针往均值缩、等于增益更小）。

机制：所有失败都是"线没擦够"，没有压力过大的。力环把接触时压力落在窗口内的时间从 73% 提到 78%（oracle 84%），带着过大压力擦过线的比例从 43% 降到 37%（oracle 27%）。残差只在 16–20% 的步上动，K = 0.2 时平均 0.6 mm，没碰到 3 mm 上限。

参考：同一任务上早先评出的 Diffusion Policy 12.5%、RDP 39.6%。

### 2. 读数质量是瓶颈：白噪声便宜，相关误差贵

真实压力加每步独立的高斯白噪声（24 Hz）：

| 白噪声 σ | 0 N | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| K = 0.2 | 93.2% | 91.7% | 91.7% | 89.6% | 83.9% |
| K = 0.4 | 96.4% | 96.4% | 94.8% | 92.2% | 87.0% |

![成功率对力读数白噪声](results/noise_curve.png)

探针的误差不是白噪声（只算接触步）：rmse 3.7 N，lag-1 自相关 0.59（相关时间约 2 步 = 80 ms），而且随压力系统性偏：真压 2–6 N 时高估 +1.0 N，14–20 N 时低估 −3.6 N，>20 N 低估 −8 N（往均值缩）。对照曲线：探针 K = 0.2 的 83.9% 正好等于 4 N 白噪声；探针 K = 0.4 的 50.5% 远低于 4 N 白噪声的 87%。相关了几步的抖动在高增益下被当成真信号追，白噪声则在机器人响应时间内被平均掉。

### 3. 给触觉指尖力读数的三条指标（力调节类任务）

1. **每帧抖动便宜**：σ ≤ 1 N 几乎免费，≤ 2 N 代价不到 4 个点，不必为它重滤波或降帧率。
2. **持续 ≥ 2 个控制步（≥ 80 ms）的误差和随压力变化的偏差才贵**：它决定能用的增益上限（K 0.2 和 0.4 差 10 个点）。
3. **量程要线性到窗口上沿以上（≥ 20 N）**：探针在 14–20 N 低估 3.6 N，正是环最该抬起的时候。

### 顺便：触觉预测当辅助损失，在这里也没用

m3（历史 64 帧 + 预测损失）单独跑和 m0 没有差别（三种子 288 回合：67.4% 对 72.2%，Fisher p = 0.24）。GRU 状态读得出当前压力（R² 0.85），几乎读不出未来（0.67，对 0.64 的"假设不变"）。把"已发出的命令"也喂进快环的版本在行为克隆下被抄成捷径，闭环 3.1%，不在本仓库里展开。

## 诚实的边界

- 仿真，一个任务。擦板的失败只有"压力出窗"一种，一个标量 P 环就够；需要接触几何的任务（插孔、理线）不一定如此。
- 预测器弱：ahead 探针只比"假设不变"强一点，又不看策略的计划，等于 now 加噪声。要公平检验"预测"，需要看得见计划的预测器（比如 f̂(t+4) = 当前压力 + 刚度 × 接下来 4 步计划的法向位移）。本仓库没做。
- 力环本身是力位混合控制的常识，这里的贡献是对照，不是方法。

## 复现

### 依赖

- NVIDIA **Isaac Gym Preview 4** 和 **TacSL 版 IsaacGymEnvs**（`isaacgymenvs.tacsl_sensors`，IndustReal 的 peg 资产）。NVIDIA 许可，自行从 NVIDIA 获取，不随本仓库分发。
- **reactive_diffusion_policy**（MIT，[xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy)）：只用它的 `ConditionalUnet1D`。
- Python 3.8，PyTorch（Isaac Gym 要求的版本），`urdfpy`、`scipy`、`zarr`、`numpy`、`matplotlib`、`opencv-python`、`imageio`。
- 一张 6 GB 显存的卡够评测（1.3 GB）和训练（3.8 GB）。

目录约定（可用环境变量 `TACSL_ROOT` 改，默认 `~/tacsl`）：

```
$TACSL_ROOT/IsaacGymEnvs/        TacSL 版 IsaacGymEnvs（pip install -e）
$TACSL_ROOT/rdp/src/             reactive_diffusion_policy 的克隆
```

### checkpoint

从本仓库的 [Release](../../releases) 下载，放到 `checkpoints/`：

```
checkpoints/wipe_irdp_v2/policy.pt     m0：ImplicitRDP 原版（擦板 v2，48000 步）
checkpoints/wipe_m3/policy.pt          m3：历史 64 + 预测损失 λ=0.3
checkpoints/wipe_irdp_v2/force_calib.pt   训 m3 时预测头用的力标定（评测不需要）
```

### 跑

```bash
conda activate tacsl
export TACSL_ROOT=~/tacsl
cd sim

# ImplicitRDP 单独（种子 2，96 回合，板晃动）
WIPE_TASK=v2 python eval_irdp_wipe.py --ckpt ../checkpoints/wipe_irdp_v2/policy.pt --perturb 1 --episodes 96 --seed 2 --no-h \
    --stats ../results/l1/wipe_m0_p1_s2.npz

# + 力环，真实压力，K = 0.4
WIPE_TASK=v2 python eval_irdp_wipe_fast.py --ckpt ../checkpoints/wipe_m3/policy.pt --perturb 1 --episodes 96 --seed 2 --no-h \
    --residual oracle --gain 0.4 --stats ../results/l1/wipe_m3_fast_oracle_g0.4_s2.npz

# 探针（要先在种子 1 上评一遍 m3、不加 --no-h，得到 wipe_m3_p1_h.npy）
WIPE_TASK=v2 python eval_irdp_wipe.py --ckpt ../checkpoints/wipe_m3/policy.pt --perturb 1 --episodes 96 --seed 1 --stats ../results/l1/wipe_m3_p1.npz
python fit_force_probe.py ../results/l1/wipe_m3_p1 ../results/probe_m3_force.npz

# 整个网格（now / ahead × K，oracle，噪声曲线），每条 96 回合约 8 分钟
bash ../scripts/run_fast_layer.sh
GAINS="0.4" ORACLE_GAINS="0.2 0.4" bash ../scripts/run_fast_layer.sh
NOISE_N="0.5 1 2 4" bash ../scripts/run_fast_layer.sh

# 分析
python mcnemar_fast.py ../results/l1
python noise_curve.py ../results/l1 ../results/noise_curve.png
python probe_error.py ../results/l1

# 录像（一个进程一段，Isaac Gym 一个进程只能开一个仿真）
WIPE_TASK=v2 python record_video.py record --label irdp --ckpt ../checkpoints/wipe_irdp_v2/policy.pt --residual none --seed 8 --out ../media/seed8/frames_irdp
WIPE_TASK=v2 python record_video.py record --label loop --ckpt ../checkpoints/wipe_m3/policy.pt --residual oracle --gain 0.4 --seed 8 --out ../media/seed8/frames_loop
python record_video.py compose --left ../media/seed8/frames_irdp --right ../media/seed8/frames_loop --out ../media/wipe
```

`results/l1/` 里是本文所有数字对应的 96 回合统计（`.npz` 逐回合，`.txt` 汇总）。

### 从头训（可选）

示教：`WIPE_TASK=v2 python wipe_expert.py --mode expert --episodes 960 --demos 480 --seed 0 --perturb 1 --record data/wipe_demos_v2.zarr`（脚本专家，板晃动，只留成功回合）。先用 `rdp_lite.py` 训一个基础 checkpoint（触觉 PCA + DP），再用 `implicit_rdp_lite.py --task wipe --base <基础> --calib <force_calib.pt>` 训 ImplicitRDP；参数见两个文件开头的 docstring。48000 步在 RTX 3060 上约 3.3 小时。

## 仓库结构

```
sim/                      仿真、专家、策略、评测、分析（都是本项目自己写的；资产见 THIRD_PARTY.md）
  insertion_env.py        基础环境：UR5 + 2F-85 + GelSight Mini（TacSL 触觉）
  wipe_env.py             擦板任务：晃动的板、相机估计误差、成功与压力规则
  wipe_expert.py          脚本专家（积分型力环）、回合统计与汇总
  rdp_lite.py             RDP 最小复现（触觉 PCA、DP、快慢两层）
  implicit_rdp_lite.py    ImplicitRDP 最小复现 + 历史 / 预测头改动
  eval_irdp_wipe.py       评测（可存每步 GRU 状态给探针用）
  eval_irdp_wipe_fast.py  评测 + 力环（now / ahead / oracle，噪声）
  fit_force_probe.py  mcnemar_fast.py  noise_curve.py  probe_error.py  record_video.py
  ur5_2f85_isaacgym/  objects/   机器人与板的 URDF / 网格
scripts/run_fast_layer.sh 整个网格的流水线
results/                  所有评测统计、探针权重、噪声曲线
media/                    GIF / mp4
```

## 来源与致谢

- ImplicitRDP：Chen et al., *ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning*, arXiv:2512.10946, 2025。本仓库是按论文的最小复现，不是官方代码。
- RDP：Xue et al., *Reactive Diffusion Policy*, RSS 2025。`ConditionalUnet1D` 来自其官方实现（MIT）。
- TacSL / IndustReal：Akinola, Xu et al., NVIDIA（IsaacGymEnvs，BSD-3）。
- 机器人资产：UR5（ROS-Industrial，BSD-3）、Robotiq 2F-85（ElectronicElephant 的 pybullet_ur5_robotiq，BSD）、GelSight Mini 网格。详见 `THIRD_PARTY.md`。

代码在 MIT 许可下。实验在 2026 年 10 月完成于个人设备，与任何机构无关。
