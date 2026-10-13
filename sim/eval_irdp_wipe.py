"""擦板任务上评测 ImplicitRDP 及其 C49 变体（wipe_eval_implicit.py 的改版；时序、一致推理、快环计时逐字沿用）。

C49 多出来的：
  history>0 的模型   每步把块起点之前 H 帧触觉也喂给 GRU（回合开头不足的填零 + 无读数标志）
  efference 的模型   GRU 输入带每步发出的命令位置增量（TCP 系），评测时用策略自己发出的命令算
  诊断记录            h (n, T, 512) float16：每步 GRU 最新状态；surprise (n, T)：||h_t - h_{t-1}||；
                      board_shake (n, T, 5)：板晃动的真实台面指令（相机看不见的那部分）；normal_force (n, T)：真实压力。
                      analyze.py 用线性探针看状态里能不能读出板的运动。
    WIPE_TASK=v2 python eval_irdp_wipe.py --ckpt ../runs/wipe_m2/policy.pt --perturb 1 --episodes 96 --seed 1 --stats ../runs/l1/wipe_m2_p1.npz
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CWD = os.getcwd()
sys.path.insert(0, HERE)
os.chdir(HERE)                                                     # 环境资产（objects/、ur5_2f85_isaacgym/）按相对路径找
os.environ.setdefault("WIPE_TASK", "v2")
from wipe_env import ESTIMATE_NOISE, WipeEnv  # noqa: E402
import wipe_expert  # noqa: E402
import eval_policy  # noqa: E402
import implicit_rdp_lite as irdp  # noqa: E402  （本目录的改版；旧 checkpoint 也能载入）
from rdp_lite import HORIZON, N_ACTION, OBS_OFFSETS, quat_to_matrix  # noqa: E402
from wipe_eval_implicit import fast_loop_steps  # noqa: E402  （快环计时，原版）

import numpy as np  # noqa: E402
import torch  # noqa: E402


def here(p):
    return p if os.path.isabs(p) else os.path.join(ORIG_CWD, p)


@torch.no_grad()
def run_episode(env, policy, latency=eval_policy.LATENCY, horizon=wipe_expert.HORIZON, log_h=True):
    assert 0 <= latency < N_ACTION and N_ACTION + latency <= HORIZON
    H, eff = int(getattr(policy, "history", 0)), bool(getattr(policy, "efference", False))
    real = getattr(env, "timing", "sync") == "real"
    obs = env.reset()
    n, dev = env.num_envs, env.device
    hold_pos, hold_quat = obs["tcp_pos"].clone(), obs["tcp_quat"].clone()
    slip0 = obs["peg_in_hand"][:, 2].clone()
    max_slip = torch.zeros(n, device=dev)
    history, embs, us = [], [], []
    active = pending = None
    steps_to_success = torch.full((n,), -1, device=dev)
    surprise = torch.full((n, horizon), float("nan"), device=dev)
    hs = torch.zeros((n, horizon, irdp.RNN_HIDDEN), device=dev, dtype=torch.float16) if log_h else None
    shake = torch.zeros((n, horizon, len(env.stage_dofs)), device=dev)
    pressure = torch.zeros((n, horizon), device=dev)
    cmds = []                  # 每步发出的命令位置
    for t in range(horizon):
        emb = policy.embed(obs["tactile"])
        embs.append(policy.norm_tactile(emb))
        history.append(policy.features(obs, emb))
        shake[:, t] = env.targets[:, env.stage_dofs]
        pressure[:, t] = obs["normal_force"]
        # 传出副本：上一步发出的命令相对再上一步的增量，写在这一步的 TCP 系下（和训练侧 Windows.u 一致）
        inc = cmds[-1] - cmds[-2] if len(cmds) >= 2 else torch.zeros((n, 3), device=dev)
        us.append((quat_to_matrix(obs["tcp_quat"]).transpose(-1, -2) @ inc.unsqueeze(-1)).squeeze(-1))
        if t % N_ACTION == 0:      # slow loop: cache the slow observation and the noise of a new chunk
            slow = policy.norm_obs(torch.stack([history[max(t + off, 0)] for off in OBS_OFFSETS], dim=1))
            pending = dict(t0=t, base=(obs["tcp_pos"].clone(), obs["tcp_quat"].clone()), slow=slow,
                           noise=torch.randn((n, HORIZON, irdp.AUG_DIM), device=dev))
        if pending is not None and t == pending["t0"] + latency:
            active, pending = pending, None
        compute_steps = 0
        if active is None:         # nothing has arrived yet: hold still
            pos, quat = hold_pos, hold_quat
        else:                      # fast loop: the chunk so far, with every tactile reading since its observation (+ history)
            t0 = active["t0"]
            k = t - t0
            seq = torch.stack(embs[max(t0 - H, 0):t + 1], dim=1)                       # (n, h+k+1, 30)
            useq = torch.stack(us[max(t0 - H, 0):t + 1], dim=1)
            valid = torch.ones(seq.shape[:2], dtype=torch.bool, device=dev)
            if H and seq.shape[1] < H + k + 1:                                         # 回合开头：历史不足，前面填零
                pad = H + k + 1 - seq.shape[1]
                seq = torch.cat([torch.zeros((n, pad, seq.shape[2]), device=dev), seq], 1)
                useq = torch.cat([torch.zeros((n, pad, 3), device=dev), useq], 1)
                valid = torch.cat([torch.zeros((n, pad), dtype=torch.bool, device=dev), valid], 1)
            unorm = policy.norm_u(useq) if eff else None
            chunk = policy.denoise(active["slow"], seq, active["noise"][:, :k + 1], valid, unorm)
            pos, quat = irdp.absolute_pose(*active["base"], chunk[:, -1, :irdp.POSE_DIM])
            h = policy.fast_states(seq, valid, unorm)                                  # (n, k+1, 512)
            if log_h:
                hs[:, t] = h[:, -1].half()
            if h.shape[1] >= 2:
                surprise[:, t] = (h[:, -1] - h[:, -2]).norm(dim=-1)
            compute_steps = fast_loop_steps(k + 1) if real else 0
        cmds.append(pos.clone())
        obs = env.step(pos, quat, compute_steps)
        max_slip = torch.maximum(max_slip, slip0 - obs["peg_in_hand"][:, 2])
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break
    stats = wipe_expert.episode_stats(obs, max_slip, steps_to_success)
    stats.update(surprise=surprise.cpu().numpy().astype(np.float16), board_shake=shake.cpu().numpy().astype(np.float32),
                 normal_force=pressure.cpu().numpy().astype(np.float32))
    if log_h:
        stats["h"] = hs.cpu().numpy()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--noise", type=float, default=ESTIMATE_NOISE)
    parser.add_argument("--height-noise", type=float, default=0.004)
    parser.add_argument("--latency", type=int, default=eval_policy.LATENCY)
    parser.add_argument("--perturb", type=float, default=0.0, help="板晃动的 env 比例")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-h", action="store_true", help="不存每步的 GRU 状态（省 npz 体积）")
    parser.add_argument("--stats", default="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    policy = torch.load(here(args.ckpt), map_location="cuda:0")["policy"].eval()
    env = WipeEnv(num_envs=args.envs, headless=True, seed=args.seed, estimate_noise=args.noise, height_noise=args.height_noise,
                  perturb=args.perturb, camera_errors=True, randomize_grasp=True)
    runs = []
    for _ in range(int(np.ceil(args.episodes / env.num_envs))):
        runs.append(run_episode(env, policy, args.latency, log_h=not args.no_h))
        if env.viewer_closed():
            break
    stats = {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}
    label = "policy implicit_rdp history %d efference %s pred_w %.3f (latency %d, wipe %s, perturb %.1f)" % (
        getattr(policy, "history", 0), getattr(policy, "efference", False), getattr(policy, "pred_w", 0.0), args.latency,
        os.environ.get("WIPE_TASK", "v1"), args.perturb)
    wipe_expert.summarize({k: v for k, v in stats.items() if k not in ("surprise", "board_shake", "normal_force", "h")}, label)
    s = stats["surprise"].astype(np.float32)
    print("   surprise ||h_t - h_{t-1}||: median %.4f, 90th pct %.4f" % (np.nanmedian(s), np.nanquantile(s, 0.9)))
    if args.stats:
        os.makedirs(os.path.dirname(os.path.abspath(here(args.stats))), exist_ok=True)
        h = stats.pop("h", None)            # h 很大（96×264×512），单独存 <stats>_h.npy（不入库）；analyze.py 会去找它
        np.savez_compressed(here(args.stats), **stats)
        if h is not None:
            np.save(here(args.stats)[:-4] + "_h.npy", h)
    env.close()
