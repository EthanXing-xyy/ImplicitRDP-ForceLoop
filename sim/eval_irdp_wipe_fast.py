"""擦板评测 + 快层参照（C 类）：eval_irdp_wipe.py 的改版，策略照常出动作块，再加一个每步的快层残差。

快层：从 GRU 状态 h_t 用线性探针读压力 f̂（--residual now：现在的压力；ahead：4 步后的压力，= 控制环死区），
      在估计的板法向上加残差 d = clip(gain · (f̂ − target), ±clip)，压力（预计）高了抬、低了压，只在已接触（f̂_now > contact）时生效。
      探针权重来自 fit_force_probe.py（在另一个评测种子的 h 上拟合）。--residual none 就是原评测；
      oracle = 用仿真的真实压力代替探针（读数的上限，也是真机上 XELA 直接给力的情形）；--noise-n 在 oracle 读数上加每步独立的高斯白噪声（N），
      画"成功率 对 力读数误差"的曲线，给融合指尖定精度指标。
问题：同一条力环，用 4 步提前量比用当前压力好吗？两者比不加好吗？
    WIPE_TASK=v2 python eval_irdp_wipe_fast.py --ckpt ../runs/wipe_m3/policy.pt --perturb 1 --episodes 96 --seed 2 --no-h \
        --residual ahead --gain 0.2 --probe ../runs/probe_m3_force.npz --stats ../runs/l1/wipe_m3_fast_ahead_g0.2_s2.npz
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
import implicit_rdp_lite as irdp  # noqa: E402  （本目录的改版）
from rdp_lite import HORIZON, N_ACTION, OBS_OFFSETS, quat_to_matrix  # noqa: E402
from wipe_eval_implicit import fast_loop_steps  # noqa: E402
from isaacgym.torch_utils import quat_apply  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def here(p):
    return p if os.path.isabs(p) else os.path.join(ORIG_CWD, p)


def load_probe(path, mode, device):
    d = np.load(here(path))
    use = "now" if mode == "now" else "ahead"
    t = lambda k: torch.tensor(np.asarray(d[k], dtype=np.float32), device=device)
    return dict(w_now=t("w_now"), b_now=t("b_now"), w_use=t("w_" + use), b_use=t("b_" + use), k_ahead=int(d["k_ahead"]))


@torch.no_grad()
def run_episode(env, policy, fast, latency=eval_policy.LATENCY, horizon=wipe_expert.HORIZON, log_h=False, on_step=None):
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
    f_now_log = torch.full((n, horizon), float("nan"), device=dev)      # 快层读到的当前压力
    f_use_log = torch.full((n, horizon), float("nan"), device=dev)      # 快层拿来算残差的压力（now 或 ahead）
    d_log = torch.zeros((n, horizon), device=dev)                         # 残差（m，沿板法向，正 = 离开板）
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    cmds = []
    for t in range(horizon):
        emb = policy.embed(obs["tactile"])
        embs.append(policy.norm_tactile(emb))
        history.append(policy.features(obs, emb))
        shake[:, t] = env.targets[:, env.stage_dofs]
        pressure[:, t] = obs["normal_force"]
        inc = cmds[-1] - cmds[-2] if len(cmds) >= 2 else torch.zeros((n, 3), device=dev)
        us.append((quat_to_matrix(obs["tcp_quat"]).transpose(-1, -2) @ inc.unsqueeze(-1)).squeeze(-1))
        if t % N_ACTION == 0:
            slow = policy.norm_obs(torch.stack([history[max(t + off, 0)] for off in OBS_OFFSETS], dim=1))
            pending = dict(t0=t, base=(obs["tcp_pos"].clone(), obs["tcp_quat"].clone()), slow=slow,
                           noise=torch.randn((n, HORIZON, irdp.AUG_DIM), device=dev))
        if pending is not None and t == pending["t0"] + latency:
            active, pending = pending, None
        compute_steps = 0
        if active is None:
            pos, quat = hold_pos, hold_quat
        else:
            t0 = active["t0"]
            k = t - t0
            seq = torch.stack(embs[max(t0 - H, 0):t + 1], dim=1)
            useq = torch.stack(us[max(t0 - H, 0):t + 1], dim=1)
            valid = torch.ones(seq.shape[:2], dtype=torch.bool, device=dev)
            if H and seq.shape[1] < H + k + 1:
                pad = H + k + 1 - seq.shape[1]
                seq = torch.cat([torch.zeros((n, pad, seq.shape[2]), device=dev), seq], 1)
                useq = torch.cat([torch.zeros((n, pad, 3), device=dev), useq], 1)
                valid = torch.cat([torch.zeros((n, pad), dtype=torch.bool, device=dev), valid], 1)
            unorm = policy.norm_u(useq) if eff else None
            chunk = policy.denoise(active["slow"], seq, active["noise"][:, :k + 1], valid, unorm)
            pos, quat = irdp.absolute_pose(*active["base"], chunk[:, -1, :irdp.POSE_DIM])
            h = policy.fast_states(seq, valid, unorm)
            if log_h:
                hs[:, t] = h[:, -1].half()
            if h.shape[1] >= 2:
                surprise[:, t] = (h[:, -1] - h[:, -2]).norm(dim=-1)
            if fast is not None:                                           # 快层参照：探针读压力，法向残差
                if fast["mode"] == "oracle":
                    f_now = f_use = obs["normal_force"].to(pos.device).float()
                    if fast["noise_n"] > 0:                                # 读数噪声：单独的生成器，不动策略的噪声序列（保持回合可配对）
                        f_now = f_use = f_now + fast["noise_n"] * torch.randn(f_now.shape, generator=fast["gen"], device=f_now.device)
                else:
                    hl = h[:, -1].float()
                    f_now = hl @ fast["w_now"] + fast["b_now"]
                    f_use = hl @ fast["w_use"] + fast["b_use"]
                touching = f_now > fast["contact"]
                d = (fast["gain"] * (f_use - fast["target"])).clamp(-fast["clip"], fast["clip"]) * touching
                normal = quat_apply(obs["board_quat_estimate"], ez)
                pos = pos + d.unsqueeze(-1) * normal
                f_now_log[:, t], f_use_log[:, t], d_log[:, t] = f_now, f_use, d
            compute_steps = fast_loop_steps(k + 1) if real else 0
        cmds.append(pos.clone())
        obs = env.step(pos, quat, compute_steps)
        if on_step is not None:                                            # 录像 / 调试：每步的观测、快层读数和残差
            on_step(t, obs, f_use_log[:, t], d_log[:, t])
        max_slip = torch.maximum(max_slip, slip0 - obs["peg_in_hand"][:, 2])
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break
    stats = wipe_expert.episode_stats(obs, max_slip, steps_to_success)
    stats.update(surprise=surprise.cpu().numpy().astype(np.float16), board_shake=shake.cpu().numpy().astype(np.float32),
                 normal_force=pressure.cpu().numpy().astype(np.float32), fast_f_now=f_now_log.cpu().numpy().astype(np.float32),
                 fast_f_use=f_use_log.cpu().numpy().astype(np.float32), fast_d=d_log.cpu().numpy().astype(np.float32))
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
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-h", action="store_true")
    parser.add_argument("--residual", choices=("none", "now", "ahead", "oracle"), default="none",
                        help="快层用哪个压力算残差：探针读的现在 / 4 步后，或 oracle = 仿真真实压力")
    parser.add_argument("--gain", type=float, default=0.2, help="mm 每 N：残差 = gain · (f̂ − target)")
    parser.add_argument("--clip", type=float, default=3.0, help="mm：残差上限")
    parser.add_argument("--target", type=float, default=10.0, help="N：压力目标（窗口 5–15 的中点）")
    parser.add_argument("--contact", type=float, default=2.0, help="N：探针读到的当前压力超过它才算接触、才加残差")
    parser.add_argument("--probe", default="", help="fit_force_probe.py 存的探针权重")
    parser.add_argument("--noise-n", type=float, default=0.0, help="N：oracle 读数上加的高斯白噪声标准差（每步独立）")
    parser.add_argument("--stats", default="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    policy = torch.load(here(args.ckpt), map_location="cuda:0")["policy"].eval()
    fast = None
    if args.residual == "oracle":
        fast = dict(k_ahead=0)
    elif args.residual != "none":
        assert args.probe, "--residual 要配 --probe"
        fast = load_probe(args.probe, args.residual, "cuda:0")
    if fast is not None:
        fast.update(gain=args.gain * 1e-3, clip=args.clip * 1e-3, target=args.target, contact=args.contact, mode=args.residual,
                    noise_n=args.noise_n, gen=torch.Generator(device="cuda:0").manual_seed(1000 + args.seed))
    env = WipeEnv(num_envs=args.envs, headless=True, seed=args.seed, estimate_noise=args.noise, height_noise=args.height_noise,
                  perturb=args.perturb, camera_errors=True, randomize_grasp=True)
    runs = []
    for _ in range(int(np.ceil(args.episodes / env.num_envs))):
        runs.append(run_episode(env, policy, fast, args.latency, log_h=not args.no_h))
        if env.viewer_closed():
            break
    stats = {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}
    label = "policy implicit_rdp history %d efference %s pred_w %.3f + fast residual %s gain %.2f mm/N clip %.1f target %.0f noise %.1f N (latency %d, wipe %s, perturb %.1f, seed %d)" % (
        getattr(policy, "history", 0), getattr(policy, "efference", False), getattr(policy, "pred_w", 0.0), args.residual, args.gain,
        args.clip, args.target, args.noise_n, args.latency, os.environ.get("WIPE_TASK", "v1"), args.perturb, args.seed)
    wipe_expert.summarize({k: v for k, v in stats.items() if k not in ("surprise", "board_shake", "normal_force", "h", "fast_f_now", "fast_f_use", "fast_d")}, label)
    if fast is not None:
        fn, fu, d, tr = stats["fast_f_now"], stats["fast_f_use"], stats["fast_d"], stats["normal_force"]
        ok = ~np.isnan(fn)
        k = fast["k_ahead"] if args.residual == "ahead" else 0
        tgt = np.concatenate([tr[:, k:], np.full((tr.shape[0], k), np.nan)], 1) if k else tr
        m = ok & ~np.isnan(tgt)
        r2 = 1 - ((fu[m] - tgt[m]) ** 2).sum() / ((tgt[m] - tgt[m].mean()) ** 2).sum()
        act = d != 0
        print("   fast layer: probe %s vs true pressure R2 %.3f (rmse %.2f N) on this run; residual active %.0f%% of steps, |d| mean %.2f mm, "
              "90th pct %.2f mm, clipped %.0f%% of active steps"
              % (args.residual, r2, np.sqrt(((fu[m] - tgt[m]) ** 2).mean()), 100 * act.mean(), 1000 * np.abs(d[act]).mean() if act.any() else 0,
                 1000 * np.quantile(np.abs(d[act]), 0.9) if act.any() else 0, 100 * (np.abs(d[act]) >= fast["clip"] - 1e-9).mean() if act.any() else 0))
    s = stats["surprise"].astype(np.float32)
    print("   surprise ||h_t - h_{t-1}||: median %.4f, 90th pct %.4f" % (np.nanmedian(s), np.nanquantile(s, 0.9)))
    if args.stats:
        os.makedirs(os.path.dirname(os.path.abspath(here(args.stats))), exist_ok=True)
        h = stats.pop("h", None)
        np.savez_compressed(here(args.stats), **stats)
        if h is not None:
            np.save(here(args.stats)[:-4] + "_h.npy", h)
    env.close()
