"""Run the ImplicitRDP policy of implicit_rdp_lite.py in the wiping scene (the counterpart of wipe_eval.py).

Consistent inference (the paper's Alg. 1) on the schedule of the DP and RDP runs: every N_ACTION steps the slow
observation and a fresh noise chunk are cached; that chunk takes over LATENCY steps later (eval_policy.py), and from
then on, at every control step, its steps so far are denoised again with all tactile readings since its
observation and the last one is executed. Until a new chunk takes over the old one keeps running the same way.

The fast loop is taken to finish inside the control step, like RDP's decoder; --time measures what it takes for
one robot on this machine (5 passes of the network on up to N_ACTION + LATENCY tokens). With the sensors on their
own clocks (insertion_env.py, timing="real") that time counts: the action of a control step is sent
fast_loop_steps() physics steps into it, and the previous action stays in force until then.

Usage (inside `conda activate tacsl`):
    python wipe_eval_implicit.py --ckpt runs/wipe_irdp/policy.pt --episodes 96 --perturb 1
    python wipe_eval_implicit.py --ckpt runs/wipe_irdp/policy.pt --time
"""
import argparse
import time

from wipe_env import ESTIMATE_NOISE, WipeEnv
import wipe_expert

import numpy as np
import torch

import eval_policy
import implicit_rdp_lite as irdp
from rdp_lite import HORIZON, N_ACTION, OBS_OFFSETS


FAST_LOOP_MS = (9.5, 0.74)      # ms: fixed part and per step of the chunk so far, one robot, RTX 3060 (laptop). Measured
                                # by --time: 11.7 ms for 3 steps and 22.8 ms for 18 right after training (GPU warm),
                                # 11.4 and 16.2 ms on a cool GPU; the slower pair is used, a control loop has to
                                # hold then too. That is 2 physics steps up to 9 steps of the chunk, 3 beyond


def fast_loop_steps(length):
    """Physics steps that pass until the fast loop has denoised `length` steps of the chunk (the arm's 125 Hz loop
    reads a command at its next cycle)."""
    return int(np.ceil((FAST_LOOP_MS[0] + FAST_LOOP_MS[1] * length) / (1000 * eval_policy.CONTROL_DT / 5)))


@torch.no_grad()
def run_episode(env, policy, latency=eval_policy.LATENCY, frozen=False, horizon=wipe_expert.HORIZON,
                stats_fn=wipe_expert.episode_stats):
    """One synchronous episode in every env. frozen: every fast token of a chunk is the tactile reading of the
    chunk's first step (what the network makes of it without feedback inside the chunk; it was not trained so)."""
    assert 0 <= latency < N_ACTION and N_ACTION + latency <= HORIZON
    real = getattr(env, "timing", "sync") == "real"
    obs = env.reset()
    hold_pos, hold_quat = obs["tcp_pos"].clone(), obs["tcp_quat"].clone()
    slip0 = obs["peg_in_hand"][:, 2].clone()
    max_slip = torch.zeros(env.num_envs, device=env.device)
    history, embs = [], []
    active = pending = None
    steps_to_success = torch.full((env.num_envs,), -1, device=env.device)
    for t in range(horizon):
        emb = policy.embed(obs["tactile"])
        embs.append(policy.norm_tactile(emb))
        history.append(policy.features(obs, emb))
        if t % N_ACTION == 0:      # slow loop: cache the slow observation and the noise of a new chunk
            slow = policy.norm_obs(torch.stack([history[max(t + off, 0)] for off in OBS_OFFSETS], dim=1))
            pending = dict(t0=t, base=(obs["tcp_pos"].clone(), obs["tcp_quat"].clone()), slow=slow,
                           noise=torch.randn((env.num_envs, HORIZON, irdp.AUG_DIM), device=env.device))
        if pending is not None and t == pending["t0"] + latency:
            active, pending = pending, None
        compute_steps = 0
        if active is None:         # nothing has arrived yet: hold still
            pos, quat = hold_pos, hold_quat
        else:                      # fast loop: the chunk so far, with every tactile reading since its observation
            k = t - active["t0"]
            tactile = torch.stack(embs[active["t0"]:t + 1], dim=1)
            if frozen:
                tactile = tactile[:, :1].expand(-1, k + 1, -1)
            chunk = policy.denoise(active["slow"], tactile, active["noise"][:, :k + 1])
            pos, quat = irdp.absolute_pose(*active["base"], chunk[:, -1, :irdp.POSE_DIM])
            compute_steps = fast_loop_steps(k + 1) if real else 0
        obs = env.step(pos, quat, compute_steps)
        max_slip = torch.maximum(max_slip, slip0 - obs["peg_in_hand"][:, 2])
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break
    return stats_fn(obs, max_slip, steps_to_success)


@torch.no_grad()
def time_fast_loop(policy, latency, device, repeats=200):
    """Seconds one fast-loop step takes for one robot, at the shortest and the longest chunk-so-far."""
    slow = torch.zeros((1, len(OBS_OFFSETS), policy.norm_obs.lo.shape[0]), device=device)
    out = {}
    for length in (latency + 1, N_ACTION + latency):
        tactile, noise = torch.zeros((1, length, irdp.TAC_DIM), device=device), torch.randn((1, length, irdp.AUG_DIM), device=device)
        for _ in range(20):
            policy.denoise(slow, tactile, noise)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeats):
            policy.denoise(slow, tactile, noise)
        torch.cuda.synchronize()
        out[length] = (time.time() - t0) / repeats
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/wipe_irdp/policy.pt")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--noise", type=float, default=ESTIMATE_NOISE)
    parser.add_argument("--height-noise", type=float, default=0.004)
    parser.add_argument("--latency", type=int, default=eval_policy.LATENCY, help="steps until a new chunk takes over")
    parser.add_argument("--perturb", type=float, default=0.0, help="fraction of envs in which the board shakes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frozen", action="store_true", help="no tactile feedback inside a chunk")
    parser.add_argument("--exact-camera", action="store_true")
    parser.add_argument("--fixed-grasp", action="store_true")
    parser.add_argument("--time", action="store_true", help="only measure the fast loop's computing time")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--stats", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = torch.load(args.ckpt, map_location="cuda:0")["policy"].eval()
    if args.time:
        for length, seconds in time_fast_loop(policy, args.latency, "cuda:0").items():
            print("fast loop, one robot, %2d steps of the chunk: %.1f ms (a control step is %.1f ms)"
                  % (length, 1000 * seconds, 1000 * eval_policy.CONTROL_DT))
        raise SystemExit
    env = WipeEnv(num_envs=1 if args.view else args.envs, headless=not args.view, seed=args.seed,
                  estimate_noise=args.noise, height_noise=args.height_noise, perturb=args.perturb,
                  camera_errors=not args.exact_camera, randomize_grasp=not args.fixed_grasp)
    runs = []
    for _ in range(int(np.ceil(args.episodes / env.num_envs))):
        runs.append(run_episode(env, policy, args.latency, args.frozen))
        if env.viewer_closed():
            break
    stats = {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}
    wipe_expert.summarize(stats, "policy implicit_rdp%s (latency %d)" % (" frozen" if args.frozen else "", args.latency))
    if args.stats:
        np.savez(args.stats, **stats)
    env.close()
