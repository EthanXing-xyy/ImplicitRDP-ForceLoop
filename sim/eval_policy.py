"""Run a trained policy from rdp_lite.py in the insertion scene and report its success rate.

  rdp         slow latent diffusion every 16 steps + fast tactile-conditioned decoder every step
  rdp_frozen  the same, but the fast decoder is fed the tactile reading from the start of the
              chunk (no feedback inside the chunk) -- shows what the fast loop contributes
  dp          diffusion policy baseline: a 32-step action chunk, 16 steps executed open loop

Inference delay: the slow policy (16 DDIM steps of the UNet) takes 70-80 ms for one robot on the
RTX 3060, i.e. LATENCY = 2 control steps at 24 Hz. It starts on the observation of step t, and its
chunk takes over at step t + LATENCY, from index LATENCY on (the official real_runner.py skips the
first latency_step actions the same way); until then the previous chunk keeps running (indices
16..17 of a 32-step chunk). The RDP decoder of the new chunk first catches up on the tactile readings
of the steps it missed. The fast decoder itself (one GRU step) takes well under a millisecond.
Tactile readings are one control step old and the arm follows its commands with the servoj lag;
both are part of the environment (insertion_env.py).

With the sensors on their own clocks (insertion_env.py, timing="real") the time of the fast loop counts too: RDP's
action of a control step is sent RDP_FAST_STEPS physics steps into it (the tactile embedding and one GRU step,
measured by wipe_eval.py --time; the arm's 125 Hz loop reads a command at its next cycle). DP's actions are
already there when the step begins.

Usage (inside `conda activate tacsl`):
    python eval_policy.py --policy rdp --episodes 96
    python eval_policy.py --policy rdp --view
"""
import argparse

from insertion_env import CONTROL_DT, InsertionEnv
from expert import HORIZON as EPISODE_STEPS, LOST_SLIP, summarize

import numpy as np
import torch

import rdp_lite
from rdp_lite import ACT_DIM, LATENT_C, LATENT_T, N_ACTION, OBS_OFFSETS

LATENCY = 2
RDP_FAST_STEPS = 1     # physics steps (8.3 ms each) until RDP's fast loop has its action: 0.7 ms measured, one robot, RTX 3060


@torch.no_grad()
def run_episode(env, policy, mode, latency=LATENCY, keep_orientation=False, horizon=EPISODE_STEPS, stats_fn=None):
    """One synchronous episode in every env. Other tasks pass their episode length and
    stats_fn(obs, max_slip, steps_to_success) -> dict of per-env arrays."""
    assert 0 <= latency < N_ACTION and N_ACTION + latency <= rdp_lite.HORIZON
    frozen = mode == "rdp_frozen"
    compute_steps = RDP_FAST_STEPS if mode != "dp" and getattr(env, "timing", "sync") == "real" else 0
    obs = env.reset()
    hold_pos, hold_quat = obs["tcp_pos"].clone(), obs["tcp_quat"].clone()
    slip0 = obs["peg_in_hand"][:, 2].clone()
    max_slip = torch.zeros(env.num_envs, device=env.device)
    min_slip = torch.zeros(env.num_envs, device=env.device)
    history, embs = [], []
    active = pending = None
    steps_to_success = torch.full((env.num_envs,), -1, device=env.device)
    for t in range(horizon):
        emb = policy.embed(obs["tactile"])
        embs.append(emb)
        history.append(policy.features(obs, emb))
        if t % N_ACTION == 0:      # slow policy, on the newest observation
            o = policy.norm_obs(torch.cat([history[max(t + off, 0)] for off in OBS_OFFSETS], dim=-1))
            pending = dict(t0=t, base=(obs["tcp_pos"].clone(), obs["tcp_quat"].clone()))
            if mode == "dp":
                pending["chunk"] = policy.norm_action.inverse(policy.dp.sample(o, rdp_lite.HORIZON, ACT_DIM))
            else:
                latent = policy.norm_latent.inverse(policy.slow.sample(o, LATENT_T, LATENT_C))
                pending["cond"], pending["hidden"] = policy.tokenizer.condition(latent), None
        if pending is not None and t == pending["t0"] + latency:    # its output arrives
            active, pending = pending, None
            for i in range(latency if mode != "dp" else 0):        # the decoder catches up on what it missed
                feedback = embs[active["t0"] if frozen else active["t0"] + i]
                _, active["hidden"] = policy.tokenizer.decode_step(active["cond"], policy.norm_tactile(feedback),
                                                                   active["hidden"])
        if active is None:         # nothing has arrived yet: hold still
            pos, quat = hold_pos, hold_quat
        else:
            k = t - active["t0"]
            if mode == "dp":
                relative = active["chunk"][:, k]
            else:                  # fast policy
                feedback = embs[active["t0"]] if frozen else emb
                action, active["hidden"] = policy.tokenizer.decode_step(active["cond"], policy.norm_tactile(feedback),
                                                                        active["hidden"])
                relative = policy.norm_action.inverse(action)
            pos, quat = rdp_lite.absolute_pose(*active["base"], relative)
            if keep_orientation:   # ablation: execute the positions only, the hand keeps the chunk-start orientation
                quat = active["base"][1]
        obs = env.step(pos, quat, 0 if active is None else compute_steps)
        slip = slip0 - obs["peg_in_hand"][:, 2]
        max_slip, min_slip = torch.maximum(max_slip, slip), torch.minimum(min_slip, slip)
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break
    if stats_fn is not None:
        return stats_fn(obs, max_slip, steps_to_success)
    seconds = steps_to_success.cpu().numpy() * CONTROL_DT
    return dict(
        success=obs["success"].cpu().numpy(),
        estimate_error=(obs["socket_estimate"][:, 0:2] - obs["socket_pos"][:, 0:2]).norm(dim=-1).cpu().numpy(),
        final_depth=obs["insertion_depth"].cpu().numpy(),
        final_lateral=obs["lateral_error"].cpu().numpy(),
        max_slip=max_slip.cpu().numpy(),
        lost_peg=(min_slip < -LOST_SLIP).cpu().numpy(),
        stopped=obs["stopped"].cpu().numpy(), gel_overload=obs["gel_overload"].cpu().numpy(),
        grasp_overload=obs["grasp_overload"].cpu().numpy(),
        peak_force=obs["peak_socket_force"].cpu().numpy(), peak_tool_force=obs["peak_tool_force"].cpu().numpy(),
        socket_moved=obs["socket_moved"].cpu().numpy(), move_called_off=obs["move_called_off"].cpu().numpy(),
        seconds=np.where(seconds > 0, seconds, np.nan),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("rdp", "rdp_frozen", "dp"), default="rdp")
    parser.add_argument("--ckpt", default="runs/rdp_lite/policy.pt")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.004)
    parser.add_argument("--height-noise", type=float, default=0.004)
    parser.add_argument("--latency", type=int, default=LATENCY, help="slow-policy inference delay in control steps")
    parser.add_argument("--keep-orientation", action="store_true", help="ablation: ignore the predicted rotations")
    parser.add_argument("--perturb", type=float, default=0.0, help="fraction of envs in which the socket moves after contact")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--stats", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = torch.load(args.ckpt, map_location="cuda:0")["policy"].eval()
    env = InsertionEnv(num_envs=1 if args.view else args.envs, headless=not args.view, seed=args.seed,
                       estimate_noise=args.noise, height_noise=args.height_noise, perturb=args.perturb)
    runs = []
    for _ in range(int(np.ceil(args.episodes / env.num_envs))):
        runs.append(run_episode(env, policy, args.policy, args.latency, args.keep_orientation))
        if env.viewer_closed():
            break
    stats = {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}
    summarize(stats, "policy %s (latency %d%s)" % (args.policy, args.latency, ", rotations ignored" if args.keep_orientation else ""))
    if args.stats:
        np.savez(args.stats, **stats)
    env.close()
