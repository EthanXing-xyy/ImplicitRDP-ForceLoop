"""Scripted wiping controllers for wipe_env.py, their scores, and demonstration recording.

Both controllers plan in the board's frame as the camera estimates it (late and a little wrong, see wipe_env.py):
the tool's tip is commanded at a point (along the line, off it, above the board) of that frame, so the hand
follows whatever the camera sees the board do. Neither is given the true pose of the board or of the peg.

  expert  goes over the estimated start of the line and lowers the tool slowly until it feels the board. Then it
          slides along the estimated line and holds the pressure by feel: it raises or lowers the tool in
          proportion to the force error, and it only moves forward while the pressure is right. The force it
          feels is the simulator's (what the gel shows as the peg being pushed up into the hand). This is the
          demonstrator: every move it makes along the board's normal is an answer to the force.
  blind   the same approach; it stops lowering when the robot's own force estimate says it has touched, presses a
          fixed distance further and slides along the estimated line, without looking at the force again. A robot
          with a camera and without tactile sensing can do this much.

Before contact both turn the hand so that the tool stands square on the estimated board with the gripper's
closing axis along the line; after contact the orientation is held.

Task "v2" (wipe_env.py): the line is an arc, which the controllers follow in the estimated board frame (the hand
keeps its orientation, closing axis along the arc's chord), and they wipe at WIPE_SPEED_V2. Its demonstrator is
  human   the expert as a person teleoperating would do it: it feels the force REACTION_TIME late and corrects
          the height more gently (HUMAN_FORCE_GAIN: a loop with that much dead time is not stable with the expert's
          gain), and no two demonstrations are alike: the force aimed at (HUMAN_FORCE_TARGET) and the speed
          (HUMAN_SPEED) are drawn for each. All values are placeholders.

Usage (inside `conda activate tacsl`):
    python wipe_expert.py --mode expert --episodes 96
    python wipe_expert.py --mode blind  --episodes 96 --perturb 1
    python wipe_expert.py --mode expert --view
    python wipe_expert.py --mode expert --episodes 480 --perturb 0.5 --record data/wipe_demos.zarr
"""
import argparse
import math
import os

from wipe_env import (ESTIMATE_NOISE, FORCE_LIMIT, FORCE_MAX, FORCE_MIN, LINE_LENGTH, OVERPRESSURE_TIME, SUCCESS_FRACTION,
                      TOUCH_FORCE, WipeEnv, arc_point)
from insertion_env import CONTROL_DT, PEG_BOTTOM, PEG_GRASP
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

import numpy as np
import torch

TIP = PEG_GRASP - PEG_BOTTOM   # m from the TCP to the tool's tip along the hand's axis, for a peg grasped where it is meant to be
HOVER = 0.015           # m above the estimated board: where the approach ends
APPROACH_SPEED = 0.04   # m/s, moving over the estimated start of the line
TURN_SPEED = np.radians(20.0)   # rad/s, turning the hand to the estimated board
DESCEND_SPEED = 0.015   # m/s, until the tool is near the board's estimated height
CONTACT_SPEED = 0.005   # m/s, in the last SLOW_ZONE before the estimated contact and until the board is felt
SLOW_ZONE = 0.008
WIPE_SPEED = 0.040      # m/s along the line (a careful robot; a person wipes at 100 mm/s and more)
WIPE_SPEED_V2 = 0.100   # m/s, task "v2"
REACTION_TIME = 5       # control steps (0.21 s) before the human demonstrator acts on a force
HUMAN_FORCE_GAIN = 0.0004   # m/s per N: with 0.2 s dead time, 0.1 s servo lag and ~6 N/mm the loop gain times its delay is
                            # ~0.8 (it oscillates from 1.57 on); the expert's gain would give 3
HUMAN_FORCE_TARGET = (8.0, 12.0)   # N
HUMAN_SPEED = (0.080, 0.120)       # m/s
FORCE_TARGET = 10.0     # N
ADVANCE_MIN, ADVANCE_MAX = FORCE_MIN + 1.0, FORCE_MAX - 2.0   # the expert moves forward only inside this (narrower) window
FORCE_GAIN = 0.0015     # m/s of tool height per N of force error. With ~6 N/mm between tool height and force
                        # (measured: diag/wipe_grip_hold.py) the error decays with a time constant of ~0.1 s. A 1-D
                        # model of the loop (servo lag, 24 Hz, camera jitter, contact noise) is stable from 3 to
                        # 12 N/mm with it, and it follows a board shaking by 2 mm
HEIGHT_SPEED_MAX = 0.030  # m/s
END_MARGIN = 0.003      # m: slide this far past the end of the line
LIFT, LIFT_SPEED = 0.015, 0.02
BLIND_TOUCH = 5.0       # N on the robot's own force estimate: touched
BLIND_PRESS = 0.0008    # m further down from there (5 N more at the nominal 6 N/mm)
HORIZON = 264           # control steps per episode (11 s at 24 Hz)

APPROACH, DESCEND, WIPE, LIFT_OFF, DONE = range(5)


def on_line(along, turn):
    """Where the line is `along` (N,) from its start, in the board's plane relative to the start, (N, 2). turn (N,):
    how far the line's direction turns over its length (rad; 0: straight). The frame's x axis is the direction of the chord."""
    kappa = turn / LINE_LENGTH
    half = torch.full_like(along, LINE_LENGTH / 2)
    return arc_point(along - half, kappa) - arc_point(-half, kappa)


def turn_towards(quat, goal, max_angle):
    """`quat` (N, 4) turned towards `goal` by at most `max_angle` (rad)."""
    delta = quat_mul(goal, quat_conjugate(quat))
    delta = delta * torch.where(delta[:, 3:4] < 0, -1.0, 1.0)
    sine = delta[:, 0:3].norm(dim=-1)
    angle = 2 * torch.atan2(sine, delta[:, 3])
    axis = delta[:, 0:3] / sine.unsqueeze(-1).clamp(min=1e-9)
    return quat_mul(quat_from_angle_axis(angle.clamp(max=max_angle), axis), quat)


def run_episode(env, mode, callback=None):
    """One synchronous episode in every env. Returns the per-step logs and per-env statistics.

    callback(t, obs, phase, tip) is called once per control step, before the action is applied; tip (N, 3) is the
    commanded tip in the estimated board frame (along the line from its start, off it, above the board).
    """
    obs = env.reset()
    n, dev = env.num_envs, env.device
    unit = lambda i: torch.eye(3, device=dev)[i].expand(n, 3)
    # the hand on the board: closing axis (TCP y) along the line (board x), tool axis (TCP z) into the board
    square = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]], device=dev).expand(n, 4)
    reach = unit(2) * TIP
    phase = torch.full((n,), APPROACH, device=dev)
    quat = obs["tcp_quat"].clone()
    # where the tip is in the estimated board frame, taking the peg to sit in the hand as intended
    in_board = lambda point, obs: quat_apply(quat_conjugate(obs["board_quat_estimate"]), point - obs["start_estimate"])
    tip = in_board(obs["tcp_pos"] + quat_apply(quat, reach), obs)
    slip0 = obs["peg_in_hand"][:, 2].clone()
    max_slip = torch.zeros(n, device=dev)
    in_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    pressed = torch.zeros(n, device=dev)            # blind: how far it has pressed down since touching
    lift_from = torch.zeros(n, device=dev)
    along = torch.zeros(n, device=dev)              # how far along the line the tip has been commanded
    turn = obs["line_turn"]
    speed_along = torch.full((n,), WIPE_SPEED_V2 if env.task == "v2" else WIPE_SPEED, device=dev)
    force_target, force_gain = torch.full((n,), FORCE_TARGET, device=dev), FORCE_GAIN
    if mode == "human":
        if not hasattr(env, "human_rng"):
            env.human_rng = torch.Generator(device="cpu").manual_seed(env.seed + 7000)
        draw = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=env.human_rng).to(dev)
        speed_along, force_target, force_gain = draw(*HUMAN_SPEED), draw(*HUMAN_FORCE_TARGET), HUMAN_FORCE_GAIN
    felt = [obs["normal_force"]] * (REACTION_TIME if mode == "human" else 0)
    steps_to_success = torch.full((n,), -1, device=dev)
    log = dict(tcp_pos=[], tcp_quat=[], socket_estimate=[], board_quat_estimate=[], line_turn=[], tactile=[], tactile_age=[], action_pos=[],
               action_quat=[], phase=[], wiped=[], normal_force=[], tip_along=[], perturbed=[])

    for t in range(HORIZON):
        felt.append(obs["normal_force"])
        force = felt.pop(0)                         # the human demonstrator's is REACTION_TIME old
        touched = force > TOUCH_FORCE if mode != "blind" else obs["tool_force"] > BLIND_TOUCH

        # --- phase transitions
        over_start = (tip - unit(2) * HOVER).norm(dim=-1) < 1e-4
        phase = torch.where((phase == APPROACH) & over_start, torch.full_like(phase, DESCEND), phase)
        phase = torch.where((phase == DESCEND) & touched, torch.full_like(phase, WIPE), phase)
        lifting = (phase == WIPE) & (along > LINE_LENGTH + END_MARGIN)
        lift_from = torch.where(lifting, tip[:, 2], lift_from)
        phase = torch.where(lifting, torch.full_like(phase, LIFT_OFF), phase)
        phase = torch.where((phase == LIFT_OFF) & (tip[:, 2] > lift_from + LIFT), torch.full_like(phase, DONE), phase)
        in_contact |= phase >= WIPE

        # --- the tip's motion in the estimated board frame for this control step
        move = torch.zeros_like(tip)
        d = unit(2) * HOVER - tip
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        move = torch.where((phase == APPROACH).unsqueeze(-1), d / dist * dist.clamp(max=APPROACH_SPEED * CONTROL_DT), move)
        speed = torch.where(tip[:, 2] < SLOW_ZONE, CONTACT_SPEED, DESCEND_SPEED)
        move[:, 2] = torch.where(phase == DESCEND, -speed * CONTROL_DT, move[:, 2])
        wiping = phase == WIPE
        if mode != "blind":
            advance = wiping & (force >= ADVANCE_MIN) & (force <= ADVANCE_MAX)           # forward only while the pressure is right
            height = (force_gain * (force - force_target)).clamp(-HEIGHT_SPEED_MAX, HEIGHT_SPEED_MAX) * CONTROL_DT
        else:
            pressing = wiping & (pressed < BLIND_PRESS)
            pressed = pressed + pressing * CONTACT_SPEED * CONTROL_DT
            advance = wiping & ~pressing
            height = -(pressing * CONTACT_SPEED * CONTROL_DT)
        further = along + wiping * advance * speed_along * CONTROL_DT
        move[:, 0:2] = torch.where(wiping.unsqueeze(-1), on_line(further, turn) - on_line(along, turn), move[:, 0:2])
        along = further
        move[:, 2] = torch.where(wiping, height, move[:, 2])
        move[:, 2] = torch.where(phase == LIFT_OFF, LIFT_SPEED * CONTROL_DT, move[:, 2])
        tip = tip + move

        # --- the hand: square on the estimated board until it touches, then as it is
        goal = quat_mul(obs["board_quat_estimate"], square)
        quat = torch.where(in_contact.unsqueeze(-1), quat, turn_towards(quat, goal, TURN_SPEED * CONTROL_DT))
        target = obs["start_estimate"] + quat_apply(obs["board_quat_estimate"], tip) - quat_apply(quat, reach)

        for key in ("tcp_pos", "tcp_quat", "socket_estimate", "board_quat_estimate", "line_turn", "tactile", "tactile_age", "wiped",
                    "normal_force", "tip_along"):
            log[key].append(obs[key].cpu().numpy())
        log["action_pos"].append(target.cpu().numpy())
        log["action_quat"].append(quat.cpu().numpy())
        log["phase"].append(phase.cpu().numpy())
        log["perturbed"].append(np.zeros(n, dtype=bool))      # no overridden steps here (expert.py's --kick has them)
        if callback is not None:
            callback(t, obs, phase, tip)

        obs = env.step(target, quat)
        max_slip = torch.maximum(max_slip, slip0 - obs["peg_in_hand"][:, 2])
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break

    return {k: np.stack(v, axis=1) for k, v in log.items()}, episode_stats(obs, max_slip, steps_to_success)  # (env, time, ...)


def undo_delay(reading, age):
    """Readings (env, T, ...) that were `age` seconds old (on average) when they were delivered, moved back to the
    control step they show, by interpolating between the two deliveries around it. Returns them for the steps
    0 .. T - k - 1, and k (the steps at the end that have no reading yet)."""
    delay = age / CONTROL_DT
    whole = int(math.floor(delay + 1e-6))
    part, k, steps = delay - whole, whole + 1, reading.shape[1]
    return (1 - part) * reading[:, whole:steps - k + whole] + part * reading[:, whole + 1:steps - k + whole + 1], k


def episode_stats(obs, max_slip, steps_to_success):
    """Per-env numbers at the end of an episode (also used by the policy evaluation)."""
    steps = steps_to_success.cpu().numpy()
    return dict(
        success=obs["success"].cpu().numpy(), wiped=obs["wiped"].cpu().numpy(), pressed_hard=obs["pressed_hard"].cpu().numpy(),
        contact_time=obs["contact_time"].cpu().numpy(), band_time=obs["band_time"].cpu().numpy(),
        hard_time=obs["hard_time"].cpu().numpy(), longest_hard=obs["longest_hard"].cpu().numpy(),
        peak_normal_force=obs["peak_normal_force"].cpu().numpy(), too_hard=obs["too_hard"].cpu().numpy(),
        score=obs["score"].cpu().numpy(), score_without_force_rules=obs["score_without_force_rules"].cpu().numpy(),
        peak_tool_force=obs["peak_tool_force"].cpu().numpy(), stopped=obs["stopped"].cpu().numpy(),
        gel_overload=obs["gel_overload"].cpu().numpy(), grasp_overload=obs["grasp_overload"].cpu().numpy(),
        shaken=obs["shaken"].cpu().numpy(), shake_normal=obs["shake_normal"].cpu().numpy(), max_slip=max_slip.cpu().numpy(),
        line_turn=obs["line_turn"].cpu().numpy(), board_friction=obs["board_friction"].cpu().numpy(),
        tilt=torch.rad2deg(torch.acos(quat_apply(obs["board_quat"], torch.tensor([[0.0, 0.0, 1.0]], device=obs["board_quat"].device)
                                                 .expand(len(max_slip), 3))[:, 2].clamp(max=1.0))).cpu().numpy(),
        estimate_error=(obs["start_estimate"] - obs["line_start"])[:, 0:2].norm(dim=-1).cpu().numpy(),
        height_error=(obs["start_estimate"] - obs["line_start"])[:, 2].cpu().numpy(),
        final_along=obs["tip_along"].cpu().numpy(),
        seconds=np.where(steps > 0, steps * CONTROL_DT, np.nan),
    )


def append_zarr(path, logs, stats, only_success=True):
    """Diffusion-policy style replay buffer, with the keys rdp_lite.py reads (as expert.append_zarr)."""
    import zarr

    root = zarr.open(path, mode="a")
    data, meta = root.require_group("data"), root.require_group("meta")
    keys = ("tcp_pos", "tcp_quat", "socket_estimate", "board_quat_estimate", "action_pos", "action_quat", "phase",
            "perturbed", "wiped", "normal_force", "tip_along", "tactile_age", "line_turn")
    ends = list(meta["episode_ends"][:]) if "episode_ends" in meta else []
    total = ends[-1] if ends else 0
    kept = 0
    for i in range(len(stats["success"])):
        if only_success and not stats["success"][i]:
            continue
        kept += 1
        episode = {name: logs[name][i] for name in keys}
        tactile = logs["tactile"][i]                       # (T, 2, points, 3)
        episode["tactile_left"], episode["tactile_right"] = tactile[:, 0], tactile[:, 1]
        for name, arr in episode.items():
            arr = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
            if name in data:
                data[name].append(arr, axis=0)
            else:
                data.create_dataset(name, data=arr, chunks=(HORIZON,) + arr.shape[1:])
        total += len(episode["tcp_pos"])
        ends.append(total)
    if "episode_ends" in meta:
        del meta["episode_ends"]
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    return kept


def summarize(stats, label):
    """Share of the line wiped, how the pressure was held, force events, and the effect of the perturbation."""
    s, n = stats, len(stats["success"])
    ok = s["success"].astype(bool)
    print("%s  episodes %d  score %.3f (1: %d, 0.5: %d, 0: %d)  success %d (%.1f%%)  score without the force rules %.3f  "
          "line wiped: mean %.1f%%, median %.1f%%, 10th percentile %.1f%%"
          % (label, n, s["score"].mean(), (s["score"] == 1).sum(), (s["score"] == 0.5).sum(), (s["score"] == 0).sum(), ok.sum(),
             100 * ok.mean(), s["score_without_force_rules"].mean(), 100 * s["wiped"].mean(), 100 * np.median(s["wiped"]),
             100 * np.quantile(s["wiped"], 0.1)))
    touched = s["contact_time"] > 0
    share = lambda x: 100 * np.median(x[touched] / s["contact_time"][touched]) if touched.any() else np.nan
    print("   on the board in %d episodes; of the time on it, pressure inside %.0f-%.0f N: median %.0f%%, above: median %.0f%%; "
          "line passed over with too much force: mean %.1f%%; time to success median %.1f s"
          % (touched.sum(), FORCE_MIN, FORCE_MAX, share(s["band_time"]), share(s["hard_time"]), 100 * s["pressed_hard"].mean(),
             np.nanmedian(s["seconds"]) if ok.any() else np.nan))
    wiped_enough, long_press, over = s["wiped"] >= SUCCESS_FRACTION, s["longest_hard"] > OVERPRESSURE_TIME, s["peak_normal_force"] > FORCE_LIMIT
    print("   not a score of 1 because: line not wiped %d, above %.0f N for more than %.1f s in one go %d, above %.0f N %d "
          "(several can apply); longest press above the window: median %.2f s, 90th percentile %.2f s, max %.2f s; "
          "tool slipped more than 2 mm in the hand in %d episodes"
          % ((~wiped_enough).sum(), FORCE_MAX, OVERPRESSURE_TIME, long_press.sum(), FORCE_LIMIT, over.sum(),
             np.median(s["longest_hard"]), np.quantile(s["longest_hard"], 0.9), s["longest_hard"].max(), (s["max_slip"] > 0.002).sum()))
    print("   protective stops %d, gel overloads %d (plus %d during the grasp, not counted); pressure peak: median %.0f N, "
          "90th percentile %.0f N, max %.0f N; force felt by the robot: max %.0f N; peg pushed up in the hand: median %.1f mm"
          % (s["stopped"].sum(), s["gel_overload"].sum(), s["grasp_overload"].sum(), np.median(s["peak_normal_force"]),
             np.quantile(s["peak_normal_force"], 0.9), s["peak_normal_force"].max(), s["peak_tool_force"].max(),
             1000 * np.median(s["max_slip"])))
    shaken = s["shaken"].astype(bool)
    for name, sel in (("board shaking", shaken), ("board still", ~shaken)):
        if shaken.any() and sel.any():
            print("   %s in %d episodes: score %.3f, wiped mean %.1f%%, success %.0f%%"
                  % (name, sel.sum(), s["score"][sel].mean(), 100 * s["wiped"][sel].mean(), 100 * ok[sel].mean()))
    for lo, hi in ((0, 5), (5, 10), (10, 15)):
        sel = (s["tilt"] >= lo) & (s["tilt"] < hi + (hi == 15))
        if sel.any():
            print("   board tipped by %2d..%2d deg: %3d episodes, wiped mean %.0f%%, success %.0f%%"
                  % (lo, hi, sel.sum(), 100 * s["wiped"][sel].mean(), 100 * ok[sel].mean()))
    for lo, hi in ((-4, -2), (-2, 0), (0, 2), (2, 4)):
        sel = (s["height_error"] >= lo / 1000) & (s["height_error"] < hi / 1000)
        if sel.any():
            print("   height estimate off by %+d..%+d mm: %3d episodes, wiped mean %.0f%%, success %.0f%%"
                  % (lo, hi, sel.sum(), 100 * s["wiped"][sel].mean(), 100 * ok[sel].mean()))


def evaluate(env, mode, episodes, record=None, verbose=True, demos=0):
    """demos > 0 (with record): go on until that many successful episodes have been recorded (at most `episodes`)."""
    all_stats, kept = [], 0
    for _ in range(int(np.ceil(episodes / env.num_envs))):
        logs, stats = run_episode(env, mode)
        if record:
            kept += append_zarr(record, logs, stats)
        all_stats.append(stats)
        if env.viewer_closed() or (demos and kept >= demos):
            break
    if record:
        print("recorded %d successful episodes" % kept)
    stats = {k: np.concatenate([s[k] for s in all_stats]) for k in all_stats[0]}
    if verbose:
        summarize(stats, "mode " + mode)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("blind", "expert", "human"), default="expert")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--noise", type=float, default=ESTIMATE_NOISE, help="max error (m) of the line-start estimate in the plane")
    parser.add_argument("--height-noise", type=float, default=0.004, help="max error (m) of its height")
    parser.add_argument("--perturb", type=float, default=0.0, help="fraction of envs in which the board shakes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-camera", action="store_true", help="ablation: the camera's estimate of the board has no error")
    parser.add_argument("--fixed-grasp", action="store_true", help="ablation: the tool sits in the hand the same way every time")
    parser.add_argument("--view", action="store_true", help="one env with a viewer window")
    parser.add_argument("--record", default="", help="append successful episodes to this zarr store")
    parser.add_argument("--demos", type=int, default=0, help="with --record: stop once this many successful episodes are recorded")
    parser.add_argument("--stats", default="", help="save per-episode statistics to this .npz")
    args = parser.parse_args()

    env = WipeEnv(num_envs=1 if args.view else args.envs, headless=not args.view, seed=args.seed,
                  estimate_noise=args.noise, height_noise=args.height_noise, perturb=args.perturb,
                  camera_errors=not args.exact_camera, randomize_grasp=not args.fixed_grasp)
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    stats = evaluate(env, args.mode, args.episodes, record=args.record, demos=args.demos)
    if args.stats:
        np.savez(args.stats, **stats)
    env.close()
