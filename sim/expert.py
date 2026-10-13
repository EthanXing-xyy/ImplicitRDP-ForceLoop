"""Scripted peg-insertion controllers, their success rates, and demonstration recording.

  blind   go to the *estimated* socket position and push straight down until the resistance lasts
          (half a second above BLIND_FORCE: the chamfer resists too, but only for a moment).
  expert  the same start, but once the peg lands on the socket rim it slides toward the true
          hole (ground truth from the simulator) and then inserts. If the peg jams on the way in it
          lifts it a few mm, brings the hand over the hole axis and tries again (RETRIES times). This is
          the demonstrator. Before contact both controllers turn the hand so that the peg is parallel
          to the hole; after contact the orientation is held.

Usage (inside `conda activate tacsl`):
    python expert.py --mode blind  --episodes 64            # success rate
    python expert.py --mode expert --episodes 64
    python expert.py --mode expert --view                   # watch one env
    python expert.py --mode expert --episodes 200 --record data/insertion_demos.zarr
    python expert.py --mode expert --episodes 200 --record data/insertion_demos.zarr --kick 1.0

--kick: in that fraction of the envs, shortly after the peg lands on the rim, the expert is overridden
for a moment and the peg is dragged in a random direction. The expert then recovers. These steps are
marked `perturbed` in the recording and are never used as training targets; what they add is the
states that follow a wrong move, where the drag felt by the sensors does not point at the hole.
"""
import argparse
import os

from insertion_env import CONTROL_DT, SOCKET_DEPTH, InsertionEnv
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

import numpy as np
import torch
import torch.nn.functional as F

APPROACH_SPEED = 0.04   # m/s, moving over the estimated socket position
DESCEND_SPEED = 0.015   # m/s, until the peg is near the socket's estimated height
CONTACT_SPEED = 0.005   # m/s, in the last SLOW_ZONE before the estimated contact (a guarded move: the real arm
SLOW_ZONE = 0.010       # is stiff, hitting the rim at full speed would trip its force limit)
SEARCH_SPEED = 0.004    # m/s, sliding on the rim
RIM_SLIP = 0.0004       # the peg pushed this far up in the hand (m) = it is resting on something
BOTTOM_SLIP = 0.0003    # inside the hole, pushed up this much more = it has reached the bottom; the expert stops here
BOTTOM_FORCE = 10.0     # N: or when the peg meets this much resistance inside the hole (what the gel would feel)
JAM_DEPTH = 0.8 * SOCKET_DEPTH   # resistance with the peg less deep than this (the task's success depth) is a jam, deeper
# is good enough. The demonstrator sees how far the peg is in; judged by travel alone, jams at 19-20 mm were taken
# for the bottom, and with a limit near the true bottom, pegs that were deep enough were pulled up again.
OFF_AGAIN = 0.002       # m: in INSERT but the peg is not in the hole and this far off its axis: search again. (More than
# the chamfer can take up, 1.5 mm: standing a tilted peg up moves its tip ~1.1 mm off, and 1 mm here made the expert
# switch between SEARCH and INSERT without end.)
RETRIES = 5             # how often a jam is answered by lifting LIFT and trying again (3 ran out with the peg still advancing)
LIFT = 0.003
SLOW_TRAVEL = 0.015     # after this much travel inside the hole the bottom is near: creep at CREEP_SPEED
CREEP_SPEED = 0.002
BLIND_FORCE = 20.0      # N: the blind controller stops after half a second of resistance above this
MAX_SLIP = 0.004        # give up pushing beyond this
LOST_SLIP = 0.002       # the peg stayed behind while the hand moved up: the hand has come off it
HORIZON = 360           # control steps per episode (15 s at 24 Hz; the guarded approach takes time, and so do
                        # retries and searching again after the socket has moved: 12 s cut off 6 of 96 episodes)
CONTACT_FORCE = 2.0     # N on the socket: the peg touches it (as the environment's PERTURB_CONTACT_FORCE)
MAX_TURN = np.radians(5.0)  # the hand never turns further than this from its start orientation. Aligning the peg needs
# at most ~2.5 deg (socket tilt + grasp tilt). The hand aligns the peg with the hole only before contact: a peg whose
# tip is held by the socket pivots between the pads, and a hand that chased it would never catch up (hand turns, peg
# stays, hand turns further: ~100 deg, until the wrist hits the table) or, limited to MAX_TURN, squeeze the peg out.
KICK_DELAY = (0, 10)    # control steps between landing on the rim and the kick
KICK_STEPS = (8, 20)    # how long the kick lasts (at SEARCH_SPEED: 1.3 .. 3.3 mm)

APPROACH, DESCEND, SEARCH, INSERT, DONE, RETRY = range(6)


def run_episode(env, mode, callback=None, kick=0.0, rng=None):
    """One synchronous episode in every env. Returns the per-step logs and per-env statistics.

    callback(t, obs, phase) is called once per control step, before the action is applied.
    kick: fraction of the envs in which the peg is dragged the wrong way for a moment (see the top of the file).
    """
    obs = env.reset()
    n, dev = env.num_envs, env.device
    phase = torch.full((n,), APPROACH, device=dev)
    target = obs["tcp_pos"].clone()
    quat = obs["tcp_quat"].clone()
    start_quat = obs["tcp_quat"].clone()
    slip0 = obs["peg_in_hand"][:, 2].clone()
    slip_at_entry = torch.zeros(n, device=dev)      # slip when the peg entered the hole; the bottom is felt relative to it
    z_at_entry = torch.zeros(n, device=dev)         # TCP height then; resistance only counts after it has gone down a bit
    z_first_entry = torch.zeros(n, device=dev)      # TCP height when the peg first went in (creep speed counts from here)
    resisted = torch.zeros(n, device=dev)            # consecutive steps of resistance (blind controller)
    retries = torch.zeros(n, device=dev)
    lift_to = torch.zeros(n, device=dev)
    up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3)
    touched = torch.zeros(n, dtype=torch.bool, device=dev)
    in_contact = torch.zeros(n, dtype=torch.bool, device=dev)   # the peg has touched the socket: hand orientation is held
    steps_to_success = torch.full((n,), -1, device=dev)
    log = dict(tcp_pos=[], tcp_quat=[], socket_estimate=[], tactile=[], action_pos=[], action_quat=[],
               phase=[], insertion_depth=[], lateral_error=[], peg_in_hand=[], perturbed=[])
    rng = rng or np.random.default_rng(0)
    draw = lambda lo, hi: torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float32, device=dev)
    kicked = draw(0, 1) < kick
    kick_angle = draw(-np.pi, np.pi)
    kick_dir = torch.stack([torch.cos(kick_angle), torch.sin(kick_angle)], dim=-1)
    kick_delay, kick_steps = draw(*KICK_DELAY).round(), draw(*KICK_STEPS).round()
    search_start = torch.full((n,), -1.0, device=dev)

    for t in range(HORIZON):
        slip = slip0 - obs["peg_in_hand"][:, 2]          # > 0: peg pushed up into the hand (TCP z points down)
        depth, lateral = obs["insertion_depth"], obs["lateral_error"]
        est_xy = obs["socket_estimate"][:, 0:2]
        peg_axis, hole_axis = quat_apply(obs["peg_quat"], up), quat_apply(obs["socket_quat"], up)
        # the hole's axis is a line (the socket may be tilted): aim at the point of it level with the peg's bottom
        def on_axis(point):
            along = ((point - obs["socket_pos"]) * hole_axis).sum(-1, keepdim=True)
            return obs["socket_pos"] + along * hole_axis
        to_hole = (on_axis(obs["peg_bottom"]) - obs["peg_bottom"])[:, 0:2]
        # once the tip is in the hole, steer the held end of the peg over the hole axis so the peg stands straight
        held_end = obs["tcp_pos"] + quat_apply(obs["tcp_quat"], obs["peg_in_hand"])
        to_axis = (on_axis(held_end) - held_end)[:, 0:2]
        tilted = to_axis.norm(dim=-1) > 0.0005
        # make the peg's axis parallel to the hole's: rotate the whole hand by the angle between them ...
        turn_axis = torch.cross(peg_axis, hole_axis, dim=-1)
        turn_angle = torch.asin(turn_axis.norm(dim=-1).clamp(max=1.0))
        turn_axis = turn_axis / turn_axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        # ... but only while the peg is free to follow the hand. Once it touches the socket its tilt comes from the contact
        # (tip held by the hole, hand off to the side): that is corrected by moving the hand, and a hand that kept turning
        # would squeeze the peg out of the pads. The orientation reached before contact is held from then on.
        in_contact |= (slip > RIM_SLIP) | (obs["insertion_depth"] > 0.0) | (obs["socket_force"] > CONTACT_FORCE)
        aligned = quat_mul(quat_from_angle_axis(turn_angle, turn_axis), obs["tcp_quat"])
        turned = quat_mul(aligned, quat_conjugate(start_quat))        # (x, y, z, w): rotation from the start orientation
        turned = turned * torch.sign(turned[:, 3:4] + 1e-12)
        angle = 2 * torch.atan2(turned[:, 0:3].norm(dim=-1), turned[:, 3])
        axis = turned[:, 0:3] / turned[:, 0:3].norm(dim=-1, keepdim=True).clamp(min=1e-12)
        aligned = quat_mul(quat_from_angle_axis(angle.clamp(max=MAX_TURN), axis), start_quat)
        quat = torch.where(in_contact.unsqueeze(-1), quat, aligned)

        # --- phase transitions
        at_estimate = (target[:, 0:2] - est_xy).norm(dim=-1) < 1e-4
        phase = torch.where((phase == APPROACH) & at_estimate, torch.full_like(phase, DESCEND), phase)
        entered = depth > 0.002
        on_rim = (slip > RIM_SLIP) & ~entered
        touched |= (phase == DESCEND) & on_rim
        if mode == "expert":
            phase = torch.where((phase == DESCEND) & on_rim, torch.full_like(phase, SEARCH), phase)
            phase = torch.where((phase == SEARCH) & (entered | (lateral < 0.0004)), torch.full_like(phase, INSERT), phase)
        phase = torch.where((phase == DESCEND) & entered, torch.full_like(phase, INSERT), phase)
        if mode == "expert":   # the hole can move away again before the peg is in (perturbation): back to searching
            off_again = (phase == INSERT) & ~entered & (lateral > OFF_AGAIN)
            phase = torch.where(off_again, torch.full_like(phase, SEARCH), phase)
            slip_at_entry = torch.where(off_again, torch.zeros_like(slip_at_entry), slip_at_entry)
            z_first_entry = torch.where(off_again, torch.zeros_like(z_first_entry), z_first_entry)
        now_inserting = (phase == INSERT) & (slip_at_entry == 0)
        slip_at_entry = torch.where(now_inserting, slip.clamp(min=1e-6), slip_at_entry)
        z_at_entry = torch.where(now_inserting, obs["tcp_pos"][:, 2], z_at_entry)
        z_first_entry = torch.where(now_inserting & (z_first_entry == 0), obs["tcp_pos"][:, 2], z_first_entry)
        # the expert stops when it feels the bottom (resistance after it has gone a few mm into the hole); resistance
        # with the peg not yet near the bottom is a jam: lift the peg, bring the hand over the hole axis (which stands the peg up), try again
        travel = z_at_entry - obs["tcp_pos"][:, 2]
        resisting = (travel > 0.003) & ((slip - slip_at_entry > BOTTOM_SLIP) | (obs["socket_force"] > BOTTOM_FORCE))
        jammed = (phase == INSERT) & resisting & (depth < JAM_DEPTH) & (retries < RETRIES) & (mode == "expert")
        retries = retries + jammed.float()
        lift_to = torch.where(jammed, obs["tcp_pos"][:, 2] + LIFT, lift_to)
        slip_at_entry = torch.where(jammed, torch.zeros_like(slip_at_entry), slip_at_entry)   # re-recorded on the next INSERT
        phase = torch.where(jammed, torch.full_like(phase, RETRY), phase)
        lifted = (phase == RETRY) & (obs["tcp_pos"][:, 2] > lift_to - 0.0005)
        phase = torch.where(lifted, torch.full_like(phase, INSERT), phase)
        z_at_entry = torch.where(lifted, obs["tcp_pos"][:, 2], z_at_entry)        # travel and slip count from here again
        slip_at_entry = torch.where(lifted, slip.clamp(min=1e-6), slip_at_entry)
        # a wedged peg can hold more than the grip (about 36 N): if the hand comes off it, stop rather than hit it later
        lost_peg = slip < -LOST_SLIP
        finished = ((phase == INSERT) & resisting & ~lifted) | (slip > MAX_SLIP) | lost_peg
        if mode == "blind":   # it cannot tell the rim from the bottom: sustained resistance while descending means stop
            resisted = torch.where(obs["socket_force"] > BLIND_FORCE, resisted + 1, torch.zeros_like(resisted))
            finished |= (phase == DESCEND) & (resisted >= 12)
        phase = torch.where(((phase == INSERT) | (phase == DESCEND) | (phase == RETRY)) & finished, torch.full_like(phase, DONE), phase)

        # --- motion for this control step
        move = torch.zeros_like(target)
        d = est_xy - target[:, 0:2]
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        approach = d / dist * torch.minimum(dist, torch.full_like(dist, APPROACH_SPEED * CONTROL_DT))
        move[:, 0:2] = torch.where((phase == APPROACH).unsqueeze(-1), approach, move[:, 0:2])
        going_down = (phase == DESCEND) | (phase == INSERT)
        if mode == "expert":
            # until the tip is in the hole, keep the tip over it (the peg may hang at an angle in the pads: standing it up
            # first would take the tip off the hole again); once it is in, straighten the peg before pushing further
            in_hole = ((phase == INSERT) | (phase == RETRY)) & entered
            going_down &= ~(in_hole & tilted)
            goal = torch.where(in_hole.unsqueeze(-1), to_axis, to_hole)
            dist = goal.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            slide = goal / dist * torch.minimum(dist, torch.full_like(dist, SEARCH_SPEED * CONTROL_DT))
            sliding = (phase == SEARCH) | (phase == INSERT) | (phase == RETRY)
            move[:, 0:2] = torch.where(sliding.unsqueeze(-1), slide, move[:, 0:2])
        above_estimate = obs["peg_bottom"][:, 2] - obs["socket_estimate"][:, 2]
        speed = torch.where(above_estimate < SLOW_ZONE, CONTACT_SPEED, DESCEND_SPEED)   # also inside the hole: the bottom is blind
        speed = torch.where((phase == INSERT) & (z_first_entry - obs["tcp_pos"][:, 2] > SLOW_TRAVEL), CREEP_SPEED, speed)
        move[:, 2] = torch.where(going_down, -speed * CONTROL_DT, 0.0)
        move[:, 2] = torch.where(phase == RETRY, CONTACT_SPEED * CONTROL_DT, move[:, 2])
        search_start = torch.where((phase == SEARCH) & (search_start < 0), torch.full_like(search_start, t), search_start)
        since = t - search_start - kick_delay
        kicking = kicked & (phase == SEARCH) & (search_start >= 0) & (since >= 0) & (since < kick_steps)
        move = torch.where(kicking.unsqueeze(-1), F.pad(kick_dir * SEARCH_SPEED * CONTROL_DT, (0, 1)), move)
        target = target + move

        for key in ("tcp_pos", "tcp_quat", "socket_estimate", "tactile", "insertion_depth", "lateral_error", "peg_in_hand"):
            log[key].append(obs[key].cpu().numpy())
        log["action_pos"].append(target.cpu().numpy())
        log["action_quat"].append(quat.cpu().numpy())
        log["phase"].append(phase.cpu().numpy())
        log["perturbed"].append(kicking.cpu().numpy())
        if callback is not None:
            callback(t, obs, phase)

        obs = env.step(target, quat)
        newly = obs["success"] & (steps_to_success < 0)
        steps_to_success = torch.where(newly, torch.full_like(steps_to_success, t + 1), steps_to_success)
        if env.viewer_closed():
            break

    stats = dict(
        success=obs["success"].cpu().numpy(),
        estimate_error=(obs["socket_estimate"][:, 0:2] - obs["socket_pos"][:, 0:2]).norm(dim=-1).cpu().numpy(),
        estimate_offset=(obs["socket_estimate"][:, 0:2] - obs["socket_pos"][:, 0:2]).cpu().numpy(),
        touched_rim=touched.cpu().numpy(),
        max_slip=(slip0.unsqueeze(1) - torch.tensor(np.stack(log["peg_in_hand"], axis=1)[..., 2], device=dev)).max(dim=1).values.cpu().numpy(),
        lost_peg=((slip0.unsqueeze(1) - torch.tensor(np.stack(log["peg_in_hand"], axis=1)[..., 2], device=dev)).min(dim=1).values < -LOST_SLIP).cpu().numpy(),
        stopped=obs["stopped"].cpu().numpy(), gel_overload=obs["gel_overload"].cpu().numpy(),
        grasp_overload=obs["grasp_overload"].cpu().numpy(),
        peak_force=obs["peak_socket_force"].cpu().numpy(), peak_tool_force=obs["peak_tool_force"].cpu().numpy(),
        final_depth=obs["insertion_depth"].cpu().numpy(),
        final_lateral=obs["lateral_error"].cpu().numpy(),
        socket_moved=obs["socket_moved"].cpu().numpy(), move_called_off=obs["move_called_off"].cpu().numpy(),
        seconds=np.where(steps_to_success.cpu().numpy() > 0, steps_to_success.cpu().numpy() * CONTROL_DT, np.nan),
    )
    return {k: np.stack(v, axis=1) for k, v in log.items()}, stats  # arrays are (env, time, ...)


def append_zarr(path, logs, stats, only_success=True):
    """Diffusion-policy style replay buffer: data/<key> concatenated over time, meta/episode_ends."""
    import zarr

    root = zarr.open(path, mode="a")
    data, meta = root.require_group("data"), root.require_group("meta")
    keys = dict(tcp_pos="tcp_pos", tcp_quat="tcp_quat", socket_estimate="socket_estimate",
                action_pos="action_pos", action_quat="action_quat", phase="phase",
                insertion_depth="insertion_depth", lateral_error="lateral_error", perturbed="perturbed")
    ends = list(meta["episode_ends"][:]) if "episode_ends" in meta else []
    total = ends[-1] if ends else 0
    kept = 0
    for i in range(len(stats["success"])):
        if only_success and not stats["success"][i]:
            continue
        kept += 1
        episode = {name: logs[src][i] for src, name in keys.items()}
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
    """Success rate, force events, and why the failed episodes failed (each counted under the first reason that applies)."""
    s, n = stats, len(stats["success"])
    print("%s  episodes %d  success %d (%.1f%%)  time to success median %.1f s"
          % (label, n, s["success"].sum(), 100 * s["success"].mean(), np.nanmedian(s["seconds"]) if s["success"].any() else np.nan))
    print("   peg pushed up in the hand: median %.1f mm, 90th percentile %.1f mm"
          % (1000 * np.median(s["max_slip"]), 1000 * np.quantile(s["max_slip"], 0.9)))
    print("   protective stops %d, gel overloads %d (plus %d during the grasp, not counted), hand came off a wedged peg %d; "
          "peak force on the socket: median %.0f N, 90th percentile %.0f N, max %.0f N"
          % (s["stopped"].sum(), s["gel_overload"].sum(), s["grasp_overload"].sum(), s["lost_peg"].sum(),
             np.median(s["peak_force"]), np.quantile(s["peak_force"], 0.9), s["peak_force"].max()))
    if "peak_tool_force" in s:
        print("   peak force felt by the robot (the protective stop's signal): median %.0f N, 90th percentile %.0f N, max %.0f N"
              % (np.median(s["peak_tool_force"]), np.quantile(s["peak_tool_force"], 0.9), s["peak_tool_force"].max()))
    if "socket_moved" in s and (s["socket_moved"].any() or s["move_called_off"].any()):
        moved = s["socket_moved"].astype(bool)
        print("   socket moved after contact in %d episodes (success %.0f%%), not moved %d (success %.0f%%; %d of them "
              "because the peg was already in the bore)"
              % (moved.sum(), 100 * s["success"][moved].mean() if moved.any() else np.nan, (~moved).sum(),
                 100 * s["success"][~moved].mean() if (~moved).any() else np.nan, s["move_called_off"].sum()))
    left = ~s["success"].astype(bool)
    for name, sel in (("protective stop", s["stopped"]), ("gel overload", s["gel_overload"]),
                      ("hand came off the peg", s["lost_peg"]), ("never entered (< 2 mm)", s["final_depth"] < 0.002),
                      ("entered, < %.0f mm deep" % (800 * SOCKET_DEPTH), s["final_depth"] < 0.8 * SOCKET_DEPTH),
                      ("deep enough, > 2 mm off the axis", np.ones(n, dtype=bool))):
        hit = left & sel.astype(bool)
        if hit.any():
            print("   failed: %-34s %3d" % (name, hit.sum()))
        left &= ~hit
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
        sel = (s["estimate_error"] >= lo / 1000) & (s["estimate_error"] < hi / 1000)
        if sel.any():
            print("   estimate off by %d-%d mm: %3d episodes, success %.0f%%" % (lo, hi, sel.sum(), 100 * s["success"][sel].mean()))


def evaluate(env, mode, episodes, record=None, verbose=True, kick=0.0, seed=0):
    all_stats = []
    rng = np.random.default_rng(seed)
    for _ in range(int(np.ceil(episodes / env.num_envs))):
        logs, stats = run_episode(env, mode, kick=kick, rng=rng)
        if record:
            append_zarr(record, logs, stats)
        all_stats.append(stats)
        if env.viewer_closed():
            break
    stats = {k: np.concatenate([s[k] for s in all_stats]) for k in all_stats[0]}
    if verbose:
        summarize(stats, "mode " + mode)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("blind", "expert"), default="expert")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.004, help="max error (m) of the socket position estimate")
    parser.add_argument("--height-noise", type=float, default=0.004, help="max error (m) of the socket height estimate")
    parser.add_argument("--kick", type=float, default=0.0, help="fraction of envs dragged the wrong way for a moment")
    parser.add_argument("--perturb", type=float, default=0.0, help="fraction of envs in which the socket moves after contact")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true", help="one env with a viewer window")
    parser.add_argument("--record", default="", help="append successful episodes to this zarr store")
    parser.add_argument("--stats", default="", help="save per-episode statistics to this .npz")
    args = parser.parse_args()

    env = InsertionEnv(num_envs=1 if args.view else args.envs, headless=not args.view, seed=args.seed,
                       estimate_noise=args.noise, height_noise=args.height_noise, perturb=args.perturb)
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    stats = evaluate(env, args.mode, args.episodes, record=args.record, kick=args.kick, seed=args.seed)
    if args.stats:
        np.savez(args.stats, **stats)
    env.close()
