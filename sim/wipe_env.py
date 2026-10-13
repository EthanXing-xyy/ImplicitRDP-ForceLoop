"""Wiping scene for the UR5 + 2F-85 + two GelSight Minis: press the peg onto a board and slide it along a line.

The robot holds the same peg as in the insertion task (insertion_env.py), now as a wiping tool. A 150 x 60 mm
board is held in front of it, tipped by up to 15 deg. A straight 100 mm line on it is to be wiped: the tool's
tip has to pass over it while pressing with a force inside a window and while moving. The board's pose is
known only roughly, and (perturb > 0) the board shakes all the time, as in a hand. The policy gets its TCP
pose, a noisy and late estimate of the board's pose, and the two tactile force fields; no force reading.

What is modelled, and where the numbers come from (placeholders until the lab's setup is measured):
  tool tip    a real eraser has a felt pad, a few mm thick, that gives ~1 mm under 10 N. The peg is rigid, so that
              compliance is put under the board instead: the board sits on a spring-damper along its normal
              (objects/wipe_board_stage.urdf, PAD_STIFFNESS). In series with the gel and the arm this gives
              roughly 6 N per mm of hand height.
  friction    felt on a melamine whiteboard: about 0.4-0.6. The peg's value is 0.3 (shared with the insertion
              task) and PhysX averages the two surfaces, so the board gets 0.5: 0.4 together.
  force       wiping a whiteboard by hand takes 5-15 N: the window FORCE_MIN..FORCE_MAX. Pressing harder does not
              wipe (a process window, as in polishing or drawing: too hard damages surface or tip); pressing
              harder for more than OVERPRESSURE_TIME without a break, or harder than FORCE_LIMIT at any moment,
              fails the episode. The 150 N protective stop and the gel limit apply as before. (The grasp itself
              gives way first: at 20 N per pad the peg starts to slide up between the pads at ~20 N, breaks
              loose at 33-37 N and then slides at 16-23 N, diag/wipe_grip_hold.py. So FORCE_LIMIT is rarely
              reached; pressing too deep shows as a long press above the window and as the tool slipping.)
  score       as in the RDP paper's wiping task (residue against the demonstrations' level): 1 if no more than
              1 - SUCCESS_FRACTION of the line is left (the expert leaves 0-1%), 0.5 if less than a third is
              left, 0 otherwise; 0 as well if a force rule was broken. `success` is a score of 1.
  rubbing     pressing without moving wipes nothing: the tip has to slide (WIPE_MIN_SPEED), and one moment at the
              right pressure is not a wipe either: a cell counts as wiped once the tip has slid over it inside the
              window for WIPE_DOSE in all (a quarter of the tip's width; a clean pass gives 12 mm). With "any
              moment inside the window" instead, the contact noise (+-2-3 N) let a pass that was too light score:
              the camera-only controller got 81% of the line that way and 55% with the dose, the expert 100%
              either way (re-scored traces, 16 episodes each).
  line        straight, always LINE_LENGTH long. Its middle is within +-socket_range of the point under the hand,
              LINE_HEIGHT above the table. In the board's plane it points along the gripper's closing axis
              +-LINE_YAW_RANGE (the hand turns to it): sliding along the closing axis pushes the peg into one pad,
              sliding across it would pivot the peg between the two (the only rotation two flat pads allow).
  tilt        the board is tipped by 0..BOARD_TILT_MAX in a random direction, about the middle of the line.
  estimate    the camera's, of the board's pose: where the line starts, off by up to estimate_noise in the plane
              (2 mm: a drawn line is seen better than the centre of a hole) and height_noise in height (4 mm:
              depth is a camera's worst axis), and the board's orientation, off by up to ORIENTATION_NOISE.
              Those errors stay for the episode. A new frame comes every CAMERA_EVERY control steps (12 Hz),
              shows the board as it was ESTIMATE_DELAY control steps before, and jitters by CAMERA_NOISE and
              CAMERA_TURN_NOISE; the tracker smooths the frames over CAMERA_SMOOTHING. What the controllers and
              the policies get is the tracker's output. It cannot follow mm-size motion at 1 Hz.
              With timing="real" (insertion_env.py: every sensor on its own clock) the camera is a D435 running
              freely at CAMERA_FPS; a frame's pose reaches the tracker CAMERA_LATENCY after the frame was taken,
              and the tracker takes in every frame as it arrives.
  shaking     the board is held by a hand, not a vice: all the time it drifts along its normal (SHAKE_NORMAL), in
              its plane (SHAKE_PLANE) and in tilt (SHAKE_TILT). Each of the five axes moves as a sum of
              SHAKE_PARTS sines with random frequencies in SHAKE_FREQ, random phases and weights, drawn anew for
              every episode; the amplitudes are bounds on the excursion. No faster than 1 Hz: a 24 Hz force
              loop through the arm's 0.1 s servo lag cannot follow more, and neither can a hand produce it at
              these amplitudes while holding still.

  touch       the pressure reaches the policy only through the gels: the board pushes the peg up between the
              pads and each gel shears under the half of that force it passes on. The tactile shear is that
              load (InsertionEnv.TACTILE_SHEAR = "load", see _ForceField): with the insertion task's reading,
              which shows slip only, the pressure could not be told from the tactile signal at all.

Everything else (robot, grasp, servo lag, tactile delay, force limits) is InsertionEnv's.

Task "v2" (2026-10-10, agreed with the user item by item; rule: closer to the real thing and never easier. All
results before that are task "v1"). The default is the environment variable WIPE_TASK, else "v1". What "v2" changes:
  timing      the tactile sensors, the robot's pose report and the controller's computing time are those of
              insertion_env.py's timing="real". The camera stays as above (12 Hz on the control clock): a D435 at
              its 30 frames/s would make the estimate better, i.e. the task easier.
  line        an arc instead of a straight line: still LINE_LENGTH long, its direction turning by ARC_TURN in all,
              to the left or to the right, through the same middle point; the board's x axis is the direction at
              the middle (and of the chord). The shape is known exactly, as the straight line's was (line_turn in
              the observation); the board's pose is not, as before.
  friction    the board's differs from episode to episode (ink, dust): BOARD_FRICTION_RANGE, which with the peg's
              0.3 gives 0.3-0.7 together. (Checked from the stage's own deflection under the sliding tool: the
              simulator's contact force report does not show the friction force properly.)
  touch       the tactile reading is no longer clean: every frame carries marker-tracking noise (TACTILE_NOISE),
              each pad's shear reading has an offset that stays for the episode (TACTILE_ZERO: the rest
              positions of the markers drift with temperature and load history), and TACTILE_CREEP of the gel's
              shear comes late and stays late (time constant TACTILE_CREEP_TAU: silicone is viscoelastic).
              All three sizes are placeholders.
The wiping speed and the demonstrator of task "v2" are in wipe_expert.py.
"""
import math
import os

from insertion_env import COLS, CONTROL_DT, ESTIMATE_DELAY, FORCE_ALPHA, ROWS, SIM_DT, TABLE_HEIGHT, InsertionEnv   # imports isaacgym before torch
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

import torch

from make_wipe_board import BOARD_TOP, LINE_LENGTH, STAGE_JOINTS   # the line: LINE_LENGTH along the board's x axis, centred

BOARD_STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objects", "wipe_board_stage.urdf")
CELL = 0.001                      # m: the line is scored in cells of this length
CELLS = int(round(LINE_LENGTH / CELL))
WIPE_RADIUS = 0.006               # m: a cell is under the tool when it is this close to the centre of the tip (the
                                  # tip is 16 mm wide; 2 mm are left off for a tip that does not lie quite flat)
FORCE_MIN, FORCE_MAX = 5.0, 15.0  # N along the board's normal: the window in which the tool wipes
FORCE_LIMIT = 40.0                # N: pressing harder than this at any moment fails the episode
WIPE_MIN_SPEED = 0.002            # m/s: slower than this is pressing, not wiping
WIPE_DOSE = 0.003                 # m of sliding inside the window over a cell before it counts as wiped
TOUCH_FORCE = 2.0                 # N: the tool is on the board
SUCCESS_FRACTION = 0.9            # of the line wiped: score 1
PARTIAL_FRACTION = 2.0 / 3.0      # more than this wiped: score 0.5
OVERPRESSURE_TIME = 0.5           # s: pressing above FORCE_MAX for longer than this in one go fails the episode. A spike
                                  # at touch-down or when the board jerks is over in 0.1-0.3 s if the force is being
                                  # controlled; pressing on for longer is what damages a surface
BOARD_FRICTION = 0.5
PAD_STIFFNESS = 1.0e4             # N/m: the felt pad of the tool (10 N/mm), as the spring under the board
PAD_DAMPING = 150.0               # N s/m (above critical for the 0.3 kg board: no ringing)
SLIDE_GAINS = (1.0e6, 2.0e3)      # N/m, N s/m: the stage's in-plane slides (stiff, as the socket's stage)
TILT_GAINS = (1.0e3, 10.0)        # N m/rad, N m s/rad: its tilt axes (20 N at the end of the line: 0.06 deg)
LINE_HEIGHT = 0.038               # m above the table: the middle of the line (a board tipped by 15 deg clears the table)
LINE_YAW_RANGE = math.radians(15.0)
BOARD_TILT_MAX = math.radians(15.0)
ESTIMATE_NOISE = 0.002            # m: largest error of the camera's estimate of the line's start, in the plane
ORIENTATION_NOISE = math.radians(2.0)   # of the camera's estimate of the board's orientation
CAMERA_EVERY = 2                  # control steps per camera frame (12 Hz)
CAMERA_FPS = 30.0                 # timing="real": the D435's colour and depth streams
CAMERA_LATENCY = 0.125            # s, timing="real": frame taken -> its pose at the tracker (= the 3 control steps of
                                  # timing="sync"; users measure 100-130 ms for a D435 stream, Intel gives no figure)
CAMERA_SLOTS = 8                  # camera frames kept per env (four are on their way at most)
CAMERA_NOISE = 0.0005             # m per axis: jitter of one frame's position (plane fit or marker at ~0.4 m)
CAMERA_TURN_NOISE = math.radians(0.3)   # jitter of one frame's orientation (rms angle)
CAMERA_SMOOTHING = 0.2            # s: time constant of the pose tracker's low-pass
SHAKE_FREQ = (0.2, 1.0)           # Hz
SHAKE_PARTS = 3                   # sines per axis
SHAKE_NORMAL = (0.001, 0.002)     # m: bound of the excursion along the normal, drawn per episode (a 1-D model of the
                                  # pressure loop holds the window 98% of the time at 1 mm, 89% at 2 mm, 75% at 3 mm)
SHAKE_PLANE = 0.003               # m: largest bound of the excursion in the board's plane
SHAKE_TILT = math.radians(2.0)    # largest bound of the tilt excursion
SHAKE_RAMP = 1.0                  # s: the shaking fades in at the start of the episode
# task "v2" (placeholders, see the top of the file)
ARC_TURN = (math.radians(10.0), math.radians(30.0))   # how far the line's direction turns from its start to its end
BOARD_FRICTION_RANGE = (0.3, 1.1)
TACTILE_NOISE = 1.0e-5            # m per marker and axis in one frame (a fraction of a pixel of a GelSight Mini image); a pad
TACTILE_MARKERS = 63              # has 7 x 9 markers, the field here more points: the noise per point is scaled so that
                                  # a pad's summed reading is as noisy as that of 63 markers
TACTILE_ZERO = 4.0e-5             # m: largest offset of a pad's shear reading (0.5 N of pressure if both pads agree)
TACTILE_CREEP, TACTILE_CREEP_TAU = 0.1, 1.0   # share of the shear that comes late; s


def minimum_jerk(s):
    """Position profile of a smooth move: s = time / duration in 0..1 -> share of the way."""
    return s * s * s * (10 - 15 * s + 6 * s * s)


def turn_part(quat, goal, share):
    """`quat` (N, 4) turned the fraction `share` of the way to `goal`."""
    delta = quat_mul(goal, quat_conjugate(quat))
    delta = delta * torch.where(delta[:, 3:4] < 0, -1.0, 1.0)
    sine = delta[:, 0:3].norm(dim=-1)
    angle = 2 * torch.atan2(sine, delta[:, 3])
    return quat_mul(quat_from_angle_axis(share * angle, delta[:, 0:3] / sine.unsqueeze(-1).clamp(min=1e-12)), quat)


def cells_under_tool(along, lateral):
    """(N, CELLS) bool: which cells of a straight line are under the tip, for a tip centre `along` the line (m from
    its start) and `lateral` off it (both (N,)). (The env itself measures in the board's plane, see _after_step.)"""
    centres = (torch.arange(CELLS, device=along.device) + 0.5) * CELL
    return (centres.unsqueeze(0) - along.unsqueeze(1)) ** 2 + lateral.unsqueeze(1) ** 2 <= WIPE_RADIUS ** 2


def arc_point(u, kappa):
    """Point of the line in the board's plane, (..., 2): `u` is the distance along the line from its middle, `kappa`
    its curvature (1/m, positive: it bends towards +y; 0: straight along x). u and kappa broadcast."""
    angle = kappa * u
    straight = angle.abs() < 1e-6
    k = torch.where(kappa.abs() < 1e-9, torch.ones_like(kappa), kappa) + torch.zeros_like(angle)
    return torch.stack([torch.where(straight, u + torch.zeros_like(angle), torch.sin(angle) / k),
                        torch.where(straight, 0.5 * angle * u, (1 - torch.cos(angle)) / k)], dim=-1)


class WipeEnv(InsertionEnv):
    STAGE_FILE = BOARD_STAGE
    STAGE_LINK = "board"
    STAGE_FRICTION = BOARD_FRICTION
    STAGE_GAINS = tuple(zip(SLIDE_GAINS, SLIDE_GAINS, TILT_GAINS, TILT_GAINS, (PAD_STIFFNESS, PAD_DAMPING)))   # per DOF
    TACTILE_SHEAR = "load"        # the gels show the load they carry, not only slip: the pressure has to be felt

    def __init__(self, *args, seed=0, estimate_noise=ESTIMATE_NOISE, camera_errors=True, task=None, camera_free=False, **kwargs):
        """camera_errors=False (an ablation): the camera's estimate of the board's pose is exact, with no bias and
        no jitter (it is still ESTIMATE_DELAY steps old). The random draws are the same, so are the layouts.
        task: "v1" or "v2" (top of the file); None: the environment variable WIPE_TASK, else "v1".
        camera_free: with timing="real", the camera runs on its own clock at CAMERA_FPS (not part of either task)."""
        self.task = task or os.environ.get("WIPE_TASK", "v1")
        assert self.task in ("v1", "v2"), self.task
        print("wiping task:", self.task)
        if self.task == "v2":
            kwargs.setdefault("timing", "real")
        self.camera_free = camera_free
        self.task_rng = torch.Generator(device="cpu").manual_seed(seed + 5000)   # separate: layouts stay as before
        self.seed = seed
        self.camera_scale = float(camera_errors)
        if not camera_errors:
            estimate_noise, kwargs["height_noise"] = 1e-9, 0.0    # not 0: the bias is drawn as a share of estimate_noise
        self.camera_rng = torch.Generator(device="cpu").manual_seed(seed + 3000)   # separate: layouts do not depend on the jitter
        super().__init__(*args, seed=seed, estimate_noise=estimate_noise, **kwargs)
        names = self.gym.get_actor_dof_names(self.envs[0], self.socket)
        assert names == list(STAGE_JOINTS), (names, STAGE_JOINTS)      # _shake() and _commanded_pose() rely on this order
        assert not camera_free or self.timing == "real"
        if self.task == "v2":
            self.gel_creep, self.gel_creep_tau = TACTILE_CREEP, TACTILE_CREEP_TAU
            self.frame_noise = TACTILE_NOISE * math.sqrt(ROWS * COLS / TACTILE_MARKERS)

    # ---------------------------------------------------------------- episode
    def _place_object(self, root):
        n, dev = self.num_envs, self.device
        assert len(self.stage_dofs) == len(STAGE_JOINTS)
        # task "v2": the line's shape, the pads' zero offsets and the board's friction, drawn anew for every episode
        v2 = float(self.task == "v2")
        draw2 = lambda *shape: torch.rand(shape, generator=self.task_rng).to(dev)
        side = torch.where(draw2(n) < 0.5, -1.0, 1.0)
        self.line_turn = v2 * side * (ARC_TURN[0] + (ARC_TURN[1] - ARC_TURN[0]) * draw2(n))        # rad, signed
        self.kappa = self.line_turn / LINE_LENGTH
        centres = (torch.arange(CELLS, device=dev) + 0.5) * CELL - LINE_LENGTH / 2
        self.cell_xy = arc_point(centres.unsqueeze(0), self.kappa.unsqueeze(1))                    # (N, CELLS, 2)
        ends = arc_point(torch.tensor([[-LINE_LENGTH / 2, LINE_LENGTH / 2]], device=dev), self.kappa.unsqueeze(1))
        flat = lambda xy: torch.cat([xy, torch.zeros((n, 1), device=dev)], dim=-1)
        self.start_local, self.end_local = flat(ends[:, 0]), flat(ends[:, 1])                      # in the board's plane
        angle, radius = 2 * math.pi * draw2(n, 2), TACTILE_ZERO * torch.sqrt(draw2(n, 2))          # uniform over a disc
        friction = BOARD_FRICTION_RANGE[0] + (BOARD_FRICTION_RANGE[1] - BOARD_FRICTION_RANGE[0]) * draw2(n)
        self.board_friction = torch.full((n,), BOARD_FRICTION, device=dev)
        if self.task == "v2":
            self.gel_zero = torch.stack([radius * torch.cos(angle), torch.zeros_like(radius), radius * torch.sin(angle)], dim=-1)
            self.board_friction = friction
            for env, value in zip(self.envs, friction.tolist()):
                shapes = self.gym.get_actor_rigid_shape_properties(env, self.socket)
                for shape in shapes:
                    shape.friction = value
                self.gym.set_actor_rigid_shape_properties(env, self.socket, shapes)
        up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3)
        axis = self.home_closing_axis[0:2] / self.home_closing_axis[0:2].norm()
        yaw = math.atan2(axis[1].item(), axis[0].item()) + (2 * self._rand(n) - 1) * LINE_YAW_RANGE
        tilt_dir = 2 * math.pi * self._rand(n)
        tilt_axis = torch.stack([torch.cos(tilt_dir), torch.sin(tilt_dir), torch.zeros(n, device=dev)], dim=-1)
        tilt = quat_from_angle_axis(self._rand(n) * BOARD_TILT_MAX, tilt_axis)
        quat = quat_mul(tilt, quat_from_angle_axis(yaw, up))                 # the line's direction first, then tipped over
        centre = torch.cat([self.home_xy + (2 * self._rand(n, 2) - 1) * self.socket_range,
                            torch.full((n, 1), TABLE_HEIGHT + LINE_HEIGHT, device=dev)], dim=-1)   # the line's middle
        root[:, self.socket, 0:3] = centre - quat_apply(quat, self._on_board(0.0))
        root[:, self.socket, 3:7] = quat
        root[:, self.socket, 7:13] = 0.0
        self.stage_pos, self.stage_quat = root[:, self.socket, 0:3].clone(), quat
        # the camera's errors: position of the line's start, orientation of the board
        angle = 2 * math.pi * self._rand(n)
        lo = (self.estimate_noise_low / self.estimate_noise) ** 2
        radius = self.estimate_noise * torch.sqrt(lo + (1 - lo) * self._rand(n))  # uniform over a disc (or ring)
        height_error = (2 * torch.rand(n, generator=self.height_rng) - 1).to(dev) * self.height_noise
        self.estimate_error = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), height_error], dim=-1)
        error_axis = torch.nn.functional.normalize(2 * self._rand(n, 3) - 1 + 1e-6, dim=-1)
        self.orientation_error = quat_from_angle_axis(self._rand(n) * ORIENTATION_NOISE * self.camera_scale, error_axis)
        # shaking (drawn for every env so that the layouts do not depend on `perturb`)
        draw = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand((n,) + shape, generator=self.perturb_rng).to(dev)
        self.will_perturb = draw(0.0, 1.0) < self.perturb
        plane, plane_dir = draw(0.0, SHAKE_PLANE), draw(0.0, 2 * math.pi)
        tilt_bound, tilt_dir = draw(0.0, SHAKE_TILT), draw(0.0, 2 * math.pi)
        self.shake_amplitude = torch.stack([plane * torch.cos(plane_dir).abs(), plane * torch.sin(plane_dir).abs(),
                                            tilt_bound * torch.cos(tilt_dir).abs(), tilt_bound * torch.sin(tilt_dir).abs(),
                                            draw(*SHAKE_NORMAL)], dim=-1) * self.will_perturb.unsqueeze(-1)
        self.shake_freq = draw(*SHAKE_FREQ, len(STAGE_JOINTS), SHAKE_PARTS)
        self.shake_phase = draw(0.0, 2 * math.pi, len(STAGE_JOINTS), SHAKE_PARTS)
        weight = draw(0.3, 1.0, len(STAGE_JOINTS), SHAKE_PARTS)
        self.shake_weight = weight / weight.sum(dim=-1, keepdim=True)        # the amplitude bounds the sum
        self.pose_queue = [self._commanded_pose(torch.zeros((n, len(STAGE_JOINTS)), device=dev))] * ESTIMATE_DELAY

    def _shake(self, time):
        """(N, 5): the stage's joint targets at `time` (s) into the episode."""
        wave = torch.sin(2 * math.pi * self.shake_freq * time + self.shake_phase)
        fade = minimum_jerk(min(1.0, time / SHAKE_RAMP))
        return self.shake_amplitude * fade * (self.shake_weight * wave).sum(dim=-1)

    def _commanded_pose(self, joints):
        """Where the stage's targets `joints` (N, 5) put the board: the start of the line (N, 3) and the board's
        orientation (N, 4). (Not where it is: under the tool the pad gives ~1 mm, which the tool hides.)"""
        unit = lambda i: torch.eye(3, device=self.device)[i].expand(self.num_envs, 3)
        quat = quat_mul(self.stage_quat, quat_mul(quat_from_angle_axis(joints[:, 2], unit(0)), quat_from_angle_axis(joints[:, 3], unit(1))))
        slide = torch.cat([joints[:, 0:2], torch.full_like(joints[:, 0:1], BOARD_TOP)], dim=-1)
        centre = self.stage_pos + quat_apply(self.stage_quat, slide) + quat_apply(quat, unit(2)) * joints[:, 4:5]
        return centre + quat_apply(quat, self.start_local), quat

    def _start_task(self):
        n, dev = self.num_envs, self.device
        self.dose = torch.zeros((n, CELLS), device=dev)                              # m of in-window sliding per cell
        self.cells = torch.zeros((n, CELLS), dtype=torch.bool, device=dev)          # wiped
        self.pressed_hard = torch.zeros_like(self.cells)                             # passed over with too much force
        self.filtered_normal = torch.zeros(n, device=dev)
        self.peak_normal = torch.zeros(n, device=dev)
        self.contact_time, self.band_time, self.hard_time = (torch.zeros(n, device=dev) for _ in range(3))
        self.hard_run, self.longest_hard = torch.zeros(n, device=dev), torch.zeros(n, device=dev)   # s above the window in one go
        self.last_tip = None
        self.frame = 0                         # control steps since the start of the episode
        self.seen = None                       # the tracker's output: line start (N, 3), board orientation (N, 4)
        if self.camera_free:                   # the camera's clock starts somewhere in its frame period
            self.camera_next = (torch.rand(n, generator=self.timing_rng, dtype=torch.float64) / CAMERA_FPS).to(dev)
            self.camera_taken = torch.full((n, CAMERA_SLOTS), math.inf, dtype=torch.float64, device=dev)   # inf: slot empty or used
            self.camera_start, self.camera_quat = torch.zeros((n, CAMERA_SLOTS, 3), device=dev), torch.zeros((n, CAMERA_SLOTS, 4), device=dev)
            self.camera_slot = torch.zeros(n, dtype=torch.long, device=dev)

    def _on_board(self, along):
        """(N, 3): the point of the line `along` (m) from its middle, in the board's frame."""
        return torch.tensor([[along, 0.0, BOARD_TOP]], device=self.device).expand(self.num_envs, 3)

    def _tip(self):
        """Centre of the tool's tip relative to the line: along it from its start, off it sideways (positive
        towards the board's +y), above the board (m)."""
        b = self._peg_bottom_in_socket()            # in the board's frame
        straight = self.kappa.abs() < 1e-9
        k = torch.where(straight, torch.ones_like(self.kappa), self.kappa)
        sign = torch.sign(k)
        rx, ry = sign * b[:, 0], sign * (b[:, 1] - 1 / k)            # from the arc's centre, which is at (0, 1 / kappa)
        along = torch.where(straight, b[:, 0], torch.atan2(rx, -ry) / k) + LINE_LENGTH / 2
        lateral = torch.where(straight, b[:, 1], sign * (1 / k.abs() - torch.sqrt(rx * rx + ry * ry)))
        return along, lateral, b[:, 2] - BOARD_TOP

    def _perturb_step(self):
        """Stage targets for the next physics step: the board shakes."""
        self.time += SIM_DT
        self.targets[:, self.stage_dofs] = self._shake(self.time)

    def _camera_frame(self, start, quat):
        """What one camera frame makes of the board at (start, quat): the episode's errors plus this frame's jitter."""
        n, dev = self.num_envs, self.device
        noise = lambda *shape: torch.randn(shape, generator=self.camera_rng).to(dev)
        turn = noise(n, 3) * CAMERA_TURN_NOISE * self.camera_scale / math.sqrt(3.0)
        jitter = quat_from_angle_axis(turn.norm(dim=-1), turn / turn.norm(dim=-1, keepdim=True).clamp(min=1e-12))
        return (start + self.estimate_error + noise(n, 3) * CAMERA_NOISE * self.camera_scale,
                quat_mul(jitter, quat_mul(self.orientation_error, quat)))

    def _capture(self):
        """timing="real", after every physics step: the tactile sensors, and the cameras whose next frame is due."""
        super()._capture()
        if not self.camera_free:
            return
        due = self.camera_next <= self.time + 1e-9
        if due.any():
            start, quat = self._camera_frame(*self._commanded_pose(self.targets[:, self.stage_dofs]))
            env = due.nonzero(as_tuple=True)[0]
            slot = self.camera_slot[env]
            self.camera_start[env, slot], self.camera_quat[env, slot] = start[env], quat[env]
            self.camera_taken[env, slot] = self.time
            self.camera_slot[env] = (slot + 1) % CAMERA_SLOTS
            self.camera_next[env] += 1.0 / CAMERA_FPS

    def _track(self):
        """timing="real": the tracker takes in the camera frames that have arrived since the last control step, oldest first."""
        rows = torch.arange(self.num_envs, device=self.device)
        share = 1.0 - math.exp(-1.0 / (CAMERA_FPS * CAMERA_SMOOTHING))
        while True:
            arrived = self.camera_taken + CAMERA_LATENCY <= self.time + 1e-9
            has = arrived.any(dim=-1)
            if not has.any():
                return
            oldest = torch.where(arrived, self.camera_taken, torch.full_like(self.camera_taken, math.inf)).argmin(dim=-1)
            start = self.seen[0] + share * (self.camera_start[rows, oldest] - self.seen[0])
            quat = turn_part(self.seen[1], torch.where(has.unsqueeze(-1), self.camera_quat[rows, oldest], self.seen[1]), share)
            self.seen = (torch.where(has.unsqueeze(-1), start, self.seen[0]), torch.where(has.unsqueeze(-1), quat, self.seen[1]))
            self.camera_taken[rows[has], oldest[has]] = math.inf

    def _after_step(self):
        """Score the physics step: the cells under the tip collect the distance it slides inside the window."""
        up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, 3)
        normal = quat_apply(self.body[:, self.socket_body, 3:7], up)
        pressing = (self.contact[:, self.socket_body] * normal).sum(-1).abs()        # friction is not part of it
        self.filtered_normal += (self._despike("normal", pressing) - self.filtered_normal) * FORCE_ALPHA
        self.peak_normal = torch.maximum(self.peak_normal, self.filtered_normal)
        tip = self._peg_bottom_in_socket()[:, 0:2]                   # in the board's plane
        speed = torch.zeros_like(tip[:, 0]) if self.last_tip is None else (tip - self.last_tip).norm(dim=-1) / SIM_DT
        self.last_tip = tip
        force = self.filtered_normal
        in_window, too_hard, sliding = (force >= FORCE_MIN) & (force <= FORCE_MAX), force > FORCE_MAX, speed > WIPE_MIN_SPEED
        under = ((self.cell_xy - tip.unsqueeze(1)) ** 2).sum(-1) <= WIPE_RADIUS ** 2
        self.dose += under * ((in_window & sliding) * speed * SIM_DT).unsqueeze(1)
        self.cells = self.dose >= WIPE_DOSE
        self.pressed_hard |= under & (too_hard & sliding).unsqueeze(1)
        self.contact_time += SIM_DT * (force > TOUCH_FORCE)
        self.band_time += SIM_DT * in_window
        self.hard_time += SIM_DT * too_hard
        self.hard_run = (self.hard_run + SIM_DT) * too_hard
        self.longest_hard = torch.maximum(self.longest_hard, self.hard_run)

    # ---------------------------------------------------------------- observation
    def observe(self):
        board_pos, board_quat = self.body[:, self.socket_body, 0:3].clone(), self.body[:, self.socket_body, 3:7].clone()
        start = board_pos + quat_apply(board_quat, self._on_board(0.0) + self.start_local)
        end = board_pos + quat_apply(board_quat, self._on_board(0.0) + self.end_local)
        # the camera: frames at 12 Hz, each ESTIMATE_DELAY control steps old and jittering, smoothed by the tracker
        # (timing="real": frames at CAMERA_FPS on the camera's own clock, each CAMERA_LATENCY old when it arrives)
        self.pose_queue.append(self._commanded_pose(self.targets[:, self.stage_dofs]))
        old_start, old_quat = self.pose_queue.pop(0)
        if self.seen is None:                  # it has watched the still board before the episode: no jitter left
            self.seen = (old_start + self.estimate_error, quat_mul(self.orientation_error, old_quat))
        elif self.camera_free:
            self._track()
        elif self.frame % CAMERA_EVERY == 0:
            frame_start, frame_quat = self._camera_frame(old_start, old_quat)
            share = 1.0 - math.exp(-CAMERA_EVERY * CONTROL_DT / CAMERA_SMOOTHING)
            self.seen = (self.seen[0] + share * (frame_start - self.seen[0]), turn_part(self.seen[1], frame_quat, share))
        self.frame += 1
        estimate, quat_estimate = self.seen[0].clone(), self.seen[1].clone()
        along, lateral, height = self._tip()
        wiped = self.cells.float().mean(dim=1)
        too_hard = (self.longest_hard > OVERPRESSURE_TIME) | (self.peak_normal > FORCE_LIMIT)
        clean = torch.where(wiped >= SUCCESS_FRACTION, 1.0, torch.where(wiped > PARTIAL_FRACTION, 0.5, 0.0))   # the paper's score
        score = clean * ~(too_hard | self.stopped | self.gel_overload)
        return dict(
            self._common_obs(),
            # what a policy may use (under both names: the policy code calls the goal estimate `socket_estimate`)
            start_estimate=estimate, socket_estimate=estimate.clone(), board_quat_estimate=quat_estimate,
            line_turn=self.line_turn.clone(),      # the line's shape: how far its direction turns (rad; 0: straight)
            # ground truth, for the scripted expert and for scoring only
            line_start=start, line_end=end, board_pos=board_pos, board_quat=board_quat,
            normal_force=self.filtered_normal.clone(), peak_normal_force=self.peak_normal.clone(),
            tip_along=along, tip_lateral=lateral, tip_height=height,
            wiped=wiped, pressed_hard=self.pressed_hard.float().mean(dim=1),
            contact_time=self.contact_time.clone(), band_time=self.band_time.clone(), hard_time=self.hard_time.clone(),
            longest_hard=self.longest_hard.clone(),
            shaken=self.will_perturb.clone(), shake_normal=self.shake_amplitude[:, 4].clone(),
            board_friction=self.board_friction.clone(),
            too_hard=too_hard, score=score, score_without_force_rules=clean * ~(self.stopped | self.gel_overload),
            success=score == 1.0,
        )
