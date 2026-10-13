"""Peg-in-hole scene for the UR5 + 2F-85 + two GelSight Minis.

Each env: the robot, a 16 mm round peg (IndustReal asset shipped with TacSL) held between
the sensors, and the matching socket on a pedestal. Episodes start with the peg already in
the gripper. The policy is given a noisy estimate of the socket position and the two tactile
force fields; it commands a TCP pose at 24 Hz. Optionally (perturb > 0) the socket, which sits on a stiff XY
stage, is moved a few mm once the peg has touched it (see PERTURB_*).

All envs are reset together and run for the same number of steps.
"""
import math
import os

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, tf_combine

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from urdfpy import URDF

from isaacgymenvs.tacsl_sensors.tacsl_sensors import TactileFieldSensor

from ee_control import EEController
from ur5_2f85_common import (ARM_GAINS, ARM_HOME, PAD_GAINS, ASSET_FILE_GSMINI_PRISMATIC, ASSET_FILE_GSMINI_PRISMATIC_BOX, ASSET_ROOT, GRIP_FORCE, PAD_DRIVER, PAD_JOINTS,
                             enable_sensor_self_collision, load_robot, make_sim, pad_limits, set_drive_gains,
                             solve_finger_joint)

SIDES = ("left", "right")
ROWS, COLS = 20, 25               # tactile points per sensor
SIM_DT = 1.0 / 120.0
SIM_STEPS_PER_ACTION = 5          # 24 Hz control, as in the RDP paper
CONTROL_DT = SIM_DT * SIM_STEPS_PER_ACTION

INDUSTREAL = os.path.join(os.path.expanduser(os.environ.get("TACSL_ROOT", "~/tacsl")), "IsaacGymEnvs/assets/industreal")   # TACSL_ROOT: where IsaacGymEnvs (TacSL) lives
PEG_FILE = "urdf/industreal_round_peg_16mm.urdf"
# (A copy of the peg tessellated to 0.8 mm was tried against the contact-force chatter: no gain, 5x slower.)
SOCKET_FILE = "urdf/industreal_round_hole_16mm.urdf"
PEG_DIAMETER = 0.015983
PEG_BOTTOM = 0.003                # the peg mesh spans z = 3 .. 53 mm in its own frame
PEG_GRASP = 0.045                 # height on the peg (own frame) that sits at the TCP
SOCKET_TOP = 0.028                # socket mesh: top face at 28 mm, 8.5 mm bore, 1 mm chamfer
SOCKET_DEPTH = 0.025              # the hole's floor is at 3 mm (upward faces of the mesh); earlier versions assumed 23 mm
TABLE_HEIGHT = 0.19
ROBOT, PEG, SOCKET, TABLE = "ur5_2f85", "peg", "socket", "table"


CLOSE_STEP = 0.00002              # pad travel per sim step while closing (4.8 mm/s; the 2F-85 does 20-150 mm/s)
CLOSE_STEP_TOUCHING = 0.000001    # once the pads feel the peg: the motor stops at its force setting, so no overshoot
# How the arm follows a commanded pose: UR's servoj blends toward the target over `lookahead_time`
# (0.03-0.2 s; 0.1 s is the value in UR's RTDE examples) after one 125 Hz controller cycle.
SERVO_TAU = 0.1                   # s, first-order lag of the TCP target
SERVO_DELAY = 1                   # sim steps (120 Hz) before a new command starts to act
# Force limits. A UR5 (CB3) goes into protective stop when the force at the tool exceeds the safety
# setting, 150 N by default; the robot then stands still until a person resets it, so the episode
# is lost. GelSight gels are damaged by large loads; the limit per pad is a placeholder.
STOP_FORCE = 150.0                # N at the tool: the external force on the robot, i.e. what its force estimate sees
GEL_FORCE_LIMIT = 80.0            # N per pad (placeholder; a jammed peg twisting in the hand reaches 40-50 N here)
# The protective stop uses the force the robot actually feels at the tool (UR estimates it from the motor currents).
# Two simulator readings measure it: what robot + peg press into the fixed world (contact forces on socket and table),
# and what the robot's links receive (from the peg through the pads, plus any direct contact). With a 27 g peg the
# two can differ by m*a < 1 N; each one alone, however, shows solver artifacts of several hundred newtons that the
# other does not: peg-socket spikes while the peg's edge slides over the chamfer (up to 3 physics steps), and
# pad-peg chatter spikes with nothing touched (diag/spike_check.py). A real force appears in both, so the robot is
# taken to feel the smaller of the two.
# Contact forces are also reported with isolated one-physics-step outliers of several hundred newtons, both
# peg-socket (chamfer) and pad-peg (contact chatter: the peg would be flung away at ~100 m/s if they were real).
# Every force signal therefore first goes through a 3-step median (removes single-step outliers, passes anything
# lasting two steps = 17 ms or longer), the usual outlier rejection in force sensing.
# Neither limit sees the simulator's raw per-step values: the robot's force estimate is a filtered signal, and
# the gel's load is only observed at the camera's frame rate. Time constants of the first-order filters:
FORCE_FILTER_TAU = 0.010          # s, robot force estimate
GEL_FILTER_TAU = 0.040            # s, one sensor frame
FORCE_ALPHA = 1.0 - math.exp(-SIM_DT / FORCE_FILTER_TAU)   # per physics step (SIM_DT / tau overshoots for tau ~ dt)
GEL_ALPHA = 1.0 - math.exp(-SIM_DT / GEL_FILTER_TAU)
# Friction (PhysX averages the two surfaces' values): peg-socket 0.3 (aluminium on plastic, placeholder),
# peg-gel 1.0 (silicone on metal is high; TacSL's default, placeholder).
FRICTION = dict(peg=0.3, socket=0.3, gel=1.7)
# Where the peg ends up in the hand differs from grasp to grasp, and fixtures are never perfectly
# level. Ranges are placeholders (no measurements of the lab's setup yet).
GRASP_HEIGHT_RANGE = 0.003        # m: the peg sits this much higher or lower in the hand
GRASP_SHIFT_RANGE = 0.002         # m: along the pads, perpendicular to the closing axis
GRASP_TILT_RANGE = math.radians(1.5)   # about the closing axis (the only tilt two flat pads allow)
SOCKET_TILT_RANGE = math.radians(1.0)  # the socket's axis is off vertical by up to this, random direction
TACTILE_DELAY = 1                 # control steps: a GelSight Mini frame is 40 ms old when the policy sees it
# Sensor timing. timing="sync" (all results up to 2026-10-09): every sensor is read on the control clock, the tactile
# reading is exactly TACTILE_DELAY control steps old and the TCP pose is the current one. timing="real": each sensor
# runs on its own clock, as the hardware does, and the controller gets the newest frame that has reached it:
#   GelSight Mini   25 frames/s, free running, each sensor with its own phase (drawn per episode). A frame reaches the
#                   policy TACTILE_LATENCY after it was taken: 10-60 ms inside the sensor, depending on exposure, USB
#                   bandwidth and resolution, plus ~10 ms of marker tracking (RDP paper, appendix B); the middle of
#                   that range. Against a 24 Hz controller the reading is therefore 45-85 ms old, sliding through
#                   that range once a second, and the two pads are up to 40 ms apart.
#   UR5 (CB3)       RTDE reports the TCP pose at 125 Hz: the newest report is one physics step (8 ms) old.
#   controller      a command that takes time to compute is sent that much after the control tick; until the arm's
#                   125 Hz loop has read it the previous command stays in force (step(..., compute_steps)).
# The camera of the wiping task is in wipe_env.py. The default comes from the environment variable SENSOR_TIMING.
TACTILE_FPS = 25.0
TACTILE_LATENCY = 0.045           # s
PROPRIO_DELAY = 1                 # physics steps
FRAME_SLOTS = 4                   # tactile frames kept per sensor (at most two are on their way)
SHEAR_STIFFNESS = 1.0   # shear force per metre of elastomer stretch (TacSL's normal force is 1 per metre of depth)
SHEAR_FRICTION = 1.0    # the contact slips once shear exceeds this times the normal force (the peg's friction)
GEL_SHEAR_STIFFNESS = 6000.0   # N/m per pad, TACTILE_SHEAR = "load": how far the gel's surface moves under the friction force
                               # it carries (a 4 mm silicone layer of 18 x 14 mm with G ~ 0.1 MPa; placeholder)
# The socket sits on a stiff, position-driven XY stage (objects/socket_stage.urdf) so that it can be moved during an
# episode; at rest it behaves as a fixed socket (a 50 N push moves it 0.05 mm).
SOCKET_STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objects", "socket_stage.urdf")
STAGE_STIFFNESS = 1.0e6           # N/m
STAGE_DAMPING = 2.0e3             # N s/m (above critical for the 0.4 kg the x joint carries)
# Perturbation after contact, the RDP paper's hardest test condition (a person moves the object once the tool touches
# it, to break the contact). Here: once the peg first presses on the socket, after a short delay the socket is moved a
# few mm in a random horizontal direction in a smooth (minimum-jerk) move. A move that would start with the peg
# already inside the bore is not made (the insertion has outrun the person). Values are placeholders.
PERTURB_CONTACT_FORCE = 2.0       # N on the socket (filtered): first contact
PERTURB_DELAY = (0.0, 0.2)        # s after first contact
PERTURB_SHIFT = (0.003, 0.006)    # m
PERTURB_TIME = 0.3                # s
PERTURB_MAX_DEPTH = 0.001         # m below the socket top: the peg is in the bore, no move
# The socket position estimate (the camera's) follows the socket, with the same error, this many control steps late.
ESTIMATE_DELAY = 3
# Envs per sim. Measured 2026-10-08 (expert, 192 episodes per setting, two seeds):
#  - beyond 61 envs (30500 tactile points per sensor) TacSL's SDF query returns garbage for the envs after the 61st,
#    without any error (negative normal forces, a handful of points in contact);
#  - with 48 or 96 envs the success rate is the same as with 16, but the peg slides down in the hand in 7% of the
#    episodes instead of 3% (p = 0.007); cause not found (not the grasp procedure, not PhysX's GPU buffer sizes).
# All results are therefore produced with 16 envs per sim; other sizes are for diagnostics only.
MAX_ENVS = 61


class _ForceField(TactileFieldSensor):
    """TacSL's tactile force field on both elastomers, with the peg as the object.

    The normal force is TacSL's (proportional to penetration depth). The shear force is not:
    TacSL's is proportional to sliding speed, so it vanishes whenever the peg is at rest in the
    hand, however hard it is being pushed. Here each tactile point keeps a stretch that follows
    the peg surface while the contact sticks and saturates at the friction limit when it slips,
    so a held load shows up as a steady shear. update() must be called after every sim step.

    That stretch only grows when the peg moves in the hand. Between rigid bodies it does not move until it slips,
    so a load that the grasp holds without slipping leaves no trace (measured on the wiping task: the reading
    explains 9% of the pressure's variance), whereas a real gel shears elastically under any load and its markers
    show it. With env.TACTILE_SHEAR = "load" the shear is therefore the gel's elastic answer to the force it
    passes on: whatever pushes on the held peg from outside (env.force_on_peg(): the object's and the table's
    contact forces, reversed) is carried by the two gels, half each, as friction. Its part in the pad's plane,
    median of three steps and smoothed over one sensor frame, limited to what friction holds
    (SHEAR_FRICTION x the grip force), divided by GEL_SHEAR_STIFFNESS, is shown at the points in contact.
    (The simulator's own contact force on the elastomer bodies is not used: with nothing touching the peg it
    reads -13 N along the tool axis, std 5 N, and does not follow the load. Not modelled: how the two gels share
    the load, and the patterns a moment on the peg leaves.)
    """

    def __init__(self, env, stiffness, damping):
        super().__init__()
        self.gym, self.sim, self.device = env.gym, env.sim, env.device
        self.env = env
        self.env_ptrs, self.num_envs = env.envs, env.num_envs
        self.actor_handles = {ROBOT: env.robot, PEG: env.peg}
        configs = [dict(
            name=side, elastomer_actor_name=ROBOT, elastomer_link_name="elastomer_" + side,
            elastomer_tip_link_name="elastomer_tip_" + side,
            elastomer_parent_urdf_path=os.path.join(ASSET_ROOT, env.asset_file),
            indenter_urdf_path=os.path.join(INDUSTREAL, PEG_FILE),
            indenter_actor_name=PEG, indenter_link_name="industreal_round_peg_16mm",
            actor_handle=env.robot, compliance_stiffness=stiffness, compliant_damping=damping,
            use_acceleration_spring=False,
        ) for side in SIDES]
        self.set_compliant_dynamics_for_tactile_sensors(configs)
        self.sdf_tensor = self.setup_tactile_force_field("physx", ROWS, COLS, configs)

    def start(self, body):
        self.body_pos, self.body_quat = body[..., 0:3], body[..., 3:7]
        self.body_linvel, self.body_angvel = body[..., 7:10], body[..., 10:13]
        self.initialize_penalty_based_tactile(num_divs=[ROWS, COLS])
        # TacSL's fixed read axes suit the GelSight R1.5; the Mini's elastomer faces -z (see tactile_demo.py).
        to_elastomer = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
        quat = torch.tensor(Rotation.from_matrix(to_elastomer).as_quat(), dtype=torch.float32, device=self.device)
        self.tactile_quat_local = quat.unsqueeze(0).repeat(ROWS * COLS, 1)
        self.reset()

    def reset(self):
        """Forget all stretch (call when the peg is not touching the sensors)."""
        points = (self.num_envs, ROWS * COLS)
        self.stretch = {side: torch.zeros(points + (3,), device=self.device) for side in SIDES}
        self.last_on_peg = {side: None for side in SIDES}
        self.force = {side: torch.zeros(points + (3,), device=self.device) for side in SIDES}
        self.carried = {side: torch.zeros((self.num_envs, 3), device=self.device) for side in SIDES}   # friction force on the pad
        self.slow = {side: torch.zeros((self.num_envs, 3), device=self.device) for side in SIDES}      # the late part of the gel's shear

    def update(self):
        """Advance every tactile point's stretch by the peg motion of the last sim step."""
        n, points = self.num_envs, ROWS * COLS
        for side, config in self.tactile_shear_field_configs_dict.items():
            peg, elastomer = config["indenter_link_rb_id"], config["elastomer_link_rb_id"]
            peg_quat, peg_pos = tf_combine(self.body_quat[:, peg], self.body_pos[:, peg],
                                           self.indenter_mesh_local_tf[0].expand(n, 4),
                                           self.indenter_mesh_local_tf[1].expand(n, 3))
            peg_quat, peg_pos = peg_quat.unsqueeze(1).expand(n, points, 4), peg_pos.unsqueeze(1).expand(n, points, 3)
            world, quat = self.get_tactile_points_in_world(self.tactile_pos_local, self.tactile_quat_local, elastomer)
            to_tactile = quat_conjugate(quat)
            on_peg = quat_apply(quat_conjugate(peg_quat), world - peg_pos)      # tactile points in the peg frame
            _, depth, normal = self.query_distance_and_normal_with_physx(on_peg.view(n, 1, points, 3))
            depth = depth.reshape(n, points).clamp(min=0.0)
            pressing = quat_apply(to_tactile, quat_apply(peg_quat, normal.reshape(n, points, 3))) * depth.unsqueeze(-1)

            stretch = self.stretch[side]
            if self.last_on_peg[side] is not None:   # how far the peg surface moved past each point
                stretch = stretch + quat_apply(to_tactile, quat_apply(peg_quat, self.last_on_peg[side] - on_peg))
            self.last_on_peg[side] = on_peg
            stretch[..., 1] = 0.0                    # the tactile frame's y axis is the surface normal
            limit = SHEAR_FRICTION * depth / SHEAR_STIFFNESS
            stretch = stretch * (limit / stretch.norm(dim=-1).clamp(min=1e-9)).clamp(max=1.0).unsqueeze(-1)
            if self.env.TACTILE_SHEAR == "load":
                carried = self.env._despike("carried_" + side, 0.5 * self.env.force_on_peg())
                self.carried[side] = self.carried[side] + (carried - self.carried[side]) * GEL_ALPHA
                shear = quat_apply(to_tactile[:, 0], self.carried[side])     # the points of a pad share one frame
                shear[:, 1] = 0.0
                held = SHEAR_FRICTION * self.env.grip_force_setting
                shear = shear * (held / shear.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
                elastic = shear / GEL_SHEAR_STIFFNESS
                if self.env.gel_creep > 0:    # part of the gel's answer comes late and stays for a while (viscoelastic)
                    self.slow[side] = self.slow[side] + (elastic - self.slow[side]) * (SIM_DT / self.env.gel_creep_tau)
                    elastic = (1 - self.env.gel_creep) * elastic + self.env.gel_creep * self.slow[side]
                if self.env.gel_zero is not None:   # the markers' rest positions have moved since they were taken
                    elastic = elastic + self.env.gel_zero[:, SIDES.index(side)]
                stretch = (depth > 0).unsqueeze(-1) * elastic.unsqueeze(1)
            self.stretch[side] = stretch
            self.force[side] = pressing + SHEAR_STIFFNESS * stretch

    def read(self):
        """(N, 2, ROWS*COLS, 3): per sensor and point, normal force and shear (x, y of the elastomer)."""
        return -torch.stack([self.force[side][..., [1, 0, 2]] for side in SIDES], dim=1)


class InsertionEnv:
    # What the peg works on: a URDF with the object's link on a position-driven stage. Other tasks on the same robot
    # subclass this and override these, _place_object(), _start_task(), _perturb_step(), _after_step() and observe().
    STAGE_FILE = SOCKET_STAGE
    STAGE_LINK = "socket"
    STAGE_FRICTION = FRICTION["socket"]
    STAGE_GAINS = (STAGE_STIFFNESS, STAGE_DAMPING)
    TACTILE_SHEAR = "stretch"     # what the tactile shear shows: "stretch" (slip in the hand) or "load" (see _ForceField)

    def __init__(self, num_envs=1, headless=True, seed=0, socket_range=0.02, estimate_noise=0.004,
                 height_noise=0.004, stiffness=8000.0, damping=0.0, grip_force=GRIP_FORCE, peg_density=2700.0,
                 camera=False, arm_gains=ARM_GAINS, randomize_grasp=True, grasp_tilt=GRASP_TILT_RANGE, substeps=2,
                 peg_damping=0.5, pad_gains=PAD_GAINS, contact_offset=0.001, thickness=0.02, gel_collision="mesh",
                 iterations=16, max_depenetration=5.0, perturb=0.0, timing=None):
        """
        socket_range:   the socket is placed uniformly within +-this (m) in x and y
        estimate_noise: the socket position given to the policy is off by up to this (m), in a random direction
        height_noise:   the socket height given to the policy is off by up to +-this (m), so that a policy
                        cannot tell from the estimate alone when the peg will touch down
        stiffness:      compliant-contact stiffness of the elastomers (sets how deep GRIP_FORCE presses them in)
        damping:        compliant-contact damping of the elastomers
        grip_force:     the pads close until each presses on the peg with this force (N), then lock
        camera:         add a camera to env 0 for snapshot() (pictures for reports; not a policy input)
        perturb:        fraction of the envs in which the socket is moved after first contact (see PERTURB_*)
        timing:         "sync" or "real" (see TACTILE_FPS above); None: the environment variable SENSOR_TIMING, else "sync"
        """
        self.timing = timing or os.environ.get("SENSOR_TIMING", "sync")
        assert self.timing in ("sync", "real"), self.timing
        print("sensor timing:", self.timing)
        self.timing_rng = torch.Generator(device="cpu").manual_seed(seed + 4000)   # separate: layouts stay as before
        self.pose_history = []             # the TCP pose after each of the last PROPRIO_DELAY + 1 physics steps
        # imperfections of the tactile reading, off unless a task turns them on (wipe_env.py, task "v2")
        self.gel_creep, self.gel_creep_tau = 0.0, 1.0     # share of the gel's shear that comes late, and how late (s)
        self.gel_zero = None                              # (N, 2, 3) m, tactile frame: offset of each pad's shear reading
        self.frame_noise = 0.0                            # m per point and axis, drawn anew for every frame (timing="real")
        self.noise_rng = torch.Generator(device="cuda:0").manual_seed(seed + 6000)
        if num_envs > MAX_ENVS:
            raise ValueError("%d envs: the tactile readings are only valid for the first %d (see MAX_ENVS)" % (num_envs, MAX_ENVS))
        self.num_envs, self.device = num_envs, "cuda:0"
        self.perturb = perturb
        self.perturb_rng = torch.Generator(device="cpu").manual_seed(seed + 2000)   # separate: layouts stay as before
        self.socket_range, self.estimate_noise, self.height_noise = socket_range, estimate_noise, height_noise
        self.grip_force_setting, self.randomize_grasp, self.grasp_tilt = grip_force, randomize_grasp, grasp_tilt
        self.asset_file = {"mesh": ASSET_FILE_GSMINI_PRISMATIC, "box": ASSET_FILE_GSMINI_PRISMATIC_BOX}[gel_collision]
        self.estimate_noise_low = 0.0   # raise to force large errors (used for report pictures)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.height_rng = torch.Generator(device="cpu").manual_seed(seed + 1000)   # separate: xy layouts stay as before
        gym = self.gym = gymapi.acquire_gym()
        sim = self.sim = make_sim(gym, dt=SIM_DT, contact_offset=contact_offset, substeps=substeps, iterations=iterations,
                                  max_depenetration=max_depenetration)

        robot_asset = load_robot(gym, sim, self.asset_file, thickness=thickness)
        # asset options as in IndustReal's own task (tasks/industreal/industreal_base.py)
        peg_opts = gymapi.AssetOptions()
        peg_opts.density = peg_density
        peg_opts.thickness = thickness
        peg_opts.linear_damping = peg_opts.angular_damping = peg_damping
        peg_opts.max_angular_velocity = 64.0
        peg_opts.enable_gyroscopic_forces = True
        peg_opts.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
        peg_asset = gym.load_asset(sim, INDUSTREAL, PEG_FILE, peg_opts)
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        fixed.thickness = thickness
        fixed.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
        socket_asset = gym.load_asset(sim, os.path.dirname(self.STAGE_FILE), os.path.basename(self.STAGE_FILE), fixed)
        table_asset = gym.create_box(sim, 0.4, 0.4, TABLE_HEIGHT, fixed)

        # Kinematics from the URDF: pad travel, and where the TCP is at the home pose.
        model = URDF.load(os.path.join(ASSET_ROOT, self.asset_file))
        self.mimic = {j.name: j.mimic.multiplier for j in model.joints if j.mimic is not None}
        lo, hi = pad_limits(model)
        self.q_open = solve_finger_joint(model, PEG_DIAMETER + 0.004, driver=PAD_DRIVER, lower=lo, upper=hi)
        self.q_close = hi            # the pads never need to go further than this
        home_tcp = {l.name: tf for l, tf in model.link_fk(cfg=dict(ARM_HOME, **{PAD_DRIVER: 0.0})).items()}["tcp"]
        self.home_xy = torch.tensor(home_tcp[:2, 3], dtype=torch.float32, device=self.device)
        self.home_closing_axis = torch.tensor(home_tcp[:3, 1], dtype=torch.float32, device=self.device)   # TCP y: the pads close along it

        self.envs = []
        per_row = int(math.ceil(math.sqrt(num_envs)))
        for i in range(num_envs):
            env = gym.create_env(sim, gymapi.Vec3(-0.6, -0.6, 0), gymapi.Vec3(0.9, 0.9, 1.2), per_row)
            self.envs.append(env)
            robot = gym.create_actor(env, robot_asset, gymapi.Transform(), ROBOT, i, 0)
            names, self.dof_props = set_drive_gains(gym, env, robot, arm_gains=arm_gains, pad_gains=pad_gains)
            gym.enable_actor_dof_force_sensors(env, robot)
            enable_sensor_self_collision(gym, env, robot)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(home_tcp[0, 3], home_tcp[1, 3], home_tcp[2, 3] - PEG_GRASP)
            peg = gym.create_actor(env, peg_asset, pose, PEG, i, 0)
            pose.p = gymapi.Vec3(home_tcp[0, 3], home_tcp[1, 3], TABLE_HEIGHT)
            socket = gym.create_actor(env, socket_asset, pose, SOCKET, i, 0)
            pose.p = gymapi.Vec3(home_tcp[0, 3], home_tcp[1, 3], TABLE_HEIGHT / 2)
            table = gym.create_actor(env, table_asset, pose, TABLE, i, 0)
            gym.set_rigid_body_color(env, table, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.55, 0.5, 0.45))
            gym.set_rigid_body_color(env, peg, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.5, 0.1))
            for actor, friction in ((peg, FRICTION["peg"]), (socket, self.STAGE_FRICTION)):
                shapes = gym.get_actor_rigid_shape_properties(env, actor)
                shapes[0].friction = friction
                gym.set_actor_rigid_shape_properties(env, actor, shapes)
            # the socket rests on the stage, not on the table: the two must not collide (shared filter bit;
            # the robot's shapes use bits 0b11, the peg 0)
            for actor in (socket, table):
                shapes = gym.get_actor_rigid_shape_properties(env, actor)
                for s in shapes:
                    s.filter = 0b100
                gym.set_actor_rigid_shape_properties(env, actor, shapes)
            stage = gym.get_actor_dof_properties(env, socket)
            stage["driveMode"][:] = gymapi.DOF_MODE_POS
            stage["stiffness"][:], stage["damping"][:] = self.STAGE_GAINS
            gym.set_actor_dof_properties(env, socket, stage)
        self.robot, self.peg, self.socket, self.names = robot, peg, socket, names
        self.actors_per_env = gym.get_actor_count(self.envs[0])
        self.finger_dofs = [i for i, n in enumerate(names) if n not in ARM_HOME]
        assert [names[i] for i in self.finger_dofs] == list(PAD_JOINTS), names
        self.finger_ratio = torch.tensor([1.0 if names[i] == PAD_DRIVER else self.mimic[names[i]] for i in self.finger_dofs],
                                         device=self.device)
        count = gym.get_actor_dof_count(self.envs[0], socket)
        self.stage_dofs = [gym.get_actor_dof_index(self.envs[0], socket, i, gymapi.DOMAIN_ENV) for i in range(count)]
        assert self.stage_dofs == list(range(len(names), len(names) + count)), self.stage_dofs   # robot DOFs first, then the stage's
        # all DOFs of an env (robot, stage); the stage's targets are zero unless the socket is being moved
        self.home = torch.tensor([ARM_HOME.get(n, 0.0) for n in names] + [0.0] * count, device=self.device)

        self.controller = EEController(gym, sim, self.envs[0], robot, ROBOT, num_envs)
        self.field = _ForceField(self, stiffness, damping)
        body_names = gym.get_actor_rigid_body_names(self.envs[0], robot)
        shape_ranges = gym.get_actor_rigid_body_shape_indices(self.envs[0], robot)
        for env in self.envs:   # after the sensor setup, which touches the same shape properties
            shapes = gym.get_actor_rigid_shape_properties(env, robot)
            for side in SIDES:
                r = shape_ranges[body_names.index("elastomer_" + side)]
                for i in range(r.start, r.start + r.count):
                    shapes[i].friction = FRICTION["gel"]
            gym.set_actor_rigid_shape_properties(env, robot, shapes)

        self.viewer = None
        if not headless:
            self.viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            look = gymapi.Vec3(home_tcp[0, 3], home_tcp[1, 3], TABLE_HEIGHT + 0.05)
            gym.viewer_camera_look_at(self.viewer, self.envs[0], gymapi.Vec3(look.x + 0.3, look.y + 0.35, look.z + 0.2), look)

        self.camera = None
        if camera:
            props = gymapi.CameraProperties()
            props.width, props.height = 1280, 960
            self.camera = gym.create_camera_sensor(self.envs[0], props)
            look = gymapi.Vec3(home_tcp[0, 3], home_tcp[1, 3], TABLE_HEIGHT + 0.06)
            gym.set_camera_location(self.camera, self.envs[0],
                                    gymapi.Vec3(look.x + 0.16, look.y + 0.20, look.z + 0.10), look)

        gym.prepare_sim(sim)
        self.controller.start()
        self.root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(num_envs, self.actors_per_env, 13)
        self.contact = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim)).view(num_envs, -1)
        find = lambda name: gym.find_actor_rigid_body_index(self.envs[0], robot, name, gymapi.DOMAIN_ENV)
        self.elastomer_bodies = [find("elastomer_" + side) for side in SIDES]
        self.socket_body = gym.find_actor_rigid_body_index(self.envs[0], socket, self.STAGE_LINK, gymapi.DOMAIN_ENV)
        self.table_body = gym.get_actor_rigid_body_index(self.envs[0], table, 0, gymapi.DOMAIN_ENV)
        self.robot_bodies = [gym.get_actor_rigid_body_index(self.envs[0], robot, i, gymapi.DOMAIN_ENV)
                             for i in range(gym.get_actor_rigid_body_count(self.envs[0], robot))]
        self.dof = self.controller.dof
        self.body = self.controller.body
        self._idx = lambda actor: torch.tensor(
            [gym.get_actor_index(e, actor, gymapi.DOMAIN_SIM) for e in self.envs], dtype=torch.int32, device=self.device)
        self.peg_sim_idx, self.socket_sim_idx = self._idx(peg), self._idx(socket)
        self.targets = self.home.repeat(num_envs, 1)
        self.field_started = False
        self.stopped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.gel_overload = torch.zeros_like(self.stopped)
        self.peak_socket_force = torch.zeros(num_envs, device=self.device)
        self.filtered_socket_force = torch.zeros(num_envs, device=self.device)
        self.filtered_tool_force = torch.zeros(num_envs, device=self.device)
        self._force_hist = {}
        self.peak_tool_force = torch.zeros(num_envs, device=self.device)
        self.filtered_grip = torch.zeros(num_envs, 2, device=self.device)
        self.stop_pos, self.stop_quat = None, None
        self.in_episode = False            # the perturbation only runs between reset() and the next reset()

    # ---------------------------------------------------------------- low level
    def _servo(self, tcp_pos, tcp_quat):
        """What the arm's own controller is heading for: the command one cycle ago, blended in over SERVO_TAU."""
        halted = self.stopped.unsqueeze(-1)
        tcp_pos, tcp_quat = torch.where(halted, self.stop_pos, tcp_pos), torch.where(halted, self.stop_quat, tcp_quat)
        self.servo_queue.append((tcp_pos.clone(), tcp_quat.clone()))
        cmd_pos, cmd_quat = self.servo_queue.pop(0)
        a = SIM_DT / SERVO_TAU
        self.servo_pos += a * (cmd_pos - self.servo_pos)
        sign = torch.sign((cmd_quat * self.servo_quat).sum(-1, keepdim=True))
        q = self.servo_quat + a * (cmd_quat * sign - self.servo_quat)
        self.servo_quat = q / q.norm(dim=-1, keepdim=True)
        return self.servo_pos, self.servo_quat

    def _sim_step(self, tcp_pos=None, tcp_quat=None, q_finger=None):
        gym, sim = self.gym, self.sim
        if tcp_pos is not None:
            self.controller.refresh()
            self.targets[:, self.controller.arm_dofs] = self.controller.joint_targets(*self._servo(tcp_pos, tcp_quat))
        if q_finger is not None:   # a scalar, or (N, pads) with the mimic ratio already applied
            self.targets[:, self.finger_dofs] = q_finger * self.finger_ratio if not torch.is_tensor(q_finger) else q_finger
        if self.in_episode:
            self._perturb_step()
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(self.targets))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        self.controller.refresh()
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        self.pose_history = (self.pose_history + [self.controller.pose()])[-(PROPRIO_DELAY + 1):]
        if self.field_started:
            self.field.update()
        if self.in_episode and self.timing == "real":
            self._capture()
        # force limits, checked every physics step on filtered signals: a protective stop freezes the arm for the episode
        self.filtered_socket_force += (self._despike("socket", self.socket_force()) - self.filtered_socket_force) * FORCE_ALPHA
        felt = torch.minimum(self._despike("world", self.world_force()), self._despike("links", self.link_force()))
        self.filtered_tool_force += (felt - self.filtered_tool_force) * FORCE_ALPHA
        self.filtered_grip += (self._despike("grip", self.grip_force()) - self.filtered_grip) * GEL_ALPHA
        self.peak_socket_force = torch.maximum(self.peak_socket_force, self.filtered_socket_force)
        self.peak_tool_force = torch.maximum(self.peak_tool_force, self.filtered_tool_force)
        newly = (self.filtered_tool_force > STOP_FORCE) & ~self.stopped
        if newly.any():
            pos, quat = self.controller.pose()
            self.stop_pos = torch.where(newly.unsqueeze(-1), pos, self.stop_pos)
            self.stop_quat = torch.where(newly.unsqueeze(-1), quat, self.stop_quat)
            self.stopped |= newly
        self.gel_overload |= self.filtered_grip.max(dim=1).values > GEL_FORCE_LIMIT
        if self.in_episode:
            self._after_step()
        if self.viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(self.viewer, sim, True)
            gym.sync_frame_time(sim)

    def _perturb_step(self):
        """Stage targets for the next physics step: start, carry out, or call off the move after first contact."""
        self.time += SIM_DT
        first_contact = self.will_perturb & (self.move_start < 0) & (self.filtered_socket_force > PERTURB_CONTACT_FORCE)
        self.move_start = torch.where(first_contact, self.time + self.perturb_delay, self.move_start)
        starting = (self.move_start >= 0) & ~self.socket_moved & ~self.move_called_off & (self.time >= self.move_start)
        in_bore = SOCKET_TOP - self._peg_bottom_in_socket()[:, 2] > PERTURB_MAX_DEPTH
        self.move_called_off |= starting & in_bore
        self.socket_moved |= starting & ~in_bore
        s = ((self.time - self.move_start) / PERTURB_TIME).clamp(0.0, 1.0) * self.socket_moved
        self.targets[:, self.stage_dofs] = self.perturb_shift * (s * s * s * (10 - 15 * s + 6 * s * s)).unsqueeze(-1)

    def _capture(self):
        """timing="real", after every physics step of an episode: the tactile sensors whose next frame is due take it."""
        due = self.tactile_next <= self.time + 1e-9                               # (N, 2)
        if due.any():
            env, side = due.nonzero(as_tuple=True)
            slot = self.tactile_slot[env, side]
            frame = self.field.read()[env, side]
            if self.frame_noise > 0:
                frame = frame + self.frame_noise * torch.randn(frame.shape, generator=self.noise_rng, device=self.device)
            self.frames[env, side, slot] = frame
            self.frame_taken[env, side, slot] = self.time
            self.tactile_slot[env, side] = (slot + 1) % FRAME_SLOTS
            self.tactile_next[env, side] += 1.0 / TACTILE_FPS

    def _newest_frames(self):
        """timing="real": per sensor the newest tactile frame that has arrived, (N, 2, points, 3), and its age (N, 2) in s."""
        arrived = self.frame_taken + TACTILE_LATENCY <= self.time + 1e-9          # (N, 2, FRAME_SLOTS)
        newest = torch.where(arrived, self.frame_taken, torch.full_like(self.frame_taken, -math.inf)).argmax(dim=-1)
        index = newest[:, :, None, None, None].expand(-1, -1, 1, *self.frames.shape[3:])
        age = self.time - self.frame_taken.gather(2, newest.unsqueeze(-1)).squeeze(-1)
        return self.frames.gather(2, index).squeeze(2), age.float()

    def _peg_bottom_in_socket(self):
        """(N, 3): the centre of the peg's bottom face in the socket's frame (z = 0 at the socket's base)."""
        peg_pos, peg_quat = self.root[:, self.peg, 0:3], self.root[:, self.peg, 3:7]
        socket_pos, socket_quat = self.body[:, self.socket_body, 0:3], self.body[:, self.socket_body, 3:7]
        bottom = peg_pos + quat_apply(peg_quat, torch.tensor([[0.0, 0.0, PEG_BOTTOM]], device=self.device).expand_as(peg_pos))
        return quat_apply(quat_conjugate(socket_quat), bottom - socket_pos)

    def snapshot(self, path):
        """Save what the report camera sees in env 0."""
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera, gymapi.IMAGE_COLOR, path)

    def viewer_closed(self):
        return self.viewer is not None and self.gym.query_viewer_has_closed(self.viewer)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.rng).to(self.device)

    def grip_force(self):
        """(N, 2): normal force on each elastomer, from the simulator's contact forces."""
        return torch.stack([self.contact[:, b].norm(dim=-1) for b in self.elastomer_bodies], dim=1)

    def _despike(self, name, value):
        """Median of this signal's last three physics steps (see the force-limit comment at the top)."""
        hist = self._force_hist.get(name)
        hist = value.unsqueeze(0).repeat(3, *[1] * value.dim()) if hist is None else torch.cat([hist[1:], value.unsqueeze(0)])
        self._force_hist[name] = hist
        return hist.median(dim=0).values

    def world_force(self):
        """(N,): magnitude of the force robot + peg exert on the fixed world (socket and table)."""
        return (self.contact[:, self.socket_body] + self.contact[:, self.table_body]).norm(dim=-1)

    def link_force(self):
        """(N,): magnitude of the summed contact forces on all robot links (the two pad normal forces cancel)."""
        return self.contact[:, self.robot_bodies].sum(dim=1).norm(dim=-1)

    def force_on_peg(self):
        """(N, 3): what pushes on the held peg from outside (the object it works on, the table), world frame."""
        return -(self.contact[:, self.socket_body] + self.contact[:, self.table_body])

    def socket_force(self):
        """(N,): force the socket feels (the peg pressing on it)."""
        return self.contact[:, self.socket_body].norm(dim=-1)

    # ---------------------------------------------------------------- episode
    def reset(self):
        """New socket positions, peg back in the gripper. Returns the first observation."""
        gym, sim, n = self.gym, self.sim, self.num_envs
        self.in_episode = False
        self.gym.refresh_actor_root_state_tensor(sim)
        root = self.root.clone()

        self._place_object(root)

        # robot at home with the fingers slightly open, the stage at zero
        state = torch.zeros((n, len(self.home), 2), device=self.device)
        state[..., 0] = self.home
        state[:, self.finger_dofs, 0] = self.q_open * self.finger_ratio
        self.targets = state[..., 0].clone()
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(state))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(self.targets))
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        self._sim_step()
        self.start_pos, self.start_quat = self.controller.pose()
        self.servo_pos, self.servo_quat = self.start_pos.clone(), self.start_quat.clone()
        self.servo_queue = [(self.start_pos.clone(), self.start_quat.clone())] * SERVO_DELAY
        self.stopped[:], self.gel_overload[:], self.peak_socket_force[:], self.peak_tool_force[:] = False, False, 0.0, 0.0
        self.filtered_socket_force[:], self.filtered_tool_force[:], self.filtered_grip[:] = 0.0, 0.0, 0.0
        self._force_hist = {}
        self.stop_pos, self.stop_quat = self.start_pos.clone(), self.start_quat.clone()
        if not self.field_started:
            self.field.start(self.body)
            self.field_started = True
        self.field.reset()   # fingers are open: no stretch; closing on the peg and its weight are tracked from here

        # Grasp, the way the real gripper does it: the pads close until each presses on the peg with the
        # set force, then they stay where they are (worm gear). The peg is held in place (upright,
        # PEG_GRASP at the TCP) while the pads close on it.
        # the peg is not in the same place every time: shifted along the pads, higher or lower, tilted about the closing axis
        r = float(self.randomize_grasp)
        shift = torch.stack([(2 * self._rand(n) - 1) * GRASP_SHIFT_RANGE * r, torch.zeros(n, device=self.device),
                             (2 * self._rand(n) - 1) * GRASP_HEIGHT_RANGE * r], dim=-1)
        grasp_point = self.start_pos + quat_apply(self.start_quat, shift)
        closing_axis = quat_apply(self.start_quat, torch.tensor([[0.0, 1.0, 0.0]], device=self.device).expand(n, 3))
        peg_quat = quat_from_angle_axis((2 * self._rand(n) - 1) * self.grasp_tilt * r, closing_axis)
        peg_state = torch.zeros((n, 13), device=self.device)
        peg_state[:, 0:3] = grasp_point - quat_apply(peg_quat, torch.tensor([[0.0, 0.0, PEG_GRASP]], device=self.device).expand(n, 3))
        peg_state[:, 3:7] = peg_quat
        q = torch.full((n,), self.q_open, device=self.device)
        for k in range(2000):   # 1 um per step once touching: enough travel for any force setting
            root = self.root.clone()
            root[:, self.peg] = peg_state
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root),
                                                    gymtorch.unwrap_tensor(self.peg_sim_idx), n)
            grip = self.grip_force()
            closing = grip.mean(dim=1) < self.grip_force_setting
            if k >= 30 and not closing.any():
                break
            step = torch.where(grip.max(dim=1).values > 1.0, CLOSE_STEP_TOUCHING, CLOSE_STEP)
            q = torch.where(closing, (q + step).clamp(max=self.q_close), q)
            self._sim_step(self.start_pos, self.start_quat, q.unsqueeze(1) * self.finger_ratio)
        for _ in range(30):  # let go of the peg and let the grasp settle
            self._sim_step(self.start_pos, self.start_quat)
        # The peg seats itself once it is free, which changes the contact; the motor keeps regulating
        # to its force setting (open or close a little) before it stops and the worm gear holds.
        q = self.targets[:, self.finger_dofs[0]].clone()
        for k in range(300):
            grip = self.grip_force().mean(dim=1)
            low, high = grip < 0.9 * self.grip_force_setting, grip > 1.1 * self.grip_force_setting
            if k >= 30 and not (low | high).any():
                break
            q = torch.where(low, q + CLOSE_STEP_TOUCHING, torch.where(high, q - CLOSE_STEP_TOUCHING, q)).clamp(max=self.q_close)
            self._sim_step(self.start_pos, self.start_quat, q.unsqueeze(1) * self.finger_ratio)
        for _ in range(30):
            self._sim_step(self.start_pos, self.start_quat)
        self.start_grip = self.grip_force()
        # The grasp is not part of the task. Closing by position steps is only an imitation of the
        # gripper's force-limited motor and can overshoot for a moment; note it, but do not fail the episode.
        self.grasp_overload = self.gel_overload.clone()
        self.stopped[:], self.gel_overload[:], self.peak_socket_force[:], self.peak_tool_force[:] = False, False, 0.0, 0.0

        self.command_pos, self.command_quat = self.start_pos.clone(), self.start_quat.clone()
        self.tactile_queue = [self.field.read()] * TACTILE_DELAY
        self.time = 0.0
        if self.timing == "real":   # each sensor's clock starts somewhere in its frame period and has been running:
            n = self.num_envs       # its last three frames (of the hand at rest) are in the buffer, on their way or arrived
            first = self.field.read()
            self.frames = first.unsqueeze(2).repeat(1, 1, FRAME_SLOTS, 1, 1)                      # (N, 2, slots, points, 3)
            self.tactile_next = (torch.rand((n, 2), generator=self.timing_rng, dtype=torch.float64) / TACTILE_FPS).to(self.device)
            self.frame_taken = torch.full((n, 2, FRAME_SLOTS), -math.inf, dtype=torch.float64, device=self.device)
            for i in range(3):
                self.frame_taken[:, :, i] = self.tactile_next - (3 - i) / TACTILE_FPS
            self.tactile_slot = torch.full((n, 2), 3, dtype=torch.long, device=self.device)
        self._start_task()
        self.in_episode = True
        return self.observe()

    def _place_object(self, root):
        """Write the object's pose for the new episode into `root` (N, actors, 13); draw the estimate and the perturbation."""
        n = self.num_envs
        socket_xy = self.home_xy + (2 * self._rand(n, 2) - 1) * self.socket_range
        root[:, self.socket, 0:2] = socket_xy
        root[:, self.socket, 2] = TABLE_HEIGHT
        tilt_dir = 2 * math.pi * self._rand(n)
        tilt_axis = torch.stack([torch.cos(tilt_dir), torch.sin(tilt_dir), torch.zeros(n, device=self.device)], dim=-1)
        root[:, self.socket, 3:7] = quat_from_angle_axis((2 * self._rand(n) - 1) * SOCKET_TILT_RANGE, tilt_axis)
        root[:, self.socket, 7:13] = 0.0
        angle = 2 * math.pi * self._rand(n)
        lo = (self.estimate_noise_low / self.estimate_noise) ** 2
        radius = self.estimate_noise * torch.sqrt(lo + (1 - lo) * self._rand(n))  # uniform over a disc (or ring)
        self.socket_estimate = torch.cat([socket_xy + torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], -1),
                                          torch.full((n, 1), TABLE_HEIGHT + SOCKET_TOP, device=self.device)], dim=-1)
        height_error = (2 * torch.rand(n, generator=self.height_rng) - 1).to(self.device) * self.height_noise
        self.socket_estimate[:, 2] += height_error
        self.socket_xy = socket_xy
        self.estimate_offset = self.socket_estimate[:, 0:2] - socket_xy   # the estimate keeps this error when the socket moves
        self.estimate_queue = [socket_xy.clone()] * ESTIMATE_DELAY
        # perturbation after contact (drawn for every env so that the layouts do not depend on `perturb`)
        draw = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=self.perturb_rng).to(self.device)
        self.will_perturb = draw(0.0, 1.0) < self.perturb
        direction = draw(-math.pi, math.pi)
        self.perturb_shift = draw(*PERTURB_SHIFT).unsqueeze(-1) * torch.stack([torch.cos(direction), torch.sin(direction)], -1)
        self.perturb_delay = draw(*PERTURB_DELAY)

    def _start_task(self):
        """Task state at the start of an episode (the peg is in the hand, nothing has been touched)."""
        n = self.num_envs
        self.move_start = torch.full((n,), -1.0, device=self.device)
        self.socket_moved = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.move_called_off = torch.zeros_like(self.socket_moved)

    def _after_step(self):
        """Called after every physics step of an episode, with the filtered forces up to date (nothing to do here)."""

    def step(self, target_pos, target_quat, compute_steps=0):
        """Hold a TCP target (N, 3), (N, 4) for one control period. compute_steps: physics steps that pass before the
        target is sent (the time the controller took to compute it); the previous target stays in force until then."""
        assert 0 <= compute_steps < SIM_STEPS_PER_ACTION
        previous = (self.command_pos, self.command_quat)
        self.command_pos, self.command_quat = target_pos, target_quat
        for i in range(SIM_STEPS_PER_ACTION):
            self._sim_step(*(previous if i < compute_steps else (target_pos, target_quat)))
        return self.observe()

    # ---------------------------------------------------------------- observation
    def _common_obs(self):
        """What every task observes: robot, peg in the hand, tactile reading, forces and the safety flags."""
        tcp_pos, tcp_quat = self.controller.pose()
        peg_pos, peg_quat = self.root[:, self.peg, 0:3].clone(), self.root[:, self.peg, 3:7].clone()
        bottom = peg_pos + quat_apply(peg_quat, torch.tensor([[0.0, 0.0, PEG_BOTTOM]], device=self.device).expand_as(peg_pos))
        # the tactile reading is as old as a real sensor frame would be
        if self.timing == "real":
            tactile, tactile_age = self._newest_frames()
            seen_pos, seen_quat = (x.clone() for x in self.pose_history[0])      # the robot's newest report
        else:
            self.tactile_queue.append(self.field.read())
            tactile = self.tactile_queue.pop(0)
            tactile_age = torch.full((self.num_envs, 2), TACTILE_DELAY * CONTROL_DT, device=self.device)
            seen_pos, seen_quat = tcp_pos, tcp_quat
        # where the peg sits in the hand: its grasp point in the TCP frame (zero for a perfect grasp)
        grasp = peg_pos + quat_apply(peg_quat, torch.tensor([[0.0, 0.0, PEG_GRASP]], device=self.device).expand_as(peg_pos))
        in_hand = quat_apply(quat_conjugate(tcp_quat), grasp - tcp_pos)
        return dict(
            # what a policy may use
            tcp_pos=seen_pos, tcp_quat=seen_quat, tactile=tactile,
            tactile_age=tactile_age,                           # s since each pad's frame was taken (a real driver's timestamp)
            # ground truth, for the scripted expert and for scoring only
            peg_bottom=bottom, peg_quat=peg_quat, peg_in_hand=in_hand,
            grip_force=self.filtered_grip.clone(), socket_force=self.filtered_socket_force.clone(),
            peak_socket_force=self.peak_socket_force.clone(),
            tool_force=self.filtered_tool_force.clone(), peak_tool_force=self.peak_tool_force.clone(),
            stopped=self.stopped.clone(), gel_overload=self.gel_overload.clone(), grasp_overload=self.grasp_overload.clone(),
        )

    def observe(self):
        socket_pos, socket_quat = self.body[:, self.socket_body, 0:3].clone(), self.body[:, self.socket_body, 3:7].clone()
        bottom_in_socket = self._peg_bottom_in_socket()
        depth = SOCKET_TOP - bottom_in_socket[:, 2]
        # the camera's estimate of the socket position: follows the socket's moves, ESTIMATE_DELAY control steps late
        # (the stage's commanded shift: its deflection under the peg's push, ~0.02 mm, is far below what a camera sees)
        self.estimate_queue.append(self.socket_xy + self.targets[:, self.stage_dofs])
        self.socket_estimate[:, 0:2] = self.estimate_queue.pop(0) + self.estimate_offset
        lateral = bottom_in_socket[:, 0:2].norm(dim=-1)
        return dict(
            self._common_obs(),
            socket_estimate=self.socket_estimate.clone(),      # a policy may use this one
            socket_pos=socket_pos, socket_quat=socket_quat,    # the rest is ground truth, as above
            insertion_depth=depth, lateral_error=lateral,
            socket_moved=self.socket_moved.clone(), move_called_off=self.move_called_off.clone(),
            success=(depth > 0.8 * SOCKET_DEPTH) & (lateral < 0.002) & ~self.stopped & ~self.gel_overload,
        )

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
