"""Shared pieces for the UR5 + Robotiq 2F-85 scripts."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.join(HERE, "ur5_2f85_isaacgym")  # built by convert_for_isaacgym.py
ASSET_FILE = "urdf/ur5_robotiq_85.urdf"
ASSET_FILE_GSMINI = "urdf/ur5_robotiq_85_gsmini.urdf"  # built by add_gsmini.py
ASSET_FILE_GSMINI_PRISMATIC = "urdf/ur5_robotiq_85_gsmini_prismatic.urdf"  # built by make_prismatic_gripper.py
ASSET_FILE_GSMINI_PRISMATIC_BOX = "urdf/ur5_robotiq_85_gsmini_prismatic_box.urdf"  # same, box collision for the gel

ARM_HOME = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.0,
}
DRIVER = "finger_joint"
PAD_DRIVER = "left_pad_joint"   # the prismatic variant; right_pad_joint mimics it
PAD_JOINTS = ("left_pad_joint", "right_pad_joint")
TCP_LINK = "tcp"  # between the sensing surfaces; only in the GelSight variant
# Position-drive gains. The finger joints are stiff so the two (independently driven) fingers
# close symmetrically under contact; with soft gains one side lagged by ~0.5 mm.
# Arm joint position drives. 4000/80 gave the TCP a stiffness of about 7 N/mm against a contact
# (measured, diag/force_check.py), which is far softer than a UR5 holding a position: 40000/400
# gives about 50 N/mm (measured, diag/grip_calib.py). The real arm is stiffer still; its contact
# force is bounded by the protective stop (STOP_FORCE in insertion_env.py), not by compliance.
ARM_GAINS = (40000.0, 400.0)
FINGER_GAINS = (2000.0, 50.0)
# Prismatic pads (insertion environment): the drive is a stiff position servo whose force is capped
# at the grip force while closing; after the grasp the cap is raised to imitate the worm gear's self-locking.
PAD_GAINS = (1.0e5, 2.0e2)      # N/m, N s/m
GRIP_FORCE = 20.0               # N per pad: the 2F-85's lowest setting (spec: 20-235 N); gel sensors are run at low force
LOCK_FORCE = 1000.0             # N: the pads do not back-drive once the grasp is made
SENSOR_LINKS = {
    "left": ("gelsight_sensor_left", "elastomer_left"),
    "right": ("gelsight_sensor_right", "elastomer_right"),
}


def other_gpu_pids():
    """Other processes that would disturb a sim on this GPU: anything with Isaac Gym loaded, and
    anything computing on the GPU (a training run, a notebook), as listed by nvidia-smi."""
    import subprocess

    pids = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if os.stat("/proc/" + pid).st_uid == os.getuid() and "isaacgym/_bindings" in open("/proc/%s/maps" % pid).read():
                pids.add(int(pid))
        except OSError:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        pids.update(int(line) for line in out.split() if line.strip().isdigit())
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(pids - {os.getpid()})


def refuse_if_other_sim():
    """Exit if anything else is using the GPU.

    On this GPU the tactile force fields come out wrong, without any error, whenever another
    process computes on it at the same time: a second sim, but also a PyTorch training run
    (measured: the same evaluation gave 61/96 alone and 34/96 next to a training run; the
    physics was identical, only the tactile readings differed).
    """
    others = other_gpu_pids()
    if others:
        raise SystemExit("Another process is using the GPU (pid %s). Anything computing on the GPU at the same time "
                         "corrupts the tactile readings without any error; stop it first." % ", ".join(map(str, others)))


def make_sim(gym, gpu_pipeline=True, dt=1.0 / 60.0, contact_offset=0.001, substeps=2, iterations=16, max_depenetration=5.0):
    """PhysX sim with a ground plane, set up the way TacSL sets up its contact-rich scenes, plus
    IndustReal's max_depenetration_velocity: without it the peg's edge passing the chamfer produced
    single-step contact forces of 10^4-10^5 N. (IndustReal's other solver settings - contact offset
    10 mm, friction offset 10 mm - made the peg jam in the bore here: 3/96 for the expert.)"""
    from isaacgym import gymapi

    refuse_if_other_sim()
    params = gymapi.SimParams()
    params.dt = dt
    params.substeps = substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.physx.solver_type = 1
    params.physx.num_position_iterations = iterations
    params.physx.num_velocity_iterations = 1
    params.physx.rest_offset = 0.0
    params.physx.contact_offset = contact_offset
    params.physx.max_depenetration_velocity = max_depenetration
    params.physx.use_gpu = True
    params.use_gpu_pipeline = gpu_pipeline
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    return sim


def load_robot(gym, sim, asset_file=ASSET_FILE_GSMINI, thickness=0.02):
    """The UR5 + 2F-85 asset: fixed base, gravity off (a real UR compensates its own weight)."""
    from isaacgym import gymapi

    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    opts.disable_gravity = True
    opts.collapse_fixed_joints = False
    opts.thickness = thickness     # Isaac Gym's default is 0.02 m; IndustReal/TacSL set 0 on every asset
    return gym.load_asset(sim, ASSET_ROOT, asset_file, opts)


def link_poses(robot, cfg):
    """World pose (4x4) of every link of a urdfpy robot; mimic joints follow the driver."""
    return {link.name: tf for link, tf in robot.link_fk(cfg=cfg).items()}


def _tip_offset(robot, q, driver=DRIVER):
    """Closing-axis offset of the right sensing surface from the left one, in the gripper base frame."""
    pose = link_poses(robot, {driver: q})
    to_base = np.linalg.inv(pose["robotiq_arg2f_base_link"])
    left = (to_base @ pose["elastomer_tip_left"])[:3, 3]
    right = (to_base @ pose["elastomer_tip_right"])[:3, 3]
    return float(right[1] - left[1])


def sensor_gap(robot, q, driver=DRIVER):
    """Distance between the two sensing surfaces (negative = overlapping)."""
    return _tip_offset(robot, q, driver) * np.sign(_tip_offset(robot, 0.0, driver))


def solve_finger_joint(robot, gap, upper=0.8, driver=DRIVER, lower=0.0):
    """Driver joint value at which the sensing surfaces are `gap` metres apart (closing = increasing)."""
    lo, hi = lower, upper
    assert sensor_gap(robot, hi, driver) < gap < sensor_gap(robot, lo, driver), "gap not reachable"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sensor_gap(robot, mid, driver) > gap else (lo, mid)
    return lo


def pad_limits(robot):
    """(lower, upper) of the prismatic pad joints."""
    j = {j.name: j for j in robot.joints}[PAD_DRIVER]
    return j.limit.lower, j.limit.upper


def set_drive_gains(gym, env, actor, pad_force=LOCK_FORCE, arm_gains=ARM_GAINS, pad_gains=PAD_GAINS):
    """Position drive on every joint, with the gains above; pad_force caps the pad drives."""
    from isaacgym import gymapi

    names = gym.get_actor_dof_names(env, actor)
    props = gym.get_actor_dof_properties(env, actor)
    props["driveMode"][:] = gymapi.DOF_MODE_POS
    for i, n in enumerate(names):
        if n in PAD_JOINTS:
            props["stiffness"][i], props["damping"][i] = pad_gains
            props["effort"][i] = pad_force
        else:
            props["stiffness"][i], props["damping"][i] = arm_gains if n in ARM_HOME else FINGER_GAINS
    gym.set_actor_dof_properties(env, actor, props)
    return names, props


def enable_sensor_self_collision(gym, env, actor):
    """Let the left and right sensors collide with each other, and nothing else on the robot with itself.

    Isaac Gym skips a pair of shapes when their filter masks share a bit. The actor must have been
    created with collision filter 0, otherwise these per-shape masks are ignored.
    """
    names = gym.get_actor_rigid_body_names(env, actor)
    ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
    props = gym.get_actor_rigid_shape_properties(env, actor)
    for p in props:
        p.filter = 0b11
    for bit, side in ((0b01, "left"), (0b10, "right")):
        for link in SENSOR_LINKS[side]:
            r = ranges[names.index(link)]
            for i in range(r.start, r.start + r.count):
                props[i].filter = bit
    gym.set_actor_rigid_shape_properties(env, actor, props)
