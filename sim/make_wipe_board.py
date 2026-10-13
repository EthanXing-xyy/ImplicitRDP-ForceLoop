"""Build the board of the wiping task (wipe_env.py): objects/wipe_board.obj and objects/wipe_board_stage.urdf.

A 150 x 60 x 10 mm slab whose top face is BOARD_TOP above the stage's base (the same height as the socket's
top, so the approach is the same as in the insertion task). The slab is a triangle mesh with an SDF, like the
socket (the peg slides on the socket's rim without trouble; a box primitive is not used because the freed peg
has been seen to pass through the table, which is one). All faces are subdivided so that contacts
are generated all over it (2.3 mm edges).

The board sits on a five-axis stage: two stiff slides in its plane, two stiff tilt axes through the middle of
the line, and a slide along its normal held by a spring-damper drive. The last is the compliance of what a real
wiping tool has at its tip (felt, sponge) and the peg here does not. The joint targets are what shakes the board.

    python make_wipe_board.py          (CPU only)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTH, WIDTH, THICKNESS = 0.150, 0.060, 0.010
BOARD_TOP = 0.028
LINE_LENGTH = 0.100   # m: the line to wipe, along the board's x axis, centred on the board (drawn on it for pictures only)
MASS = 0.3      # kg: a 150 x 60 x 10 mm melamine-faced board (~1.2 g/cm3) weighs about 0.1 kg; with its carrier

STAGE_JOINTS = ("stage_x", "stage_y", "stage_rx", "stage_ry", "stage_normal")   # the stage's DOFs, in this order

CARRIER = """    <link name="{name}">
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
"""

URDF = """<?xml version="1.0"?>
<!-- Built by make_wipe_board.py. The board of the wiping task on a stage that can move it in its own plane
     (stage_x, stage_y), tip it about the middle of the line (stage_rx, stage_ry; the pivot is on the board's
     top face) and move it along its normal (stage_normal, spring-loaded: the compliance of the tool's pad). -->
<robot name="wipe_board_stage">
    <link name="stage_base"/>
{carriers}    <link name="board">
        <inertial>
            <origin xyz="0 0 {cz:.4f}"/>
            <mass value="{mass}"/>
            <inertia ixx="{ixx:.3e}" iyy="{iyy:.3e}" izz="{izz:.3e}" ixy="0" ixz="0" iyz="0"/>
        </inertial>
        <visual>
            <geometry>
                <mesh filename="wipe_board.obj"/>
            </geometry>
            <material name="board">
                <color rgba="0.95 0.95 0.93 1"/>
            </material>
        </visual>
        <visual>
            <origin xyz="0 0 {ink:.4f}"/>
            <geometry>
                <box size="{line} 0.002 0.0004"/>
            </geometry>
            <material name="ink">
                <color rgba="0.1 0.2 0.8 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <mesh filename="wipe_board.obj"/>
            </geometry>
            <sdf resolution="256"/>
        </collision>
    </link>
    <joint name="stage_x" type="prismatic">
        <parent link="stage_base"/>
        <child link="carrier_x"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.02" upper="0.02" effort="1000" velocity="1"/>
    </joint>
    <joint name="stage_y" type="prismatic">
        <parent link="carrier_x"/>
        <child link="carrier_y"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.02" upper="0.02" effort="1000" velocity="1"/>
    </joint>
    <joint name="stage_rx" type="revolute">
        <parent link="carrier_y"/>
        <child link="carrier_rx"/>
        <origin xyz="0 0 {top}"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.2" upper="0.2" effort="1000" velocity="10"/>
    </joint>
    <joint name="stage_ry" type="revolute">
        <parent link="carrier_rx"/>
        <child link="carrier_ry"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.2" upper="0.2" effort="1000" velocity="10"/>
    </joint>
    <joint name="stage_normal" type="prismatic">
        <parent link="carrier_ry"/>
        <child link="board"/>
        <origin xyz="0 0 -{top}"/>
        <axis xyz="0 0 1"/>
        <limit lower="-0.02" upper="0.02" effort="1000" velocity="1"/>
    </joint>
</robot>
"""

if __name__ == "__main__":
    import trimesh

    box = trimesh.creation.box(extents=(LENGTH, WIDTH, THICKNESS))
    box.apply_translation((0.0, 0.0, BOARD_TOP - THICKNESS / 2))
    mesh = box
    for _ in range(6):          # midpoint subdivision keeps the mesh closed: 150 mm / 2^6 = 2.3 mm edges
        mesh = mesh.subdivide()
    assert mesh.is_watertight, "the board mesh must be closed for the SDF"
    out = os.path.join(HERE, "objects")
    mesh.export(os.path.join(out, "wipe_board.obj"))
    with open(os.path.join(out, "wipe_board_stage.urdf"), "w") as f:
        f.write(URDF.format(carriers="".join(CARRIER.format(name="carrier_" + j[6:]) for j in STAGE_JOINTS[:4]),
                            top=BOARD_TOP, ink=BOARD_TOP + 0.0001, line=LINE_LENGTH, cz=BOARD_TOP - THICKNESS / 2, mass=MASS,
                            ixx=MASS * (WIDTH ** 2 + THICKNESS ** 2) / 12, iyy=MASS * (LENGTH ** 2 + THICKNESS ** 2) / 12,
                            izz=MASS * (LENGTH ** 2 + WIDTH ** 2) / 12))
    print("board: %d vertices, %d faces, bounds %s" % (len(mesh.vertices), len(mesh.faces), mesh.bounds.round(4).tolist()))
