"""Cartesian (end-effector pose) control for the UR5 in Isaac Gym.

The real UR5 is commanded with joint positions, so this is differential inverse
kinematics rather than torque control: each step the TCP pose error is mapped
through the damped pseudo-inverse of the Jacobian to a joint step, and the joint
position drives do the rest.
"""
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul

import torch

from ur5_2f85_common import ARM_HOME, TCP_LINK


def orientation_error(target, current):
    """Rotation vector (small-angle) taking `current` to `target`; quaternions are (x, y, z, w)."""
    q = quat_mul(target, quat_conjugate(current))
    return 2.0 * q[..., 0:3] * torch.sign(q[..., 3:4])


class EEController:
    """TCP pose -> arm joint position targets, for every env at once.

    Create it before gym.prepare_sim(); call start() after prepare_sim().
    """

    def __init__(self, gym, sim, env, actor, actor_name, num_envs=1, damping=0.05, max_joint_step=0.05):
        self.gym, self.sim, self.num_envs = gym, sim, num_envs
        self.damping, self.max_joint_step = damping, max_joint_step
        names = gym.get_actor_dof_names(env, actor)
        self.arm_dofs = [names.index(n) for n in ARM_HOME]
        self.tcp_body = gym.find_actor_rigid_body_index(env, actor, TCP_LINK, gymapi.DOMAIN_ENV)
        tcp_in_actor = gym.find_actor_rigid_body_index(env, actor, TCP_LINK, gymapi.DOMAIN_ACTOR)
        assert tcp_in_actor > 0, "link '%s' not found" % TCP_LINK
        self.jacobian_row = tcp_in_actor - 1  # fixed base: the root link has no Jacobian row
        self.actor_name = actor_name

    def start(self):
        gym, sim = self.gym, self.sim
        self.body = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(self.num_envs, -1, 13)
        self.dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(self.num_envs, -1, 2)
        self.jacobian = gymtorch.wrap_tensor(gym.acquire_jacobian_tensor(sim, self.actor_name))
        self.refresh()

    def refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def pose(self):
        """Current TCP position (N, 3) and quaternion (N, 4) in the world frame."""
        return self.body[:, self.tcp_body, 0:3].clone(), self.body[:, self.tcp_body, 3:7].clone()

    def arm_positions(self):
        return self.dof[:, self.arm_dofs, 0].clone()

    def joint_targets(self, target_pos, target_quat):
        """Arm joint position targets (N, 6) that move the TCP toward the target pose. Call refresh() first."""
        pos, quat = self.pose()
        error = torch.cat([target_pos - pos, orientation_error(target_quat, quat)], dim=-1).unsqueeze(-1)
        j = self.jacobian[:, self.jacobian_row][:, :, self.arm_dofs]
        jt = j.transpose(1, 2)
        eye = torch.eye(6, device=j.device).unsqueeze(0) * self.damping ** 2
        step = (jt @ torch.linalg.solve(j @ jt + eye, error)).squeeze(-1)
        step = step.clamp(-self.max_joint_step, self.max_joint_step)
        return self.arm_positions() + step
