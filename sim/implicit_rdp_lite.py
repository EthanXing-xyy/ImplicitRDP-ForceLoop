"""ImplicitRDP（arXiv:2512.10946）的最小复现 + C49 改动（历史、传出副本、触觉预测头；原件逐字保留，改动处都标了 C49）。
模块名必须仍叫 implicit_rdp_lite（checkpoint 按这个名字 pickle）。

改动：
  --history H   快 token 的 GRU 不再每块从零开始，而是先吃块起点之前的 H 帧触觉（回合开头不足的帧填零 + 一位"无读数"标志），
                状态跨块延续。H=0 就是原版。
  --efference   GRU 输入加传出副本：上一步发出的命令相对再上一步的位置增量（当前 TCP 系，3 维）。预测"已发出、未生效的命令会带来什么触觉"要有它。
  --pred-w λ    预测头：从 GRU 输出 h_s 和接下来 K_MAX 步的计划动作，预测 k∈PRED_K 步后的触觉变化（按维标准化），
                L1 损失乘 λ 一起反传进 GRU。λ=0 不训这个头。部署时不用。
  --pred-target 预测什么：force = 触觉标定出的 3 维力（VRR 用的那个估计，擦板默认：任务调的就是它，比嵌入干净）；emb = 30 维 PCA 嵌入。
  探针           同样结构的头接在 detach 的 h_s 上，所有变体都训（不影响策略），用来比较"去噪损失自己让 h 有多少预测力"。
  --task        插孔或擦板（数据默认路径与回合长度）。
  训练日志里多打印 denoise / pred / probe 三项损失，训完报告预测头与探针相对"复制上一帧"（Δ=0）的 L1 比值。

下面是原件说明：

ImplicitRDP (Chen et al., RA-L 2026, arXiv 2512.10946) for the wiping task, without cameras.

What follows the paper / official code (github.com/Chen-Wendi/ImplicitRDP, config
train_reactive_diffusion_transformer_real_image_workspace.yaml + train_implicitrdp.sh):
  * one network: a Transformer decoder denoises the action chunk; its cross-attention reads "slow" tokens (the
    observation at the chunk start) and "fast" tokens (one per action step, a GRU over the fast signal)
  * temporally causal structure: action token s attends to action tokens <= s and to fast tokens <= s only, so a
    chunk can be denoised for its first s steps as soon as s fast readings are in
  * consistent inference: the slow tokens and the initial noise are drawn once per chunk; at every control step the
    chunk so far is denoised again (DDIM, eta = 0, 5 steps) with one more fast token and its last action is executed
  * virtual-target regularisation (VRR): the denoised vector is [action, virtual target, stiffness]; the virtual
    target is x_real + f / k_adp with the adaptive stiffness of the paper's Eq. 5 (0.5 N..5 N -> 10000..200 N/m);
    only the action part is executed
  * velocity prediction, Euler angles for the (relative) rotation, input perturbation 0.1, squared-cosine schedule
    with 100 training steps, EMA, AdamW 1e-4 with warm-up and cosine decay
  * sizes: 768-wide, 7 layers, 4 heads, attention dropout 0.1, GRU with 512 units
"""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import rdp_lite  # noqa: E402
from rdp_lite import HORIZON, N_ACTION, OBS_OFFSETS, MinMax, quat_to_matrix, matrix_to_quat  # noqa: E402

POSE_DIM = 6                 # relative TCP pose: position (3) + Euler angles (3)
AUG_DIM = 2 * POSE_DIM + 1   # action, virtual target, stiffness
TAC_DIM = 2 * rdp_lite.EMB
N_EMB, N_LAYER, N_HEAD, RNN_HIDDEN = 768, 7, 4, 512
TRAIN_STEPS, DDIM_STEPS = 100, 5
INPUT_PERTURB = 0.1
K_MAX, K_MIN, F_LOW, F_HIGH = 10000.0, 200.0, 0.5, 5.0    # N/m, N/m, N, N (paper, Sec. IV-A)
TACTILE_DELAY = 1            # control steps a tactile reading is old (insertion_env.py); undone when labelling
CONTROL_DT = 1.0 / 24.0      # s (insertion_env.py)
PRED_K = (4, 8)              # C49：预测几步后的触觉变化
PRED_KMAX = max(PRED_K)      # C49：预测头看到的计划动作步数


# ----------------------------------------------------------------------------- rotations
def matrix_to_euler(m):
    """Rotation matrix -> roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll). Used on relative rotations only
    (a few degrees), far from the singularity at pitch = +-90 deg."""
    return torch.stack([torch.atan2(m[..., 2, 1], m[..., 2, 2]),
                        -torch.asin(m[..., 2, 0].clamp(-1.0, 1.0)),
                        torch.atan2(m[..., 1, 0], m[..., 0, 0])], -1)


def euler_to_matrix(e):
    cr, sr, cp, sp, cy, sy = e[..., 0].cos(), e[..., 0].sin(), e[..., 1].cos(), e[..., 1].sin(), e[..., 2].cos(), e[..., 2].sin()
    return torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr], -1).reshape(*e.shape[:-1], 3, 3)


def relative_pose(base_pos, base_rot, pos, rot):
    """Pose (pos, rotation matrix) in the frame of the base pose, as position + Euler angles."""
    base_t = base_rot.transpose(-1, -2)
    return torch.cat([(base_t @ (pos - base_pos).unsqueeze(-1)).squeeze(-1), matrix_to_euler(base_t @ rot)], -1)


def absolute_pose(base_pos, base_quat, rel):
    """Inverse of relative_pose: 6 numbers in the base frame -> world position and quaternion."""
    base = quat_to_matrix(base_quat)
    return (base_pos + (base @ rel[..., 0:3].unsqueeze(-1)).squeeze(-1), matrix_to_quat(base @ euler_to_matrix(rel[..., 3:6])))


# ----------------------------------------------------------------------------- force and virtual target
def tactile_force(calib, left, right):
    """Tactile fields of the two pads (..., points, 3) -> estimated force on the held tool, TCP frame (..., 3)."""
    x = torch.cat([left.sum(-2), right.sum(-2)], -1)
    return x @ calib[:6] + calib[6]


def adaptive_stiffness(force):
    """Paper Eq. 5: stiff while the force is small, soft once it is clearly a contact."""
    mag = force.norm(dim=-1, keepdim=True)
    return (K_MAX - (K_MAX - K_MIN) * (mag - F_LOW) / (F_HIGH - F_LOW)).clamp(K_MIN, K_MAX)


class Windows:
    """Every (episode, time) of the demos as a training sample, on the GPU (rdp_lite.Windows, with ImplicitRDP's targets).
    C49：多了块起点之前的 history 帧（tactile() 返回 (B, H+32, 30) 与有效位）和预测目标（pred_targets()）。"""

    def __init__(self, data, ends, pca, rot_ref, calib, history=0, pred_target="force"):
        device = data["tcp_pos"].device
        starts = np.concatenate([[0], ends[:-1]])
        first = torch.tensor(np.concatenate([np.full(e - s, s) for s, e in zip(starts, ends)]), device=device)
        last = torch.tensor(np.concatenate([np.full(e - s, e - 1) for s, e in zip(starts, ends)]), device=device)
        t = torch.arange(len(first), device=device)
        self.emb = torch.cat([pca(data["tactile_left"]), pca(data["tactile_right"])], dim=-1)          # (T, 30)
        feat = rdp_lite.features(data["socket_estimate"], data["tcp_pos"], data["tcp_quat"], self.emb, rot_ref,
                                 data.get("board_quat_estimate"), data.get("line_turn"))
        obs_idx = torch.maximum(torch.stack([(t + o).clamp(min=0) for o in OBS_OFFSETS], dim=1), first.unsqueeze(1))
        self.obs = feat[obs_idx]                                                                       # (T, 2, FEAT)
        self.chunk = torch.minimum(t.unsqueeze(1) + torch.arange(HORIZON, device=device), last.unsqueeze(1))
        self.history, self.pred_target = history, pred_target
        self.first, self.last = first, last                                                            # C49
        # C49 传出副本：第 t-1 步发出的命令相对第 t-2 步命令的位置增量，写在第 t 步 TCP 系下（第 t 步的命令在评测时还没算出来，
        # 所以快层只能拿到 t-1 之前的命令）；回合前两步为 0
        a1 = data["action_pos"][(t - 1).clamp(min=0)]
        a2 = data["action_pos"][(t - 2).clamp(min=0)]
        inc = torch.where((t - 2 >= first).unsqueeze(1), a1 - a2, torch.zeros_like(a1))
        self.u = (quat_to_matrix(data["tcp_quat"]).transpose(-1, -2) @ inc.unsqueeze(-1)).squeeze(-1)                  # (T, 3)

        # the force the tool puts on the board, world frame, at every step: the reading that shows the force of step
        # t arrives TACTILE_DELAY steps later; in a recorded episode it can be put back where it belongs. Demos
        # recorded with the sensors on their own clocks (insertion_env.py, timing="real") carry each reading's age:
        # there the delay is their mean age, a fraction of a step, undone by interpolating between two readings
        delay = float(data["tactile_age"].mean()) / CONTROL_DT if "tactile_age" in data else float(TACTILE_DELAY)
        whole = int(math.floor(delay + 1e-6))
        self.label_delay = delay
        estimate = torch.cat([tactile_force(calib, data["tactile_left"][i:i + 8192], data["tactile_right"][i:i + 8192])
                              for i in range(0, len(t), 8192)])
        on_tool = ((1 - (delay - whole)) * estimate[torch.minimum(t + whole, last)]
                   + (delay - whole) * estimate[torch.minimum(t + whole + 1, last)])
        rot = quat_to_matrix(data["tcp_quat"])
        push = -(rot @ on_tool.unsqueeze(-1)).squeeze(-1)
        stiffness = adaptive_stiffness(on_tool)
        virtual = data["tcp_pos"] + push / stiffness                    # Eq. 4: x_real + K^-1 f, along the force
        self.force, self.stiffness = on_tool, stiffness
        self.fest = estimate                                             # C49：每步读数算出的力估计（未做延迟校正，评测时就是这个）

        nxt = torch.minimum(self.chunk + 1, last.unsqueeze(1))          # an action's outcome: the pose and force one step on
        parts = []
        for i in range(0, len(t), 8192):
            s = slice(i, i + 8192)
            base_pos, base_rot = data["tcp_pos"][s, None], rot[s, None]
            action_rot = quat_to_matrix(data["action_quat"][self.chunk[s]])
            action = relative_pose(base_pos, base_rot, data["action_pos"][self.chunk[s]], action_rot)
            target = relative_pose(base_pos, base_rot, virtual[nxt[s]], action_rot)     # same rotation as the action
            parts.append(torch.cat([action, target, stiffness[nxt[s]]], -1))
        self.action = torch.cat(parts)                                                                 # (T, 32, 13)
        self.index = t
        if "perturbed" in data:
            keep = ~data["perturbed"].bool()[self.chunk].any(dim=1)
            self.obs, self.chunk, self.action, self.index = self.obs[keep], self.chunk[keep], self.action[keep], t[keep]
        self.n = len(self.index)
        # C49：预测目标的标准化尺度（按维），在随机 20k 个样本上估计
        g = torch.Generator(device=device).manual_seed(0)
        i = torch.randint(0, self.n, (min(20000, self.n),), device=device, generator=g)
        tgt, mask = self.pred_targets(i)
        self.pred_scale = torch.stack([tgt[:, k][mask[:, k]].std(dim=0) for k in range(len(PRED_K))]).clamp(min=1e-6)  # (K, D)
        self.tgt_dim = self.pred_scale.shape[1]
        self.pred_mean_abs = torch.stack([(tgt[:, k][mask[:, k]] / self.pred_scale[k]).abs().mean() for k in range(len(PRED_K))])

    def tactile(self, i):
        """(B, H+32, 30) 触觉嵌入序列（history 帧在前），(B, H+32) 有效位（回合开头之前的 history 帧无效、填零），
        (B, H+32, 3) 同一序列的传出副本 u。"""
        chunk = self.chunk[i]
        seq, valid, u = self.emb[chunk], torch.ones(chunk.shape, dtype=torch.bool, device=chunk.device), self.u[chunk]
        if self.history:
            t0 = self.index[i].unsqueeze(1)
            hist = t0 - self.history + torch.arange(self.history, device=chunk.device)
            hvalid = hist >= self.first[self.index[i]].unsqueeze(1)
            hc = hist.clamp(min=0)
            seq = torch.cat([self.emb[hc] * hvalid.unsqueeze(-1), seq], 1)
            u = torch.cat([self.u[hc] * hvalid.unsqueeze(-1), u], 1)
            valid = torch.cat([hvalid, valid], 1)
        return seq, valid, u

    def pred_targets(self, i):
        """(B, K, 32, D) 每步 s 往后 k 步的触觉变化 x_{s+k}-x_s（未标准化；x = 标定力 3 维或嵌入 30 维），
        (B, K, 32) 有效位（s+k 不超过回合末也不超过块末）。"""
        chunk, last = self.chunk[i], self.last[self.index[i]].unsqueeze(1)
        src = self.fest if self.pred_target == "force" else self.emb
        tgts, masks = [], []
        for k in PRED_K:
            fut = torch.minimum(chunk + k, last)
            tgts.append(src[fut] - src[chunk])
            s = torch.arange(HORIZON, device=chunk.device).unsqueeze(0)
            masks.append((chunk + k <= last) & (s + k < HORIZON))
        return torch.stack(tgts, 1), torch.stack(masks, 1)

    def future_actions(self, i):
        """(B, 32, PRED_KMAX, 6) 每步 s 之后的 PRED_KMAX 步计划动作（相对块起点的位姿，和 action 一样），块末不足的填零。"""
        a = self.action[i][..., :POSE_DIM]                                                              # (B, 32, 6)
        pad = torch.cat([a, torch.zeros_like(a[:, :PRED_KMAX])], 1)
        return torch.stack([pad[:, k:k + HORIZON] for k in range(1, PRED_KMAX + 1)], 2)

    def batch(self, size):
        i = torch.randint(0, self.n, (size,), device=self.obs.device)
        seq, valid, u = self.tactile(i)
        tgt, mask = self.pred_targets(i)
        return self.obs[i], self.action[i], seq, valid, u, tgt / self.pred_scale[None, :, None, :], mask, self.future_actions(i)


# ----------------------------------------------------------------------------- network
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        arg = x[:, None].float() * freq[None]
        return torch.cat([arg.sin(), arg.cos()], dim=-1)


class Denoiser(nn.Module):
    """The official TransformerForDiffusion with its GRU fast encoder; the image encoder is replaced by two linear
    layers that turn each observed frame into a pose token and a tactile token.
    C49：history>0 时 GRU 输入多一位"有读数"标志，序列前 history 帧是块起点之前的触觉；efference 时再多 3 维命令增量（归一化）；
    forward 多返回 GRU 输出（块内部分）。旧 checkpoint 没有这些属性 = 原版。"""

    def __init__(self, pose_feat, n_emb=N_EMB, n_layer=N_LAYER, n_head=N_HEAD, p_drop_attn=0.1, history=0, efference=False):
        super().__init__()
        self.pose_feat = pose_feat
        self.history, self.efference = history, efference
        self.input_emb = nn.Linear(AUG_DIM, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, HORIZON, n_emb))
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.pose_token = nn.Linear(pose_feat, n_emb)
        self.tactile_token = nn.Linear(TAC_DIM, n_emb)
        self.n_slow = 1 + 2 * len(OBS_OFFSETS)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.n_slow + HORIZON, n_emb))
        self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))
        self.rnn = nn.GRU(TAC_DIM + (1 if history else 0) + (3 if efference else 0), RNN_HIDDEN, 1, batch_first=True)
        self.rnn_fc = nn.Linear(RNN_HIDDEN, n_emb)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(n_emb, n_head, 4 * n_emb, p_drop_attn, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, AUG_DIM)
        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                nn.init.normal_(p, std=0.02) if name.startswith("weight") else nn.init.zeros_(p)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.normal_(m.in_proj_weight, std=0.02)
            nn.init.zeros_(m.in_proj_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def fast_states(self, tactile, valid=None, u=None):
        """(B, H+s, 30) 归一化触觉（前 H 帧是历史），valid (B, H+s)，u (B, H+s, 3) 归一化命令增量 → GRU 输出的块内部分 (B, s, RNN_HIDDEN)。"""
        H, eff = getattr(self, "history", 0), getattr(self, "efference", False)
        x = tactile
        if H or eff:
            if valid is None:
                valid = torch.ones(tactile.shape[:2], dtype=torch.bool, device=tactile.device)
            parts = [tactile * valid.unsqueeze(-1)]
            if H:
                parts.append(valid.unsqueeze(-1).to(tactile.dtype))
            if eff:
                assert u is not None, "efference 模型要给命令增量 u"
                parts.append(u * valid.unsqueeze(-1))
            x = torch.cat(parts, -1)
        out = self.rnn(x)[0]
        return out[:, H:] if H else out

    def forward(self, sample, timestep, obs, tactile, valid=None, u=None):
        """sample (B, s, 13) noisy chunk so far, timestep (B,), obs (B, 2, FEAT) normalised, tactile (B, H+s, 30) normalised.
        Returns (B, s, 13) and the GRU states (B, s, RNN_HIDDEN)."""
        s = sample.shape[1]
        slow = torch.cat([self.time_emb(timestep).unsqueeze(1), self.pose_token(obs[..., :self.pose_feat]),
                          self.tactile_token(obs[..., self.pose_feat:])], dim=1)
        slow = self.encoder(slow + self.cond_pos_emb[:, :self.n_slow])
        h = self.fast_states(tactile, valid, u)
        assert h.shape[1] == s, (h.shape, s)
        fast = self.rnn_fc(h) + self.cond_pos_emb[:, self.n_slow:self.n_slow + s]
        causal = torch.triu(torch.ones((s, s), device=sample.device), diagonal=1).bool()
        memory_mask = torch.zeros((s, self.n_slow + s), device=sample.device).bool()
        memory_mask[:, self.n_slow:] = causal
        x = self.input_emb(sample) + self.pos_emb[:, :s]
        memory = torch.cat([slow, fast], dim=1)
        for block in self.blocks:
            x = block(x, memory, tgt_mask=causal, memory_mask=memory_mask)
        return self.head(self.ln_f(x)), h

    def optim_groups(self, weight_decay):
        """Weight decay on the weights of linear, attention and GRU layers only (official get_optim_groups)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            leaf = name.split(".")[-1]
            plain = "bias" in leaf or leaf in ("pos_emb", "cond_pos_emb") or ".norm" in name or name.startswith("ln_f")
            (no_decay if plain else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def pred_head(tgt_dim):
    """C49：GRU 状态 + 之后 PRED_KMAX 步计划动作 → 各 k 的触觉变化（标准化，D 维）。"""
    return nn.Sequential(nn.Linear(RNN_HIDDEN + PRED_KMAX * POSE_DIM, 256), nn.Mish(), nn.Linear(256, len(PRED_K) * tgt_dim))


class Policy(nn.Module):
    """Everything needed at deployment. PCA and the orientation reference are those of the DP / RDP checkpoint."""

    def __init__(self, base, windows, history=0, pred_w=0.0, efference=False, u_noise=0.0):
        super().__init__()
        self.pca = base.pca
        self.register_buffer("rot_ref", base.rot_ref.clone())
        self.board_input = bool(getattr(base, "board_input", False))
        self.turn_input = bool(getattr(base, "turn_input", False))
        pose_feat = windows.obs.shape[-1] - TAC_DIM
        self.norm_obs, self.norm_action = MinMax(windows.obs), MinMax(windows.action)
        self.norm_tactile = MinMax(windows.emb)
        self.history, self.pred_w, self.efference = history, pred_w, efference          # C49
        self.u_noise = u_noise                                                           # C49 训练时传出副本的抖动（归一化单位）
        self.register_buffer("pred_scale", windows.pred_scale.clone())                   # C49 (K, D)
        self.pred_target, self.tgt_dim = windows.pred_target, windows.tgt_dim
        self.norm_u = MinMax(windows.u)                                                  # C49 命令增量的归一化
        self.net = Denoiser(pose_feat, history=history, efference=efference)
        self.pred = pred_head(self.tgt_dim) if pred_w > 0 else None                      # C49：训进 GRU 的预测头
        self.probe = pred_head(self.tgt_dim)                                             # C49：detach 探针，所有变体都有
        steps = torch.arange(TRAIN_STEPS + 1, dtype=torch.float64) / TRAIN_STEPS
        bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2          # squaredcos_cap_v2
        betas = (1 - bar[1:] / bar[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0).float())

    def embed(self, tactile):
        """(B, 2, points, 3) tactile fields of both sensors -> (B, 30)."""
        return torch.cat([self.pca(tactile[:, 0]), self.pca(tactile[:, 1])], dim=-1)

    def features(self, obs, emb):
        board = obs.get("board_quat_estimate") if self.board_input else None
        turn = obs.get("line_turn") if getattr(self, "turn_input", False) else None
        return rdp_lite.features(obs["socket_estimate"], obs["tcp_pos"], obs["tcp_quat"], emb, self.rot_ref, board, turn)

    def _pred_loss(self, head, h, fut, tgt, mask):
        """head 在 h (B,32,512) 与 fut (B,32,KMAX,6) 上预测 (B,K,32,D)，L1 对 mask 平均。"""
        p = head(torch.cat([h, fut.flatten(2)], -1)).view(h.shape[0], HORIZON, len(PRED_K), self.tgt_dim).transpose(1, 2)
        err = (p - tgt).abs().mean(-1)                                    # (B, K, 32)
        return (err * mask).sum() / mask.sum().clamp(min=1)

    def loss(self, obs, action, tactile, valid, u, tgt, mask, fut):
        """Velocity-prediction loss on the whole chunk; the causal masks make it a loss on every prefix at once.
        C49：返回 dict(denoise, pred, probe)；总损失 = denoise + pred_w * pred + probe（probe 不回传进网络）。"""
        x = self.norm_action(action)
        t = torch.randint(0, TRAIN_STEPS, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        a = self.alpha_bar[t].view(-1, 1, 1)
        noisy = a.sqrt() * x + (1 - a).sqrt() * (noise + INPUT_PERTURB * torch.randn_like(x))
        target = a.sqrt() * noise - (1 - a).sqrt() * x
        un = self.norm_u(u)
        if self.u_noise > 0:                                              # 评测时 u 来自策略自己的命令，比专家的毛糙；训练时抖一下
            un = un + self.u_noise * torch.randn_like(un) * valid.unsqueeze(-1).to(un.dtype)
        out, h = self.net(noisy, t, self.norm_obs(obs), self.norm_tactile(tactile), valid, un)
        fut = fut / self.norm_action.span[:POSE_DIM]                      # 计划动作按动作归一化的尺度缩放
        losses = dict(denoise=F.mse_loss(out, target), probe=self._pred_loss(self.probe, h.detach(), fut, tgt, mask))
        if self.pred is not None:
            losses["pred"] = self._pred_loss(self.pred, h, fut, tgt, mask)
        return losses

    def total(self, losses):
        return losses["denoise"] + self.pred_w * losses.get("pred", 0.0) + losses["probe"]

    @torch.no_grad()
    def denoise(self, obs, tactile, noise, valid=None, u=None):
        """DDIM, eta = 0: the first s steps of a chunk from its cached slow observation (normalised), the s tactile
        embeddings so far (normalised; C49：前面可带 history 帧) and the first s steps of its cached noise. Returns (B, s, 13), denormalised."""
        x = noise
        ts = list(range(TRAIN_STEPS - TRAIN_STEPS // DDIM_STEPS, -1, -(TRAIN_STEPS // DDIM_STEPS)))    # 80, 60, 40, 20, 0
        for i, t in enumerate(ts):
            a = self.alpha_bar[t]
            v = self.net(x, torch.full((x.shape[0],), t, device=x.device), obs, tactile, valid, u)[0]
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-1, 1)
            if i + 1 == len(ts):
                return self.norm_action.inverse(x0)
            eps = a.sqrt() * v + (1 - a).sqrt() * x
            a_next = self.alpha_bar[ts[i + 1]]
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps

    @torch.no_grad()
    def fast_states(self, tactile, valid=None, u=None):
        """C49 诊断：GRU 状态 (B, s, RNN_HIDDEN)，评测时记 ||h_t - h_{t-1}|| 当"意外"曲线。tactile、u 都已归一化。"""
        return self.net.fast_states(tactile, valid, u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("insertion", "wipe"), default="insertion")
    parser.add_argument("--data", default="")
    parser.add_argument("--base", default="", help="DP / RDP checkpoint: its tactile PCA is reused")
    parser.add_argument("--calib", default="")
    parser.add_argument("--out", default="runs/irdp")
    parser.add_argument("--steps", type=int, default=48000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=0, help="C49：块起点之前喂给 GRU 的触觉帧数（0 = 原版，每块从零开始）")
    parser.add_argument("--pred-w", type=float, default=0.0, help="C49：触觉预测损失的权重（0 = 不训预测头，只有 detach 探针）")
    parser.add_argument("--efference", action="store_true", help="C49：GRU 输入加每步发出的命令位置增量（传出副本）")
    parser.add_argument("--pred-target", choices=("force", "emb"), default="force", help="C49：预测标定力（3 维）还是嵌入（30 维）")
    parser.add_argument("--u-noise", type=float, default=0.05, help="C49：训练时加在归一化传出副本上的高斯抖动（评测时命令来自策略自己）")
    args = parser.parse_args()
    RA = os.path.dirname(os.path.abspath(__file__))      # data/ and runs/ next to this file
    if not args.data:
        args.data = f"{RA}/data/insertion_demos.zarr" if args.task == "insertion" else f"{RA}/data/wipe_demos.zarr"
    if not args.base:
        args.base = f"{RA}/runs/v5/policy.pt" if args.task == "insertion" else f"{RA}/runs/wipe/policy.pt"
    if not args.calib:
        args.calib = f"{RA}/runs/wipe_irdp/force_calib.pt" if args.task == "wipe" else ""
    assert args.calib, "--calib：插孔要先跑 insertion_force_calib.py"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True     # 0.24 -> 0.17 s per step on the RTX 3060
    device = "cuda:0"
    os.makedirs(args.out, exist_ok=True)

    base = torch.load(args.base, map_location=device)["policy"]
    calib = torch.load(args.calib, map_location=device)["w"]
    data, ends = rdp_lite.load_demos(args.data, device)
    windows = Windows(data, ends, base.pca, base.rot_ref, calib, history=args.history, pred_target=args.pred_target)
    del data
    torch.cuda.empty_cache()
    touching = windows.force.norm(dim=-1) > F_HIGH
    print("demos: %d episodes, %d steps, %d training samples" % (len(ends), int(ends[-1]), windows.n))
    print("force labels: a tactile reading is taken to show the force %.2f control steps before it was delivered" % windows.label_delay)
    print("estimated force: above %.1f N (softest stiffness) in %.0f%% of the steps, median there %.1f N; the virtual target "
          "is then %.0f mm past the tool on average"
          % (F_HIGH, 100 * touching.float().mean(), windows.force.norm(dim=-1)[touching].median(),
             1000 * (windows.force.norm(dim=-1) / windows.stiffness[:, 0])[touching].mean()), flush=True)
    print("C49: history %d frames, efference %s, pred_w %.3f, target %s (%d-d), k=%s, standardised target mean |x| = %s (copy-last baseline = this)"
          % (args.history, args.efference, args.pred_w, args.pred_target, windows.tgt_dim, PRED_K,
             [round(v, 3) for v in windows.pred_mean_abs.tolist()]), flush=True)
    policy = Policy(base, windows, history=args.history, pred_w=args.pred_w, efference=args.efference,
                    u_noise=args.u_noise if args.efference else 0.0).to(device)
    print("network: %.1f M parameters" % (sum(p.numel() for p in policy.net.parameters()) / 1e6), flush=True)

    params = list(policy.net.optim_groups(1e-6))
    params.append({"params": list(policy.probe.parameters()) + (list(policy.pred.parameters()) if policy.pred is not None else []),
                   "weight_decay": 1e-6})
    net = policy.net
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.95, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * max(0, s - args.warmup) / max(1, args.steps - args.warmup))))
    running, t0, history = {}, time.time(), []
    gru = list(net.rnn.parameters())
    gnorm = lambda loss_: math.sqrt(sum(g.pow(2).sum().item() for g in torch.autograd.grad(loss_, gru, retain_graph=True, allow_unused=True) if g is not None))
    for step in range(1, args.steps + 1):
        losses = policy.loss(*windows.batch(args.batch))
        loss = policy.total(losses)
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0 and policy.pred is not None:
            running["gru_grad_denoise"] = running.get("gru_grad_denoise", 0.0) + 500 * gnorm(losses["denoise"])
            running["gru_grad_pred"] = running.get("gru_grad_pred", 0.0) + 500 * policy.pred_w * gnorm(losses["pred"])
        loss.backward()
        opt.step()
        sched.step()
        decay = min(0.9999, 1 - (1 + step) ** -0.75)           # the official EMAModel (inv_gamma 1, power 0.75)
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), net.parameters()):
                p_ema.lerp_(p, 1 - decay)
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()
        if step % 500 == 0:
            avg = {k: v / 500 for k, v in running.items()}
            history.append((step, avg))
            print("implicit   step %6d / %d  loss %s  (%.0f s)" % (step, args.steps, " ".join("%s %.5f" % kv for kv in avg.items()),
                                                                  time.time() - t0), flush=True)
            running = {}
    net.load_state_dict(ema.state_dict())      # deploy the EMA weights
    policy.eval()

    # how well the deployed sampler reproduces the demonstrated chunk, and whether the tactile sequence matters to it
    del opt, ema
    torch.cuda.empty_cache()
    g = torch.Generator(device=device).manual_seed(1)
    errs, gaps, pl = [], [], {}
    for _ in range(8):
        i = torch.randint(0, windows.n, (256,), device=device, generator=g)
        obs, action = policy.norm_obs(windows.obs[i]), windows.action[i]
        seq, valid, u = windows.tactile(i)
        tactile, u = policy.norm_tactile(seq), policy.norm_u(u)
        noise = torch.randn(action.shape, device=device, generator=g)
        err = lambda out: (out - action)[..., 0:3].norm(dim=-1).mean().item()
        recon = policy.denoise(obs, tactile, noise, valid, u)
        H = args.history
        held = torch.cat([tactile[:, :H], tactile[:, H:H + 1].expand(-1, HORIZON, -1)], 1)   # tactile frozen at the chunk start
        held = policy.denoise(obs, held, noise, valid, u)
        prefix = policy.denoise(obs, tactile[:, :H + N_ACTION], noise[:, :N_ACTION], valid[:, :H + N_ACTION], u[:, :H + N_ACTION])
        errs.append((err(recon), err(held)))
        gaps.append((prefix - recon[:, :N_ACTION])[..., 0:3].norm(dim=-1).max().item())
        # C49：预测头 / 探针相对"复制上一帧"（预测 Δ=0，其 L1 = 标准化目标的平均 |x|）的比值，越小越有预测力
        with torch.no_grad():
            tgt, mask = windows.pred_targets(i)
            tgt = tgt / windows.pred_scale[None, :, None, :]
            fut = windows.future_actions(i) / policy.norm_action.span[:POSE_DIM]
            h = policy.fast_states(tactile, valid, u)
            for name, head in (("probe", policy.probe), ("pred", policy.pred)):
                if head is None:
                    continue
                p = head(torch.cat([h, fut.flatten(2)], -1)).view(len(i), HORIZON, len(PRED_K), policy.tgt_dim).transpose(1, 2)
                for kk, k in enumerate(PRED_K):
                    m = mask[:, kk]
                    pl.setdefault((name, k), []).append(((p[:, kk] - tgt[:, kk]).abs().mean(-1)[m].mean() / tgt[:, kk].abs().mean(-1)[m].mean()).item())
    log = dict(history=history, recon_mm=1000 * np.mean([e[0] for e in errs]), recon_frozen_mm=1000 * np.mean([e[1] for e in errs]),
               prefix_gap_mm=1000 * max(gaps), args=vars(args),
               pred_ratio={"%s_k%d" % k: float(np.mean(v)) for k, v in pl.items()})
    print("action error against the demos (position, mean over the chunk): %.3f mm; with the tactile sequence frozen at "
          "the chunk start: %.3f mm" % (log["recon_mm"], log["recon_frozen_mm"]))
    print("a %d-step prefix denoised alone differs from the same steps of the full chunk by at most %.4f mm "
          "(0 = the structure is causal)" % (N_ACTION, log["prefix_gap_mm"]))
    print("C49 prediction L1 relative to copy-last (1.0 = no predictive power): %s" % log["pred_ratio"])
    torch.save({"policy": policy, "log": log}, os.path.join(args.out, "policy.pt"))
    print("saved", os.path.join(args.out, "policy.pt"))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # 本目录要排在 ~/tacsl/robot_assets 前面（否则导入原版）
    import implicit_rdp_lite  # run under the module's real name so the saved policy can be loaded elsewhere

    assert implicit_rdp_lite.__file__ == os.path.abspath(__file__), implicit_rdp_lite.__file__
    implicit_rdp_lite.main()
