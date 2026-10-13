"""A compact, low-dimensional version of Reactive Diffusion Policy (RDP) for the insertion task.

What follows the paper / official code (github.com/xiaoxiaoxh/reactive_diffusion_policy):
  * tactile input = the per-sensor tactile field reduced to 15 numbers with PCA
  * fast policy   = "asymmetric tokenizer": a 1-D CNN encodes a 32-step action chunk (24 Hz,
                    relative to the TCP at the chunk start) into an 8x16 latent; a GRU decodes it
                    back step by step, conditioned on the tactile embedding of each step
                    (architecture and sizes of config/at/at_wipe_lift.yaml, the config for pose actions;
                    at_peel.yaml differs only in n_latent_dims = ENC_C)
  * slow policy   = a diffusion model over that latent (the official ConditionalUnet1D,
                    100 training steps, squared-cosine schedule, epsilon prediction, EMA)
  * baseline "DP" = the same UNet diffusing the 32-step action chunk directly, executed open loop

  * actions are 6-DoF TCP poses relative to the TCP at the chunk start, 9 numbers each (position in
    the TCP frame + the first two columns of the rotation matrix), as in the official configs with
    relative_action: True (common/action_utils.py)

What is different:
  * no cameras: the slow policy sees (noisy socket estimate - TCP position), the TCP orientation
    (rotation vector relative to the start pose) and the tactile embeddings of the last two
    observations instead of images
  * the chunk starts at the current step (the official code starts it 3 steps in the past)
  * DDIM sampling with 16 steps instead of 100; a smaller UNet (256/512/1024 channels)
  * "latent time jitter" when training the tokenizer: the latent is computed from the action chunk
    that starts up to JITTER steps earlier or later, while the decoder must still output the chunk
    that matches its tactile input. As published (no jitter, KL weight 1e-6) the latent stores the
    whole chunk of these scripted demonstrations, timing included, and the decoder ignores the
    tactile input: delaying the tactile sequence by 4 steps moved the decoded landing by 0 steps.
    With the jitter the latent still says what to do, but when things happen has to come from
    touch (same test: 3.9 steps). A larger KL weight also does this but erases the direction of
    motion from the latent. Measured in diag/at_sweep.py.

Usage (inside `conda activate tacsl`):
    python rdp_lite.py --data data/insertion_demos.zarr --out runs/rdp_lite
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

sys.path.insert(0, os.path.join(os.path.expanduser(os.environ.get("TACSL_ROOT", "~/tacsl")), "rdp/src"))   # TACSL_ROOT/rdp = reactive_diffusion_policy clone (ConditionalUnet1D)
from reactive_diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402

HORIZON = 32        # action chunk length (1.33 s at 24 Hz)
N_ACTION = 16       # steps executed before the slow policy runs again
OBS_OFFSETS = (-2, 0)  # the slow policy sees these steps (2 observations, every other frame)
EMB = 15            # PCA components per sensor
LATENT_T, LATENT_C = 8, 16
ENC_C = 8           # encoder output channels (official n_latent_dims): 8 in at_wipe_lift.yaml, the config for pose
                    # actions with rotation; 4 in at_peel.yaml (position-only actions). With 4 the tokenizer dropped the
                    # rotation entirely (diag/at_recon.py: rotation error = the rotation itself)
ACT_DIM = 9         # relative TCP pose: position (3) + 6-D rotation
FEAT_DIM = 3 + 3 + 2 * EMB   # per observation: socket estimate - TCP, TCP rotation vector, tactile embeddings
OBS_DIM = len(OBS_OFFSETS) * FEAT_DIM
JITTER = 6          # latent time jitter (steps) when training the tokenizer; 0 = as published
TRAIN_STEPS, DDIM_STEPS = 100, 16


# ----------------------------------------------------------------------------- poses (quaternions are x, y, z, w)
def quat_to_matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m):
    """Shepperd's method: of the four algebraically equal formulas take the best-conditioned one."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    t = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    r = 0.5 * t.clamp(min=1e-12).sqrt()
    f = 0.25 / r
    cand = torch.stack([
        torch.stack([(m21 - m12) * f[..., 0], (m02 - m20) * f[..., 0], (m10 - m01) * f[..., 0], r[..., 0]], -1),
        torch.stack([r[..., 1], (m01 + m10) * f[..., 1], (m02 + m20) * f[..., 1], (m21 - m12) * f[..., 1]], -1),
        torch.stack([(m01 + m10) * f[..., 2], r[..., 2], (m12 + m21) * f[..., 2], (m02 - m20) * f[..., 2]], -1),
        torch.stack([(m02 + m20) * f[..., 3], (m12 + m21) * f[..., 3], r[..., 3], (m10 - m01) * f[..., 3]], -1)], -2)
    q = torch.gather(cand, -2, t.argmax(-1)[..., None, None].expand(*t.shape[:-1], 1, 4)).squeeze(-2)
    return F.normalize(q, dim=-1)


def rot6d(m):
    """First two columns of the rotation matrix (official order: R[:3, :2].T.flatten())."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], -1)


def rot6d_to_matrix(d):
    b1 = F.normalize(d[..., 0:3], dim=-1)
    b2 = F.normalize(d[..., 3:6] - (b1 * d[..., 3:6]).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def rotvec(m):
    """Rotation matrix -> rotation vector (axis * angle)."""
    q = matrix_to_quat(m)
    q = q * torch.where(q[..., 3:4] < 0, -1.0, 1.0)
    s = q[..., 0:3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., 3:4])
    return q[..., 0:3] * torch.where(s > 1e-8, angle / s.clamp(min=1e-8), torch.full_like(s, 2.0))


def relative_pose(base_pos, base_quat, pos, quat):
    """Pose (pos, quat) in the frame of the base pose, as 9 numbers (official absolute_actions_to_relative_actions)."""
    base_t = quat_to_matrix(base_quat).transpose(-1, -2)
    p = (base_t @ (pos - base_pos).unsqueeze(-1)).squeeze(-1)
    return torch.cat([p, rot6d(base_t @ quat_to_matrix(quat))], -1)


def absolute_pose(base_pos, base_quat, rel):
    """Inverse of relative_pose: 9 numbers in the base frame -> world position and quaternion."""
    base = quat_to_matrix(base_quat)
    return (base_pos + (base @ rel[..., 0:3].unsqueeze(-1)).squeeze(-1),
            matrix_to_quat(base @ rot6d_to_matrix(rel[..., 3:9])))


# A hand standing square on a board (wiping task): TCP x = board y, TCP y (closing axis) = board x, TCP z = into the board
SQUARE_ON_BOARD = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


def features(socket_estimate, tcp_pos, tcp_quat, emb, rot_ref, board_quat=None, line_turn=None):
    """What the slow policy sees of one observation: socket estimate relative to the TCP (world frame),
    the TCP orientation as a rotation vector relative to the start orientation rot_ref, tactile embeddings.
    Wiping task (board_quat given: the camera's estimate of the board's orientation): also the orientation the
    hand would have standing square on that board, in the same form. line_turn (wiping task "v2"): how far the
    line's direction turns over its length, the one number that tells its shape."""
    turn = rotvec(quat_to_matrix(tcp_quat) @ rot_ref.transpose(-1, -2))
    parts = [socket_estimate - tcp_pos, turn]
    if board_quat is not None:
        square = torch.tensor(SQUARE_ON_BOARD, dtype=rot_ref.dtype, device=rot_ref.device)
        parts.append(rotvec(quat_to_matrix(board_quat) @ square @ rot_ref.transpose(-1, -2)))
    if line_turn is not None:
        parts.append(line_turn.unsqueeze(-1))
    return torch.cat(parts + [emb], dim=-1)


# ----------------------------------------------------------------------------- data
class MinMax(nn.Module):
    """Scale each dimension to [-1, 1] (the 'limits' normalizer of diffusion policy)."""

    def __init__(self, data):
        super().__init__()
        flat = data.reshape(-1, data.shape[-1])
        lo, hi = flat.min(0).values, flat.max(0).values
        self.register_buffer("lo", lo)
        self.register_buffer("span", (hi - lo).clamp(min=1e-6))

    def forward(self, x):
        return 2 * (x - self.lo) / self.span - 1

    def inverse(self, x):
        return (x + 1) / 2 * self.span + self.lo


class TactilePCA(nn.Module):
    """Tactile field of one sensor (points x 3) -> EMB numbers. One PCA shared by both sensors."""

    def __init__(self, fields):
        super().__init__()
        flat = fields.reshape(fields.shape[0], -1)
        mean = flat.mean(0)
        _, s, v = torch.pca_lowrank(flat - mean, q=EMB, center=False, niter=4)
        self.register_buffer("mean", mean)
        self.register_buffer("basis", v[:, :EMB])
        total = ((flat - mean) ** 2).sum()
        self.explained = float((s[:EMB] ** 2).sum() / total)

    def forward(self, field):
        return (field.reshape(*field.shape[:-2], -1) - self.mean) @ self.basis


def load_demos(path, device):
    import zarr

    root = zarr.open(path, mode="r")
    data = {k: torch.tensor(root["data"][k][:], device=device) for k in
            ("tcp_pos", "tcp_quat", "socket_estimate", "board_quat_estimate", "action_pos", "action_quat", "tactile_left",
             "tactile_right", "phase", "perturbed", "tactile_age", "line_turn")
            if k in root["data"]}
    ends = root["meta"]["episode_ends"][:]
    return data, ends


class Windows:
    """Every (episode, time) of the demos as a training sample, built on the GPU.

    Steps marked `perturbed` (the expert was overridden, see expert.py --kick) are not expert
    actions, so samples whose action chunk contains one are left out. They still appear as
    observations of later samples.
    """

    def __init__(self, data, ends, pca, rot_ref):
        device = data["tcp_pos"].device
        starts = np.concatenate([[0], ends[:-1]])
        first = torch.tensor(np.concatenate([np.full(e - s, s) for s, e in zip(starts, ends)]), device=device)
        last = torch.tensor(np.concatenate([np.full(e - s, e - 1) for s, e in zip(starts, ends)]), device=device)
        t = torch.arange(len(first), device=device)
        emb = torch.cat([pca(data["tactile_left"]), pca(data["tactile_right"])], dim=-1)   # (T, 30)
        feat = features(data["socket_estimate"], data["tcp_pos"], data["tcp_quat"], emb, rot_ref,
                        data.get("board_quat_estimate"), data.get("line_turn"))           # (T, FEAT_DIM, or 3 or 4 more)
        self.turn_input = "line_turn" in data

        obs_idx = torch.stack([(t + o).clamp(min=0) for o in OBS_OFFSETS], dim=1)
        obs_idx = torch.maximum(obs_idx, first.unsqueeze(1))            # repeat the first frame of the episode
        self.obs = feat[obs_idx].flatten(1)                                                # (T, OBS_DIM)
        chunk = torch.minimum(t.unsqueeze(1) + torch.arange(HORIZON, device=device), last.unsqueeze(1))
        self.action = torch.cat([relative_pose(data["tcp_pos"][i:i + 8192, None], data["tcp_quat"][i:i + 8192, None],
                                               data["action_pos"][chunk[i:i + 8192]], data["action_quat"][chunk[i:i + 8192]])
                                 for i in range(0, len(t), 8192)])                         # (T, 32, 9) relative
        self.tactile = emb[chunk]                                                          # (T, 32, 30)
        self.index, self.chunk, self.first, self.last = t, chunk, first, last
        self.all_action, self.all_valid = self.action, torch.ones_like(t, dtype=torch.bool)
        self.episode = torch.tensor(np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(zip(starts, ends))]),
                                    device=device)
        if "perturbed" in data:
            keep = ~data["perturbed"].bool()[chunk].any(dim=1)
            for name in ("obs", "action", "tactile", "index", "chunk", "episode", "first", "last"):
                setattr(self, name, getattr(self, name)[keep])
            self.all_valid = keep
        self.n = len(self.index)

    def jittered_action(self, i, jitter):
        """The action chunk that starts up to `jitter` steps before or after sample i (same episode)."""
        shift = torch.randint(-jitter, jitter + 1, i.shape, device=i.device)
        j = torch.minimum(torch.maximum(self.index[i] + shift, self.first[i]), self.last[i])
        return self.all_action[torch.where(self.all_valid[j], j, self.index[i])]

    def batch(self, size):
        i = torch.randint(0, self.n, (size,), device=self.obs.device)
        return self.obs[i], self.action[i], self.tactile[i]


# ----------------------------------------------------------------------------- fast policy
class AsymmetricTokenizer(nn.Module):
    """CNN encoder over the action chunk, GRU decoder conditioned on per-step tactile embeddings."""

    def __init__(self, hidden=32, latent_t=LATENT_T, enc_c=ENC_C):
        super().__init__()
        self.latent_t = latent_t
        layers, channels = [], ACT_DIM
        for _ in range(int(math.log2(HORIZON // latent_t)) - 1):       # each conv halves the length
            layers += [nn.Conv1d(channels, hidden, 5, stride=2, padding=2), nn.ReLU()]
            channels = hidden
        self.encoder = nn.Sequential(*layers, nn.Conv1d(hidden, enc_c, 5, stride=2, padding=2))
        self.quant = nn.Conv1d(enc_c, 2 * LATENT_C, 1)
        self.post_quant = nn.Conv1d(LATENT_C, enc_c, 1)
        self.rnn = nn.GRU(latent_t * enc_c + 2 * EMB, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, ACT_DIM)

    def encode(self, action, sample=True):
        """(B, 32, 9) normalised actions -> latent (B, 8, 16) and its KL to a unit Gaussian."""
        mean, logvar = self.quant(self.encoder(action.transpose(1, 2))).chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        kl = 0.5 * (mean ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=(1, 2))
        return z.transpose(1, 2), kl

    def condition(self, latent):
        """Latent (B, 8, 16) -> the constant part of the decoder input (B, 32)."""
        return self.post_quant(latent.transpose(1, 2)).transpose(1, 2).flatten(1)

    def decode(self, latent, tactile):
        """Whole chunk at once (training): tactile is (B, 32, 30) normalised."""
        cond = self.condition(latent).unsqueeze(1).expand(-1, tactile.shape[1], -1)
        out, _ = self.rnn(torch.cat([cond, tactile], dim=-1))
        return self.fc(out)

    def decode_step(self, cond, tactile, hidden):
        """One control step (deployment): tactile is (B, 30), hidden is the GRU state or None."""
        out, hidden = self.rnn(torch.cat([cond, tactile], dim=-1).unsqueeze(1), hidden)
        return self.fc(out[:, 0]), hidden


# ----------------------------------------------------------------------------- diffusion
class Diffusion(nn.Module):
    """Epsilon-prediction DDPM training and DDIM sampling around a ConditionalUnet1D."""

    def __init__(self, channels, kernel_size, obs_dim=OBS_DIM):
        super().__init__()
        self.net = ConditionalUnet1D(input_dim=channels, global_cond_dim=obs_dim, diffusion_step_embed_dim=128,
                                     down_dims=(256, 512, 1024), kernel_size=kernel_size, n_groups=8,
                                     cond_predict_scale=True)
        steps = torch.arange(TRAIN_STEPS + 1, dtype=torch.float64) / TRAIN_STEPS
        bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2          # squaredcos_cap_v2
        betas = (1 - bar[1:] / bar[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0).float())

    def loss(self, x, obs):
        t = torch.randint(0, TRAIN_STEPS, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        a = self.alpha_bar[t].view(-1, 1, 1)
        return F.mse_loss(self.net(a.sqrt() * x + (1 - a).sqrt() * noise, t, global_cond=obs), noise)

    @torch.no_grad()
    def sample(self, obs, length, channels, generator=None):
        x = torch.randn((obs.shape[0], length, channels), device=obs.device, generator=generator)
        ts = torch.linspace(TRAIN_STEPS - 1, 0, DDIM_STEPS).round().long().tolist()
        for i, t in enumerate(ts):
            a = self.alpha_bar[t]
            eps = self.net(x, torch.full((obs.shape[0],), t, device=obs.device), global_cond=obs)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            if i + 1 == len(ts):
                return x0
            a_next = self.alpha_bar[ts[i + 1]]
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps


class Policy(nn.Module):
    """Everything needed at deployment: PCA, normalisers, tokenizer, and the two diffusion models."""

    def __init__(self, pca, windows, rot_ref, enc_c=ENC_C):
        super().__init__()
        self.pca = pca
        self.register_buffer("rot_ref", rot_ref)   # TCP orientation at the start of the demos
        self.norm_obs, self.norm_action = MinMax(windows.obs), MinMax(windows.action)
        self.norm_tactile = MinMax(windows.tactile)
        self.tokenizer = AsymmetricTokenizer(enc_c=enc_c)
        self.norm_latent = MinMax(torch.zeros(2, LATENT_C))     # refitted after the tokenizer is trained
        self.board_input = windows.obs.shape[1] > OBS_DIM       # wiping demos: the board's orientation is an input too
        self.turn_input = bool(getattr(windows, "turn_input", False))   # wiping task "v2": the line's shape as well
        self.slow = Diffusion(LATENT_C, kernel_size=3, obs_dim=windows.obs.shape[1])   # RDP: diffusion over the latent
        self.dp = Diffusion(ACT_DIM, kernel_size=5, obs_dim=windows.obs.shape[1])      # baseline: diffusion over the actions

    def embed(self, tactile):
        """(B, 2, points, 3) tactile fields of both sensors -> (B, 30)."""
        return torch.cat([self.pca(tactile[:, 0]), self.pca(tactile[:, 1])], dim=-1)

    def features(self, obs, emb):
        board = obs.get("board_quat_estimate") if getattr(self, "board_input", False) else None   # (older checkpoints: no such flag)
        turn = obs.get("line_turn") if getattr(self, "turn_input", False) else None
        return features(obs["socket_estimate"], obs["tcp_pos"], obs["tcp_quat"], emb, self.rot_ref, board, turn)


def train_loop(name, params, loss_fn, steps, lr, batch_fn, ema=None, log=None):
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)
    running, t0, history = 0.0, time.time(), []
    for step in range(1, steps + 1):
        loss = loss_fn(*batch_fn())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ema is not None:
            decay = min(0.9999, (1 + step) / (10 + step))
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), ema.source.parameters()):
                    p_ema.lerp_(p, 1 - decay)
        running += loss.item()
        if step % 500 == 0:
            history.append((step, running / 500))
            print("%-10s step %6d / %d  loss %.5f  (%.0f s)" % (name, step, steps, running / 500, time.time() - t0), flush=True)
            running = 0.0
    if log is not None:
        log[name] = history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/insertion_demos.zarr")
    parser.add_argument("--out", default="runs/rdp_lite")
    parser.add_argument("--at-steps", type=int, default=20000)
    parser.add_argument("--diffusion-steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl", type=float, default=1e-6, help="weight of the tokenizer's KL term")
    parser.add_argument("--enc-channels", type=int, default=ENC_C, help="tokenizer encoder channels (official n_latent_dims)")
    parser.add_argument("--jitter", type=int, default=JITTER, help="latent time jitter in steps (0 = as published)")
    parser.add_argument("--dp-from", default="", help="reuse the PCA, normalisers and DP baseline of this checkpoint "
                                                      "and train only the tokenizer and the slow policy")
    parser.add_argument("--pca-from", default="", help="reuse only the tactile PCA of this checkpoint")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda:0"
    os.makedirs(args.out, exist_ok=True)

    data, ends = load_demos(args.data, device)
    touching = (data["tactile_left"][..., 0].sum(-1) > 0)
    fit = torch.cat([data["tactile_left"][touching], data["tactile_right"][touching]])
    fit = fit[torch.randperm(len(fit), device=device)[:20000]]
    reuse = args.dp_from or args.pca_from
    pca = torch.load(reuse, map_location=device)["policy"].pca if reuse else TactilePCA(fit).to(device)
    rot_ref = quat_to_matrix(data["tcp_quat"][0])   # every episode starts in the same pose
    windows = Windows(data, ends, pca, rot_ref)
    del data
    torch.cuda.empty_cache()
    print("demos: %d episodes, %d steps, %d usable as training samples; tactile PCA keeps %.1f%% of the variance in %d numbers"
          % (len(ends), int(ends[-1]), windows.n, 100 * pca.explained, EMB), flush=True)
    policy = Policy(pca, windows, rot_ref, args.enc_channels).to(device)
    log = {}
    if args.dp_from:
        policy.dp = torch.load(args.dp_from, map_location=device)["policy"].dp

    # 1. fast policy (asymmetric tokenizer): L1 reconstruction + a very small KL term, latent from a time-shifted chunk
    tok = policy.tokenizer

    def at_loss(i):
        a = policy.norm_action(windows.action[i])
        z, kl = tok.encode(policy.norm_action(windows.jittered_action(i, args.jitter)) if args.jitter else a)
        return (tok.decode(z, policy.norm_tactile(windows.tactile[i])) - a).abs().mean() + args.kl * kl.mean()

    train_loop("tokenizer", tok.parameters(), at_loss, args.at_steps, 1e-3,
               lambda: (torch.randint(0, windows.n, (args.batch,), device=device),), log=log)
    tok.requires_grad_(False)
    with torch.no_grad():
        z = torch.cat([tok.encode(policy.norm_action(windows.action[i:i + 4096]))[0] for i in range(0, windows.n, 4096)])
        policy.norm_latent = MinMax(z).to(device)
        recon = torch.cat([tok.decode(tok.encode(policy.norm_action(windows.action[i:i + 4096]), sample=False)[0],
                                      policy.norm_tactile(windows.tactile[i:i + 4096])) for i in range(0, windows.n, 4096)])
        err = (policy.norm_action.inverse(recon) - windows.action)[..., 0:3].norm(dim=-1)
        log["tokenizer_recon_mm"] = float(1000 * err.mean())
        print("tokenizer reconstruction error (position): mean %.3f mm, 99th percentile %.3f mm"
              % (1000 * err.mean(), 1000 * err.flatten()[::97].quantile(0.99)), flush=True)

    # 2. slow policy: diffusion over the latent; 3. baseline: diffusion over the action chunk
    def slow_loss(obs, action, tactile):
        with torch.no_grad():
            z, _ = tok.encode(policy.norm_action(action))
        return policy.slow.loss(policy.norm_latent(z), policy.norm_obs(obs))

    def dp_loss(obs, action, tactile):
        return policy.dp.loss(policy.norm_action(action), policy.norm_obs(obs))

    for name, model, loss_fn in (("slow", policy.slow, slow_loss), ("dp", policy.dp, dp_loss)):
        if name == "dp" and args.dp_from:
            continue
        ema = copy.deepcopy(model).requires_grad_(False)
        ema.source = model
        train_loop(name, model.parameters(), loss_fn, args.diffusion_steps, 1e-4, lambda: windows.batch(args.batch),
                   ema=ema, log=log)
        del ema.source
        model.load_state_dict(ema.state_dict())      # deploy the EMA weights

    torch.save({"policy": policy, "log": log}, os.path.join(args.out, "policy.pt"))
    print("saved", os.path.join(args.out, "policy.pt"))


if __name__ == "__main__":
    import rdp_lite  # run under the module's real name so the saved policy can be loaded elsewhere

    rdp_lite.main()
