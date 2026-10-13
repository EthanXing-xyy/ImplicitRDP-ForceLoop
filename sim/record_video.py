"""Record one wiping episode as frames (one process per policy; Isaac Gym allows one sim per process), then compose
the two runs side by side with a live pressure trace into media/wipe.gif and media/wipe.mp4.

    python record_video.py record --label irdp  --ckpt ../checkpoints/wipe_irdp_v2/policy.pt --residual none   --seed 2 --out ../media/frames_irdp
    python record_video.py record --label loop  --ckpt ../checkpoints/wipe_m3/policy.pt      --residual oracle --gain 0.4 --seed 2 --out ../media/frames_loop
    python record_video.py compose --left ../media/frames_irdp --right ../media/frames_loop --out ../media/wipe
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def record(args):
    os.chdir(HERE)
    os.environ.setdefault("WIPE_TASK", "v2")
    from wipe_env import ESTIMATE_NOISE, WipeEnv      # isaacgym must be imported before torch
    import eval_irdp_wipe_fast as ev
    import torch
    os.makedirs(args.out, exist_ok=True)
    policy = torch.load(args.ckpt, map_location="cuda:0")["policy"].eval()
    fast = None
    if args.residual != "none":
        fast = dict(k_ahead=0) if args.residual == "oracle" else ev.load_probe(args.probe, args.residual, "cuda:0")
        fast.update(gain=args.gain * 1e-3, clip=3e-3, target=10.0, contact=2.0, mode=args.residual, noise_n=0.0,
                    gen=torch.Generator(device="cuda:0").manual_seed(1000 + args.seed))
    env = WipeEnv(num_envs=1, headless=True, seed=args.seed, estimate_noise=ESTIMATE_NOISE, height_noise=0.004, perturb=1.0,
                  camera_errors=True, randomize_grasp=True, camera=True)
    trace = []

    def on_step(t, obs, f_use, d):
        env.snapshot(os.path.join(args.out, "f%04d.png" % t))
        trace.append(dict(t=t, force=float(obs["normal_force"][0]), f_hat=float(f_use[0]) if fast is not None else None,
                          d_mm=float(d[0]) * 1e3, success=bool(obs["success"][0])))

    torch.manual_seed(args.seed)
    stats = ev.run_episode(env, policy, fast, on_step=on_step)
    json.dump(dict(label=args.label, trace=trace, wiped=float(stats["wiped"][0]), success=bool(stats["success"][0])),
              open(os.path.join(args.out, "trace.json"), "w"))
    print("%s: success %s, wiped %.0f%%, %d frames" % (args.label, bool(stats["success"][0]), 100 * stats["wiped"][0], len(trace)))
    env.close()


def compose(args):
    import cv2
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    runs = []
    for d in (args.left, args.right):
        tr = json.load(open(os.path.join(d, "trace.json")))
        frames = sorted(glob.glob(os.path.join(d, "f*.png")))
        runs.append((tr, frames))
    n = min(len(r[1]) for r in runs)
    W, H = args.width, int(args.width * 0.75)
    titles = {"irdp": "ImplicitRDP (policy alone)", "loop": "ImplicitRDP + force loop"}
    fps = 24 / args.every
    out_frames = []
    fig, ax = plt.subplots(figsize=(2 * W / 100, 1.9), dpi=100)
    for t in range(0, n, args.every):
        tiles = []
        for tr, frames in runs:
            img = cv2.imread(frames[t])[..., ::-1]
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            st = tr["trace"][t]
            label = titles.get(tr["label"], tr["label"])
            cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            txt = "pressure %4.1f N" % st["force"] + ("   residual %+.1f mm" % st["d_mm"] if st["f_hat"] is not None else "")
            cv2.putText(img, txt, (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            if st["success"]:
                cv2.putText(img, "line wiped", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 2, cv2.LINE_AA)
            tiles.append(np.ascontiguousarray(img))
        top = np.concatenate(tiles, axis=1)
        ax.clear()
        ax.axhspan(5, 15, color="#cfe8cf", alpha=0.7, lw=0)
        for (tr, _), c, lab in zip(runs, ("#d62728", "#1f77b4"), ("policy alone", "+ force loop")):
            f = [s["force"] for s in tr["trace"][:t + 1]]
            ax.plot(np.arange(len(f)) / 24.0, f, color=c, lw=1.4, label=lab)
        ax.set_xlim(0, n / 24.0)
        ax.set_ylim(0, 30)
        ax.set_ylabel("normal force (N)", fontsize=8)
        ax.set_xlabel("time (s); green band = 5-15 N window", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plot = cv2.resize(plot, (top.shape[1], int(plot.shape[0] * top.shape[1] / plot.shape[1])))
        out_frames.append(np.concatenate([top, plot], axis=0))
    print("frames", len(out_frames), out_frames[0].shape)
    imageio.mimsave(args.out + ".gif", out_frames, duration=1000 / fps, loop=0)
    vw = cv2.VideoWriter(args.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_frames[0].shape[1], out_frames[0].shape[0]))
    for fr in out_frames:
        vw.write(np.ascontiguousarray(fr[..., ::-1]))
    vw.release()
    print("saved", args.out + ".gif", "%.1f MB" % (os.path.getsize(args.out + ".gif") / 1e6), "and .mp4")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--label", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--residual", choices=("none", "now", "ahead", "oracle"), default="none")
    r.add_argument("--gain", type=float, default=0.4)
    r.add_argument("--probe", default="../results/probe_m3_force.npz")
    r.add_argument("--seed", type=int, default=2)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compose")
    c.add_argument("--left", required=True)
    c.add_argument("--right", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--width", type=int, default=400, help="pixels per tile")
    c.add_argument("--every", type=int, default=2, help="keep every n-th control step (24 Hz)")
    a = p.parse_args()
    record(a) if a.cmd == "record" else compose(a)
