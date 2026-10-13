"""成功率 对 力读数白噪声 的曲线（oracle 压力 + 高斯噪声，m3 + 法向力环）。用法：python noise_curve.py <C49_执行/runs/l1> <out.png>"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
rows = {}   # (gain, noise) -> {seed: (success, wiped, inside)}
for f in glob.glob(L + "/wipe_m3_fast_oracle_*_s*.npz"):
    m = re.match(r".*wipe_m3_fast_oracle_(?:n([\d.]+)_)?g([\d.]+)_s(\d)\.npz", f)
    nz, g, s = float(m.group(1) or 0), float(m.group(2)), int(m.group(3))
    d = np.load(f)
    rows.setdefault((g, nz), {})[s] = (d["success"].mean(), d["wiped"].mean())
# 探针版（now）当参照点：系统性误差 rmse ~1.6 N
probe = {}
for f in glob.glob(L + "/wipe_m3_fast_now_g*_s*.npz"):
    m = re.match(r".*now_g([\d.]+)_s(\d)\.npz", f)
    probe.setdefault(float(m.group(1)), {})[int(m.group(2))] = np.load(f)["success"].mean()
base = {n: np.mean([np.load("%s/%s_p1_s%d.npz" % (L, n, s))["success"].mean() for s in (2, 3)]) for n in ("wipe_m3", "wipe_m0")}
print("| 增益 mm/N | 噪声 σ (N) | 种子 2 | 种子 3 | 合并 | 擦掉比例 |")
print("|---|---|---|---|---|---|")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for g in sorted({k[0] for k in rows}):
    pts = sorted((nz, v) for (gg, nz), v in rows.items() if gg == g)
    xs = [nz for nz, v in pts]
    ys = [100 * np.mean([r[0] for r in v.values()]) for nz, v in pts]
    for nz, v in pts:
        ss = sorted(v.items())
        print("| %.1f | %.1f | %s | %.1f%% | %.0f%% |" % (g, nz, " | ".join("%.1f" % (100 * r[0]) for s, r in ss), 100 * np.mean([r[0] for r in v.values()]),
                                                         100 * np.mean([r[1] for r in v.values()])))
    ax.plot(xs, ys, "o-", label="gain %.1f mm/N" % g)
for g, v in sorted(probe.items()):
    ax.plot([1.6], [100 * np.mean(list(v.values()))], "x", ms=8, color="k" if g == 0.2 else "gray")
    ax.annotate("probe %.1f" % g, (1.6, 100 * np.mean(list(v.values()))), textcoords="offset points", xytext=(6, -4), fontsize=7)
ax.axhline(100 * base["wipe_m3"], ls="--", color="gray", lw=0.8)
ax.annotate("m3 alone %.0f%%" % (100 * base["wipe_m3"]), (0.05, 100 * base["wipe_m3"] + 1), fontsize=7, color="gray")
ax.axhline(100 * base["wipe_m0"], ls=":", color="gray", lw=0.8)
ax.annotate("ImplicitRDP %.0f%%" % (100 * base["wipe_m0"]), (2.6, 100 * base["wipe_m0"] + 1), fontsize=7, color="gray")
ax.set_xlabel("white noise on the force reading, std (N), 24 Hz")
ax.set_ylabel("success (%), seeds 2+3, 192 episodes")
ax.set_title("m3 + normal-force P loop: success vs force-reading noise", fontsize=9)
ax.set_ylim(40, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
