"""快层参照用的力探针：从 m3 评测时存下的 GRU 状态 h 线性回归真实压力（现在 / 4 步后），权重存成 npz 给评测用。
用 seed 1 的 96 回合拟合（按回合留一半报留出 R2），评测在 seed 2、3 上做，不碰训练数据。
python3 fit_force_probe.py ../../C49_执行/runs/l1/wipe_m3_p1 ../../C49_执行/runs/probe_m3_force.npz"""
import sys

import numpy as np

K_AHEAD = 4
base, out = sys.argv[1], sys.argv[2]
h = np.load(base + "_h.npy").astype(np.float32)            # (n, T, 512)
d = np.load(base + ".npz")
f = d["normal_force"].astype(np.float32)                    # (n, T) 真实压力
n, T, D = h.shape
print("h", h.shape, "force", f.shape, "episodes", n)


def fit(X, y, lam=1e-2):
    Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xb.T @ y)
    return w[:-1], w[-1]


def r2(X, y, w, b):
    p = X @ w + b
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt(((p - y) ** 2).mean())


res = {}
for name, k in (("now", 0), ("ahead", K_AHEAD)):
    X = h[:, : T - k].reshape(-1, D)
    y = f[:, k:].reshape(-1)
    ep = np.repeat(np.arange(n), T - k)
    fit_m, hold_m = ep < n // 2, ep >= n // 2
    w, b = fit(X[fit_m], y[fit_m])
    rh, rmse = r2(X[hold_m], y[hold_m], w, b)
    w_all, b_all = fit(X, y)
    r_in, rmse_in = r2(X, y, w_all, b_all)
    print("%-6s held-out R2 %.3f (rmse %.2f N)   all-episode fit R2 %.3f (rmse %.2f N)" % (name, rh, rmse, r_in, rmse_in))
    res["w_" + name], res["b_" + name], res["r2_heldout_" + name] = w_all, np.float32(b_all), np.float32(rh)
# 复制上一帧的参照：用现在的真实压力猜 4 步后
y0, y4 = f[:, : T - K_AHEAD].reshape(-1), f[:, K_AHEAD:].reshape(-1)
print("copy-last baseline for 4 steps ahead: R2 %.3f (rmse %.2f N)" % (1 - ((y0 - y4) ** 2).sum() / ((y4 - y4.mean()) ** 2).sum(), np.sqrt(((y0 - y4) ** 2).mean())))
np.savez(out, **res, k_ahead=K_AHEAD, source=base)
print("saved", out)
