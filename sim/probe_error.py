"""探针误差是白噪声还是慢变 / 随压力系统性的偏差。用法：python probe_error.py <C49_执行/runs/l1>"""
import sys

import numpy as np

L = sys.argv[1].rstrip("/")
for s in (2, 3):
    d = np.load("%s/wipe_m3_fast_now_g0.2_s%d.npz" % (L, s))
    fh, ft = d["fast_f_now"], d["normal_force"]
    m = ~np.isnan(fh) & (ft > 2)
    e = np.where(m, fh - ft, np.nan)
    ac = {}
    for lag in (1, 4, 12):
        a, b = e[:, :-lag], e[:, lag:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        ac[lag] = np.corrcoef(a[ok], b[ok])[0, 1]
    bias = np.nanmean(e, axis=1)
    within = np.nanstd(e - bias[:, None])
    print("seed %d: rmse %.2f N | autocorr lag1 %.2f lag4 %.2f lag12 %.2f | per-episode bias: mean %+.2f N, std across episodes %.2f N | within-episode std %.2f N"
          % (s, np.sqrt(np.nanmean(e ** 2)), ac[1], ac[4], ac[12], np.nanmean(bias), np.nanstd(bias), within))
    for lo, hi in ((2, 6), (6, 10), (10, 14), (14, 20), (20, 40)):
        k = m & (ft >= lo) & (ft < hi)
        print("    true %2d-%2d N: n=%6d  mean error %+.2f N" % (lo, hi, k.sum(), (fh[k] - ft[k]).mean()))
