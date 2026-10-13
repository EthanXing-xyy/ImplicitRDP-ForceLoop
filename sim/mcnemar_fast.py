"""配对 McNemar：同种子逐回合条件相同，快层变体对基线按回合配对。用法：python mcnemar_fast.py <C49_执行/runs/l1>"""
import glob
import math
import os
import re
import sys

import numpy as np

L = sys.argv[1].rstrip("/")
SEEDS = (2, 3)


def succ(name):
    return np.concatenate([np.load("%s/%s_s%d.npz" % (L, name, s))["success"].astype(bool) for s in SEEDS])


def have(name):
    return all(os.path.exists("%s/%s_s%d.npz" % (L, name, s)) for s in SEEDS)


def mcnemar(a, b):
    if not (have(a) and have(b)):
        return
    x, y = succ(a), succ(b)
    b_, c_ = int((x & ~y).sum()), int((~x & y).sum())
    n = b_ + c_
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b_, c_) + 1)) / 2 ** n) if n else 1.0
    print("%-28s vs %-28s  %5.1f%% vs %5.1f%%  (+%d / -%d of %d pairs)  p=%.3f" % (a, b, 100 * x.mean(), 100 * y.mean(), b_, c_, len(x), p))


variants = sorted({re.sub(r"_s\d\.npz$", "", os.path.basename(f)) for f in glob.glob(L + "/wipe_m3_fast_*_s*.npz")})
gains = sorted({v.rsplit("_g", 1)[1] for v in variants})
print("== each variant vs the policy alone (m3) and vs ImplicitRDP (m0), seeds 2+3 paired")
for v in variants:
    mcnemar(v, "wipe_m3_p1")
    mcnemar(v, "wipe_m0_p1")
print("== same gain: now vs ahead, oracle vs now")
for g in gains:
    mcnemar("wipe_m3_fast_now_g" + g, "wipe_m3_fast_ahead_g" + g)
    mcnemar("wipe_m3_fast_oracle_g" + g, "wipe_m3_fast_now_g" + g)
print("== same mode: neighbouring gains")
for mode in ("now", "ahead", "oracle"):
    for g1, g2 in zip(gains, gains[1:]):
        mcnemar("wipe_m3_fast_%s_g%s" % (mode, g2), "wipe_m3_fast_%s_g%s" % (mode, g1))
