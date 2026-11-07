"""Correlation experiment (Table 6): low and high correlation levels.

The moderate level is the main simulation (results/sim_raw_interventional.csv), which this
design reproduces exactly, so it is not rerun. Synthetic data only.
3 scenarios x 3 sample sizes (139, 300, 1,000) x 100 replications per level. Resumable.

Usage:  python run_correlation.py [LEVELS] [OUT_CSV] [TIME_BUDGET_SECONDS] [N_JOBS]
Default: LEVELS=low,high  OUT_CSV=results/corr_raw.csv
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import correlation_lib
from common import RESULTS

LEVELS = (sys.argv[1] if len(sys.argv) > 1 else "low,high").split(",")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "corr_raw.csv")
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else float("inf")
NJ = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SIZES = [139, 300, 1000]
_C = {}


def job(level, n, sc, r, nl, truth):
    if level not in _C:
        sys.path.insert(0, HERE); import correlation_lib as cl; _C[level] = cl.load(level)
    sim = _C[level]; sim.NL_SCALE = nl
    o = sim.one_run(n, sc, r, truth, nl)
    if o is not None:
        o["corr"] = level
    return o


if __name__ == "__main__":
    meta, TR, NL = {}, {}, {}
    for lv in LEVELS:
        sim = correlation_lib.load(lv); NL[lv] = sim.NL_SCALE
        F = sim.latent(100_000, np.random.default_rng(1))
        TR[lv] = {sc: (sim.true_importance(sc) if sc != "null" else None) for sc in ("linear", "nonlinear", "null")}
        off = sim.CORR[~np.eye(len(sim.CORR), dtype=bool)]
        meta[lv] = {"mean_abs_r": float(np.abs(off).mean()), "max_r": float(off.max()),
                    "min_eig": float(np.linalg.eigvalsh(sim.CORR).min()),
                    "linear_scale": sim.LSCALE, "nl_scale": sim.NL_SCALE, "target_auc": sim.TARGET,
                    "oracle_auc": {s: sim._oracle_auc(sim.g(s, F), F) for s in ("linear", "nonlinear")},
                    "true_importance": {s: dict(zip(sim.NAMES, np.round(TR[lv][s], 4).tolist())) for s in ("linear", "nonlinear")}}
    json.dump(meta, open(RESULTS / ("corr_meta_" + "_".join(LEVELS) + ".json"), "w"), indent=1)
    jobs = [(lv, n, s, r) for lv in LEVELS for s in ("linear", "nonlinear", "null") for n in SIZES for r in range(100)]
    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, keep_default_na=False, na_values=[""]); rows = d.to_dict("records")
        done = {(a, int(b), c, int(e)) for a, b, c, e in zip(d["corr"], d.n, d.scenario, d.rep)}
    todo = sorted([j for j in jobs if j not in done], key=lambda j: (j[3], j[0], j[2], j[1]))
    t0 = time.time(); print("todo", len(todo), flush=True)
    for i in range(0, len(todo), 24):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); break
        res = Parallel(n_jobs=NJ)(delayed(job)(lv, n, s, r, NL[lv], TR[lv][s]) for lv, n, s, r in todo[i:i + 24])
        rows += [x for x in res if x is not None]
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("progress", len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
    else:
        print("ALLDONE", flush=True)
