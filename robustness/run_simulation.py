"""Main calibrated simulation rerun with interventional SHAP for all models (Tables 5, Figs 1-3).

Synthetic data only. 3 scenarios x 6 sample sizes x 100 replications = 1,800 datasets.
Resumable: rerun the same command after an interruption and finished rows are kept.

Usage:  python run_simulation.py [OUT_CSV] [TIME_BUDGET_SECONDS] [N_JOBS]
Default OUT_CSV = results/sim_raw_interventional.csv
"""
import os, sys, time, warnings
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_sim

OUT = sys.argv[1] if len(sys.argv) > 1 else str(RESULTS / "sim_raw_interventional.csv")
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SIZES = [100, 139, 200, 300, 500, 1000]
_SIM = None


def job(n, sc, r, nl, truth):
    global _SIM
    if _SIM is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import load_sim as ls
        _SIM = ls()
    _SIM.NL_SCALE = nl
    return _SIM.one_run(n, sc, r, truth, nl)


if __name__ == "__main__":
    sim = load_sim(); nl = sim._scale_for_nonlinear(); sim.NL_SCALE = nl
    TR = {sc: (sim.true_importance(sc) if sc != "null" else None) for sc in ("linear", "nonlinear", "null")}
    jobs = [(n, s, r) for s in ("linear", "nonlinear", "null") for n in SIZES for r in range(100)]
    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, keep_default_na=False, na_values=[""])  # keep the scenario label "null"
        rows = d.to_dict("records"); done = {(int(a), b, int(c)) for a, b, c in zip(d.n, d.scenario, d.rep)}
    todo = sorted([j for j in jobs if j not in done], key=lambda j: (j[2], j[1], j[0]))
    t0 = time.time(); print("todo", len(todo), flush=True)
    for i in range(0, len(todo), 24):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); break
        res = Parallel(n_jobs=NJ)(delayed(job)(n, s, r, nl, TR[s]) for n, s, r in todo[i:i + 24])
        rows += [x for x in res if x is not None]
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("progress", len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
    else:
        print("ALLDONE", flush=True)
