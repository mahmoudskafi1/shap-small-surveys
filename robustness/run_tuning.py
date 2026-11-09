"""Tuning check for the simulation (Section 5.2, Limitations).

The main simulation uses fixed hyperparameters. This script reruns the nonlinear scenario at
n = 139, 300 and 1,000 (100 datasets each, the same seeds and therefore the same datasets as in
the main simulation) with the hyperparameters of every model selected by an inner stratified
5-fold grid search on ROC-AUC inside each outer training fold. Everything else, including the
interventional SHAP values and all outcome measures, is as in run_simulation.py.

Grids (the fixed values of the main simulation are included in each grid):
  LR : C in {0.01, 0.1, 1, 10}
  RF : max_depth in {2, 4, None} x min_samples_leaf in {3, 5, 10}      (250 trees)
  XGB: max_depth in {1, 2, 3} x reg_lambda in {1, 5, 10}              (200 trees, learning rate 0.05)

Synthetic data only. Resumable.
Usage:  python run_tuning.py [OUT_CSV] [TIME_BUDGET_SECONDS] [N_JOBS]
        python run_tuning.py --summary [OUT_CSV]
"""
import os, sys, time, warnings
from itertools import combinations
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_sim

SIZES = [139, 300, 1000]
SCEN = "nonlinear"
GRIDS = {"LR": {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
         "RF": {"max_depth": [2, 4, None], "min_samples_leaf": [3, 5, 10]},
         "XGB": {"max_depth": [1, 2, 3], "reg_lambda": [1.0, 5.0, 10.0]}}
_SIM = None


def job(n, rep, nl, truth):
    global _SIM
    if _SIM is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import load_sim as ls
        _SIM = ls()
    sim = _SIM; sim.NL_SCALE = nl
    seed = sim.SCEN_ID[SCEN] * 10_000_000 + n * 1000 + rep          # same seed as the main simulation
    X, y = sim.make_data(n, SCEN, seed)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    out = {"n": n, "scenario": SCEN, "rep": rep}
    imps = {m: [] for m in GRIDS}; preds = {m: np.zeros(n) for m in GRIDS}; chosen = {m: [] for m in GRIDS}
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        for name, est in sim.models(seed).items():
            gs = GridSearchCV(est, GRIDS[name], cv=StratifiedKFold(5, shuffle=True, random_state=seed + f),
                              scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr])
            m = gs.best_estimator_; chosen[name].append(str(gs.best_params_))
            preds[name][te] = m.predict_proba(X[te])[:, 1]
            imps[name].append(sim.shap_imp(name, m, X[tr], X[te]))
    for name in GRIDS:
        out[f"auc_{name}"] = roc_auc_score(y, preds[name])
        M = np.vstack(imps[name])
        out[f"stab_{name}"] = float(np.nanmean([kendalltau(a, b)[0] for a, b in combinations(M, 2)]))
        mi = M.mean(0); top = np.argsort(-mi)
        out[f"dominance_{name}"] = float(mi[top[0]] / mi.sum())
        out[f"tau_truth_{name}"] = float(kendalltau(mi, truth)[0])
        out[f"top1_{name}"] = int(top[0] == np.argmax(truth))
        out[f"top3_{name}"] = len(set(top[:3]) & set(np.argsort(-truth)[:3])) / 3
        out[f"params_{name}"] = " | ".join(chosen[name])
    return out


def summary(path):
    t = pd.read_csv(path); t["setting"] = "tuned"
    f = pd.read_csv(RESULTS / "sim_raw_interventional.csv", keep_default_na=False, na_values=[""])
    f = f[(f.scenario == SCEN) & f.n.isin(SIZES)].copy(); f["setting"] = "fixed"
    f = f.merge(t[["n", "rep"]], on=["n", "rep"])                      # the same datasets
    D = pd.concat([f, t.drop(columns=[c for c in t if c.startswith("params_")])])
    for m in ("RF", "XGB"):
        D[f"gain_{m}"] = D[f"auc_{m}"] - D["auc_LR"]
    cols = ["auc_LR", "auc_RF", "auc_XGB", "gain_RF", "gain_XGB", "stab_LR", "stab_RF", "stab_XGB",
            "top1_LR", "top1_RF", "top1_XGB", "tau_truth_LR", "tau_truth_RF", "tau_truth_XGB"]
    S = D.groupby(["n", "setting"])[cols].mean().round(3)
    print(S.T.to_string())
    w = t.merge(f, on=["n", "rep"], suffixes=("_t", "_f"))
    for m in ("RF", "XGB"):
        d = (w[f"auc_{m}_t"] - w["auc_LR_t"]) - (w[f"auc_{m}_f"] - w["auc_LR_f"])
        print(m, "change in gain (tuned - fixed), mean by n:", d.groupby(w.n).mean().round(3).to_dict(),
              "sd:", d.groupby(w.n).std().round(3).to_dict())
    S.to_csv(os.path.splitext(path)[0] + "_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "sim_tuned_nonlinear.csv")); sys.exit()
    OUT = sys.argv[1] if len(sys.argv) > 1 else str(RESULTS / "sim_tuned_nonlinear.csv")
    BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")
    NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    sim = load_sim(); nl = sim._scale_for_nonlinear(); sim.NL_SCALE = nl; truth = sim.true_importance(SCEN)
    jobs = [(n, r) for n in SIZES for r in range(100)]
    rows, done = [], set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); rows = d.to_dict("records"); done = {(int(a), int(b)) for a, b in zip(d.n, d.rep)}
    todo = sorted([j for j in jobs if j not in done], key=lambda j: (j[1], j[0]))
    t0 = time.time(); print("todo", len(todo), flush=True)
    for i in range(0, len(todo), 6):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); break
        res = Parallel(n_jobs=NJ)(delayed(job)(n, r, nl, truth) for n, r in todo[i:i + 6])
        rows += [x for x in res if x is not None]
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("progress", len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
    else:
        print("ALLDONE", flush=True)
