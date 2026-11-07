"""Worker functions for the permutation null on the 2019 survey (Table 4).

Paths are passed through environment variables so that parallel workers can import this
module: PL_DATA = confidential analysis CSV (never committed)."""
import os, sys
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_ra, load_sim

ra = load_ra(); sim = load_sim()
df = pd.read_csv(os.environ["PL_DATA"]); X = df[ra.FEATS].to_numpy(float); y = df["Adpt"].to_numpy(int)
TM = ra.FEATS.index([f for f in ra.FEATS if ra.LABEL[f] == "Top management support"][0])


def fold_stab(y, splits, seed0, return_all=False):
    """Mean pairwise Kendall tau between the five fold rankings, averaged over `splits` repetitions
    (the simulation protocol: fixed hyperparameters, SHAP on the held-out fold)."""
    out = {m: [] for m in ("LR", "RF", "XGB")}; auc = {m: [] for m in out}
    from sklearn.metrics import roc_auc_score
    for s in range(splits):
        imps = {m: [] for m in out}; pred = {m: np.zeros(len(y)) for m in out}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed0 + s).split(X, y):
            for name, m in sim.models(seed0 + s).items():
                m.fit(X[tr], y[tr]); imps[name].append(sim.shap_imp(name, m, X[tr], X[te]))
                pred[name][te] = m.predict_proba(X[te])[:, 1]
        for name in out:
            out[name].append(np.mean([stats.kendalltau(a, b)[0] for a, b in combinations(imps[name], 2)]))
            auc[name].append(roc_auc_score(y, pred[name]))
    if return_all:
        return out, auc
    return {m: float(np.mean(v)) for m, v in out.items()}


def boot(y, B, seed):
    rng = np.random.default_rng(seed); R = {m: [] for m in ("LR", "RF", "XGB")}
    for b in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for name, m in ra.fixed_models(b).items():
            m.fit(X[idx], y[idx]); R[name].append(stats.rankdata(-ra.shap_mean_abs(name, m, X[idx])))
    res = {}
    for name, r in R.items():
        r = np.vstack(r); first = (r == 1).mean(0); top3 = (r <= 3).mean(0)
        res[name] = dict(max_first=float(first.max()), argmax_first=int(first.argmax()), tm_first=float(first[TM]),
                         max_top3=float(top3.max()), tm_top3=float(top3[TM]),
                         tau=float(np.mean([stats.kendalltau(a, c)[0] for a, c in combinations(r, 2)])))
    return res


def one(k, splits=20, B=200):
    yp = np.random.default_rng(10_000 + k).permutation(y)
    return dict(k=k, fold=fold_stab(yp, splits, 500 + k * 100), boot=boot(yp, B, 700 + k))
