"""Organisation-aware analyses of the 2019 survey (Section 5.1, Limitations).

All respondents from the same organisation are kept in the same fold (outer and inner loops),
and the bootstrap resamples organisations rather than respondents. Respondents who left the
employer field blank (firm_id 0 in the analysis file) are treated as separate organisations.

  1. Nested CV with stratified grouped folds (10 x 5-fold outer, 5-fold inner grid search):
     ROC-AUC, balanced accuracy, Brier score; corrected resampled t-test for ML minus LR.
  2. Permutation test of the logistic regression AUC (C = 1) with grouped folds.
  3. Fold stability of the SHAP rankings with grouped folds (simulation protocol, 20 x 5-fold).
  4. Organisation-level bootstrap of the SHAP rankings (200 refits).

Requires the CONFIDENTIAL analysis file produced by real_data/audit_2019.py
(survey2019_analysis.csv), which is not in this repository and must never be committed.
Only aggregate statistics are written.

Usage:  python grouped_cv.py PATH/TO/survey2019_analysis.csv [OUT_JSON] [N_JOBS]
"""
import json, os, sys, warnings
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS, load_ra, load_sim

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "real_grouped_cv.json")
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def load():
    ra = load_ra(); df = pd.read_csv(DATA)
    X = df[ra.FEATS].to_numpy(float); y = df["Adpt"].to_numpy(int)
    g = df["firm_id"].to_numpy().copy()
    blank = g == 0
    g[blank] = 100_000 + np.arange(blank.sum())          # blank employer field: one group per respondent
    return ra, X, y, g


def outer_repeat(r):
    ra, X, y, g = load(); rows = []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g)):
        for name, (est, grid) in ra.specs(r).items():
            if grid:
                inner = StratifiedGroupKFold(5, shuffle=True, random_state=100 * r + f)
                m = GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr], groups=g[tr]).best_estimator_
            else:
                m = est.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            rows.append(dict(rep=r, fold=f, model=name, n_train=len(tr), n_test=len(te), auc=roc_auc_score(y[te], p),
                             bal_acc=balanced_accuracy_score(y[te], (p >= .5).astype(int)), brier=brier_score_loss(y[te], p)))
    return rows


def lr_cv_auc(X, y, g, seed, ra):
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        m = ra.fixed_models(0)["LR"].fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def perm_chunk(ks):
    ra, X, y, g = load()
    return [lr_cv_auc(X, np.random.default_rng(10_000 + k).permutation(y), g, k, ra) for k in ks]


def fold_stab_repeat(s):
    ra, X, y, g = load(); sim = load_sim(); imps = {m: [] for m in ("LR", "RF", "XGB")}
    pred = {m: np.zeros(len(y)) for m in imps}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
        for name, m in sim.models(s).items():
            m.fit(X[tr], y[tr]); imps[name].append(sim.shap_imp(name, m, X[tr], X[te]))
            pred[name][te] = m.predict_proba(X[te])[:, 1]
    return ({m: float(np.mean([stats.kendalltau(a, b)[0] for a, b in combinations(v, 2)])) for m, v in imps.items()},
            {m: float(roc_auc_score(y, pred[m])) for m in pred})


def boot_chunk(bs):
    ra, X, y, g = load(); orgs = np.unique(g); members = {o: np.where(g == o)[0] for o in orgs}; out = []
    for b in bs:
        rng = np.random.default_rng(b)
        idx = np.concatenate([members[o] for o in rng.choice(orgs, len(orgs), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        r = {}
        for name, m in ra.fixed_models(b).items():
            m.fit(X[idx], y[idx]); r[name] = stats.rankdata(-ra.shap_mean_abs(name, m, X[idx])).tolist()
        out.append(dict(b=b, n=int(len(idx)), ranks=r))
    return out


def corrected(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    se = np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1)); t = d.mean() / se
    q = stats.t.ppf(0.975, k - 1)
    return dict(mean_diff=round(float(d.mean()), 3), t=round(float(t), 2), p=round(float(2 * stats.t.sf(abs(t), k - 1)), 3),
                ci95=[round(float(d.mean() - q * se), 3), round(float(d.mean() + q * se), 3)])


if __name__ == "__main__":
    ra, X, y, g = load()
    R = {"n": int(len(y)), "n_groups": int(len(np.unique(g))),
         "largest_group": int(pd.Series(g).value_counts().max())}
    # 1. nested CV
    cv = pd.DataFrame([x for r in Parallel(n_jobs=NJ)(delayed(outer_repeat)(r) for r in range(10)) for x in r])
    s = cv.groupby("model")[["auc", "bal_acc", "brier"]].agg(["mean", "std"]).round(3)
    R["nested_cv"] = {m: {f"{a}_{b}": float(s.loc[m, (a, b)]) for a in ("auc", "bal_acc", "brier") for b in ("mean", "std")} for m in s.index}
    R["test_fold_size"] = {"min": int(cv.n_test.min()), "max": int(cv.n_test.max())}
    k = cv[cv.model == "LR"].sort_values(["rep", "fold"])
    ntr, nte = float(k.n_train.mean()), float(k.n_test.mean())
    R["ml_minus_lr_auc"] = {m: corrected(cv[cv.model == m].sort_values(["rep", "fold"]).auc.values, k.auc.values, ntr, nte)
                            for m in ("RF", "XGB")}
    # 2. permutation test (LR, C = 1)
    base = float(np.mean([lr_cv_auc(X, y, g, s_, ra) for s_ in range(5)]))
    perm = [v for c in Parallel(n_jobs=NJ)(delayed(perm_chunk)(list(range(i, i + 50))) for i in range(0, 500, 50)) for v in c]
    R["permutation_test_LR"] = {"cv_auc": round(base, 3), "p": float((1 + np.sum(np.array(perm) >= base)) / 501),
                                "null_auc_95pct": round(float(np.percentile(perm, 95)), 3)}
    # 3. fold stability
    fs = Parallel(n_jobs=NJ)(delayed(fold_stab_repeat)(s_) for s_ in range(20))
    R["fold_stability"] = {m: round(float(np.mean([f[0][m] for f in fs])), 3) for m in ("LR", "RF", "XGB")}
    R["fold_protocol_auc"] = {m: round(float(np.mean([f[1][m] for f in fs])), 3) for m in ("LR", "RF", "XGB")}
    # 4. organisation bootstrap
    bo = [x for c in Parallel(n_jobs=NJ)(delayed(boot_chunk)(list(range(i, i + 20))) for i in range(0, 200, 20)) for x in c]
    TM = [ra.LABEL[f] for f in ra.FEATS].index("Top management support")
    names = [ra.LABEL[f] for f in ra.FEATS]; R["org_bootstrap"] = {"refits": len(bo), "n_range": [min(b["n"] for b in bo), max(b["n"] for b in bo)]}
    for m in ("LR", "RF", "XGB"):
        Rk = np.array([b["ranks"][m] for b in bo])
        R["org_bootstrap"][m] = {"share_first": dict(zip(names, (Rk == 1).mean(0).round(3).tolist())),
                                 "share_top3": dict(zip(names, (Rk <= 3).mean(0).round(3).tolist())),
                                 "tau_all": round(float(np.mean([stats.kendalltau(a, c)[0] for a, c in combinations(Rk, 2)])), 3)}
    json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R, indent=1))
