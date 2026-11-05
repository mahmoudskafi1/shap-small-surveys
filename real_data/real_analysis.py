"""
Real-data analysis for the methods paper (2019 Lebanese survey, n = 139; published
in Skafi, Yunis & Zekri, IEEE Access 2020 - reanalysed here, disclosed).

Steps
 1. Repeated nested CV (10 x 5-fold outer, 5-fold inner grid search) for LR, RF, XGB
    and a majority baseline; ROC-AUC, balanced accuracy, Brier score.
 2. ML-vs-LR difference: corrected resampled t-test (Nadeau & Bengio, 2003) on paired fold AUCs.
 3. Label-permutation test of the LR AUC (is the signal above chance?).
 4. Explanation stability: 200 bootstrap refits; for each model, distribution of the
    rank of every construct, frequency with which each construct is ranked first,
    and agreement (Kendall tau) between SHAP ranking and |standardised LR coefficient| ranking.
 5. Sensitivity: excluding the one organisation with 18 respondents.

Usage: python -I real_analysis.py ANALYSIS_CSV OUTDIR
(ANALYSIS_CSV = survey2019_analysis.csv written by audit_2019.py)
"""
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
FEATS = ["CCbenefitSc", "CosteffectSC", "CmptbltSC", "ComplxtSC", "TrialabilitySC",
         "InnovtSC", "TchnExpSC", "CmptPrsrSC", "TopMgtSC"]
LABEL = {"CCbenefitSc": "Relative advantage", "CosteffectSC": "Cost", "CmptbltSC": "Compatibility",
         "ComplxtSC": "Complexity", "TrialabilitySC": "Trialability", "InnovtSC": "Innovativeness",
         "TchnExpSC": "Technological experience", "CmptPrsrSC": "Competitive pressure",
         "TopMgtSC": "Top management support"}


def specs(seed):
    return {
        "Majority": (DummyClassifier(strategy="most_frequent"), None),
        "LR": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
               {"m__C": [0.01, 0.1, 1.0, 10.0]}),
        "RF": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
               {"max_depth": [2, 4, None], "min_samples_leaf": [3, 10]}),
        "XGB": (XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0),
                {"max_depth": [1, 2, 3], "reg_lambda": [1.0, 10.0]}),
    }


def nested_cv(X, y, repeats=10, seed=0):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    rows = []
    for f, (tr, te) in enumerate(outer.split(X, y)):
        for name, (est, grid) in specs(seed).items():
            if grid:
                gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=f),
                                  scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr])
                m = gs.best_estimator_
            else:
                m = est.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            rows.append({"fold": f, "model": name, "auc": roc_auc_score(y[te], p),
                         "bal_acc": balanced_accuracy_score(y[te], (p >= .5).astype(int)),
                         "brier": brier_score_loss(y[te], p)})
    return pd.DataFrame(rows)


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(d.mean()), float(t), float(p)


def fixed_models(seed):
    return {
        "LR": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                     random_state=seed, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=10.0, eval_metric="logloss",
                             random_state=seed, n_jobs=1, verbosity=0),
    }


def shap_mean_abs(name, m, X):
    if name == "LR":
        sc, lr = m.named_steps["sc"], m.named_steps["m"]
        sv = shap.LinearExplainer(lr, sc.transform(X)).shap_values(sc.transform(X))
    else:
        sv = shap.TreeExplainer(m).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    return np.abs(sv).mean(0)


def bootstrap_stability(X, y, B=200, seed=0):
    rng = np.random.default_rng(seed)
    ranks = {m: [] for m in ("LR", "RF", "XGB")}
    taus = {m: [] for m in ("RF", "XGB")}
    for b in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        imps = {}
        for name, m in fixed_models(b).items():
            m.fit(X[idx], y[idx])
            imps[name] = shap_mean_abs(name, m, X[idx])
            ranks[name].append(stats.rankdata(-imps[name]))
        for name in ("RF", "XGB"):
            taus[name].append(stats.kendalltau(imps[name], imps["LR"])[0])
    out = {}
    for name, R in ranks.items():
        R = np.vstack(R)
        out[name] = {
            "median_rank": dict(zip([LABEL[f] for f in FEATS], np.median(R, 0).tolist())),
            "rank_IQR": dict(zip([LABEL[f] for f in FEATS],
                                 (np.percentile(R, 75, 0) - np.percentile(R, 25, 0)).tolist())),
            "share_ranked_first": dict(zip([LABEL[f] for f in FEATS], (R == 1).mean(0).round(3).tolist())),
            "share_in_top3": dict(zip([LABEL[f] for f in FEATS], (R <= 3).mean(0).round(3).tolist())),
            "mean_pairwise_kendall_tau_between_bootstraps": float(np.mean(
                [stats.kendalltau(a, b)[0] for a, b in combinations(R[:60], 2)])),
        }
    out["kendall_tau_SHAP_vs_LR"] = {k: float(np.mean(v)) for k, v in taus.items()}
    return out, ranks


def main(path, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    X, y = df[FEATS].to_numpy(float), df["Adpt"].to_numpy(int)
    R = {"n": int(len(y)), "positives": int(y.sum())}

    cv = nested_cv(X, y)
    cv.to_csv(outdir / "nested_cv_folds.csv", index=False)
    summ = cv.groupby("model")[["auc", "bal_acc", "brier"]].agg(["mean", "std"]).round(3)
    summ.to_csv(outdir / "nested_cv_summary.csv")
    R["nested_cv"] = {m: {f"{a}_{b}": float(summ.loc[m, (a, b)]) for a in ("auc", "bal_acc", "brier")
                          for b in ("mean", "std")} for m in summ.index}
    lr = cv[cv.model == "LR"].sort_values("fold").auc.values
    R["ml_minus_lr_auc"] = {}
    for m in ("RF", "XGB"):
        d, t, p = corrected_ttest(cv[cv.model == m].sort_values("fold").auc.values, lr,
                                  n_train=0.8 * len(y), n_test=0.2 * len(y))
        R["ml_minus_lr_auc"][m] = {"mean_diff": round(d, 3), "t": round(t, 2), "p": round(p, 3)}

    # permutation test for LR (fixed C=1) AUC
    base = cross_val_score(fixed_models(0)["LR"], X, y, scoring="roc_auc",
                           cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)).mean()
    rng = np.random.default_rng(0)
    perm = [cross_val_score(fixed_models(0)["LR"], X, rng.permutation(y), scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=i)).mean() for i in range(500)]
    R["permutation_test_LR"] = {"cv_auc": round(float(base), 3),
                                "p": float((1 + np.sum(np.array(perm) >= base)) / (1 + len(perm))),
                                "null_auc_95pct": round(float(np.percentile(perm, 95)), 3)}

    # LR standardized coefficients (full sample)
    lrm = fixed_models(0)["LR"].fit(X, y)
    R["lr_std_coef"] = dict(zip([LABEL[f] for f in FEATS], lrm.named_steps["m"].coef_[0].round(3).tolist()))

    stab, ranks = bootstrap_stability(X, y)
    R["bootstrap_shap"] = stab
    np.save(outdir / "bootstrap_ranks.npy", {k: np.vstack(v) for k, v in ranks.items()}, allow_pickle=True)

    # sensitivity: drop the 18-respondent organisation (largest named firm cluster)
    counts = df.firm_id.value_counts()
    big = [f for f in counts.index if counts[f] == 18]
    sub = df[~df.firm_id.isin(big)]
    Xs, ys = sub[FEATS].to_numpy(float), sub["Adpt"].to_numpy(int)
    cvs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    R["sensitivity_without_largest_org"] = {
        "n": int(len(ys)),
        **{m: round(float(cross_val_score(fixed_models(0)[m], Xs, ys, cv=cvs, scoring="roc_auc").mean()), 3)
           for m in ("LR", "RF", "XGB")}}

    (outdir / "real_results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
