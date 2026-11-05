"""
Calibrated simulation study (SYNTHETIC DATA, used only as a reported simulation).

Question: in technology-adoption surveys of the size typical in IS research,
(a) do random forest / XGBoost improve prediction over logistic regression, and
(b) do SHAP importance rankings recover the true drivers, and are they stable?

Data-generating process calibrated to the authors' 2019 survey (calibration_2019.json):
  * 9 latent constructs with the disattenuated inter-construct correlations observed in 2019
  * each construct measured by its real number of 5-point items, with the real
    first-component loadings and the real item response distributions
  * scale score = mean of items (what analysts actually model)
  * adoption ~ Bernoulli(logistic(b0 + g(latent))) with base rate 0.64 (2019)

Scenarios for g():
  linear     : real 2019 effect profile (one strong driver, several small, two ~0)
  nonlinear  : same drivers, but a threshold, a saturation and an interaction,
               rescaled so the best achievable AUC matches the linear scenario
  null       : no effects (adoption is pure noise)

Usage: python calibrated_sim.py OUTDIR [REPS]
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import kendalltau, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
CAL = json.loads((HERE / "calibration_2019.json").read_text())
NAMES = CAL["constructs"]
P = len(NAMES)
IDX = {k: i for i, k in enumerate(NAMES)}
CORR = np.array(CAL["latent_corr"])
CHOL = np.linalg.cholesky(CORR)
BETA = np.array([CAL["beta_latent_approx"][k] for k in NAMES])

# Per-item thresholds reproducing the observed 2019 response distributions.
THRESH = {}
for k in NAMES:
    th = []
    for probs in CAL["item_marginals"][k]:
        p = np.clip(np.cumsum(probs)[:-1], 1e-4, 1 - 1e-4)
        th.append(norm.ppf(p))
    THRESH[k] = th


# ----------------------------------------------------------------- scenarios
def g_linear(F):
    return F @ BETA


def _g_nonlinear_raw(F):
    tm, te, tr, cp = (F[:, IDX[c]] for c in ("TopMgt", "TechExp", "Trial", "CompPress"))
    return (1.6 * np.maximum(tm, 0.0)            # threshold: support matters only above average
            + 0.8 * np.tanh(1.5 * te)            # saturation
            + 0.6 * tr * te                      # interaction
            + 0.4 * (cp > 0.5)                   # step
            + F[:, IDX["Innov"]] * BETA[IDX["Innov"]]
            + F[:, IDX["Compat"]] * BETA[IDX["Compat"]])


# importance attribution of each term to constructs (interaction split equally)
NL_TERMS = [
    (["TopMgt"], lambda F: 1.6 * np.maximum(F[:, IDX["TopMgt"]], 0)),
    (["TechExp"], lambda F: 0.8 * np.tanh(1.5 * F[:, IDX["TechExp"]])),
    (["Trial", "TechExp"], lambda F: 0.6 * F[:, IDX["Trial"]] * F[:, IDX["TechExp"]]),
    (["CompPress"], lambda F: 0.4 * (F[:, IDX["CompPress"]] > 0.5)),
    (["Innov"], lambda F: F[:, IDX["Innov"]] * BETA[IDX["Innov"]]),
    (["Compat"], lambda F: F[:, IDX["Compat"]] * BETA[IDX["Compat"]]),
]


def latent(n, rng):
    return rng.standard_normal((n, P)) @ CHOL.T


def _scale_for_nonlinear():
    """Rescale the nonlinear signal so its oracle AUC equals the linear one."""
    F = latent(100_000, np.random.default_rng(1))
    target = _oracle_auc(g_linear(F), F)
    raw = _g_nonlinear_raw(F)
    lo, hi = 0.1, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _oracle_auc(mid * raw, F) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _intercept(lin, rate):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + lin)))) < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _oracle_auc(lin, F, rate=CAL["adoption_rate"]):
    b0 = _intercept(lin, rate)
    p = 1 / (1 + np.exp(-(b0 + lin)))
    y = np.random.default_rng(2).binomial(1, p)
    return roc_auc_score(y, p)


NL_SCALE = None


def g(scenario, F):
    if scenario == "linear":
        return g_linear(F)
    if scenario == "nonlinear":
        return NL_SCALE * _g_nonlinear_raw(F)
    return np.zeros(len(F))


def true_importance(scenario):
    """Mean |centred contribution| of each construct to the true logit (large sample)."""
    F = latent(200_000, np.random.default_rng(3))
    imp = np.zeros(P)
    if scenario == "linear":
        imp = np.abs(BETA) * np.mean(np.abs(F - F.mean(0)), axis=0)
    elif scenario == "nonlinear":
        contrib = np.zeros_like(F)
        for cols, fn in NL_TERMS:
            t = NL_SCALE * fn(F)
            t = t - t.mean()
            for c in cols:
                contrib[:, IDX[c]] += t / len(cols)
        imp = np.mean(np.abs(contrib), axis=0)
    return imp


# ----------------------------------------------------------------- data
def make_data(n, scenario, seed):
    rng = np.random.default_rng(seed)
    F = latent(n, rng)
    lin = g(scenario, F)
    ref = g(scenario, latent(100_000, np.random.default_rng(4)))
    b0 = _intercept(ref, CAL["adoption_rate"])
    y = rng.binomial(1, 1 / (1 + np.exp(-(b0 + lin))))
    X = np.empty((n, P))
    for k in NAMES:
        j = IDX[k]
        lam = np.array(CAL["loadings"][k])
        raw = F[:, [j]] * lam + rng.standard_normal((n, len(lam))) * np.sqrt(1 - lam ** 2)
        items = np.column_stack([np.searchsorted(THRESH[k][i], raw[:, i]) + 1 for i in range(len(lam))])
        X[:, j] = items.mean(1)
    return X, y


# ----------------------------------------------------------------- models
def models(seed):
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RF": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=5.0, eval_metric="logloss",
                             random_state=seed, n_jobs=1, verbosity=0),
    }


def shap_imp(name, m, Xtr, Xte):
    if name == "LR":
        sc, lr = m.steps[0][1], m.steps[1][1]
        sv = shap.LinearExplainer(lr, sc.transform(Xtr)).shap_values(sc.transform(Xte))
    else:
        sv = shap.TreeExplainer(m).shap_values(Xte)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    return np.abs(sv).mean(0)


SCEN_ID = {"linear": 0, "nonlinear": 1, "null": 2}


def one_run(n, scenario, rep, truth, nl_scale):
    global NL_SCALE
    NL_SCALE = nl_scale                      # workers are fresh processes
    seed = SCEN_ID[scenario] * 10_000_000 + n * 1000 + rep
    X, y = make_data(n, scenario, seed)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {"n": n, "scenario": scenario, "rep": rep}
    fold_imps = {m: [] for m in ("LR", "RF", "XGB")}
    preds = {m: np.zeros(n) for m in ("LR", "RF", "XGB")}
    for tr, te in cv.split(X, y):
        for name, m in models(seed).items():
            m.fit(X[tr], y[tr])
            preds[name][te] = m.predict_proba(X[te])[:, 1]
            fold_imps[name].append(shap_imp(name, m, X[tr], X[te]))
    order_true = truth
    for name in ("LR", "RF", "XGB"):
        out[f"auc_{name}"] = roc_auc_score(y, preds[name])
        M = np.vstack(fold_imps[name])
        out[f"stab_{name}"] = float(np.nanmean([kendalltau(a, b)[0] for a, b in combinations(M, 2)]))
        mean_imp = M.mean(0)
        top = np.argsort(-mean_imp)
        out[f"dominance_{name}"] = float(mean_imp[top[0]] / mean_imp.sum())
        if order_true is not None:
            out[f"tau_truth_{name}"] = float(kendalltau(mean_imp, order_true)[0])
            out[f"top1_{name}"] = int(top[0] == np.argmax(order_true))
            true_top3 = set(np.argsort(-order_true)[:3])
            out[f"top3_{name}"] = len(set(top[:3]) & true_top3) / 3
    return out


def main(outdir: Path, reps: int):
    global NL_SCALE
    outdir.mkdir(parents=True, exist_ok=True)
    NL_SCALE = _scale_for_nonlinear()
    F = latent(100_000, np.random.default_rng(1))
    meta = {"nonlinear_scale": NL_SCALE,
            "oracle_auc": {s: _oracle_auc(g(s, F), F) if s != "null" else 0.5 for s in ("linear", "nonlinear", "null")},
            "true_importance": {s: dict(zip(NAMES, np.round(true_importance(s), 4).tolist()))
                                for s in ("linear", "nonlinear")}}
    (outdir / "sim_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    truths = {s: (true_importance(s) if s != "null" else None) for s in ("linear", "nonlinear", "null")}
    sizes = [100, 139, 200, 300, 500, 1000]
    jobs = [(n, s, r) for s in ("linear", "nonlinear", "null") for n in sizes for r in range(reps)]
    rows = []
    batch = 60
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        res = Parallel(n_jobs=2)(delayed(one_run)(n, s, r, truths[s], NL_SCALE) for n, s, r in chunk)
        rows += [r for r in res if r is not None]
        pd.DataFrame(rows).to_csv(outdir / "sim_raw.csv", index=False)
        print(f"progress {min(i + batch, len(jobs))}/{len(jobs)}", flush=True)
    raw = pd.DataFrame(rows)
    agg = raw.drop(columns="rep").groupby(["scenario", "n"]).mean().round(3)
    agg.to_csv(outdir / "sim_summary.csv")
    print(agg.to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 50)
