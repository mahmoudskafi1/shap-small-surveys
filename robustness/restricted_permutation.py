"""Permutation tests that keep the respondents of each organisation together (Section 5.1).

Restricted (block) permutation: the outcomes of whole organisations are exchanged between
organisations of the same size, and the order of respondents within an organisation is kept.
Single respondents (including those with a blank employer field) are therefore shuffled
freely among themselves, the seven organisations with two respondents exchange their pairs of
outcomes, and organisations whose size is unique (4 and 18 respondents) keep their outcomes,
which makes the tests conservative.

  auc   : permutation test of the logistic regression AUC (C = 1) with stratified grouped
          5-fold cross-validation, 500 permutations (as in grouped_cv.py, step 2).
  boot  : noise baseline for the organisation-level bootstrap of the SHAP rankings,
          100 permutations x 200 refits (as in grouped_null.py). Resumable.

Requires the CONFIDENTIAL analysis file (survey2019_analysis.csv), which is not in this
repository and must never be committed. Writes aggregate statistics only.

Usage:  python restricted_permutation.py auc  DATA [OUT_JSON]
        python restricted_permutation.py boot DATA [OUT_JSONL] [K] [TIME_BUDGET_SECONDS]
        python restricted_permutation.py --summary [OUT_JSONL]
"""
import json, os, sys, time, warnings
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS


def load(data):
    sys.path.insert(0, HERE)
    from common import load_ra
    ra = load_ra(); df = pd.read_csv(data)
    X = df[ra.FEATS].to_numpy(float); y = df["Adpt"].to_numpy(int)
    g = df["firm_id"].to_numpy().copy(); blank = g == 0
    g[blank] = 100_000 + np.arange(blank.sum())          # blank employer field: one group per respondent
    return ra, X, y, g


def restricted_perm(y, g, rng):
    """Exchange the outcome vectors of whole organisations of equal size."""
    members = {o: np.where(g == o)[0] for o in np.unique(g)}
    by_size = {}
    for o, idx in members.items():
        by_size.setdefault(len(idx), []).append(o)
    yp = y.copy()
    for size, orgs in by_size.items():
        orgs = sorted(orgs); src = [orgs[i] for i in rng.permutation(len(orgs))]
        for o, s in zip(orgs, src):
            yp[members[o]] = y[members[s]]
    return yp


def lr_cv_auc(ra, X, y, g, seed):
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        m = ra.fixed_models(0)["LR"].fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def auc_chunk(data, ks):
    ra, X, y, g = load(data)
    return [lr_cv_auc(ra, X, restricted_perm(y, g, np.random.default_rng(20_000 + k)), g, k) for k in ks]


def boot_one(k, data, B=200):
    ra, X, y, g = load(data)
    TM = [ra.LABEL[f] for f in ra.FEATS].index("Top management support")
    yp = restricted_perm(y, g, np.random.default_rng(20_000 + k))
    orgs = np.unique(g); members = {o: np.where(g == o)[0] for o in orgs}
    R = {m: [] for m in ("LR", "RF", "XGB")}
    for b in range(B):
        rng = np.random.default_rng(1_000_000 + 1_000 * k + b)   # same organisation draws as grouped_null.py
        idx = np.concatenate([members[o] for o in rng.choice(orgs, len(orgs), replace=True)])
        if yp[idx].min() == yp[idx].max():
            continue
        for name, m in ra.fixed_models(b).items():
            m.fit(X[idx], yp[idx]); R[name].append(stats.rankdata(-ra.shap_mean_abs(name, m, X[idx])))
    out = {"k": k, "boot": {}}
    for name, r in R.items():
        r = np.vstack(r); first = (r == 1).mean(0); top3 = (r <= 3).mean(0)
        out["boot"][name] = dict(max_first=float(first.max()), argmax_first=int(first.argmax()), tm_first=float(first[TM]),
                                 max_top3=float(top3.max()), tm_top3=float(top3[TM]),
                                 tau=float(np.mean([stats.kendalltau(a, c)[0] for a, c in combinations(r, 2)])))
    return out


def summary(path):
    P = [json.loads(L) for L in open(path)]
    G = json.load(open(RESULTS / "real_grouped_cv.json"))["org_bootstrap"]
    TMn = "Top management support"
    print(f"K = {len(P)}")
    print(f"{'statistic':10s} {'model':5s} {'obs':>6s} {'mean':>6s} {'p95':>6s} {'max':>6s} {'p':>5s}")
    for stat in ("first", "top3", "tau"):
        for m in ("LR", "RF", "XGB"):
            obs = {"first": G[m]["share_first"][TMn], "top3": G[m]["share_top3"][TMn], "tau": G[m]["tau_all"]}[stat]
            null = np.array([p["boot"][m]["max_" + stat] if stat != "tau" else p["boot"][m]["tau"] for p in P])
            pv = (np.sum(null >= obs) + 1) / (len(null) + 1)
            print(f"{stat:10s} {m:5s} {obs:6.3f} {null.mean():6.3f} {np.percentile(null, 95):6.3f} {null.max():6.3f} {pv:5.2f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "restricted_null.jsonl")); sys.exit()
    DATA = os.path.abspath(sys.argv[2])
    if mode == "auc":
        OUT = sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "restricted_auc_test.json")
        ra, X, y, g = load(DATA)
        base = float(np.mean([lr_cv_auc(ra, X, y, g, s) for s in range(5)]))
        perm = np.array([v for c in Parallel(n_jobs=2)(delayed(auc_chunk)(DATA, list(range(i, i + 50))) for i in range(0, 500, 50)) for v in c])
        R = {"scheme": "outcomes exchanged between organisations of equal size", "permutations": 500,
             "cv_auc": round(base, 3), "null_auc_mean": round(float(perm.mean()), 3),
             "null_auc_95pct": round(float(np.percentile(perm, 95)), 3), "null_auc_max": round(float(perm.max()), 3),
             "p": float((1 + np.sum(perm >= base)) / 501)}
        json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R, indent=1))
    elif mode == "boot":
        OUT = sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "restricted_null.jsonl")
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else float("inf")
        done = {json.loads(L)["k"] for L in open(OUT)} if os.path.exists(OUT) else set()
        todo = [k for k in range(K) if k not in done]; t0 = time.time(); print("todo", len(todo), flush=True)
        for i in range(0, len(todo), 2):
            if time.time() - t0 > BUDGET:
                print("BUDGET STOP: rerun the same command to continue", flush=True); break
            res = Parallel(n_jobs=2)(delayed(boot_one)(k, DATA) for k in todo[i:i + 2])
            with open(OUT, "a") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")
            print("progress", len(done) + i + len(res), round(time.time() - t0), "s", flush=True)
        else:
            print("ALLDONE", flush=True)
