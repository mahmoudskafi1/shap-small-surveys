"""Organisation-aware noise baseline for the organisation-level bootstrap (Section 5.1).

For each of 100 permutations of the adoption outcome (the same permutations as in
permutation_lib.py), the 200-refit bootstrap that resamples organisations is repeated and the
same statistics as in Table 4 are recorded: the highest first-place and top-three shares of
any construct, the shares of top management support, and the mean pairwise Kendall tau.
Respondents with a blank employer field are treated as separate organisations.

Requires the CONFIDENTIAL analysis file (survey2019_analysis.csv), which is not in this
repository and must never be committed. Writes aggregate statistics only; resumable.

Usage:  python grouped_null.py PATH/TO/survey2019_analysis.csv [OUT_JSONL] [K] [TIME_BUDGET_SECONDS]
        python grouped_null.py --summary [OUT_JSONL]
"""
import json, os, sys, time, warnings
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS


def one(k, data, B=200):
    sys.path.insert(0, HERE)
    from common import load_ra
    ra = load_ra(); df = pd.read_csv(data)
    X = df[ra.FEATS].to_numpy(float); y = df["Adpt"].to_numpy(int)
    g = df["firm_id"].to_numpy().copy(); blank = g == 0; g[blank] = 100_000 + np.arange(blank.sum())
    TM = [ra.LABEL[f] for f in ra.FEATS].index("Top management support")
    yp = np.random.default_rng(10_000 + k).permutation(y)        # same permutation as permutation_lib.one(k)
    orgs = np.unique(g); members = {o: np.where(g == o)[0] for o in orgs}
    R = {m: [] for m in ("LR", "RF", "XGB")}
    for b in range(B):
        rng = np.random.default_rng(1_000_000 + 1_000 * k + b)
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
    for m in ("LR", "RF", "XGB"):
        print(m, "TopMgt first under null:", round(np.mean([p["boot"][m]["tm_first"] for p in P]), 3),
              "top three:", round(np.mean([p["boot"][m]["tm_top3"] for p in P]), 3))


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "grouped_null.jsonl")); sys.exit()
    DATA = os.path.abspath(sys.argv[1])
    OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "grouped_null.jsonl")
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else float("inf")
    done = {json.loads(L)["k"] for L in open(OUT)} if os.path.exists(OUT) else set()
    todo = [k for k in range(K) if k not in done]; t0 = time.time(); print("todo", len(todo), flush=True)
    for i in range(0, len(todo), 2):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); break
        res = Parallel(n_jobs=2)(delayed(one)(k, DATA) for k in todo[i:i + 2])
        with open(OUT, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print("progress", len(done) + i + len(res), round(time.time() - t0), "s", flush=True)
    else:
        print("ALLDONE", flush=True)
