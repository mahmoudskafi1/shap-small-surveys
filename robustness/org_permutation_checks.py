"""Two checks of the restricted (block) permutation tests (Section 4.1, Section 5.1, Limitations).

In restricted_permutation.py, the organisations with 4 and 18 respondents have no partner of
the same size, so their outcomes stay fixed in every permutation. This script checks what
that choice does.

  sub   DATA [OUT_JSON]
        Sensitivity without the two organisations (n = 117). Every remaining organisation
        can exchange its outcomes, so no outcome is fixed. Runs the permutation test of the
        logistic regression AUC (500 restricted permutations) and the observed
        organisation-level bootstrap (200 refits, the same draws as grouped_cv.py).
  subnull DATA [OUT_JSONL] [K] [TIME_BUDGET_SECONDS]
        Noise baseline for that bootstrap (K = 100 permutations x 200 refits), as in
        restricted_permutation.py. Resumable.
  calib DATA [OUT_JSONL] [N_SIM] [TIME_BUDGET_SECONDS]
        Type I error of the AUC permutation test. The real constructs and the real
        organisation structure are kept, and the outcome is simulated with no link to the
        constructs but with a random organisation effect, so that respondents of the same
        organisation tend to give the same answer (intra-class correlation 0, 0.2 and 0.5 on
        the latent scale; adoption rate 64%). Each simulated dataset is tested with the
        restricted permutation and, for comparison, with a shuffle across respondents
        (99 permutations each). A valid test rejects at most about 5% of datasets at the
        0.05 level. Resumable.
  --summary [DIR]

Requires the CONFIDENTIAL analysis file (survey2019_analysis.csv), which is not in this
repository and must never be committed. Writes aggregate statistics only.
"""
import json, os, sys, time, warnings
from itertools import combinations
import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.optimize import brentq
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS
import restricted_permutation as rp

ICC = [0.0, 0.2, 0.5]
N_PERM_CALIB = 99


def load_sub(data):
    """Drop the organisations whose size is unique (4 and 18 respondents)."""
    ra, X, y, g = rp.load(data)
    o, c = np.unique(g, return_counts=True)
    size = dict(zip(o, c)); counts = {s: list(c).count(s) for s in set(c)}
    keep = np.array([counts[size[x]] > 1 for x in g])
    return ra, X[keep], y[keep], g[keep]


def boot_ranks(ra, X, y, g, seeds):
    orgs = np.unique(g); members = {o: np.where(g == o)[0] for o in orgs}
    R = {m: [] for m in ("LR", "RF", "XGB")}
    for b in seeds:
        rng = np.random.default_rng(b)
        idx = np.concatenate([members[o] for o in rng.choice(orgs, len(orgs), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        for name, m in ra.fixed_models(b).items():
            m.fit(X[idx], y[idx]); R[name].append(stats.rankdata(-ra.shap_mean_abs(name, m, X[idx])))
    return R


def boot_stats(ra, R):
    TM = [ra.LABEL[f] for f in ra.FEATS].index("Top management support"); out = {}
    for name, r in R.items():
        r = np.vstack(r); first = (r == 1).mean(0); top3 = (r <= 3).mean(0)
        out[name] = dict(max_first=float(first.max()), argmax_first=int(first.argmax()), tm_first=float(first[TM]),
                         max_top3=float(top3.max()), tm_top3=float(top3[TM]),
                         tau=float(np.mean([stats.kendalltau(a, c)[0] for a, c in combinations(r, 2)])))
    return out


def sub_obs_chunk(data, seeds):
    ra, X, y, g = load_sub(data); return boot_ranks(ra, X, y, g, seeds)


def sub_auc_chunk(data, ks):
    ra, X, y, g = load_sub(data)
    return [rp.lr_cv_auc(ra, X, rp.restricted_perm(y, g, np.random.default_rng(20_000 + k)), g, k) for k in ks]


def sub_null_one(k, data, B=200):
    ra, X, y, g = load_sub(data)
    yp = rp.restricted_perm(y, g, np.random.default_rng(20_000 + k))
    return {"k": k, "boot": boot_stats(ra, boot_ranks(ra, X, yp, g, [1_000_000 + 1_000 * k + b for b in range(B)]))}


def intercept(sigma, rate=0.64):
    z = np.random.default_rng(0).standard_normal(200_000) * sigma
    return brentq(lambda a: expit(a + z).mean() - rate, -10, 10)


def calib_one(level, i, data):
    ra, X, _, g = rp.load(data)
    icc = ICC[level]; sigma = np.sqrt(icc / (1 - icc) * np.pi ** 2 / 3) if icc > 0 else 0.0
    rng = np.random.default_rng(30_000_000 + 100_000 * level + i)
    orgs = np.unique(g); u = dict(zip(orgs, rng.standard_normal(len(orgs)) * sigma))
    y = (rng.random(len(g)) < expit(intercept(sigma) + np.array([u[o] for o in g]))).astype(int)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return None
    obs = float(np.mean([rp.lr_cv_auc(ra, X, y, g, s) for s in range(5)]))
    pr = np.random.default_rng(40_000_000 + 100_000 * level + i)
    restr = [rp.lr_cv_auc(ra, X, rp.restricted_perm(y, g, pr), g, k) for k in range(N_PERM_CALIB)]
    free = [rp.lr_cv_auc(ra, X, pr.permutation(y), g, k) for k in range(N_PERM_CALIB)]
    p = lambda null: float((1 + np.sum(np.array(null) >= obs)) / (len(null) + 1))
    return {"icc": icc, "i": i, "rate": float(y.mean()), "obs_auc": obs, "p_restricted": p(restr), "p_free": p(free)}


def summary(d):
    S = json.load(open(os.path.join(d, "org_sub117.json")))
    print("n = 117:", json.dumps(S["auc_test"]))
    G = S["org_bootstrap"]
    pth = os.path.join(d, "org_sub117_null.jsonl")
    if os.path.exists(pth):
        P = [json.loads(L) for L in open(pth)]; print(f"bootstrap baseline, K = {len(P)}")
        for stat in ("first", "top3", "tau"):
            for m in ("LR", "RF", "XGB"):
                obs = {"first": G[m]["tm_first"], "top3": G[m]["tm_top3"], "tau": G[m]["tau"]}[stat]
                null = np.array([p["boot"][m]["max_" + stat] if stat != "tau" else p["boot"][m]["tau"] for p in P])
                print(f"{stat:6s} {m:4s} obs {obs:.3f} null mean {null.mean():.3f} p95 {np.percentile(null, 95):.3f} "
                      f"p {(np.sum(null >= obs) + 1) / (len(null) + 1):.2f}")
    pth = os.path.join(d, "org_calibration.jsonl")
    if os.path.exists(pth):
        C = [json.loads(L) for L in open(pth)]
        for icc in ICC:
            c = [x for x in C if x["icc"] == icc]
            if not c:
                continue
            r = np.array([x["p_restricted"] <= 0.05 for x in c]); f = np.array([x["p_free"] <= 0.05 for x in c])
            se = lambda v: np.sqrt(v.mean() * (1 - v.mean()) / len(v))
            print(f"ICC {icc:.1f}: datasets {len(c)}, rejection at 0.05: restricted {r.mean():.3f} (SE {se(r):.3f}), "
                  f"across respondents {f.mean():.3f} (SE {se(f):.3f}); mean adoption rate {np.mean([x['rate'] for x in c]):.2f}")


def resumable(OUT, todo_all, key, fn, BUDGET, step=2):
    done = set()
    if os.path.exists(OUT):
        done = {key(json.loads(L)) for L in open(OUT)}
    todo = [t for t in todo_all if t not in done]; t0 = time.time(); print("todo", len(todo), flush=True)
    for i in range(0, len(todo), step):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); return
        res = Parallel(n_jobs=2)(delayed(fn)(*t) for t in todo[i:i + step])
        with open(OUT, "a") as f:
            for t, r in zip(todo[i:i + step], res):
                f.write(json.dumps(r if r is not None else {"skipped": list(t)}) + "\n")
        print("progress", len(done) + i + len(res), round(time.time() - t0), "s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else str(RESULTS)); sys.exit()
    DATA = os.path.abspath(sys.argv[2])
    if mode == "sub":
        OUT = sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "org_sub117.json")
        ra, X, y, g = load_sub(DATA)
        base = float(np.mean([rp.lr_cv_auc(ra, X, y, g, s) for s in range(5)]))
        perm = np.array([v for c in Parallel(n_jobs=2)(delayed(sub_auc_chunk)(DATA, list(range(i, i + 50))) for i in range(0, 500, 50)) for v in c])
        Rs = Parallel(n_jobs=2)(delayed(sub_obs_chunk)(DATA, list(range(i, i + 20))) for i in range(0, 200, 20))
        R = {m: [r for x in Rs for r in x[m]] for m in ("LR", "RF", "XGB")}
        o, c = np.unique(g, return_counts=True)
        out = {"n": int(len(y)), "organisations": int(len(o)), "sizes": {int(s): int(k) for s, k in zip(*np.unique(c, return_counts=True))},
               "adoption_rate": round(float(y.mean()), 3),
               "auc_test": {"permutations": 500, "cv_auc": round(base, 3), "null_auc_mean": round(float(perm.mean()), 3),
                            "null_auc_95pct": round(float(np.percentile(perm, 95)), 3), "p": float((1 + np.sum(perm >= base)) / 501)},
               "org_bootstrap": dict(refits=len(R["LR"]), **boot_stats(ra, R))}
        json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out, indent=1))
    elif mode == "subnull":
        OUT = sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "org_sub117_null.jsonl")
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else float("inf")
        resumable(OUT, [(k,) for k in range(K)], lambda r: (r["k"],), lambda k: sub_null_one(k, DATA), BUDGET)
    elif mode == "calib":
        OUT = sys.argv[3] if len(sys.argv) > 3 else str(RESULTS / "org_calibration.jsonl")
        N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else float("inf")
        todo = [(l, i) for i in range(N) for l in range(len(ICC))]
        key = lambda r: tuple(r["skipped"]) if "skipped" in r else (ICC.index(r["icc"]), r["i"])
        resumable(OUT, todo, key, lambda l, i: calib_one(l, i, DATA), BUDGET, step=6)
