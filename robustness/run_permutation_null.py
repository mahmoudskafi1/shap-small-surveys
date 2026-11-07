"""Permutation null for the real-data stability statistics (Table 4): 100 permuted outcomes,
each with 20 x 5-fold fold stability and 200 bootstrap refits. Resumable; one JSON line per
permutation (aggregate statistics only).

Usage:  python run_permutation_null.py PATH/TO/survey2019_analysis.csv [OUT_JSONL] [K] [TIME_BUDGET_SECONDS]
"""
import json, os, sys, time
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PL_DATA"] = os.path.abspath(sys.argv[1])
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, HERE)
from common import RESULTS
OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "perm_null_interventional.jsonl")
K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else float("inf")


def run(k):
    import permutation_lib
    return permutation_lib.one(k, 20, 200)


if __name__ == "__main__":
    t0 = time.time(); done = set()
    if os.path.exists(OUT):
        for L in open(OUT):
            done.add(json.loads(L)["k"])
    todo = [k for k in range(K) if k not in done]
    for i in range(0, len(todo), 2):
        if time.time() - t0 > BUDGET:
            print("BUDGET STOP: rerun the same command to continue", flush=True); break
        res = Parallel(n_jobs=2)(delayed(run)(k) for k in todo[i:i + 2])
        with open(OUT, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print("progress", len(done) + i + len(res), round(time.time() - t0), "s", flush=True)
    else:
        print("ALLDONE", flush=True)
