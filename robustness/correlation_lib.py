"""Correlation-level variant of the calibrated simulation (Section 4.2, Table 6).

Levels: low = independent constructs; moderate = calibrated 2019 matrix (the main
simulation, unchanged); high = 0.5 * calibrated + 0.5 * equicorrelation(0.6).
At each level the linear and nonlinear signals are rescaled so that the oracle AUC equals
the calibrated linear oracle AUC (0.778); seeds are identical across levels.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_sim


def corr_matrix(level, R):
    P = len(R)
    if level == "low":
        return np.eye(P)
    if level == "moderate":
        return R.copy()
    if level == "high":
        E = np.full((P, P), 0.6); np.fill_diagonal(E, 1.0)
        return 0.5 * R + 0.5 * E
    raise ValueError(level)


def load(level):
    sim = load_sim()
    F0 = sim.latent(100_000, np.random.default_rng(1))            # calibrated matrix
    target = sim._oracle_auc(sim.g_linear(F0), F0)
    sim.CORR = corr_matrix(level, np.array(sim.CAL["latent_corr"]))
    sim.CHOL = np.linalg.cholesky(sim.CORR)
    F = sim.latent(100_000, np.random.default_rng(1))
    base = F @ sim.BETA
    lo, hi = 0.2, 5.0
    for _ in range(40):                                            # bisection on the linear scale
        mid = (lo + hi) / 2
        if sim._oracle_auc(mid * base, F) < target:
            lo = mid
        else:
            hi = mid
    ls = (lo + hi) / 2 if level != "moderate" else 1.0
    beta = sim.BETA.copy()
    sim.g_linear = lambda F, _b=beta, _s=ls: _s * (F @ _b)
    sim.NL_SCALE = sim._scale_for_nonlinear()
    sim.LSCALE = ls; sim.TARGET = target
    return sim
