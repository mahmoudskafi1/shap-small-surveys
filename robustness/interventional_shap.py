"""Interventional SHAP for all three models (revision of the original analysis).

The original scripts used interventional SHAP for logistic regression and the default
path-dependent TreeSHAP for random forest and XGBoost. This module replaces the SHAP
functions of real_data/real_analysis.py and simulation/calibrated_sim.py so that all
three models are explained with interventional SHAP, which is the definition used for
the true importance in the simulation.

Tree models: TreeExplainer(feature_perturbation="interventional") with a background of
up to 100 cases drawn at random from the training data of each fit.
Logistic regression: LinearExplainer with the training data as background.
"""
import numpy as np
import shap


def tree_interv(m, Xbg, Xev, nbg=100, seed=0):
    bg = Xbg if len(Xbg) <= nbg else Xbg[np.random.default_rng(seed).choice(len(Xbg), nbg, replace=False)]
    sv = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional").shap_values(Xev, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(0)


def apply(ra, sim):
    """Patch the real-data module `ra` and/or the simulation module `sim` (either may be None)."""
    def shap_mean_abs(name, m, X):
        if name == "LR":
            sc, lr = m.named_steps["sc"], m.named_steps["m"]
            return np.abs(shap.LinearExplainer(lr, sc.transform(X)).shap_values(sc.transform(X))).mean(0)
        return tree_interv(m, X, X)

    def shap_imp(name, m, Xtr, Xte):
        if name == "LR":
            sc, lr = m.steps[0][1], m.steps[1][1]
            return np.abs(shap.LinearExplainer(lr, sc.transform(Xtr)).shap_values(sc.transform(Xte))).mean(0)
        return tree_interv(m, Xtr, Xte)

    if ra is not None:
        ra.shap_mean_abs = shap_mean_abs
    if sim is not None:
        sim.shap_imp = shap_imp
