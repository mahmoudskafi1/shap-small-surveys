"""Paths and module loaders shared by the robustness scripts."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM_PATH = ROOT / "simulation" / "calibrated_sim.py"
RA_PATH = ROOT / "real_data" / "real_analysis.py"
RESULTS = HERE / "results"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sim(interventional=True):
    import interventional_shap
    sim = load_module(SIM_PATH, "sim")
    if interventional:
        interventional_shap.apply(None, sim)
    return sim


def load_ra(interventional=True):
    import interventional_shap
    ra = load_module(RA_PATH, "ra")
    if interventional:
        interventional_shap.apply(ra, None)
    return ra
