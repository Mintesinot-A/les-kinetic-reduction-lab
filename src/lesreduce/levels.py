"""Build the four mechanism 'levels' used throughout the lab."""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np

from .kinetics import (Mechanism, Reaction, toy_mechanism, sample_states,
                       drgep_importance, skeletal, fit_global_one_step)

# DRGEP thresholds that give the 6- and 5-species skeletal mechanisms
EPS_LEVELS = {"skeletal-6sp": 1e-2, "skeletal-5sp": 0.3}


def build_levels(cache_path=None, verbose=True):
    """Return (OrderedDict name->Mechanism, DRGEP importance dict, fit info).

    The global one-step fit takes ~30 s, so its (lnA, Ea) is cached in JSON.
    """
    det = toy_mechanism("detailed-8sp")
    C, T = sample_states(det)
    imp = drgep_importance(det, C, T)
    levels = OrderedDict([("detailed-8sp", det)])
    for name, eps in EPS_LEVELS.items():
        levels[name] = skeletal(det, imp, eps, name=name)

    info = None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            info = json.load(f)
    if info is None:
        if verbose:
            print("fitting global one-step mechanism (about 30 s) ...")
        _, info = fit_global_one_step(det)
        info = {k: float(v) for k, v in info.items()}
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(info, f, indent=2)
    levels["global-1step"] = Mechanism(
        ["F", "O", "P"],
        [Reaction({"F": 1, "O": 1}, {"P": 1}, float(np.exp(info["lnA"])), info["Ea"],
                  float(det.q.max()))],
        rho_cv=det.rho_cv, name="global-1step")
    return levels, imp, info
