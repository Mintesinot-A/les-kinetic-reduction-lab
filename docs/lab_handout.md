# Lab handout: how much chemistry does an LES need?

**Duration:** about 2 × 50 min. **Prerequisites:** basic Python; a first course in CFD and in combustion. **Setup:** `pip install -r requirements.txt` (NumPy, SciPy, Matplotlib). No other software.

## Learning goals

After this lab you can:

1. Explain what DRGEP does and why a skeletal mechanism is smaller but not "wrong".
2. Compare reduced mechanisms in 0D *and* in a flow, and explain why 0D agreement is necessary but not sufficient.
3. Separate three error sources in a reacting LES: kinetics reduction, grid resolution, and the sub-grid model.
4. Quote a cost/accuracy trade-off with numbers.

## Background in five lines

* The jet core is cold fuel (700 K), the co-flow is hot oxidiser (1150 K). Mixing sets the temperature along a "mixing line" T(Z); the mixture auto-ignites after an *ignition delay* τ(Z).
* The mechanism (8 species, 12 reactions) is a **toy** chain-branching system, not a real fuel.
* DRGEP ranks species by how much the targets (F, O, P) depend on them; species below a threshold ε are removed along with their reactions.
* The LES is 2D, incompressible, with a Smagorinsky model. Chemistry does not feed back on the flow, so runs differ *only* by mechanism and grid.
* Time is non-dimensional; 1 time unit = 1 ms in the chemistry.

## Part A: reduction in 0D (about 40 min)

```bash
python examples/01_reduction_0d.py
```

**A1.** Open `figures/0d_reduction.png`. Which species have DRGEP importance below 0.2? What role do they play in the mechanism (see `toy_mechanism` in `src/lesreduce/kinetics.py`)?

**A2.** Fill in the table from `results/reduction_summary.csv`. At which Z is the 5-species mechanism's error largest, and why might that be?

**A3.** The ignition delay has a minimum at intermediate Z. Is that where you would expect ignition in a mixing layer? Give two reasons it might not be.

**A4.** Change the threshold and see what happens:

```python
from lesreduce import toy_mechanism, sample_states, drgep_importance, skeletal, tau_mixing_line
import numpy as np
det = toy_mechanism()
C, T = sample_states(det)
imp = drgep_importance(det, C, T)
Z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
for eps in (1e-3, 1e-2, 0.2, 0.99):
    sk = skeletal(det, imp, eps)
    err = np.abs(tau_mixing_line(sk, Z) / tau_mixing_line(det, Z) - 1)
    print(eps, sk.species, f"{100*err.max():.1f} %")
```

Which threshold would you choose and why?

**A5 (short).** The one-step mechanism was fitted to match τ. Look at the red curve in the middle panel. Why can a single Arrhenius reaction not reproduce the detailed curve?

## Part B: LES with reduced mechanisms (about 60 min)

```bash
python examples/02_les_jet.py            # N=64, four mechanisms, ~2 min
```

Look at `figures/les_snapshots.png`, `figures/les_history.png` and `results/les_summary.csv`.

**B1.** Where (which Z) and when does the first ignition occur for each mechanism? Compare with the 0D optimum from Part A.

**B2.** Rank the mechanisms by ignition-time error and by cost. Plot error against chemistry cost per step. Where is the "knee"?

**B3. Grid vs chemistry.** Run a coarser and, if time allows, a finer grid:

```bash
python examples/02_les_jet.py --N 48 --mechs detailed-8sp
python examples/02_les_jet.py --N 96 --mechs detailed-8sp     # ~2-3 min
python examples/02_les_jet.py --summary-only
```

Compare the ignition-time change from N=48 → 64 → 96 with the change from detailed → 5-species. Which matters more? What would you refine first in a real project?

**B4. Sub-grid model.** Switch the Smagorinsky model off and compare:

```python
from lesreduce import toy_mechanism
from lesreduce.les2d import Jet2D, ignition_time
for Cs in (0.0, 0.17):
    sim = Jet2D(toy_mechanism(), N=48, Cs=Cs)
    hist, snaps, info = sim.run(3.0)
    print(Cs, ignition_time(hist))
```

How does the effect compare with the mechanism choice? (Remember the caveat that Smagorinsky in 2D is a regulariser.)

**B5. Failure analysis.** The one-step mechanism matches τ in 0D but fails in the LES. Write three sentences explaining why, using the ignition-Z and burnt-area results.

## Part C: stretch exercises (pick one)

1. **Add your own mechanism level.** Use QSSA to remove HO (algebraic instead of transported) and compare with the DRGEP 5-species result.
2. **Turbulence–chemistry interaction.** Multiply the chemistry source by a PaSR factor κ = τ_c / (τ_c + τ_mix) with τ_mix from the sub-grid viscosity. How does it change ignition time and the grid sensitivity?
3. **Different conditions.** Change `T_OX`, `C_FUEL0` in `kinetics.py`; re-run Part A. Does the same skeletal mechanism still work? (This is the "range of validity" question.)
4. **Real chemistry.** Install Cantera, load a published NH₃/H₂ mechanism and re-implement the DRGEP step on it (the algorithm in `drgep_importance` only needs rates of progress and a stoichiometry matrix).

## What to hand in

A short report (2 pages): the cost/accuracy plot from B2, one paragraph on grid vs chemistry vs sub-grid error (B3 and B4), one paragraph on the failure of the one-step mechanism (B5), and a clear statement of what the toy setup *cannot* tell you.

---

## Notes for instructors

* **Expected results (N=64):** detailed and 6-species ignite at t ≈ 1.74 (Z ≈ 0.41); 5-species at 1.68 (−3 %); one-step at 3.40 (Z ≈ 0.26). Against an N=128 reference (t ≈ 1.59), N=64 is ≈ 10 % late and N=48 ≈ 15 % late.
* **Run times** on a laptop: Part A ~1 min, Part B ~2 min for the four N=64 runs, ~6 min for N=128.
* **Results are for one flow realisation** (fixed perturbation seed). Let students vary `seed` in `Jet2D(...)` to see the spread; that is a good discussion point.
* **Honest caveats to raise in class:** toy mechanism, 2D, one-way coupling, no TCI model, NumPy timings are not representative of compiled solvers.
