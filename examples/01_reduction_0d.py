"""Step 1 - mechanism reduction in 0D.

* DRGEP species ranking of the toy mechanism
* ignition delay along the LES mixing line for every reduction level
* error vs number of species

Outputs: results/reduction_summary.csv, figures/0d_reduction.png
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lesreduce import tau_mixing_line, mixing_line, sample_states, drgep_importance, skeletal
from lesreduce.levels import build_levels

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

levels, imp, fit = build_levels(os.path.join(ROOT, "results", "global_fit.json"))
det = levels["detailed-8sp"]

print("DRGEP importance:")
for s, v in sorted(imp.items(), key=lambda kv: -kv[1]):
    print(f"  {s:3s} {v:8.4f}")

Zs = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
tau = {name: tau_mixing_line(m, Zs) for name, m in levels.items()}
ref = tau["detailed-8sp"]

rows = []
for name, m in levels.items():
    sel = (Zs >= 0.1) & (Zs <= 0.5)                    # range that matters in the jet
    err = np.abs(tau[name][sel] - ref[sel]) / ref[sel]
    rows.append(dict(mechanism=name, species=m.ns, reactions=m.nr,
                     max_tau_error_pct=100 * np.nanmax(err),
                     mean_tau_error_pct=100 * np.nanmean(err)))
with open(os.path.join(ROOT, "results", "reduction_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\nmechanism        ns  nr  max err   mean err   (tau error, 0.1<=Z<=0.5)")
for r in rows:
    print(f"{r['mechanism']:15s} {r['species']:3d} {r['reactions']:3d} "
          f"{r['max_tau_error_pct']:7.1f}%  {r['mean_tau_error_pct']:7.1f}%")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
names = sorted(imp, key=lambda s: -imp[s])
ax[0].bar(names, [imp[s] for s in names], color="#3b6ea5")
ax[0].set_yscale("log")
ax[0].set_ylabel("DRGEP importance")
ax[0].set_title("Species ranking")
for name, m in levels.items():
    ax[1].semilogy(Zs, tau[name] * 1e3, "o-" if "detailed" in name else ".--",
                   label=f"{name} ({m.ns} sp)")
ax[1].set_xlabel("mixture fraction Z")
ax[1].set_ylabel("ignition delay [ms]")
ax[1].set_title("0D ignition delay on the mixing line")
ax[1].legend(fontsize=8)
for r in rows:
    err = max(r["max_tau_error_pct"], 0.05)          # floor so 0 % shows on a log axis
    ax[2].plot(r["species"], err, "o", ms=8)
    ax[2].annotate(f"{r['mechanism']}\n{r['max_tau_error_pct']:.1f} %", (r["species"], err),
                   textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[2].set_yscale("log")
ax[2].set_xlabel("number of species")
ax[2].set_ylabel("max ignition-delay error [%]  (0.1 <= Z <= 0.5)")
ax[2].set_title("Accuracy vs size")
ax[2].set_xlim(2, 9.5)
ax[2].set_ylim(0.03, 800)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "0d_reduction.png"), dpi=140)
print("\nwrote figures/0d_reduction.png")
