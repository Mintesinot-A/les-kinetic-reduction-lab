"""Step 2 - 2D LES of a temporally evolving jet with detailed vs reduced kinetics.

Usage
-----
    python examples/02_les_jet.py                       # N=64, all four mechanisms (~4 min)
    python examples/02_les_jet.py --N 128 --mechs detailed-8sp   # fine-grid reference (~10 min)
    python examples/02_les_jet.py --summary-only        # just rebuild tables / figures from cache

Each case is cached in results/les_N<N>_<mechanism>.npz, so runs can be
launched separately and the summary assembled afterwards.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from lesreduce.levels import build_levels
from lesreduce.les2d import Jet2D, ignition_time

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SNAP_TIMES = (1.5, 2.5, 4.0)
T_COMPARE = 3.0          # time at which burnt-area fraction is compared


def cache_file(N, name):
    return os.path.join(RES, f"les_N{N}_{name}.npz")


def run_case(name, mech, N, t_end, force=False):
    f = cache_file(N, name)
    if os.path.exists(f) and not force:
        print(f"[cache] {f}")
        return
    print(f"[run]   N={N} mechanism={name} ({mech.ns} sp, {mech.nr} rx)")
    sim = Jet2D(mech, N=N)
    hist, snaps, info = sim.run(t_end, diag_every=5 if N <= 64 else 10,
                                snap_times=SNAP_TIMES, verbose=True)
    info.update(N=N, mechanism=name, ns=mech.ns, nr=mech.nr)
    out = {f"h_{k}": v for k, v in hist.items()}
    for t, s in snaps.items():
        out[f"snap_T_{t}"] = s["T"]
        out[f"snap_Z_{t}"] = s["Z"]
    np.savez(f, info=json.dumps(info), **out)
    print(f"        wall {info['wall']:.0f} s, steps {info['nstep']}")


def load(N, name):
    f = cache_file(N, name)
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=False)
    info = json.loads(str(d["info"]))
    hist = {k[2:]: d[k] for k in d.files if k.startswith("h_")}
    snaps = {float(k.split("_")[-1]): d[k] for k in d.files if k.startswith("snap_T_")}
    return dict(info=info, hist=hist, snaps=snaps)


def metrics(case):
    h, info = case["hist"], case["info"]
    t_ign, z_ign = ignition_time(h)
    burnt = float(np.interp(T_COMPARE, h["t"], h["burnt"]))
    return dict(mechanism=info["mechanism"], N=info["N"], species=info["ns"],
                reactions=info["nr"], t_ign=t_ign, Z_at_ignition=z_ign,
                burnt_at_t3=burnt, Hmean_end=float(h["Hmean"][-1]),
                ms_chem_per_step=1e3 * info["t_chem"] / info["nstep"],
                ms_total_per_step=1e3 * info["wall"] / info["nstep"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--mechs", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    levels, _, _ = build_levels(os.path.join(RES, "global_fit.json"))
    if not a.summary_only:
        for name in (a.mechs or list(levels)):
            run_case(name, levels[name], a.N, a.t_end, a.force)

    # ------------------------------------------------------------ summary
    cases = [("ref N=128", 128, "detailed-8sp")] + [(n, 64, n) for n in levels]
    cases = [(lab, load(N, n)) for lab, N, n in cases]
    cases = [(lab, c) for lab, c in cases if c is not None]
    if not cases:
        return
    # extra grids (e.g. N=48, 96) with the detailed mechanism: grid-sensitivity rows (table only)
    extra = []
    for fpath in sorted(glob.glob(os.path.join(RES, "les_N*_detailed-8sp.npz"))):
        n = int(re.search(r"les_N(\d+)_", fpath).group(1))
        if n not in (64, 128):
            extra.append((f"detailed N={n}", load(n, "detailed-8sp")))
    rows = [dict(label=lab, **metrics(c)) for lab, c in cases + extra]
    base = next((r for r in rows if r["label"] == "detailed-8sp"), None)
    ref = next((r for r in rows if r["label"] == "ref N=128"), None)
    for r in rows:
        if base:
            r["t_ign_err_vs_N64_detailed_pct"] = 100 * (r["t_ign"] - base["t_ign"]) / base["t_ign"]
            if r["N"] == base["N"]:                      # cost ratios only make sense on the same grid
                r["chem_speedup_vs_detailed"] = base["ms_chem_per_step"] / r["ms_chem_per_step"]
        if ref:
            r["t_ign_err_vs_N128_ref_pct"] = 100 * (r["t_ign"] - ref["t_ign"]) / ref["t_ign"]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for r in rows:
        for k in keys:
            r.setdefault(k, "")
    with open(os.path.join(RES, "les_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n" + "label".ljust(14) + "N    sp  t_ign  Z_ign  burnt(t=3)  chem ms/step  speedup")
    for r in rows:
        sp = r.get("chem_speedup_vs_detailed", "")
        sp = "" if sp == "" else sp
        print(f"{r['label']:13s} {r['N']:4d} {r['species']:3d} {r['t_ign']:6.2f} "
              f"{r['Z_at_ignition']:6.2f} {r['burnt_at_t3']:9.3f} {r['ms_chem_per_step']:11.1f} "
              f"{sp if sp == '' else format(sp, '8.2f')}")

    # ------------------------------------------------------------ figures
    ncol = len(cases)
    fig, axs = plt.subplots(len(SNAP_TIMES), ncol, figsize=(2.6 * ncol, 2.5 * len(SNAP_TIMES)),
                            squeeze=False)
    for j, (lab, c) in enumerate(cases):
        for i, t in enumerate(SNAP_TIMES):
            im = axs[i, j].imshow(c["snaps"][t].T, origin="lower", extent=[0, 1, 0, 1],
                                  cmap="inferno", vmin=700, vmax=2800)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title(f"{lab}\n{c['info']['ns']} sp", fontsize=9)
            if j == 0:
                axs[i, j].set_ylabel(f"t = {t}", fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.6, label="T [K]")
    plt.savefig(os.path.join(FIG, "les_snapshots.png"), dpi=140, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for lab, c in cases:
        h = c["hist"]
        ls = "k--" if lab == "ref N=128" else "-"
        ax[0].plot(h["t"], h["Hmax"], ls, label=lab)
        ax[1].plot(h["t"], h["burnt"], ls, label=lab)
    ax[0].set(xlabel="flow time t", ylabel="max chemical temperature rise [K]",
              xlim=(0, 4), title="Ignition")
    ax[1].set(xlabel="flow time t", ylabel="burnt-area fraction (dT > 200 K)",
              xlim=(0, 4), title="Flame spreading")
    ax[0].legend(fontsize=8)
    names = [lab for lab, c in cases if c["info"]["N"] == 64]
    cost = [metrics(c)["ms_chem_per_step"] for lab, c in cases if c["info"]["N"] == 64]
    ax[2].bar(names, cost, color="#3b6ea5")
    ax[2].set(ylabel="chemistry wall time per step [ms]", title="Cost (N=64, numpy)")
    ax[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "les_history.png"), dpi=140)
    print("\nwrote figures/les_snapshots.png, figures/les_history.png, results/les_summary.csv")


if __name__ == "__main__":
    main()
