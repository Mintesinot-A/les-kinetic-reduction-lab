"""
kinetics.py - a tiny, dependency-light chemical-kinetics toolkit for teaching.

Contents
--------
* ``Reaction`` / ``Mechanism``  : mass-action kinetics, Arrhenius rates,
                                  analytic Jacobian, batched linearly-implicit
                                  chemistry integrator (used inside the LES).
* ``toy_mechanism``             : an 8-species / 12-reaction chain-branching
                                  mechanism (NOT a real fuel, see README).
* ``ignition_delay``            : 0D constant-volume ignition (scipy BDF).
* ``drgep_importance``          : DRGEP species ranking (Pepiot-Desjardins &
                                  Pitsch style path-dependent coefficients).
* ``skeletal``                  : build a reduced mechanism from a threshold.
* ``fit_global_one_step``       : classical global 1-step fit to ignition delay.

Units are SI: mol/m^3, K, s, J/mol.  Density and rho*cv are held constant
(constant-volume, low-Mach, one-way coupled to the flow - see README).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314462618  # J/mol/K

# --------------------------------------------------------------------------
# Stream conditions shared by the 0D tests and the LES (mixing-line model)
# --------------------------------------------------------------------------
T_FUEL = 700.0      # K, cold fuel stream (jet core)
T_OX = 1150.0       # K, hot oxidiser stream (co-flow)
C_FUEL0 = 2.0       # mol/m^3 fuel in the fuel stream (diluted)
C_OX0 = 2.6         # mol/m^3 oxidiser in the oxidiser stream


def mixing_line(Z):
    """Species concentrations and temperature of a frozen mixture at Z."""
    Z = np.asarray(Z, dtype=float)
    return {"F": Z * C_FUEL0, "O": (1.0 - Z) * C_OX0}, T_OX + (T_FUEL - T_OX) * Z


@dataclass(frozen=True)
class Reaction:
    reactants: dict          # {species: order}
    products: dict           # {species: stoichiometric coefficient}
    A: float                 # pre-exponential (SI)
    Ea: float                # activation energy, J/mol
    q: float = 0.0           # heat release, J per mol of reaction progress
    label: str = ""

    def __str__(self):
        f = lambda d: " + ".join((f"{int(n)} " if n != 1 else "") + s for s, n in d.items())
        return f"{f(self.reactants)} -> {f(self.products)}"


class Mechanism:
    """Mass-action mechanism with constant rho*cv (so T is an extra ODE)."""

    def __init__(self, species, reactions, rho_cv=350.0, name="mech"):
        self.species = list(species)
        self.reactions = list(reactions)
        self.rho_cv = float(rho_cv)
        self.name = name
        self.ns, self.nr = len(self.species), len(self.reactions)
        self.idx = {s: i for i, s in enumerate(self.species)}
        nu_r = np.zeros((self.nr, self.ns))
        nu_p = np.zeros((self.nr, self.ns))
        for k, r in enumerate(self.reactions):
            for s, n in r.reactants.items():
                nu_r[k, self.idx[s]] += n
            for s, n in r.products.items():
                nu_p[k, self.idx[s]] += n
        self.nu_r, self.nu_p, self.nu = nu_r, nu_p, nu_p - nu_r
        self.A = np.array([r.A for r in self.reactions])
        self.Ea = np.array([r.Ea for r in self.reactions])
        self.q = np.array([r.q for r in self.reactions])
        self._reac = [[(self.idx[s], int(n)) for s, n in r.reactants.items()]
                      for r in self.reactions]

    # ---- rates ---------------------------------------------------------
    def kf(self, T):
        return self.A[:, None] * np.exp(-self.Ea[:, None] / (R_GAS * T[None, :]))

    def progress(self, C, T):
        """Rates of progress r_k, shape (nr, N).  C: (ns, N), T: (N,)."""
        r = self.kf(T)
        for k, reac in enumerate(self._reac):
            for i, n in reac:
                r[k] *= C[i] ** n
        return r

    def source(self, C, T):
        """(dC/dt, dT/dt) from chemistry."""
        r = self.progress(C, T)
        return self.nu.T @ r, (self.q @ r) / self.rho_cv

    def jacobian(self, C, T):
        """Analytic Jacobian of [dC/dt, dT/dt] wrt [C, T]: (N, ns+1, ns+1)."""
        N, ns = T.size, self.ns
        kf = self.kf(T)
        J = np.zeros((N, ns + 1, ns + 1))
        for k, reac in enumerate(self._reac):
            rk = kf[k].copy()
            for i, n in reac:
                rk *= C[i] ** n
            drdT = rk * self.Ea[k] / (R_GAS * T ** 2)
            drdC = {}
            for j, nj in reac:
                d = kf[k] * nj * C[j] ** (nj - 1)
                for i, ni in reac:
                    if i != j:
                        d = d * C[i] ** ni
                drdC[j] = d
            for i in np.nonzero(self.nu[k])[0]:
                for j, d in drdC.items():
                    J[:, i, j] += self.nu[k, i] * d
                J[:, i, ns] += self.nu[k, i] * drdT
            qk = self.q[k] / self.rho_cv
            if qk != 0.0:
                for j, d in drdC.items():
                    J[:, ns, j] += qk * d
                J[:, ns, ns] += qk * drdT
        return J

    # ---- batched integrator used by the LES ----------------------------
    def chem_step(self, C, T, dt, nsub=2):
        """Advance chemistry by dt with ``nsub`` linearly-implicit Euler steps.

        C: (ns, N), T: (N,).  Returns new (C, T).  First order, stable for
        stiff chemistry, vectorised over all cells (one batched solve each).
        """
        C, T = C.copy(), T.copy()
        h = dt / nsub
        eye = np.eye(self.ns + 1)[None]
        for _ in range(nsub):
            dC, dT = self.source(C, T)
            J = self.jacobian(C, T)
            f = np.vstack([dC, dT[None]]).T                    # (N, ns+1)
            dy = np.linalg.solve(eye - h * J, h * f[..., None])[..., 0]
            C = np.maximum(C + dy[:, : self.ns].T, 0.0)
            T = T + dy[:, self.ns]
        return C, T


# --------------------------------------------------------------------------
# The toy mechanism
# --------------------------------------------------------------------------
def toy_mechanism(name="detailed"):
    """8 species / 12 reactions.  Chain branching + termination crossover
    (an 'explosion limit' in T), a delayed HO channel and two minor side
    channels (M1, M2) that DRGEP should be able to discard.
    """
    QP = 4.0e5  # J/mol released per mol of P formed
    rx = [
        Reaction({"F": 1, "O": 1}, {"R": 2}, 5e8, 1.6e5, 0, "initiation"),
        Reaction({"F": 1, "R": 1}, {"I": 1, "R": 1}, 5e7, 6.0e4, 0, "F attack"),
        Reaction({"R": 1, "O": 1}, {"R": 2}, 1e8, 1.0e5, 0, "branching"),
        Reaction({"R": 1, "O": 1}, {"HO": 1}, 2e3, 1.0e4, 0, "R -> HO"),
        Reaction({"HO": 1, "F": 1}, {"I": 1, "R": 1}, 1e8, 9.0e4, 0, "HO reactivation"),
        Reaction({"I": 1, "O": 1}, {"P": 1, "R": 1}, 1e8, 7.0e4, QP, "oxidation"),
        Reaction({"R": 2}, {"P": 1}, 1e7, 0.0, QP, "termination"),
        Reaction({"I": 1, "R": 1}, {"M1": 1}, 1e6, 3.0e4, 0, "side 1"),
        Reaction({"M1": 1, "O": 1}, {"P": 1, "R": 1}, 1e7, 5.0e4, QP, "side 1b"),
        Reaction({"HO": 2}, {"M2": 1, "O": 1}, 1e6, 3.0e4, 0, "side 2"),
        Reaction({"M2": 1, "R": 1}, {"P": 1, "R": 1}, 1e7, 4.0e4, QP, "side 2b"),
        Reaction({"HO": 1, "R": 1}, {"P": 1}, 1e7, 0.0, QP, "termination 2"),
    ]
    return Mechanism(["F", "O", "R", "HO", "I", "P", "M1", "M2"], rx, name=name)


# --------------------------------------------------------------------------
# 0D ignition
# --------------------------------------------------------------------------
def _initial_state(mech, T0, comp):
    y0 = np.zeros(mech.ns + 1)
    for s, c in comp.items():
        if s in mech.idx:
            y0[mech.idx[s]] = c
    y0[mech.ns] = T0
    return y0


def ignition_delay(mech, T0, comp, t_max=0.2, rtol=1e-8, atol=1e-14,
                   min_rise=50.0, return_traj=False):
    """Time of maximum dT/dt in a constant-volume reactor (inf if no ignition).

    ``min_rise``: required total chemical temperature rise [K] to count as ignition.
    """
    ns = mech.ns

    def f(t, y):
        dC, dT = mech.source(np.maximum(y[:ns], 0)[:, None], y[ns:])
        return np.concatenate([dC[:, 0], dT])

    def jac(t, y):
        return mech.jacobian(np.maximum(y[:ns], 0)[:, None], y[ns:])[0]

    y0 = _initial_state(mech, T0, comp)
    sol = solve_ivp(f, (0, t_max), y0, method="BDF", jac=jac, rtol=rtol,
                    atol=atol, dense_output=True)
    t = np.geomspace(1e-6, t_max, 3000)
    Y = sol.sol(t)
    T = Y[ns]
    dTdt = np.gradient(T, t)
    i = int(np.argmax(dTdt))
    tau = t[i] if (T[-1] - T0 > min_rise and i < len(t) - 3) else np.inf
    if return_traj:
        return tau, t, Y
    return tau


def tau_mixing_line(mech, Z):
    """Ignition delay along the LES mixing line for an array of Z."""
    comp, T = mixing_line(Z)
    return np.array([ignition_delay(mech, T[i], {"F": comp["F"][i], "O": comp["O"][i]})
                     for i in range(len(Z))])


# --------------------------------------------------------------------------
# DRGEP
# --------------------------------------------------------------------------
def sample_states(mech, Zs=(0.05, 0.1, 0.2, 0.35, 0.5), t_max=0.2, n=60):
    """Sample thermochemical states along 0D ignition trajectories (mixing line)."""
    Cs, Ts = [], []
    for Z in Zs:
        comp, T0 = mixing_line(Z)
        tau, t, Y = ignition_delay(mech, float(T0), {"F": float(comp["F"]),
                                                     "O": float(comp["O"])},
                                   t_max=t_max, return_traj=True)
        t_end = min(t_max, 3 * tau) if np.isfinite(tau) else t_max
        ts = np.unique(np.concatenate([np.geomspace(1e-6, t_end, n),
                                       np.linspace(0.5 * tau, 1.5 * tau, n)
                                       if np.isfinite(tau) else []]))
        ts = ts[ts <= t_max]
        Yt = np.array([np.interp(ts, t, row) for row in Y])
        Cs.append(np.maximum(Yt[: mech.ns], 0))
        Ts.append(Yt[mech.ns])
    return np.hstack(Cs), np.concatenate(Ts)


def drgep_importance(mech, C, T, targets=("F", "O", "P")):
    """Overall DRGEP importance of each species (max over states and targets)."""
    r = mech.progress(C, T)                                    # (nr, M)
    omega = mech.nu.T[:, :, None] * r[None]                    # (ns, nr, M)
    prod = np.where(omega > 0, omega, 0).sum(1)
    cons = np.where(omega < 0, -omega, 0).sum(1)
    denom = np.maximum(np.maximum(prod, cons), 1e-300)         # (ns, M)
    part = ((mech.nu_r + mech.nu_p) > 0).astype(float)         # (nr, ns)
    dic = np.einsum("akm,kb->abm", np.abs(omega), part) / denom[:, None, :]
    for a in range(mech.ns):
        dic[a, a, :] = 0.0
    dic = np.minimum(dic, 1.0)
    imp = np.zeros(mech.ns)
    for m in range(C.shape[1]):
        for tname in targets:
            t0 = mech.idx[tname]
            best = np.zeros(mech.ns)
            best[t0] = 1.0
            heap = [(-1.0, t0)]
            while heap:                                        # max-product Dijkstra
                v, a = heapq.heappop(heap)
                v = -v
                if v < best[a]:
                    continue
                for b in range(mech.ns):
                    cand = v * dic[a, b, m]
                    if cand > best[b]:
                        best[b] = cand
                        heapq.heappush(heap, (-cand, b))
            imp = np.maximum(imp, best)
    return dict(zip(mech.species, imp))


def skeletal(mech, importance, eps, keep=("F", "O", "P"), name=None):
    """Remove species with importance < eps and every reaction touching them."""
    kept = [s for s in mech.species if importance[s] >= eps or s in keep]
    rx = [r for r in mech.reactions
          if all(s in kept for s in list(r.reactants) + list(r.products))]
    return Mechanism(kept, rx, rho_cv=mech.rho_cv, name=name or f"skeletal_eps{eps:g}")


# --------------------------------------------------------------------------
# Global one-step mechanism fitted to the detailed ignition delays
# --------------------------------------------------------------------------
def fit_global_one_step(detailed, Zs=(0.1, 0.15, 0.2, 0.3, 0.4), n_Ea=14):
    """F + O -> P with (A, Ea) fitted to detailed ignition delay on the mixing line.

    For a one-step Arrhenius model tau scales exactly as 1/A, so for each trial
    Ea the optimal A follows analytically (least squares on log tau); only Ea
    is scanned.  A fair, cheap 'global mechanism' baseline.
    """
    Zs = np.asarray(Zs, dtype=float)
    tau_ref = tau_mixing_line(detailed, Zs)
    Zs, tau_ref = Zs[np.isfinite(tau_ref)], tau_ref[np.isfinite(tau_ref)]
    comp, T = mixing_line(Zs)
    QP = float(detailed.q.max())

    def build(lnA, Ea):
        return Mechanism(["F", "O", "P"],
                         [Reaction({"F": 1, "O": 1}, {"P": 1}, np.exp(lnA), Ea, QP)],
                         rho_cv=detailed.rho_cv, name="global_1step")

    best = None
    for Ea in np.linspace(0.2e5, 2.5e5, n_Ea):
        lnA0 = np.log(1e3) + Ea / (R_GAS * 1050.0)      # k(1050 K) = 1e3
        m0 = build(lnA0, Ea)
        tau = np.array([ignition_delay(m0, T[i], {"F": comp["F"][i], "O": comp["O"][i]},
                                       t_max=0.5, rtol=1e-6, atol=1e-10)
                        for i in range(len(Zs))])
        if not np.all(np.isfinite(tau)):
            continue
        shift = np.mean(np.log(tau) - np.log(tau_ref))   # tau ~ 1/A
        cost = np.sum((np.log(tau) - shift - np.log(tau_ref)) ** 2)
        if best is None or cost < best[0]:
            best = (cost, lnA0 + shift, Ea)
    cost, lnA, Ea = best
    return build(lnA, Ea), dict(lnA=lnA, Ea=Ea, cost=cost)
