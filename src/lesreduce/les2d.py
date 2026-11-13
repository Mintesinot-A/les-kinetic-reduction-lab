"""
les2d.py - 2D temporally-evolving planar jet LES with finite-rate chemistry.

Numerics
--------
* Velocity   : incompressible, Fourier pseudo-spectral (2/3 de-aliasing),
               Smagorinsky sub-grid viscosity, SSP-RK3.
* Scalars    : mixture fraction Z, chemical temperature rise H and species
               concentrations; finite-volume, MUSCL/van Leer TVD advection,
               eddy diffusivity nu_t/Sc_t, SSP-RK3.
* Chemistry  : Strang-split, linearly-implicit Euler with analytic Jacobian,
               vectorised over all cells ("laminar chemistry": no sub-grid
               turbulence-chemistry interaction model).
* Coupling   : constant density, one-way (no dilatation / buoyancy), so that
               the ONLY difference between runs is the kinetic mechanism and
               the grid.  T = T_mix(Z) + H, where H is the chemical heat release.

Everything is non-dimensional in space and velocity (box L=1, jet velocity 1);
``t_ref`` [s] converts flow time to physical time for the chemistry.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import fft as sfft

from .kinetics import T_FUEL, T_OX, mixing_line, C_FUEL0, C_OX0


def _vanleer_slope(dm, dp):
    prod = dm * dp
    den = dm + dp
    den = np.where(den == 0.0, 1.0, den)
    return np.where(prod > 0.0, 2.0 * prod / den, 0.0)


class Jet2D:
    def __init__(self, mech, N=64, nu=1.25e-4, Cs=0.17, delta_factor=1.0,
                 Sc=0.7, Sct=0.9, t_ref=1e-3, h=0.25, theta=0.025,
                 amp=0.03, seed=0, nsub=2, cfl=0.4, chem_tol=1e-4, rad_tol=1e-6):
        self.mech, self.N = mech, N
        self.nu, self.Cs, self.Sc, self.Sct = nu, Cs, Sc, Sct
        self.t_ref, self.nsub, self.cfl, self.chem_tol = t_ref, nsub, cfl, chem_tol
        self.rad_tol = rad_tol
        L = 1.0
        self.dx = L / N
        self.delta = delta_factor * self.dx
        x = np.arange(N) * self.dx
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")      # axis0 = x, axis1 = y
        k = 2 * np.pi * sfft.fftfreq(N, d=self.dx)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        K2 = self.KX ** 2 + self.KY ** 2
        self.K2 = np.where(K2 == 0, 1.0, K2)
        kmax = np.abs(k).max()
        self.mask = ((np.abs(self.KX) < 2 / 3 * kmax) & (np.abs(self.KY) < 2 / 3 * kmax)).astype(float)
        self.ns = mech.ns
        self.t = 0.0
        self.t_chem = self.t_flow = 0.0
        self._init_fields(h, theta, amp, seed)

    # ------------------------------------------------------------------
    def _init_fields(self, h, theta, amp, seed):
        N, X, Y = self.N, self.X, self.Y
        y1, y2 = 0.5 - h / 2, 0.5 + h / 2
        s = 0.5 * (np.tanh((Y - y1) / theta) - np.tanh((Y - y2) / theta))
        u = s.copy()
        # divergence-free perturbation from a stream function (same modes for every N)
        rng = np.random.default_rng(seed)
        a, ph = rng.uniform(0.5, 1.0, 8), rng.uniform(0, 2 * np.pi, 8)
        env = np.exp(-((Y - y1) / 0.05) ** 2) + np.exp(-((Y - y2) / 0.05) ** 2)
        psi = sum(a[m] * amp / (2 * np.pi * (m + 1)) * np.cos(2 * np.pi * (m + 1) * X + ph[m])
                  for m in range(8)) * env
        psih = sfft.fft2(psi)
        self.uh = sfft.fft2(u) + 1j * self.KY * psih
        self.vh = -1j * self.KX * psih
        # scalars: [Z, H, species...]
        Z = s
        comp, _ = mixing_line(Z)
        self.Phi = np.zeros((2 + self.ns, N, N))
        self.Phi[0] = Z
        for sp, c in comp.items():
            if sp in self.mech.idx:
                self.Phi[2 + self.mech.idx[sp]] = c

    # ------------------------------------------------------------------
    def velocity(self):
        return sfft.ifft2(self.uh).real, sfft.ifft2(self.vh).real

    def vorticity(self):
        return sfft.ifft2(1j * self.KX * self.vh - 1j * self.KY * self.uh).real

    def temperature(self, Phi=None):
        Phi = self.Phi if Phi is None else Phi
        return T_OX + (T_FUEL - T_OX) * np.clip(Phi[0], 0, 1) + Phi[1]

    # ------------------------------------------------------------------
    def _rhs(self, uh, vh, Phi):
        KX, KY, d = self.KX, self.KY, sfft
        u, v = d.ifft2(uh).real, d.ifft2(vh).real
        ux, uy = d.ifft2(1j * KX * uh).real, d.ifft2(1j * KY * uh).real
        vx, vy = d.ifft2(1j * KX * vh).real, d.ifft2(1j * KY * vh).real
        S11, S22, S12 = ux, vy, 0.5 * (uy + vx)
        nut = (self.Cs * self.delta) ** 2 * np.sqrt(2 * (S11 ** 2 + S22 ** 2 + 2 * S12 ** 2))
        nue = self.nu + nut
        t11h, t12h, t22h = (d.fft2(2 * nue * S11), d.fft2(2 * nue * S12), d.fft2(2 * nue * S22))
        Nu = -d.fft2(u * ux + v * uy) + 1j * (KX * t11h + KY * t12h)
        Nv = -d.fft2(u * vx + v * vy) + 1j * (KX * t12h + KY * t22h)
        Nu *= self.mask
        Nv *= self.mask
        div = KX * Nu + KY * Nv
        Nu = Nu - KX * div / self.K2
        Nv = Nv - KY * div / self.K2
        # ---- scalars -------------------------------------------------
        dx = self.dx
        phi = Phi
        dp, dm = np.roll(phi, -1, 1) - phi, phi - np.roll(phi, 1, 1)
        sl = _vanleer_slope(dm, dp)
        uf = 0.5 * (u + np.roll(u, -1, 0))
        fx = uf * np.where(uf > 0, phi + 0.5 * sl, np.roll(phi, -1, 1) - 0.5 * np.roll(sl, -1, 1))
        dp, dm = np.roll(phi, -1, 2) - phi, phi - np.roll(phi, 1, 2)
        sl = _vanleer_slope(dm, dp)
        vf = 0.5 * (v + np.roll(v, -1, 1))
        fy = vf * np.where(vf > 0, phi + 0.5 * sl, np.roll(phi, -1, 2) - 0.5 * np.roll(sl, -1, 2))
        adv = (fx - np.roll(fx, 1, 1)) / dx + (fy - np.roll(fy, 1, 2)) / dx
        G = self.nu / self.Sc + nut / self.Sct
        Gx, Gy = 0.5 * (G + np.roll(G, -1, 0)), 0.5 * (G + np.roll(G, -1, 1))
        qx = Gx * (np.roll(phi, -1, 1) - phi) / dx
        qy = Gy * (np.roll(phi, -1, 2) - phi) / dx
        dif = (qx - np.roll(qx, 1, 1)) / dx + (qy - np.roll(qy, 1, 2)) / dx
        return Nu, Nv, dif - adv, nut.max(), (np.abs(u).max(), np.abs(v).max())

    def _chem(self, dt_nd):
        """Half or full chemistry step.  Only 'active' cells are integrated:
        those where fuel and oxidiser coexist (above ``chem_tol`` mol/m^3) or
        where intermediates are already present.  Set chem_tol=0 for all cells.
        """
        N, m = self.N, self.mech
        Phi = self.Phi
        Z = np.clip(Phi[0], 0, 1).ravel()
        T = (T_OX + (T_FUEL - T_OX) * Z + Phi[1].ravel())
        C = Phi[2:].reshape(self.ns, -1)
        if self.chem_tol > 0:
            others = [i for s, i in m.idx.items() if s not in ("F", "O", "P")]
            act = (C[m.idx["F"]] > self.chem_tol) & (C[m.idx["O"]] > self.chem_tol)
            if others:
                act |= C[others].sum(0) > self.rad_tol
        else:
            act = np.ones(T.size, dtype=bool)
        self.active_frac = act.mean()
        if not act.any():
            return
        C2, T2 = m.chem_step(C[:, act], T[act], dt_nd * self.t_ref, nsub=self.nsub)
        Cn = C.copy()
        Cn[:, act] = C2
        dH = np.zeros_like(T)
        dH[act] = T2 - T[act]
        self.Phi[2:] = Cn.reshape(self.ns, N, N)
        self.Phi[1] += dH.reshape(N, N)

    def stable_dt(self, umax, vmax, nutmax):
        dt_adv = self.cfl * self.dx / (umax + vmax + 1e-12)
        dt_dif = 0.2 * self.dx ** 2 / (self.nu + nutmax + 1e-12)
        return min(dt_adv, dt_dif, 0.01)

    def step(self, dt=None):
        if dt is None:
            _, _, _, nutm, (um, vm) = self._rhs(self.uh, self.vh, self.Phi)
            dt = self.stable_dt(um, vm, nutm)
        t0 = time.perf_counter()
        self._chem(0.5 * dt)
        t1 = time.perf_counter()
        y0 = (self.uh, self.vh, self.Phi)
        f0 = self._rhs(*y0)
        y1 = tuple(a + dt * f for a, f in zip(y0, f0[:3]))
        f1 = self._rhs(*y1)
        y2 = tuple(0.75 * a + 0.25 * (b + dt * f) for a, b, f in zip(y0, y1, f1[:3]))
        f2 = self._rhs(*y2)
        y3 = tuple(a / 3 + 2 / 3 * (b + dt * f) for a, b, f in zip(y0, y2, f2[:3]))
        self.uh, self.vh, self.Phi = y3
        self.Phi[0] = np.clip(self.Phi[0], 0, 1)
        self.Phi[1] = np.maximum(self.Phi[1], 0)
        self.Phi[2:] = np.maximum(self.Phi[2:], 0)
        t2 = time.perf_counter()
        self._chem(0.5 * dt)
        t3 = time.perf_counter()
        self.t += dt
        self.t_chem += (t1 - t0) + (t3 - t2)
        self.t_flow += (t2 - t1)
        return dt

    # ------------------------------------------------------------------
    def run(self, t_end, diag_every=10, snap_times=(), verbose=False):
        self.t_chem = self.t_flow = 0.0
        hist = {k: [] for k in ("t", "Hmax", "Hmean", "Z_at_Hmax", "burnt", "Pmean", "Tmax")}
        snaps, pending, nstep = {}, sorted(snap_times), 0
        wall0 = time.perf_counter()
        while self.t < t_end:
            dt = self.step()
            nstep += 1
            if nstep % diag_every == 0:
                H = self.Phi[1]
                i = np.unravel_index(np.argmax(H), H.shape)
                hist["t"].append(self.t)
                hist["Hmax"].append(H.max())
                hist["Hmean"].append(H.mean())
                hist["Z_at_Hmax"].append(self.Phi[0][i])
                hist["burnt"].append(float((H > 200.0).mean()))
                hist["Pmean"].append(self.Phi[2 + self.mech.idx["P"]].mean())
                hist["Tmax"].append(self.temperature().max())
            while pending and self.t >= pending[0]:
                tt = pending.pop(0)
                snaps[tt] = dict(Z=self.Phi[0].copy(), T=self.temperature(),
                                 H=self.Phi[1].copy(), omega=self.vorticity(), t=self.t)
            if verbose and nstep % 200 == 0:
                print(f"  t={self.t:6.3f} step={nstep} Hmax={self.Phi[1].max():7.1f}")
        hist = {k: np.array(v) for k, v in hist.items()}
        info = dict(nstep=nstep, wall=time.perf_counter() - wall0,
                    t_chem=self.t_chem, t_flow=self.t_flow)
        return hist, snaps, info


def ignition_time(hist, H_thresh=200.0):
    """First time the maximum chemical temperature rise exceeds H_thresh [K]."""
    i = np.nonzero(hist["Hmax"] > H_thresh)[0]
    return (hist["t"][i[0]], hist["Z_at_Hmax"][i[0]]) if len(i) else (np.inf, np.nan)
