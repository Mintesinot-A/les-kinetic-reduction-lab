import numpy as np
import pytest

from lesreduce import (toy_mechanism, ignition_delay, tau_mixing_line, mixing_line,
                       sample_states, drgep_importance, skeletal)


@pytest.fixture(scope="module")
def det():
    return toy_mechanism()


def test_jacobian_matches_finite_differences(det):
    rng = np.random.default_rng(1)
    C = rng.uniform(0.01, 1.0, (det.ns, 1))
    T = np.array([1050.0])
    J = det.jacobian(C, T)[0]
    f0 = np.concatenate([det.source(C, T)[0][:, 0], det.source(C, T)[1]])
    for j in range(det.ns + 1):
        C2, T2 = C.copy(), T.copy()
        h = 1e-6 * (T[0] if j == det.ns else C[j, 0])
        if j == det.ns:
            T2 = T + h
        else:
            C2[j, 0] += h
        f1 = np.concatenate([det.source(C2, T2)[0][:, 0], det.source(C2, T2)[1]])
        fd = (f1 - f0) / h
        np.testing.assert_allclose(J[:, j], fd, rtol=2e-4, atol=1e-6 * np.abs(f0).max())


def test_ignition_delay_has_interior_minimum(det):
    Z = np.array([0.05, 0.3, 0.6])
    tau = tau_mixing_line(det, Z)
    assert np.all(np.isfinite(tau))
    assert tau[1] < tau[0] and tau[1] < tau[2]
    assert 1e-4 < tau[1] < 1e-2          # milliseconds


def test_les_chemistry_integrator_matches_bdf(det):
    """The linearly-implicit step used in the LES reproduces the 0D BDF delay."""
    comp, T0 = mixing_line(0.3)
    tau_ref = ignition_delay(det, float(T0), {"F": float(comp["F"]), "O": float(comp["O"])})
    C = np.zeros((det.ns, 1))
    C[det.idx["F"]], C[det.idx["O"]] = comp["F"], comp["O"]
    T = np.array([float(T0)])
    dt, t, ts, Ts = 4.5e-6, 0.0, [], []          # ~ LES time step
    while t < 3 * tau_ref:
        C, T = det.chem_step(C, T, dt, nsub=4)
        t += dt
        ts.append(t)
        Ts.append(T[0])
    tau = ts[int(np.argmax(np.gradient(Ts, ts)))]
    assert abs(tau - tau_ref) / tau_ref < 0.05


def test_drgep_ranking_and_skeletal(det):
    C, T = sample_states(det)
    imp = drgep_importance(det, C, T)
    assert min(imp, key=imp.get) == "M2"
    assert all(imp[s] > 0.99 for s in ("F", "O", "P"))
    sk = skeletal(det, imp, 1e-2)
    assert set(sk.species) == {"F", "O", "R", "HO", "I", "P"}
    Z = np.array([0.1, 0.3, 0.5])
    err = np.abs(tau_mixing_line(sk, Z) / tau_mixing_line(det, Z) - 1)
    assert err.max() < 0.02
