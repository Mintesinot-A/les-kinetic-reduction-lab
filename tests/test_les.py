import numpy as np
from scipy import fft as sfft

from lesreduce import toy_mechanism, T_FUEL
from lesreduce.les2d import Jet2D


def test_short_run_is_physical():
    sim = Jet2D(toy_mechanism(), N=32)
    sim.run(0.15, diag_every=1000)
    div = sfft.ifft2(1j * sim.KX * sim.uh + 1j * sim.KY * sim.vh).real
    assert np.abs(div).max() < 1e-8                       # incompressible
    assert sim.Phi[0].min() >= 0 and sim.Phi[0].max() <= 1  # Z bounded
    assert sim.Phi[2:].min() >= 0                         # concentrations
    assert sim.temperature().min() >= T_FUEL - 1.0
    u, _ = sim.velocity()
    assert abs(u.mean() - 0.25) < 1e-3                    # mean momentum (jet h=0.25) conserved


def test_same_perturbation_for_any_resolution():
    a = Jet2D(toy_mechanism(), N=32)
    b = Jet2D(toy_mechanism(), N=64)
    # low-wavenumber vorticity must agree between the two grids
    wa, wb = sfft.fft2(a.vorticity()) / 32 ** 2, sfft.fft2(b.vorticity()) / 64 ** 2
    for kx in range(1, 5):
        assert abs(wa[kx, 0] - wb[kx, 0]) < 1e-3 * (abs(wb[kx, 0]) + 1)
