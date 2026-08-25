"""Test the canonical CeBr3 detector-efficiency response."""

import numpy as np

from spectrum.response_matrix import cebr3_efficiency


def test_efficiency_low_energy_threshold():
    """Efficiency must be zero below the detector threshold."""
    val = cebr3_efficiency(20.0)
    assert val == 0.0


def test_efficiency_high_vs_low_energy():
    """Efficiency near 100 keV must exceed efficiency near 1 MeV."""
    eff_100 = cebr3_efficiency(100.0)
    eff_1000 = cebr3_efficiency(1000.0)
    assert eff_100 > eff_1000


def test_efficiency_array_input():
    """Vector input must preserve shape and threshold behavior."""
    energies = np.array([0.0, 50.0, 100.0])
    eff = cebr3_efficiency(energies)
    assert eff.shape == energies.shape
    assert eff[0] == 0.0
    assert eff[1] > 0.0


def test_efficiency_ratios_and_order():
    """Efficiency must follow the declared CeBr3-like high-energy trend."""
    e1 = cebr3_efficiency(59.5)
    e2 = cebr3_efficiency(662.0)
    e3 = cebr3_efficiency(1332.0)
    assert e1 > e2 > e3
    ratio = e1 / e3
    assert 3.5 < ratio < 6.0
    assert cebr3_efficiency(500.0) > cebr3_efficiency(1332.0)
    assert cebr3_efficiency(1332.0) > cebr3_efficiency(2000.0)
