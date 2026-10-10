"""Unit tests for the OFDM PHY model. Run with:  python -m pytest -q"""
import numpy as np
import pytest

from phyval.phy.ofdm import (BITS_PER_SYMBOL, DATA, FAULTS, FaultConfig, channel_taps, constellation,
                             demodulate, modulate, simulate_link)

AWGN = np.array([1.0 + 0j])


@pytest.mark.parametrize("mod", list(BITS_PER_SYMBOL))
def test_constellation_has_unit_power(mod):
    pts = constellation(mod)
    assert len(pts) == 2 ** BITS_PER_SYMBOL[mod]
    assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0)


@pytest.mark.parametrize("mod", ["QPSK", "QAM16", "QAM64"])
def test_gray_coding_neighbours_differ_by_one_bit(mod):
    pts = constellation(mod)
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = d[d > 0].min()
    for i in range(len(pts)):
        for j in np.where(np.isclose(d[i], dmin))[0]:
            assert bin(i ^ j).count("1") == 1


@pytest.mark.parametrize("mod", list(BITS_PER_SYMBOL))
def test_modulate_demodulate_roundtrip(mod):
    bits = np.random.default_rng(0).integers(0, 2, 48 * BITS_PER_SYMBOL[mod] * 10, dtype=np.int8)
    assert np.array_equal(demodulate(modulate(bits, mod), mod), bits)


def test_clean_link_is_error_free_at_high_snr():
    r = simulate_link("QAM64", 45, AWGN, 0.0, FaultConfig(), np.random.default_rng(1))
    assert r.ber == 0.0
    assert r.evm_pct < 2.0


@pytest.mark.parametrize("cfo", [-0.4, -0.1, 0.0, 0.2, 0.45])
def test_cfo_estimator_is_accurate(cfo):
    r = simulate_link("QPSK", 30, AWGN, cfo, FaultConfig(), np.random.default_rng(2))
    assert abs(r.diagnostics["cfo_est"] - cfo) < 0.02
    assert r.ber == 0.0                      # offset is fully corrected


def test_multipath_within_cp_is_equalised():
    h = channel_taps([0, 2, 5], [0, -3, -6], np.random.default_rng(3))
    r = simulate_link("QPSK", 40, h, 0.1, FaultConfig(), np.random.default_rng(4))
    assert r.ber == 0.0


# For each fault: a test condition where it must clearly hurt, and the signature diagnostic it should move.
FAULT_CASES = {
    "short_cp":        dict(channel="VEH", cfo=0.0),
    "iq_imbalance":    dict(channel="AWGN", cfo=0.0, diag=("image_corr", 0.2)),
    "adc_clipping":    dict(channel="AWGN", cfo=0.0, diag=("sat_ratio", 0.002)),
    "cfo_uncorrected": dict(channel="AWGN", cfo=0.3, diag=("cpe_slope_rad", 0.12)),
    "timing_offset":   dict(channel="AWGN", cfo=0.0),
    "stale_channel":   dict(channel="VEH", cfo=0.0),
}


@pytest.mark.parametrize("fault", [f for f in FAULTS if f != "none"])
def test_each_fault_degrades_evm_and_shows_its_signature(fault):
    case = FAULT_CASES[fault]
    if case["channel"] == "VEH":
        h = channel_taps([0, 2, 5, 9, 13], [0, -2, -5, -8, -12], np.random.default_rng(5))
        h_old = channel_taps([0, 2, 5, 9, 13], [0, -2, -5, -8, -12], np.random.default_rng(6))
    else:
        h, h_old = AWGN, AWGN
    clean = simulate_link("QAM16", 35, h, case["cfo"], FaultConfig(), np.random.default_rng(7), h_stale=h_old)
    bad = simulate_link("QAM16", 35, h, case["cfo"], FaultConfig.from_name(fault, 0.8), np.random.default_rng(7), h_stale=h_old)
    assert bad.evm_pct > 1.5 * clean.evm_pct
    if "diag" in case:
        key, thr = case["diag"]
        assert abs(bad.diagnostics[key]) > thr >= abs(clean.diagnostics[key])


def test_fault_is_harmless_where_physics_says_it_should_be():
    # A stale channel estimate does nothing on a flat (AWGN) channel; an uncorrected-CFO bug does nothing without CFO.
    for fault, cfo in (("stale_channel", 0.15), ("cfo_uncorrected", 0.0)):
        r = simulate_link("QAM16", 35, AWGN, cfo, FaultConfig.from_name(fault, 1.0), np.random.default_rng(8), h_stale=AWGN)
        assert r.evm_pct < 4.0


def test_unknown_fault_name_rejected():
    with pytest.raises(ValueError):
        FaultConfig.from_name("does_not_exist")


def test_data_subcarrier_count():
    assert len(DATA) == 48
