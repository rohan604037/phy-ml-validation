"""
OFDM baseband physical layer (PHY) model.

OFDM  = Orthogonal Frequency-Division Multiplexing
PHY   = Physical Layer
FFT   = Fast Fourier Transform, IFFT = Inverse FFT
CP    = Cyclic Prefix
CFO   = Carrier Frequency Offset (expressed as a fraction of the subcarrier spacing)
EVM   = Error Vector Magnitude, BER = Bit Error Rate
AGC   = Automatic Gain Control, ADC = Analog-to-Digital Converter

Frame format (loosely modelled on IEEE 802.11a/g Wi-Fi):

    [CP | STS] [CP | LTS] [CP | DATA_1] ... [CP | DATA_n]

    STS (Short Training Symbol): only even subcarriers are used, so the time-domain
         signal repeats every 32 samples. The receiver uses the repetition to
         estimate the CFO.
    LTS (Long Training Symbol): every used subcarrier carries a known value, so
         the receiver can estimate the channel on each subcarrier.
    DATA: 48 data subcarriers + 4 pilot subcarriers per symbol. Pilots are used
         to track and remove the common phase error of each symbol.

The receiver also produces *diagnostics* (counters and estimates that a real
modem exposes to firmware/logs). These diagnostics are what the ML triage
models learn from - the models never see the injected fault label directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Numerology
# ----------------------------------------------------------------------------
N_FFT = 64                      # FFT size (number of subcarriers)
CP_LEN = 16                     # cyclic prefix length in samples
SYM_LEN = N_FFT + CP_LEN        # samples per OFDM symbol including CP
USED = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])   # 52 used subcarriers
PILOTS = np.array([-21, -7, 7, 21])
PILOT_VALUES = np.array([1, 1, 1, -1], dtype=complex)
DATA = np.array([k for k in USED if k not in set(PILOTS.tolist())])  # 48 data subcarriers

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "QAM16": 4, "QAM64": 6}
ADC_FULL_SCALE_NOMINAL = 4.0    # ADC clip level in multiples of signal RMS (clean design)


def _bins(k) -> np.ndarray:
    """Map signed subcarrier index (-32..31) to FFT bin (0..63)."""
    return np.asarray(k) % N_FFT


# Index helpers inside the USED vector
_USED_POS = {int(k): i for i, k in enumerate(USED)}
DATA_IN_USED = np.array([_USED_POS[int(k)] for k in DATA])
PILOT_IN_USED = np.array([_USED_POS[int(k)] for k in PILOTS])
# pairs (k, -k) where both are data subcarriers -> used to detect I/Q image leakage
_DATA_POS = {int(k): i for i, k in enumerate(DATA)}
MIRROR_PAIRS = np.array([(_DATA_POS[int(k)], _DATA_POS[int(-k)]) for k in DATA if int(-k) in _DATA_POS])


# ----------------------------------------------------------------------------
# Modulation (Gray-coded BPSK / QPSK / 16-QAM / 64-QAM)
# ----------------------------------------------------------------------------
_CONSTELLATION_CACHE: dict = {}


def constellation(mod: str) -> np.ndarray:
    """Return the unit-average-power constellation; index = integer value of the bit group."""
    if mod in _CONSTELLATION_CACHE:
        return _CONSTELLATION_CACHE[mod]
    m = BITS_PER_SYMBOL[mod]
    if mod == "BPSK":
        pts = np.array([-1 + 0j, 1 + 0j])
    else:
        k = m // 2
        levels = 2 ** k
        pts = np.zeros(2 ** m, dtype=complex)
        for i in range(levels):
            for q in range(levels):
                gray_i = i ^ (i >> 1)          # Gray code: neighbours differ by one bit
                gray_q = q ^ (q >> 1)
                value = (gray_i << k) | gray_q
                pts[value] = (2 * i - levels + 1) + 1j * (2 * q - levels + 1)
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    _CONSTELLATION_CACHE[mod] = pts
    return pts


def modulate(bits: np.ndarray, mod: str) -> np.ndarray:
    m = BITS_PER_SYMBOL[mod]
    groups = bits.reshape(-1, m)
    values = groups @ (1 << np.arange(m - 1, -1, -1))
    return constellation(mod)[values]


def demodulate(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Hard-decision (nearest constellation point) demodulation back to bits."""
    m = BITS_PER_SYMBOL[mod]
    pts = constellation(mod)
    values = np.argmin(np.abs(symbols.reshape(-1, 1) - pts.reshape(1, -1)), axis=1)
    return ((values[:, None] >> np.arange(m - 1, -1, -1)) & 1).reshape(-1).astype(np.int8)


# ----------------------------------------------------------------------------
# Training symbols (fixed, known to transmitter and receiver)
# ----------------------------------------------------------------------------
def _make_training():
    rng = np.random.default_rng(20260101)
    sts = np.zeros(N_FFT, dtype=complex)
    even = USED[USED % 2 == 0]
    sts[_bins(even)] = np.sqrt(2) * (rng.choice([-1, 1], len(even)) + 1j * rng.choice([-1, 1], len(even))) / np.sqrt(2)
    lts = np.zeros(N_FFT, dtype=complex)
    lts[_bins(USED)] = rng.choice([-1.0, 1.0], len(USED))
    return sts, lts


STS_FREQ, LTS_FREQ = _make_training()


def ofdm_symbol(freq: np.ndarray, cp: int) -> np.ndarray:
    """Frequency-domain symbol -> time-domain samples with cyclic prefix."""
    x = np.fft.ifft(freq) * np.sqrt(N_FFT)
    return np.concatenate([x[N_FFT - cp:], x]) if cp > 0 else x


# ----------------------------------------------------------------------------
# Fault model
# ----------------------------------------------------------------------------
FAULTS = {
    "none":            "Golden (bug-free) design",
    "short_cp":        "Framing configured with a cyclic prefix shorter than the channel delay spread",
    "iq_imbalance":    "TX RF I/Q gain and phase imbalance (image leakage onto mirror subcarriers)",
    "adc_clipping":    "RX ADC full-scale set too low, so signal peaks are clipped",
    "cfo_uncorrected": "RX CFO correction gain bug: estimated offset is only partly removed",
    "timing_offset":   "RX symbol-timing bug: FFT window starts late and picks up the next symbol",
    "stale_channel":   "RX channel estimator mixes in an outdated channel estimate",
}


@dataclass
class FaultConfig:
    """Concrete hardware/firmware parameters derived from (fault name, severity 0..1)."""
    name: str = "none"
    severity: float = 0.0
    cp_len: int = CP_LEN
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    adc_full_scale: float = ADC_FULL_SCALE_NOMINAL
    cfo_correction_gain: float = 1.0
    rx_timing_offset: int = 0
    stale_weight: float = 0.0

    @staticmethod
    def from_name(name: str, severity: float = 0.0) -> "FaultConfig":
        if name not in FAULTS:
            raise ValueError(f"Unknown fault '{name}'. Choose from: {', '.join(FAULTS)}")
        s = float(np.clip(severity, 0.0, 1.0))
        f = FaultConfig(name=name, severity=s)
        if name == "short_cp":
            f.cp_len = CP_LEN - (4 + int(round(s * 8)))             # CP of 12 .. 4 samples
        elif name == "iq_imbalance":
            f.iq_gain_db = 0.3 + 1.7 * s
            f.iq_phase_deg = 1.0 + 7.0 * s
        elif name == "adc_clipping":
            f.adc_full_scale = 2.6 - 1.2 * s                         # 2.6 .. 1.4 x RMS
        elif name == "cfo_uncorrected":
            f.cfo_correction_gain = 1.0 - s
        elif name == "timing_offset":
            f.rx_timing_offset = 1 + int(round(s * 2))               # 1..3 samples late
        elif name == "stale_channel":
            f.stale_weight = 0.03 + 0.17 * s                         # 3% .. 20% stale
        return f


# ----------------------------------------------------------------------------
# Channel
# ----------------------------------------------------------------------------
def channel_taps(delays, powers_db, rng: np.random.Generator) -> np.ndarray:
    """Block-fading multipath channel (Rayleigh taps), normalised to unit energy."""
    delays = np.asarray(delays, dtype=int)
    p = 10 ** (np.asarray(powers_db, dtype=float) / 10)
    p = p / p.sum()
    h = np.zeros(delays.max() + 1, dtype=complex)
    if len(delays) == 1:
        h[delays[0]] = 1.0
        return h
    h[delays] = np.sqrt(p / 2) * (rng.standard_normal(len(delays)) + 1j * rng.standard_normal(len(delays)))
    return h / np.linalg.norm(h)


def freq_response(h: np.ndarray) -> np.ndarray:
    return np.fft.fft(h, N_FFT)


# ----------------------------------------------------------------------------
# Link simulation
# ----------------------------------------------------------------------------
@dataclass
class LinkResult:
    mod: str
    snr_db: float
    evm_pct: float
    evm_db: float
    ber: float
    diagnostics: dict = field(default_factory=dict)
    rx_symbols: Optional[np.ndarray] = None     # equalised data symbols (for constellation plots)


def simulate_link(mod: str, snr_db: float, h: np.ndarray, cfo: float,
                  fault: FaultConfig, rng: np.random.Generator,
                  n_symbols: int = 20, h_stale: Optional[np.ndarray] = None,
                  keep_symbols: bool = False) -> LinkResult:
    """Transmit one frame through the channel and receive it, returning metrics and diagnostics."""
    m = BITS_PER_SYMBOL[mod]
    n_data = len(DATA)

    # ---------------- Transmitter ----------------
    bits = rng.integers(0, 2, n_symbols * n_data * m, dtype=np.int8)
    tx_data = modulate(bits, mod).reshape(n_symbols, n_data)
    freq_syms = [STS_FREQ, LTS_FREQ]
    for i in range(n_symbols):
        X = np.zeros(N_FFT, dtype=complex)
        X[_bins(DATA)] = tx_data[i]
        X[_bins(PILOTS)] = PILOT_VALUES
        freq_syms.append(X)
    cp = fault.cp_len
    sym_len = N_FFT + cp
    tx = np.concatenate([ofdm_symbol(X, cp) for X in freq_syms])

    if fault.iq_gain_db or fault.iq_phase_deg:              # TX RF I/Q imbalance
        a = 10 ** (fault.iq_gain_db / 20)
        ph = np.deg2rad(fault.iq_phase_deg)
        mu = (1 + a * np.exp(-1j * ph)) / 2
        nu = (1 - a * np.exp(1j * ph)) / 2
        tx = mu * tx + nu * np.conj(tx)

    # ---------------- Channel ----------------
    rx = np.convolve(tx, h)[: len(tx)]
    n = np.arange(len(rx))
    rx = rx * np.exp(1j * 2 * np.pi * cfo * n / N_FFT)
    noise_var = 10 ** (-snr_db / 10)
    rx = rx + np.sqrt(noise_var / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))

    # ---------------- Receiver front end: AGC + ADC ----------------
    agc_gain = 1.0 / np.sqrt(np.mean(np.abs(rx) ** 2))
    r = rx * agc_gain
    mag = np.abs(r)
    sat = mag >= fault.adc_full_scale
    r[sat] = r[sat] / mag[sat] * fault.adc_full_scale
    sat_ratio = float(np.mean(sat))
    r = np.concatenate([r, np.zeros(3 * SYM_LEN, dtype=complex)])   # guard for late windows

    def window(i: int) -> np.ndarray:
        start = i * sym_len + cp + fault.rx_timing_offset
        return r[start: start + N_FFT]

    # ---------------- Synchronisation: CFO estimate from repeated STS halves ----------------
    s = window(0)
    cfo_est = float(np.angle(np.sum(np.conj(s[:32]) * s[32:64])) / np.pi)
    cfo_applied = cfo_est * fault.cfo_correction_gain
    nn = np.arange(len(r))
    r = r * np.exp(-1j * 2 * np.pi * cfo_applied * nn / N_FFT)

    # ---------------- Channel estimation from LTS ----------------
    Y_lts = np.fft.fft(window(1)) / np.sqrt(N_FFT)
    H_est = Y_lts[_bins(USED)] / LTS_FREQ[_bins(USED)]
    if fault.stale_weight > 0 and h_stale is not None:
        H_old = freq_response(h_stale)[_bins(USED)] * agc_gain
        H_est = (1 - fault.stale_weight) * H_est + fault.stale_weight * H_old

    # ---------------- Data: equalise, track pilot phase, demodulate ----------------
    Y = np.stack([np.fft.fft(window(2 + i)) / np.sqrt(N_FFT) for i in range(n_symbols)])
    Z = Y[:, _bins(USED)] / H_est[None, :]
    cpe = np.angle(np.sum(Z[:, PILOT_IN_USED] * np.conj(PILOT_VALUES)[None, :], axis=1))
    Z = Z * np.exp(-1j * cpe)[:, None]
    rx_data = Z[:, DATA_IN_USED]
    rx_bits = demodulate(rx_data.reshape(-1), mod)
    ber = float(np.mean(rx_bits != bits))

    # ---------------- Metrics and diagnostics ----------------
    err = rx_data - tx_data
    err_pow = np.abs(err) ** 2
    evm_lin = np.sqrt(np.mean(err_pow) / np.mean(np.abs(tx_data) ** 2))
    evm_pct = float(100 * evm_lin)
    evm_db = float(20 * np.log10(max(evm_lin, 1e-6)))

    sym_idx = np.arange(n_symbols)
    per_sym_db = 10 * np.log10(np.mean(err_pow, axis=1) + 1e-12)
    evm_slope = float(np.polyfit(sym_idx, per_sym_db, 1)[0])            # dB per symbol
    cpe_slope = float(np.polyfit(sym_idx, np.unwrap(cpe), 1)[0])        # rad per symbol

    per_sc = np.mean(err_pow, axis=0)
    sc_spread = float(np.std(per_sc) / (np.mean(per_sc) + 1e-12))      # frequency selectivity of error

    a_idx, b_idx = MIRROR_PAIRS[:, 0], MIRROR_PAIRS[:, 1]
    e_k = err[:, a_idx]
    x_mk = tx_data[:, b_idx]
    image_corr = float(np.abs(np.sum(e_k * x_mk)) /
                       np.sqrt(np.sum(np.abs(e_k) ** 2) * np.sum(np.abs(x_mk) ** 2) + 1e-12))

    neg = H_est[:26]
    pos = H_est[26:]
    rot = np.sum(neg[1:] * np.conj(neg[:-1])) + np.sum(pos[1:] * np.conj(pos[:-1]))
    group_delay = float(-np.angle(rot) * N_FFT / (2 * np.pi))          # samples

    pilot_err = Z[:, PILOT_IN_USED] - PILOT_VALUES[None, :]
    pilot_evm_db = float(10 * np.log10(np.mean(np.abs(pilot_err) ** 2) + 1e-12))
    h_mag = np.abs(H_est) / (np.mean(np.abs(H_est)) + 1e-12)
    chan_min_db = float(20 * np.log10(np.min(h_mag) + 1e-6))

    diagnostics = {
        "cfo_est": cfo_est,
        "sat_ratio": sat_ratio,
        "evm_slope_db": evm_slope,
        "cpe_slope_rad": cpe_slope,
        "sc_spread": sc_spread,
        "image_corr": image_corr,
        "group_delay": group_delay,
        "pilot_evm_db": pilot_evm_db,
        "chan_min_db": chan_min_db,
    }
    return LinkResult(mod=mod, snr_db=snr_db, evm_pct=evm_pct, evm_db=evm_db, ber=ber,
                      diagnostics=diagnostics, rx_symbols=rx_data.reshape(-1) if keep_symbols else None)
