"""Torch-based chirp-phase extractor used as a conditioning signal.

The interferer is a periodic linear up-chirp sawtooth. The only hidden
variable not captured by (pos, az, el) is the phase offset into the chirp
cycle at which the recording window opens. Each sample is encoded as
(sin(2π φ/T), cos(2π φ/T)) and appended to the condition vector.

Calibrated constants come from tests/extract_chirp_phase.py (self-calibrated
on 2000 training samples; residual to the linear chirp model is 1.18% of BW).
"""

import math
import torch

# ── calibrated chirp constants ────────────────────────────────────────────────
FS_HZ            = 40.5e6        # sampling rate, matches utils.visualization._GNSS_FS
SLOPE_HZ_S       = 2.002e11      # median of per-segment slope fits
BW_HZ            = 6.64e6        # observed bandwidth at receiver
T_CHIRP_S        = 33.19e-6      # BW / SLOPE
F_LOW_HZ         = -3.48e6       # observed lower bound of ridge
DETECT_THRESH_HZ = 2.5e6         # ridge drop above this counts as a wrap

# ── STFT settings (mirror utils.visualization._gnss_spectrogram_db) ───────────
_N_FFT = 128
_NOVER = 64
_HOP   = _N_FFT - _NOVER         # 64


def extract_phase_sincos(iq):
    """Compute per-sample chirp phase, encoded as (sin, cos) of the angle 2π φ/T.

    Args:
        iq: complex tensor of shape [B, C, L] (batch, antennas, samples).

    Returns:
        cond: float tensor of shape [B, 2]. Antennas are averaged via a
              circular mean since they share the chirp phase up to a tiny
              AoA delay (≪ T).
    """
    B, C, L = iq.shape
    device = iq.device
    flat = iq.reshape(B * C, L)

    window = torch.blackman_window(_N_FFT, device=device, dtype=torch.float32)
    stft = torch.stft(
        flat,
        n_fft=_N_FFT,
        hop_length=_HOP,
        win_length=_N_FFT,
        window=window,
        center=False,
        normalized=False,
        onesided=False,
        return_complex=True,
    )                                    # [B*C, _N_FFT, n_frames]
    Sxx = torch.fft.fftshift(stft.abs().pow(2), dim=1)

    freqs = torch.fft.fftshift(
        torch.fft.fftfreq(_N_FFT, d=1.0 / FS_HZ, device=device)
    ).to(torch.float32)                  # [_N_FFT]
    n_frames = Sxx.shape[-1]
    t = (torch.arange(n_frames, device=device, dtype=torch.float32) * _HOP
         + _N_FFT / 2.0) / FS_HZ         # [n_frames]

    ridge_idx = Sxx.argmax(dim=1)        # [B*C, n_frames]
    ridge_f   = freqs[ridge_idx]         # [B*C, n_frames]

    # detect sawtooth wraps and unwrap by adding BW after each wrap
    df = ridge_f[:, 1:] - ridge_f[:, :-1]
    wrap_mask = (df < -DETECT_THRESH_HZ).to(torch.float32)
    wrap_cum = torch.zeros_like(ridge_f)
    wrap_cum[:, 1:] = wrap_mask.cumsum(dim=1)
    ridge_unwrapped = ridge_f + BW_HZ * wrap_cum

    # fit only the intercept; trust the calibrated slope
    b = (ridge_unwrapped - SLOPE_HZ_S * t.unsqueeze(0)).mean(dim=1)
    phase_s = (b - F_LOW_HZ) / SLOPE_HZ_S         # any real
    ang = phase_s / T_CHIRP_S * 2.0 * math.pi     # mod 2π implicitly via sin/cos

    sin_phi = ang.sin().reshape(B, C)
    cos_phi = ang.cos().reshape(B, C)
    sin_mean = sin_phi.mean(dim=1)
    cos_mean = cos_phi.mean(dim=1)
    norm = (sin_mean * sin_mean + cos_mean * cos_mean).sqrt().clamp(min=1e-8)
    return torch.stack([sin_mean / norm, cos_mean / norm], dim=1)   # [B, 2]
