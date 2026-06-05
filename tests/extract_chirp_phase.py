"""
Extract the sawtooth chirp phase offset for each sample, and test whether the
conditioning vector (pos, az, el) predicts it.

Physical model
--------------
The interferer is a periodic linear up-chirp sawtooth with:
  period T_chirp ≈ 82 µs   (from slope_max ≈ 2.44e11 Hz/s, BW = 20 MHz)
  slope  α      ≈ 2.44e11 Hz/s     (positive — instantaneous frequency rises)
  f(t_abs) = -10 MHz + α · (t_abs mod T_chirp)

Each sample's window opens at some absolute time t_open and lasts
  L ≈ 25.3 µs   (1024 / fs, fs = 40.5 MHz)
The phase offset is the only hidden parameter:
  φ = t_open mod T_chirp   ∈  [0, T_chirp)
A sawtooth wrap is visible inside the window iff  φ > T_chirp − L.

Extraction
----------
For each sample we track the spectrogram ridge over time, unwrap the
sawtooth jump (a drop of ~20 MHz), fit a straight line to the unwrapped
ridge, and read off φ from the intercept.

Run:  python tests/extract_chirp_phase.py [--n 2000]
"""

import argparse
import os
import numpy as np
from scipy.signal import spectrogram

from UniversalDataLoader import UniversalDataset

# ── physical constants (derived from earlier diagnostics) ─────────────────────
FS_HZ      = 40.5e6           # matches utils.visualization._GNSS_FS
BW_HZ      = 20.0e6           # 20 MHz chirp bandwidth
SLOPE_HZ_S = 2.44e11          # chirp rate (positive)
T_CHIRP_S  = BW_HZ / SLOPE_HZ_S        # ≈ 82 µs
L_WIN_S    = 1024 / FS_HZ              # ≈ 25.3 µs
F_LOW_HZ   = -BW_HZ / 2                # chirp starts at -10 MHz
WRAP_THRESH_HZ = 0.5 * BW_HZ           # ridge drop > 10 MHz ⇒ wrap


def extract_phase_offset(iq):
    """Return (phase_offset_s, has_wrap, wrap_time_s_or_nan, residual_hz).

    Track ridge via argmax, detect a single wrap as a drop > WRAP_THRESH_HZ,
    unwrap, fit a line, recover the intercept-based phase offset.
    """
    f, t, Sxx = spectrogram(iq, fs=FS_HZ, nperseg=128, noverlap=64,
                            window='blackman', return_onesided=False,
                            detrend=False, mode='psd')
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    ridge = f[np.argmax(Sxx, axis=0)]                # [n_t]

    # detect wrap: large negative jump between adjacent columns
    df = np.diff(ridge)
    jump_idx = np.where(df < -WRAP_THRESH_HZ)[0]
    has_wrap = len(jump_idx) > 0
    wrap_time = float(t[jump_idx[0] + 1]) if has_wrap else float('nan')

    # unwrap: add BW to every sample after the first wrap
    ridge_unwrapped = ridge.copy()
    if has_wrap:
        ridge_unwrapped[jump_idx[0] + 1:] += BW_HZ

    # fit a line:  f_unwrapped(t) = SLOPE · t + b
    # with b = F_LOW + SLOPE · phase_offset   ⇒   phase_offset = (b - F_LOW) / SLOPE
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, ridge_unwrapped, rcond=None)
    slope_fit, b = coef
    phase_offset = (b - F_LOW_HZ) / SLOPE_HZ_S
    phase_offset = phase_offset % T_CHIRP_S          # wrap into [0, T_chirp)

    residual = float(np.std(ridge_unwrapped - (slope_fit * t + b)))
    return float(phase_offset), bool(has_wrap), wrap_time, residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, default=132)
    ap.add_argument('--mode', type=str, default='train')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='/tmp/chirp_phase.npz')
    args = ap.parse_args()

    print(f'T_chirp = {T_CHIRP_S*1e6:.2f} µs   L_win = {L_WIN_S*1e6:.2f} µs')
    print(f'wrap visible iff phase_offset > {(T_CHIRP_S - L_WIN_S)*1e6:.2f} µs')
    print(f'expected wrap fraction (uniform φ): {(L_WIN_S/T_CHIRP_S)*100:.1f}%')

    ds = UniversalDataset(task_id=args.task, mode=args.mode, angle_mode='sincos')
    N = len(ds)
    print(f'\nDataset: task {args.task}, mode {args.mode}, N={N}')

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)

    phase     = np.zeros(len(idx))
    has_wrap  = np.zeros(len(idx), dtype=bool)
    wrap_t    = np.zeros(len(idx))
    residual  = np.zeros(len(idx))
    pos       = np.zeros((len(idx), 3))
    az_deg    = np.zeros(len(idx))
    el_deg    = np.zeros(len(idx))

    for k, i in enumerate(idx):
        if k % 200 == 0:
            print(f'  {k}/{len(idx)}')
        sig, (p, az_sc, el_sc) = ds[int(i)]
        # average phase across antennas (same chirp; slight measurement noise)
        per_ant = [extract_phase_offset(sig[a].numpy()) for a in range(sig.shape[0])]
        phases = np.array([x[0] for x in per_ant])
        # circular mean (phase is mod T_chirp)
        ang = phases / T_CHIRP_S * 2 * np.pi
        circ_mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
        phase[k]    = circ_mean / (2 * np.pi) * T_CHIRP_S
        has_wrap[k] = any(x[1] for x in per_ant)
        wrap_t[k]   = np.nanmean([x[2] for x in per_ant])
        residual[k] = np.mean([x[3] for x in per_ant])
        pos[k]      = p.numpy()
        az_deg[k]   = float(np.degrees(np.arctan2(az_sc[0], az_sc[1])))
        el_deg[k]   = float(np.degrees(np.arctan2(el_sc[0], el_sc[1])))

    # ── stats ─────────────────────────────────────────────────────────────────
    print(f'\nresidual ridge-fit std (mean over samples): {residual.mean():.2e} Hz')
    print(f'  → if << BW, our line model fits well')

    print(f'\nwrap fraction observed: {has_wrap.mean()*100:.1f}%   (expected {L_WIN_S/T_CHIRP_S*100:.1f}%)')

    # phase histogram
    print('\n── phase offset distribution (should be ~uniform on [0, T_chirp)) ──')
    bins = np.linspace(0, T_CHIRP_S, 21)
    hist, _ = np.histogram(phase, bins=bins)
    for h, lo, hi in zip(hist, bins[:-1], bins[1:]):
        bar = '#' * int(40 * h / max(hist.max(), 1))
        print(f'  [{lo*1e6:5.2f}, {hi*1e6:5.2f}] µs   {h:5d}  {bar}')

    # correlations of phase with condition (using circular variables for phase)
    s, c = np.sin(phase / T_CHIRP_S * 2 * np.pi), np.cos(phase / T_CHIRP_S * 2 * np.pi)
    feats = {
        'pos_x':  pos[:, 0],
        'pos_y':  pos[:, 1],
        'pos_z':  pos[:, 2],
        '|pos|':  np.linalg.norm(pos, axis=1),
        'az_deg': az_deg,
        'el_deg': el_deg,
    }
    print('\n── correlation of (sin φ, cos φ) with condition features ──')
    print('  (both must be near zero for the condition to carry no phase info)')
    for name, x in feats.items():
        r_s = np.corrcoef(x, s)[0, 1]
        r_c = np.corrcoef(x, c)[0, 1]
        print(f'  {name:8s}: corr(·, sin φ) = {r_s:+.3f}   corr(·, cos φ) = {r_c:+.3f}')

    # also: do the same condition predict whether a wrap is present?
    print('\n── does the condition predict the *presence* of a wrap? ──')
    for name, x in feats.items():
        r = np.corrcoef(x, has_wrap.astype(float))[0, 1]
        print(f'  corr({name:8s}, has_wrap) = {r:+.3f}')

    np.savez(args.out,
             idx=idx, phase=phase, has_wrap=has_wrap, wrap_time=wrap_t,
             residual=residual, pos=pos, az_deg=az_deg, el_deg=el_deg,
             T_chirp=T_CHIRP_S, L_win=L_WIN_S, slope=SLOPE_HZ_S)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
