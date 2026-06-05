"""
Extract the sawtooth chirp phase offset for each sample, and test whether the
conditioning vector (pos, az, el) predicts it.

Self-calibrating: the chirp slope, observed bandwidth, and period are
estimated from the data itself rather than hardcoded.

Algorithm
---------
Pass 1 (per sample):
  * Compute spectrogram (same settings as utils.visualization).
  * Track ridge via argmax.
  * Detect wraps as ridge drops larger than DETECT_THRESH_HZ (set well above
    the local chirp slope but well below the chirp BW).
  * For each contiguous wrap-free segment, fit a line; record slope, intercept,
    and the f_min/f_max of the ridge across the segment.

Calibration (across samples):
  * SLOPE  = median of positive segment slopes (the true chirp rate is a
             constant; medians are robust to outliers from short segments).
  * BW     = (global max ridge value)  -  (global min ridge value).
  * PERIOD = BW / SLOPE.

Pass 2 (per sample):
  * Unwrap the ridge using detected wrap positions (adds BW after each wrap).
  * Fit a line; intercept b gives phase_offset = (b - F_LOW) / SLOPE,
    reduced mod PERIOD.

Then we test whether (pos_x, pos_y, pos_z, az, el) predicts phase_offset
by correlating its sin/cos encoding with the conditioning features.

Run:  python tests/extract_chirp_phase.py [--n 2000]
"""

import argparse
import numpy as np
from scipy.signal import spectrogram

from UniversalDataLoader import UniversalDataset

FS_HZ = 40.5e6                  # matches utils.visualization._GNSS_FS
DETECT_THRESH_HZ = 2.5e6        # ridge drop > 2.5 MHz ⇒ wrap. Chosen to be
                                # well above per-step slope·dt and well below
                                # a typical sawtooth reset.


# ── per-sample raw ridge extraction ───────────────────────────────────────────
def ridge_and_segments(iq):
    """Return (t, ridge, wrap_idx) for a single antenna IQ trace."""
    f, t, Sxx = spectrogram(iq, fs=FS_HZ, nperseg=128, noverlap=64,
                            window='blackman', return_onesided=False,
                            detrend=False, mode='psd')
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    ridge = f[np.argmax(Sxx, axis=0)]
    df = np.diff(ridge)
    wrap_idx = np.where(df < -DETECT_THRESH_HZ)[0]
    return t, ridge, wrap_idx


def segment_fits(t, ridge, wrap_idx):
    """Yield (slope, intercept, f_min, f_max, length) for each wrap-free segment."""
    boundaries = [0] + list(wrap_idx + 1) + [len(ridge)]
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s < 3:
            continue
        ts, rs = t[s:e], ridge[s:e]
        A = np.vstack([ts, np.ones_like(ts)]).T
        coef, *_ = np.linalg.lstsq(A, rs, rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        yield slope, intercept, float(rs.min()), float(rs.max()), e - s


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, default=132)
    ap.add_argument('--mode', type=str, default='train')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='/tmp/chirp_phase.npz')
    args = ap.parse_args()

    ds = UniversalDataset(task_id=args.task, mode=args.mode, angle_mode='sincos')
    N = len(ds)
    print(f'Dataset: task {args.task}, mode {args.mode}, N={N}')

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)

    # ── Pass 1: per-sample raw ridge data + segment slopes ────────────────────
    print('\nPass 1: extracting per-sample ridges and wrap-free segment slopes')
    raw = []                    # one entry per sample (avg over 4 antennas)
    all_seg_slopes = []
    global_f_min, global_f_max = np.inf, -np.inf

    for k, i in enumerate(idx):
        if k % 200 == 0:
            print(f'  {k}/{len(idx)}')
        sig, (p, az_sc, el_sc) = ds[int(i)]
        per_ant_rid = []
        per_ant_wrap = []
        per_ant_t = None
        for a in range(sig.shape[0]):
            t, ridge, wrap = ridge_and_segments(sig[a].numpy())
            per_ant_rid.append(ridge)
            per_ant_wrap.append(wrap)
            per_ant_t = t
            for slope, b, fmn, fmx, L in segment_fits(t, ridge, wrap):
                if slope > 0 and L >= 5:        # only credible positive slopes
                    all_seg_slopes.append(slope)
            if ridge.size:
                global_f_min = min(global_f_min, float(ridge.min()))
                global_f_max = max(global_f_max, float(ridge.max()))
        raw.append({
            'idx': int(i),
            't': per_ant_t,
            'ridges': per_ant_rid,
            'wraps':  per_ant_wrap,
            'pos': p.numpy(),
            'az_deg': float(np.degrees(np.arctan2(az_sc[0], az_sc[1]))),
            'el_deg': float(np.degrees(np.arctan2(el_sc[0], el_sc[1]))),
        })

    # ── Calibration ───────────────────────────────────────────────────────────
    all_seg_slopes = np.array(all_seg_slopes)
    SLOPE_HZ_S = float(np.median(all_seg_slopes))
    BW_HZ      = global_f_max - global_f_min
    T_CHIRP_S  = BW_HZ / SLOPE_HZ_S
    F_LOW_HZ   = global_f_min
    L_WIN_S    = float(raw[0]['t'][-1] - raw[0]['t'][0])

    print('\n── self-calibrated chirp constants ──')
    print(f'  slope           = {SLOPE_HZ_S:.3e} Hz/s   '
          f'(median of {len(all_seg_slopes)} wrap-free segments)')
    print(f'  slope IQR       = [{np.percentile(all_seg_slopes, 25):.3e}, '
          f'{np.percentile(all_seg_slopes, 75):.3e}] Hz/s')
    print(f'  ridge f range   = [{global_f_min:+.2e}, {global_f_max:+.2e}] Hz '
          f'(BW ≈ {BW_HZ:.2e} Hz)')
    print(f'  derived period  = {T_CHIRP_S*1e6:.2f} µs')
    print(f'  window length   = {L_WIN_S*1e6:.2f} µs')
    print(f'  wrap visible iff phase > {(T_CHIRP_S - L_WIN_S)*1e6:.2f} µs '
          f'(if T > L)' if T_CHIRP_S > L_WIN_S else '  T ≤ L ⇒ every window contains a wrap')
    expected_wrap_frac = min(1.0, L_WIN_S / T_CHIRP_S)
    print(f'  expected wrap fraction (uniform φ): {expected_wrap_frac*100:.1f}%')

    # ── Pass 2: phase per sample using calibrated constants ───────────────────
    print('\nPass 2: extracting phase offset per sample using calibrated constants')
    phase    = np.zeros(len(raw))
    has_wrap = np.zeros(len(raw), dtype=bool)
    wrap_t   = np.zeros(len(raw))
    residual = np.zeros(len(raw))
    pos      = np.zeros((len(raw), 3))
    az_deg   = np.zeros(len(raw))
    el_deg   = np.zeros(len(raw))

    for k, r in enumerate(raw):
        t = r['t']
        ant_phases, ant_resid, ant_wrap_t, ant_has_wrap = [], [], [], []
        for ridge, wrap in zip(r['ridges'], r['wraps']):
            unwrapped = ridge.copy()
            if len(wrap) > 0:
                ant_has_wrap.append(True)
                ant_wrap_t.append(float(t[wrap[0] + 1]))
                for w in wrap:
                    unwrapped[w + 1:] += BW_HZ
            else:
                ant_has_wrap.append(False)
                ant_wrap_t.append(np.nan)
            A = np.vstack([t, np.ones_like(t)]).T
            coef, *_ = np.linalg.lstsq(A, unwrapped, rcond=None)
            b = float(coef[1])
            phi = ((b - F_LOW_HZ) / SLOPE_HZ_S) % T_CHIRP_S
            ant_phases.append(phi)
            ant_resid.append(float(np.std(unwrapped - (coef[0] * t + b))))

        # circular mean of antenna phases (modular variable)
        ang = np.array(ant_phases) / T_CHIRP_S * 2 * np.pi
        circ_mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
        phase[k]    = circ_mean / (2 * np.pi) * T_CHIRP_S
        has_wrap[k] = any(ant_has_wrap)
        wrap_t[k]   = np.nanmean(ant_wrap_t) if has_wrap[k] else np.nan
        residual[k] = float(np.mean(ant_resid))
        pos[k]      = r['pos']
        az_deg[k]   = r['az_deg']
        el_deg[k]   = r['el_deg']

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f'\nresidual ridge-fit std: mean={residual.mean():.2e} Hz   '
          f'median={np.median(residual):.2e} Hz   max={residual.max():.2e} Hz')
    print(f'  (compare to BW={BW_HZ:.2e} Hz; should be <<)')

    print(f'\nwrap fraction observed: {has_wrap.mean()*100:.1f}%   '
          f'(expected {expected_wrap_frac*100:.1f}%)')

    print('\n── phase offset histogram (should be ~uniform on [0, period)) ──')
    bins = np.linspace(0, T_CHIRP_S, 21)
    hist, _ = np.histogram(phase, bins=bins)
    for h, lo, hi in zip(hist, bins[:-1], bins[1:]):
        bar = '#' * int(40 * h / max(hist.max(), 1))
        print(f'  [{lo*1e6:6.2f}, {hi*1e6:6.2f}] µs   {h:5d}  {bar}')

    s = np.sin(phase / T_CHIRP_S * 2 * np.pi)
    c = np.cos(phase / T_CHIRP_S * 2 * np.pi)
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

    print('\n── does the condition predict the *presence* of a wrap? ──')
    if has_wrap.std() == 0:
        print('  (all samples wrap or none do — correlation undefined)')
    else:
        for name, x in feats.items():
            r = np.corrcoef(x, has_wrap.astype(float))[0, 1]
            print(f'  corr({name:8s}, has_wrap) = {r:+.3f}')

    np.savez(args.out,
             idx=[r['idx'] for r in raw],
             phase=phase, has_wrap=has_wrap, wrap_time=wrap_t,
             residual=residual, pos=pos, az_deg=az_deg, el_deg=el_deg,
             T_chirp=T_CHIRP_S, L_win=L_WIN_S, slope=SLOPE_HZ_S, bw=BW_HZ,
             f_low=F_LOW_HZ)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
