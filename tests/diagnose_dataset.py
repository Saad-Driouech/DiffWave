"""
Thorough dataset diagnostic for the chirp-discontinuity failure mode.

The model produces 'two-piece' chirps: line starts at one frequency, jumps to
another mid-signal. This script tests hypotheses for why:

  H1  Conditioning ambiguity: samples with the same (pos, az, el) have
      different chirp parameters (start freq, slope, time offset).
  H2  Multi-modal chirp starts: the distribution of starting frequencies is
      bimodal/multi-modal, not unimodal.
  H3  Multiple chirps per sample (multipath): more than one ridge per
      spectrogram column.
  H4  Per-sample amplitude variation destroyed by z-score: real amplitudes
      span orders of magnitude.
  H5  Hidden discrete factor (e.g. the 5 reflection levels) that is not in
      the condition vector.

Run:  python tests/diagnose_dataset.py [--n 2000] [--task 132]
"""

import argparse
import numpy as np
import torch
from scipy.signal import spectrogram
from collections import defaultdict

from UniversalDataLoader import UniversalDataset


# ── chirp parameter estimation ────────────────────────────────────────────────
def estimate_chirp(iq, fs=4.32e7, nperseg=64, noverlap=48):
    """Estimate (f_start, f_end, slope, n_ridges, ridge_strength) from one antenna IQ.

    iq: complex 1D, length L
    Returns dict with keys: f_start, f_end, slope_hz_per_s, n_ridges_mean,
                            peak_db, snr_db, t_break (None if continuous).
    """
    f, t, Sxx = spectrogram(iq, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            return_onesided=False, scaling='density')
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    # ridge: argmax frequency per time column
    ridge_idx = np.argmax(Sxx_db, axis=0)
    ridge_f = f[ridge_idx]
    ridge_strength = Sxx_db[ridge_idx, np.arange(Sxx_db.shape[1])]

    # noise floor: median of below-peak rows
    noise_db = np.median(Sxx_db)
    snr_db = ridge_strength.mean() - noise_db

    # detect discontinuity: large jump in ridge_f between adjacent columns
    df = np.diff(ridge_f)
    jump_thresh = (f.max() - f.min()) * 0.3
    jumps = np.where(np.abs(df) > jump_thresh)[0]
    t_break = t[jumps[0] + 1] if len(jumps) > 0 else None

    # count ridges per column: peaks above (max - 10 dB) within the column
    n_ridges_per_col = []
    for col in range(Sxx_db.shape[1]):
        col_db = Sxx_db[:, col]
        peak = col_db.max()
        # count well-separated peaks within 10 dB of the max
        above = col_db > peak - 10
        # group adjacent True into runs
        diff = np.diff(above.astype(int))
        n_peaks = max(1, (diff == 1).sum())
        n_ridges_per_col.append(n_peaks)
    n_ridges_mean = float(np.mean(n_ridges_per_col))

    # linear fit on the "clean" segment if no jump, else on first segment
    if t_break is None:
        seg = slice(0, len(ridge_f))
    else:
        seg = slice(0, jumps[0] + 1)
    if seg.stop - seg.start >= 4:
        coef = np.polyfit(t[seg], ridge_f[seg], 1)
        slope = float(coef[0])
        f_start = float(coef[1])
        f_end = float(coef[0] * t[-1] + coef[1])
    else:
        slope = float(np.nan)
        f_start = float(ridge_f[0])
        f_end = float(ridge_f[-1])

    return {
        'f_start': f_start,
        'f_end': f_end,
        'slope': slope,
        'n_ridges_mean': n_ridges_mean,
        'peak_db': float(ridge_strength.mean()),
        'snr_db': float(snr_db),
        't_break': float(t_break) if t_break is not None else None,
    }


# ── grouping by condition ─────────────────────────────────────────────────────
def condition_key(pos, az_deg, el_deg, round_pos=2, round_ang=1):
    """Hashable key for grouping samples with near-identical conditioning."""
    return (
        round(float(pos[0]), round_pos),
        round(float(pos[1]), round_pos),
        round(float(pos[2]), round_pos),
        round(float(az_deg), round_ang),
        round(float(el_deg), round_ang),
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=int, default=132)
    ap.add_argument('--mode', type=str, default='train')
    ap.add_argument('--n', type=int, default=2000, help='subsample size for chirp analysis')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ds = UniversalDataset(task_id=args.task, mode=args.mode, angle_mode='sincos')
    N = len(ds)
    print(f'\nDataset: task {args.task}, mode {args.mode}, N={N}')

    # ── 1. Raw amplitude distribution (H4) ────────────────────────────────────
    print('\n── [H4] amplitude distribution across samples ──')
    rng = np.random.default_rng(args.seed)
    idx_amp = rng.choice(N, size=min(500, N), replace=False)
    amp_means, amp_stds, amp_maxes = [], [], []
    for i in idx_amp:
        sig, _ = ds[int(i)]                       # [4, 1024] complex
        a = sig.abs().numpy()
        amp_means.append(a.mean())
        amp_stds.append(a.std())
        amp_maxes.append(a.max())
    amp_means = np.array(amp_means)
    amp_maxes = np.array(amp_maxes)
    print(f'  mean amplitude across samples : min={amp_means.min():.3e}  '
          f'max={amp_means.max():.3e}  ratio={amp_means.max()/amp_means.min():.1f}x')
    print(f'  peak amplitude across samples : min={amp_maxes.min():.3e}  '
          f'max={amp_maxes.max():.3e}  ratio={amp_maxes.max()/amp_maxes.min():.1f}x')
    print('  → If ratio >> 10x, per-sample z-score destroys absolute scale info '
          'the model could use to disambiguate chirp phase.')

    # ── 2. Chirp parameter estimation on subsample (H1, H2, H3) ───────────────
    print(f'\n── estimating chirp parameters on {min(args.n, N)} samples ──')
    idx_chirp = rng.choice(N, size=min(args.n, N), replace=False)

    records = []
    n_discontinuous = 0
    for k, i in enumerate(idx_chirp):
        if k % 200 == 0:
            print(f'  {k}/{len(idx_chirp)}')
        sig, (pos, az_sc, el_sc) = ds[int(i)]
        # average chirp params across the 4 antennas (they should be the same chirp)
        params = [estimate_chirp(sig[a].numpy()) for a in range(sig.shape[0])]
        f_start = np.mean([p['f_start'] for p in params])
        f_end   = np.mean([p['f_end']   for p in params])
        slope   = np.nanmean([p['slope'] for p in params])
        n_ridges = np.mean([p['n_ridges_mean'] for p in params])
        snr_db   = np.mean([p['snr_db'] for p in params])
        has_break = any(p['t_break'] is not None for p in params)
        if has_break:
            n_discontinuous += 1

        az_deg = float(np.degrees(np.arctan2(az_sc[0], az_sc[1])))
        el_deg = float(np.degrees(np.arctan2(el_sc[0], el_sc[1])))
        records.append({
            'idx': int(i),
            'pos': pos.numpy(),
            'az_deg': az_deg,
            'el_deg': el_deg,
            'f_start': f_start, 'f_end': f_end, 'slope': slope,
            'n_ridges': n_ridges, 'snr_db': snr_db,
            'has_break': has_break,
        })

    f_starts = np.array([r['f_start'] for r in records])
    f_ends   = np.array([r['f_end']   for r in records])
    slopes   = np.array([r['slope']   for r in records])
    n_ridges = np.array([r['n_ridges']for r in records])

    print(f'\n  f_start [Hz]: min={f_starts.min():.2e}  max={f_starts.max():.2e}  '
          f'mean={f_starts.mean():.2e}  std={f_starts.std():.2e}')
    print(f'  f_end   [Hz]: min={f_ends.min():.2e}  max={f_ends.max():.2e}  '
          f'mean={f_ends.mean():.2e}  std={f_ends.std():.2e}')
    print(f'  slope [Hz/s]: min={slopes.min():.2e}  max={slopes.max():.2e}  '
          f'mean={slopes.mean():.2e}  std={slopes.std():.2e}')
    print(f'  ridges/col  : mean={n_ridges.mean():.2f}  max={n_ridges.max():.2f}')
    print(f'  samples with detected discontinuity in REAL data: '
          f'{n_discontinuous}/{len(records)} ({100*n_discontinuous/len(records):.1f}%)')
    print('  → If real samples are discontinuous, the model is faithfully reproducing the dataset.')

    # ── 3. Multi-modality of f_start (H2) ─────────────────────────────────────
    print('\n── [H2] f_start distribution histogram ──')
    bins = np.linspace(f_starts.min(), f_starts.max(), 21)
    hist, _ = np.histogram(f_starts, bins=bins)
    for h, lo, hi in zip(hist, bins[:-1], bins[1:]):
        bar = '#' * int(40 * h / max(hist.max(), 1))
        print(f'  [{lo:+.2e}, {hi:+.2e}]  {h:5d}  {bar}')
    print('  → Bimodal/multi-modal => the model has to choose between modes.')

    # ── 4. Conditioning ambiguity (H1) ────────────────────────────────────────
    print('\n── [H1] within-group variance for samples with near-identical condition ──')
    for rp, ra, label in [(2, 1, 'tight  (pos to 0.01, ang to 0.1°)'),
                          (1, 0, 'loose  (pos to 0.1,  ang to 1°)')]:
        groups = defaultdict(list)
        for r in records:
            key = condition_key(r['pos'], r['az_deg'], r['el_deg'],
                                round_pos=rp, round_ang=ra)
            groups[key].append(r)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        print(f'\n  {label}: {len(multi)} groups with >1 sample, '
              f'{sum(len(v) for v in multi.values())} samples total')
        if not multi:
            continue
        within_fstart = np.array([np.std([r['f_start'] for r in v]) for v in multi.values()])
        within_slope  = np.array([np.nanstd([r['slope'] for r in v]) for v in multi.values()])
        total_fstart  = f_starts.std()
        total_slope   = np.nanstd(slopes)
        print(f'    f_start: within-group std mean = {within_fstart.mean():.2e}  '
              f'(global std = {total_fstart:.2e})  '
              f'ratio = {within_fstart.mean()/total_fstart:.2f}')
        print(f'    slope  : within-group std mean = {within_slope.mean():.2e}  '
              f'(global std = {total_slope:.2e})  '
              f'ratio = {within_slope.mean()/total_slope:.2f}')
        print('    → ratio close to 1 ⇒ condition does not constrain chirp params (ambiguity).')
        print('    → ratio near 0    ⇒ condition fully determines them.')

        # show the worst (most ambiguous) group
        worst_key = max(multi, key=lambda k: np.std([r['f_start'] for r in multi[k]]))
        worst = multi[worst_key]
        print(f'    worst group: condition={worst_key}, n={len(worst)}')
        print(f'      f_start values: {[f"{r['f_start']:+.2e}" for r in worst[:6]]}')
        print(f'      slope   values: {[f"{r['slope']:+.2e}"  for r in worst[:6]]}')

    # ── 5. Multipath / extra ridges (H3) ──────────────────────────────────────
    print('\n── [H3] multi-ridge samples ──')
    multi_ridge = (n_ridges > 1.5).sum()
    print(f'  samples with mean ridges/col > 1.5: {multi_ridge}/{len(records)} '
          f'({100*multi_ridge/len(records):.1f}%)')
    print('  → high count ⇒ multipath in the real data; explains "two-piece" generated chirps.')

    # ── 6. Hidden discrete factor (H5) ────────────────────────────────────────
    print('\n── [H5] checking dataset for non-conditioned attributes ──')
    attrs = [a for a in dir(ds) if not a.startswith('_')]
    interesting = [a for a in attrs if any(k in a.lower() for k in
                   ['reflect', 'level', 'snr', 'time', 'phase', 'offset',
                    'delay', 'path', 'mode', 'label'])]
    print(f'  candidate hidden attributes on dataset object: {interesting}')
    print('  → if any of these vary per sample but are NOT in the condition vector, '
          'the model sees them as noise it cannot explain.')

    # ── 7. Save records for further analysis ──────────────────────────────────
    out = '/tmp/diagnose_dataset.npz'
    np.savez(out,
             idx=[r['idx'] for r in records],
             pos=np.stack([r['pos'] for r in records]),
             az_deg=[r['az_deg'] for r in records],
             el_deg=[r['el_deg'] for r in records],
             f_start=f_starts, f_end=f_ends, slope=slopes,
             n_ridges=n_ridges, has_break=[r['has_break'] for r in records])
    print(f'\nSaved per-sample chirp records to {out}')


if __name__ == '__main__':
    main()
