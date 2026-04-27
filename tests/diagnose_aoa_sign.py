"""
Diagnose the AoA sign flip seen in `Physics/AoA_Regression`.

Three possible causes:
  A) Dataset uses opposite convention (theory line wrong, model correct)
  B) Measurement code has swapped conjugate (plot wrong, dataset+model correct)
  C) Model learned wrong sign (dataset+measurement correct, model wrong)

This script applies the SAME measurement used in
`DiffusionVisualizer.log_aoa_regression` to REAL samples whose true theta
is known. It then reports which case fits.

Run:  python tests/diagnose_aoa_sign.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from UniversalDataLoader import UniversalDataset


N_SAMPLES = 200            # how many test samples to scan
N_ANTENNAS = 4
ANGLE_MODE = 'sincos'


def decode_theta(y_cond, mode):
    """Return θ in radians regardless of how the dataset encoded it.

    mode='sincos' → y_cond is [sin θ, cos θ]
    mode='rad'    → y_cond is θ in radians (scalar or 1-element)
    mode='deg'    → y_cond is θ in degrees (scalar or 1-element)
    """
    arr = np.atleast_1d(np.asarray(y_cond, dtype=np.float64))
    if mode == 'sincos':
        return float(np.arctan2(arr[0], arr[1]))
    if mode == 'rad':
        return float(arr.flat[0])
    if mode == 'deg':
        return float(np.deg2rad(arr.flat[0]))
    raise ValueError(f'unknown angle_mode: {mode}')


def measure_dphi_forward(x_complex):
    """Δφ as defined in log_aoa_regression: angle(ant_{k+1} * conj(ant_k))."""
    diffs = []
    for k in range(N_ANTENNAS - 1):
        a_k   = x_complex[k]
        a_kp1 = x_complex[k + 1]
        diffs.append(np.angle(np.mean(a_kp1 * np.conj(a_k))))
    return float(np.mean(diffs))


def measure_dphi_reversed(x_complex):
    """Same but with conjugate swapped — to check sign of the measurement."""
    diffs = []
    for k in range(N_ANTENNAS - 1):
        a_k   = x_complex[k]
        a_kp1 = x_complex[k + 1]
        diffs.append(np.angle(np.mean(np.conj(a_kp1) * a_k)))
    return float(np.mean(diffs))


def main():
    ds = UniversalDataset(task_id=132, mode='test', angle_mode=ANGLE_MODE)
    n = min(N_SAMPLES, len(ds))
    print(f'Scanning {n} test samples...\n')

    thetas, dphi_fwd, dphi_rev, theory = [], [], [], []
    for i in range(n):
        x, y = ds[i]
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x  # [4, 1024] complex
        theta = decode_theta(y[1], ANGLE_MODE)
        thetas.append(theta)
        dphi_fwd.append(measure_dphi_forward(x_np))
        dphi_rev.append(measure_dphi_reversed(x_np))
        theory.append(np.pi * np.sin(theta))

    thetas    = np.asarray(thetas)
    dphi_fwd  = np.asarray(dphi_fwd)
    dphi_rev  = np.asarray(dphi_rev)
    theory    = np.asarray(theory)

    # --- Sign correlation: forward measurement vs theory ----------------------
    # Use sign agreement on samples where |theory| > 0.3 rad (avoid noise near 0)
    mask = np.abs(theory) > 0.3
    if mask.sum() < 5:
        print('Too few non-broadside samples to judge sign. Aborting.')
        return

    agree_fwd = np.mean(np.sign(dphi_fwd[mask]) == np.sign(theory[mask]))
    agree_rev = np.mean(np.sign(dphi_rev[mask]) == np.sign(theory[mask]))

    # --- Magnitude check ------------------------------------------------------
    # If dataset matches theory in magnitude, |measured| ≈ |theory|
    mag_corr_fwd = np.corrcoef(np.abs(dphi_fwd), np.abs(theory))[0, 1]
    err_fwd      = np.mean(np.abs(np.abs(dphi_fwd) - np.abs(theory)))

    print('=' * 60)
    print('REAL DATA — measurement vs theory (π·sin θ)')
    print('=' * 60)
    print(f'forward   Δφ = angle(ant_{{k+1}} * conj(ant_k))')
    print(f'  sign agreement with theory : {agree_fwd*100:5.1f}%')
    print(f'  |measured| vs |theory| corr: {mag_corr_fwd:+.3f}')
    print(f'  mean |·| absolute error    : {err_fwd:.3f} rad')
    print()
    print(f'reversed  Δφ = angle(conj(ant_{{k+1}}) * ant_k)')
    print(f'  sign agreement with theory : {agree_rev*100:5.1f}%')
    print()

    # --- Verdict --------------------------------------------------------------
    print('=' * 60)
    print('VERDICT')
    print('=' * 60)

    if agree_fwd > 0.85:
        print('Real data matches theory with the FORWARD measurement.')
        print('  → Dataset convention agrees with theoretical line.')
        print('  → Measurement code is correct.')
        print('  → The flip seen at training time is CASE C: the MODEL')
        print('    learned a sign-inverted mapping. Investigate the')
        print('    conditioning path (DiffWaveRF.cond_mlp / how y[1] is used).')
    elif agree_fwd < 0.15:
        print('Real data is the SIGN-INVERSE of the theoretical line.')
        print('  → Dataset uses the OPPOSITE Δφ convention.')
        print('  → Measurement code is also using that opposite convention.')
        print('  → This is CASE A: the model is correct (it matches the')
        print('    data); the red theoretical line in log_aoa_regression')
        print('    is the wrong sign. Fix it by negating either:')
        print('      theoretical = -np.pi * np.sin(np.deg2rad(angles_deg))')
        print('    OR swap the conjugate in measure_dphi to flip the')
        print('    measurement sign so the green curve matches the red.')
        if agree_rev > 0.85:
            print('  (Confirmed: the REVERSED measurement matches theory,')
            print('   so swapping the conj order is a one-line fix.)')
    else:
        print(f'Sign agreement is ambiguous ({agree_fwd*100:.1f}%).')
        print('  → Likely CASE B: a measurement bug or per-sample averaging')
        print('    artifact. Inspect a few samples manually.')

    if mag_corr_fwd < 0.5:
        print()
        print('NOTE: |measured| vs |theory| correlation is weak '
              f'({mag_corr_fwd:.2f}).')
        print('   Even after fixing sign, the magnitude relationship is off.')
        print('   Antenna spacing may not be λ/2, or the sin/cos encoding')
        print('   in y[1] does not encode the AoA you assume.')

    # --- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(thetas)
    ax.scatter(np.rad2deg(thetas), dphi_fwd, s=8, alpha=0.5,
               label='measured (forward)')
    ax.scatter(np.rad2deg(thetas), dphi_rev, s=8, alpha=0.5,
               label='measured (reversed conj)')
    ax.plot(np.rad2deg(thetas[order]), theory[order], 'r--',
            label='theory: π·sin(θ)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('decoded θ from y[1] (degrees)')
    ax.set_ylabel('Δφ (rad)')
    ax.set_title('REAL data: measured Δφ vs theoretical π·sin(θ)')
    ax.grid(alpha=0.3)
    ax.legend()
    out = 'tests/diagnose_aoa_sign.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nSaved scatter → {out}')


if __name__ == '__main__':
    main()
