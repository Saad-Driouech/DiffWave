"""
Verify per-baseline Δφ for the 2×2 planar array against the correct
far-field geometric prediction:

    Δφ_ij = (2π/λ) · (r_j − r_i) · k̂(az, el)
    k̂    = (cos(el)·cos(az),  cos(el)·sin(az),  sin(el))

Compares measured Δφ_ij (from the complex signal) to theory for all
6 antenna pairs of the 4-element array. A correct geometric model and
correct sign convention should give measured ≈ theory (or measured ≈
−theory if the chosen wave-vector convention is reversed; either way
the relationship is linear with slope ±1).

Pass antenna positions on the command line — defaults are zeros so
nothing real ends up in git.

Run (cluster):
  python tests/aoa_geometry_test.py \\
      --ant-x  X0 X1 X2 X3 \\
      --ant-y  Y0 Y1 Y2 Y3 \\
      --fc 1.575e9
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from UniversalDataLoader import UniversalDataset


# === PLACEHOLDERS — fill in via CLI on the cluster, do NOT commit real values
DEFAULT_ANT_X = [0.0, 0.0, 0.0, 0.0]      # metres, length 4
DEFAULT_ANT_Y = [0.0, 0.0, 0.0, 0.0]
DEFAULT_ANT_Z = [0.0, 0.0, 0.0, 0.0]
DEFAULT_CARRIER_HZ = 1.575e9              # GPS L1
# ============================================================================

ANGLE_MODE = 'sincos'
TASK_ID = 132
N_SAMPLES = 500
C_LIGHT = 299_792_458.0


def decode_angle(y_field, mode):
    arr = np.atleast_1d(np.asarray(y_field, dtype=np.float64))
    if mode == 'sincos':
        return float(np.arctan2(arr[0], arr[1]))
    if mode == 'rad':
        return float(arr.flat[0])
    if mode == 'deg':
        return float(np.deg2rad(arr.flat[0]))
    raise ValueError(f'unknown angle_mode: {mode}')


def k_hat(az, el):
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def wrap(x):
    """Wrap angle(s) into (-π, π]."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ant-x', nargs=4, type=float, default=DEFAULT_ANT_X)
    p.add_argument('--ant-y', nargs=4, type=float, default=DEFAULT_ANT_Y)
    p.add_argument('--ant-z', nargs=4, type=float, default=DEFAULT_ANT_Z)
    p.add_argument('--fc', type=float, default=DEFAULT_CARRIER_HZ,
                   help='Carrier frequency in Hz')
    p.add_argument('--n', type=int, default=N_SAMPLES)
    p.add_argument('--split', default='test', choices=['train', 'test'])
    args = p.parse_args()

    pos = np.stack([args.ant_x, args.ant_y, args.ant_z], axis=1)   # [4, 3]
    lam = C_LIGHT / args.fc
    k_mag = 2 * np.pi / lam
    print(f'λ = {lam*100:.2f} cm   (fc = {args.fc/1e9:.4f} GHz)')
    print(f'k = 2π/λ = {k_mag:.3f} rad/m')
    print(f'antenna positions [m]:\n{pos}\n')

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]   # 6 pairs

    ds = UniversalDataset(task_id=TASK_ID, mode=args.split,
                          angle_mode=ANGLE_MODE)
    n = min(args.n, len(ds))
    print(f'Scanning {n} {args.split} samples...\n')

    measured = {pr: [] for pr in pairs}
    theory   = {pr: [] for pr in pairs}

    for s in range(n):
        x, y = ds[s]
        x_np = x.numpy()                               # [4, 1024] complex
        az = decode_angle(y[1], ANGLE_MODE)
        el = decode_angle(y[2], ANGLE_MODE)
        kh = k_hat(az, el)
        for (i, j) in pairs:
            measured[(i, j)].append(
                np.angle(np.mean(x_np[j] * np.conj(x_np[i]))))
            theory[(i, j)].append(k_mag * np.dot(pos[j] - pos[i], kh))

    # ---- per-pair statistics -------------------------------------------------
    print(f'{"pair":>6}  {"baseline (m)":>22}  '
          f'{"sign agree":>11}  {"corr":>7}  {"|err| mean":>11}')
    print('-' * 70)
    for pr in pairs:
        m = np.array(measured[pr])
        t = wrap(np.array(theory[pr]))                # wrap theory to match
        b = pos[pr[1]] - pos[pr[0]]
        mask = np.abs(t) > 0.2
        if mask.sum() < 5:
            sa = corr = err = float('nan')
        else:
            sa   = np.mean(np.sign(m[mask]) == np.sign(t[mask]))
            corr = np.corrcoef(m, t)[0, 1]
            err  = np.mean(np.abs(wrap(m - t)))
        print(f'{str(pr):>6}  {str(np.round(b, 4)):>22}  '
              f'{sa*100:10.1f}%  {corr:+7.3f}  {err:11.3f}')

    # ---- scatter plots -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, pr in zip(axes.ravel(), pairs):
        m = np.array(measured[pr])
        t = wrap(np.array(theory[pr]))
        ax.scatter(t, m, s=6, alpha=0.5)
        lim = np.pi
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8,
                label='measured = +theory')
        ax.plot([-lim, lim], [lim, -lim], 'r--', linewidth=0.8,
                label='measured = −theory')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('theoretical Δφ (rad)')
        ax.set_ylabel('measured Δφ (rad)')
        b = pos[pr[1]] - pos[pr[0]]
        ax.set_title(f'pair {pr}   Δr = {np.round(b, 3)} m')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left')

    plt.suptitle(f'Per-baseline Δφ — measured vs geometric theory '
                 f'({args.split}, n={n})', fontsize=13)
    plt.tight_layout()
    out = 'tests/aoa_geometry_test.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
