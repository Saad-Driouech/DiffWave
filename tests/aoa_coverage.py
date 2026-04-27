"""
Scan the dataset and report (azimuth, elevation) coverage.

Tells you the actual θ range the model trains on, so you know what
range to sweep in `log_aoa_regression` (and whether ±80° is realistic
or pure extrapolation).

Run:  python tests/aoa_coverage.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from UniversalDataLoader import UniversalDataset


ANGLE_MODE = 'sincos'
TASK_ID = 132


def decode_angle(y_field, mode):
    """y_field is y[1] (az) or y[2] (el) from a dataset item."""
    arr = np.atleast_1d(np.asarray(y_field, dtype=np.float64))
    if mode == 'sincos':
        return float(np.arctan2(arr[0], arr[1]))   # radians
    if mode == 'rad':
        return float(arr.flat[0])
    if mode == 'deg':
        return float(np.deg2rad(arr.flat[0]))
    raise ValueError(f'unknown angle_mode: {mode}')


def scan(split):
    ds = UniversalDataset(task_id=TASK_ID, mode=split, angle_mode=ANGLE_MODE)
    az, el = [], []
    for i in range(len(ds)):
        _, y = ds[i]
        az.append(decode_angle(y[1], ANGLE_MODE))
        el.append(decode_angle(y[2], ANGLE_MODE))
    return np.array(az), np.array(el)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    print(f'{"split":>5}  {"n":>6}  '
          f'{"az min":>8}  {"az max":>8}  '
          f'{"el min":>8}  {"el max":>8}')
    print('-' * 60)

    for col, split in enumerate(['train', 'test']):
        az, el = scan(split)
        az_d, el_d = np.rad2deg(az), np.rad2deg(el)

        ax = axes[0, col]
        ax.hist(az_d, bins=80, color='C0', alpha=0.75)
        ax.set_title(f'{split} — azimuth distribution (n={len(az)})')
        ax.set_xlabel('azimuth (deg)'); ax.set_ylabel('count')
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        sc = ax.scatter(az_d, el_d, s=4, alpha=0.4)
        ax.set_title(f'{split} — (az, el) coverage')
        ax.set_xlabel('azimuth (deg)'); ax.set_ylabel('elevation (deg)')
        ax.grid(alpha=0.3)

        print(f'{split:>5}  {len(az):>6}  '
              f'{az_d.min():8.2f}  {az_d.max():8.2f}  '
              f'{el_d.min():8.2f}  {el_d.max():8.2f}')

    plt.tight_layout()
    out = 'tests/aoa_coverage.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
