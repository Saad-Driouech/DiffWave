"""
Save spectrograms of a few down-chirp samples from the dataset.

Uses /tmp/diagnose_dataset.npz (produced by diagnose_dataset.py) to pick
samples with slope < 0, then plots and saves their 4-antenna spectrograms.

Run:  python tests/save_down_chirp_samples.py [--n 4] [--output_dir ./down_chirps]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from UniversalDataLoader import UniversalDataset
from utils.visualization import _gnss_spectrogram_db


def plot_sample(sig, idx, slope, f_start, az_deg, el_deg, out_path):
    """sig: complex [4, L].  Save a 4-row spectrogram figure."""
    fig, axes = plt.subplots(4, 1, figsize=(7, 12), sharex=True)
    for a in range(4):
        f, t, Sxx_db = _gnss_spectrogram_db(sig[a].numpy())
        im = axes[a].pcolormesh(t * 1e3, f, Sxx_db, shading='auto', cmap='turbo',
                                vmin=-140, vmax=-60)
        axes[a].set_ylabel(f'Antenna {a+1}\nf [Hz]')
        fig.colorbar(im, ax=axes[a], label='dB-Hz')
    axes[-1].set_xlabel('t [ms]')
    fig.suptitle(f'Down-chirp sample idx={idx}\n'
                 f'slope={slope:+.2e} Hz/s  f_start={f_start:+.2e} Hz  '
                 f'az={az_deg:+.1f}°  el={el_deg:+.1f}°',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, default='/tmp/diagnose_dataset.npz')
    ap.add_argument('--task', type=int, default=132)
    ap.add_argument('--mode', type=str, default='train')
    ap.add_argument('--n', type=int, default=4, help='number of samples to save')
    ap.add_argument('--output_dir', type=str, default='./down_chirps')
    ap.add_argument('--strongest', action='store_true',
                    help='pick samples with the most negative slopes (default: random)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    d = np.load(args.npz)
    idx_all   = d['idx']
    slope_all = d['slope']
    fstart    = d['f_start']
    az_all    = d['az_deg']
    el_all    = d['el_deg']

    down_mask = slope_all < 0
    print(f'down-chirps in npz: {down_mask.sum()} / {len(slope_all)}')

    cand = np.where(down_mask)[0]
    if args.strongest:
        order = np.argsort(slope_all[cand])  # most negative first
        cand = cand[order]
    else:
        np.random.default_rng(0).shuffle(cand)
    picked = cand[:args.n]

    ds = UniversalDataset(task_id=args.task, mode=args.mode, angle_mode='sincos')

    for j, k in enumerate(picked):
        ds_idx = int(idx_all[k])
        sig, _ = ds[ds_idx]
        out = os.path.join(args.output_dir,
                           f'down_chirp_{j:02d}_idx{ds_idx}.png')
        plot_sample(sig, ds_idx, slope_all[k], fstart[k],
                    az_all[k], el_all[k], out)
        print(f'  saved {out}  (slope={slope_all[k]:+.2e})')


if __name__ == '__main__':
    main()
