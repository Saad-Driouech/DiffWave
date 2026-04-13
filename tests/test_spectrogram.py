"""
Test spectrogram time-resolution fix.
The signal is near DC — the "broken diagonal" is caused by only 15 time frames
(nperseg=128, noverlap=64). Increasing noverlap gives more frames → smoother band.

Compares three overlap settings side by side.

Run with:  python tests/test_spectrogram.py
Saves:     tests/test_spectrogram.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
from UniversalDataLoader import UniversalDataset

_GNSS_FS = 40.5e6

VARIANTS = [
    ("noverlap=64  (~15 frames)",  64),   # current / broken
    ("noverlap=112 (~57 frames)", 112),   # moderate
    ("noverlap=120 (~113 frames)", 120),  # high — target
]


def compute_spectrogram(x, fs=_GNSS_FS, noverlap=64):
    f, t, Sxx = scipy.signal.spectrogram(
        x, fs=fs, nperseg=128, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def main():
    ds = UniversalDataset(task_id=132, mode="test", angle_mode='sincos')
    n_ant = 4
    n_samples = 2

    n_cols = len(VARIANTS)
    n_rows = n_samples * n_ant

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    fig.suptitle("Spectrogram time-resolution comparison (nperseg=128)", fontsize=13)

    for s in range(n_samples):
        x, _ = ds[s]
        x_np = x.numpy()  # (4, 1024) complex

        for i in range(n_ant):
            sig = x_np[i]
            row = s * n_ant + i

            spectrograms = [compute_spectrogram(sig, noverlap=nov) for _, nov in VARIANTS]
            vmin = min(S.min() for _, _, S in spectrograms)
            vmax = max(S.max() for _, _, S in spectrograms)

            for col, ((label, _), (f, t, S)) in enumerate(zip(VARIANTS, spectrograms)):
                ax = axes[row, col]
                extent = [t[0]*1e3, t[-1]*1e3, f[0], f[-1]]
                ax.imshow(S, aspect='auto', origin='lower', cmap='turbo',
                          vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
                ax.set_title(f"S{s+1} Ant{i+1} — {label}", fontsize=8)
                ax.set_xlabel("t [ms]", fontsize=7)
                ax.set_ylabel("f [Hz]", fontsize=7)
                n_frames = S.shape[1]
                ax.set_title(f"S{s+1} Ant{i+1} — {label}\n({n_frames} time frames)", fontsize=8)

    plt.tight_layout()
    out = "tests/test_spectrogram.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved → {out}")
    for label, nov in VARIANTS:
        n_frames = (1024 - nov) // (128 - nov)
        print(f"  {label}: {n_frames} time frames")


if __name__ == "__main__":
    main()
