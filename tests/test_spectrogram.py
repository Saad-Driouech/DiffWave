"""
Test the spectrogram fix: compare old (wrapping) vs new (down-converted) approach.
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


def spectrogram_old(x, fs=_GNSS_FS, noverlap=64):
    """Original — no frequency correction, wraps near Nyquist."""
    f, t, Sxx = scipy.signal.spectrogram(
        x, fs=fs, nperseg=128, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def spectrogram_new(x, fs=_GNSS_FS, noverlap=64):
    """New — down-converts to baseband first so band stays continuous."""
    fft_mag = np.abs(np.fft.fft(x))
    peak_bin = np.argmax(fft_mag[:len(fft_mag) // 2])   # positive-freq peak
    f_carrier = peak_bin / len(x) * fs
    t_vec = np.arange(len(x)) / fs
    x_shifted = x * np.exp(-1j * 2 * np.pi * f_carrier * t_vec)

    f, t, Sxx = scipy.signal.spectrogram(
        x_shifted, fs=fs, nperseg=128, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f) + f_carrier, t, Sxx_db


def main():
    ds = UniversalDataset(task_id=132, mode="test", angle_mode='sincos')
    n_ant = 4
    n_samples = 3   # show 3 different dataset samples

    fig, axes = plt.subplots(n_samples * n_ant, 2,
                             figsize=(14, n_samples * n_ant * 3))
    fig.suptitle("Spectrogram: old (left) vs new/down-converted (right)", fontsize=13)

    for s in range(n_samples):
        x, _ = ds[s]                            # [4, 1024] complex
        x_np = x.numpy()                        # (4, 1024) complex64

        for i in range(n_ant):
            sig = x_np[i]                       # (1024,) complex

            row = s * n_ant + i
            ax_old = axes[row, 0]
            ax_new = axes[row, 1]

            f_old, t_old, S_old = spectrogram_old(sig)
            f_new, t_new, S_new = spectrogram_new(sig)

            ext_old = [t_old[0]*1e3, t_old[-1]*1e3, f_old[0], f_old[-1]]
            ext_new = [t_new[0]*1e3, t_new[-1]*1e3, f_new[0], f_new[-1]]

            vmin = S_old.min(); vmax = S_old.max()

            ax_old.imshow(S_old, aspect='auto', origin='lower', cmap='turbo',
                          vmin=vmin, vmax=vmax, extent=ext_old, interpolation='nearest')
            ax_old.set_title(f"Sample {s+1}, Ant {i+1} — OLD")
            ax_old.set_xlabel("t [ms]"); ax_old.set_ylabel("f [Hz]")

            ax_new.imshow(S_new, aspect='auto', origin='lower', cmap='turbo',
                          vmin=vmin, vmax=vmax, extent=ext_new, interpolation='nearest')
            ax_new.set_title(f"Sample {s+1}, Ant {i+1} — NEW")
            ax_new.set_xlabel("t [ms]"); ax_new.set_ylabel("f [Hz]")

    plt.tight_layout()
    out = "tests/test_spectrogram.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved → {out}")

    # Also print carrier frequency estimate for each antenna of sample 0
    x0, _ = ds[0]
    x0_np = x0.numpy()
    print("\nCarrier frequency estimate per antenna (sample 0):")
    for i in range(n_ant):
        sig = x0_np[i]
        peak_bin = np.argmax(np.abs(np.fft.fft(sig))[:512])
        f_c = peak_bin / 1024 * _GNSS_FS
        print(f"  Antenna {i+1}: {f_c/1e6:.2f} MHz")


if __name__ == "__main__":
    main()
