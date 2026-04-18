"""
Definitive diagnostic: is the spectrogram step real or a processing artifact?

Shows 3 columns for sample 0, all 4 antennas:
  col 1 — Raw complex signal (zero processing)
  col 2 — Per-sample z-score normalized (training pipeline)
  col 3 — Instantaneous frequency over time

If col 1 has the step → it's physical (chirp crossing DC).
If col 1 is smooth but col 2 has the step → normalization is the cause.
Col 3 shows whether the frequency smoothly crosses 0 Hz.

Run:  python tests/diagnose_step.py
"""

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from UniversalDataLoader import UniversalDataset

FS = 40.5e6
NPERSEG = 128
NOVERLAP = 64
N_ANT = 4


def spectrogram_db(sig, noverlap=NOVERLAP):
    f, t, Sxx = scipy.signal.spectrogram(
        sig, fs=FS, nperseg=NPERSEG, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def instantaneous_freq(sig):
    """Instantaneous frequency from complex signal via d(phase)/dt."""
    phase = np.unwrap(np.angle(sig))
    freq = np.diff(phase) / (2 * np.pi) * FS
    return freq


def main():
    ds = UniversalDataset(task_id=132, mode='test', angle_mode='sincos')
    x_complex, _ = ds[0]  # [4, 1024] complex

    # Raw complex per antenna
    raw = x_complex.numpy()  # [4, 1024] complex

    # Normalized (training pipeline): split I/Q, per-sample z-score, recombine
    I = x_complex.real.float()
    Q = x_complex.imag.float()
    inp = torch.cat([I, Q], dim=0)  # [8, 1024]
    mean = inp.mean()
    std = inp.std() + 1e-8
    inp_norm = ((inp - mean) / std).numpy()
    norm = inp_norm[:N_ANT] + 1j * inp_norm[N_ANT:]  # [4, 1024] complex

    fig, axes = plt.subplots(N_ANT, 3, figsize=(18, N_ANT * 4))

    for ant in range(N_ANT):
        # Col 1: Raw spectrogram
        f, t, S_raw = spectrogram_db(raw[ant])
        extent = [t[0]*1e3, t[-1]*1e3, f[0], f[-1]]
        ax = axes[ant, 0]
        im = ax.imshow(S_raw, aspect='auto', origin='lower', cmap='turbo',
                       extent=extent, interpolation='nearest')
        ax.set_title(f'Ant {ant+1} — RAW (no processing)', fontsize=10)
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('f [Hz]')
        fig.colorbar(im, ax=ax, format='%+.0f dB')

        # Col 2: Normalized spectrogram
        f, t, S_norm = spectrogram_db(norm[ant])
        ax = axes[ant, 1]
        im = ax.imshow(S_norm, aspect='auto', origin='lower', cmap='turbo',
                       extent=extent, interpolation='nearest')
        ax.set_title(f'Ant {ant+1} — NORMALIZED (training pipeline)', fontsize=10)
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('f [Hz]')
        fig.colorbar(im, ax=ax, format='%+.0f dB')

        # Col 3: Instantaneous frequency
        inst_f = instantaneous_freq(raw[ant])
        ax = axes[ant, 2]
        ax.plot(np.arange(len(inst_f)) / FS * 1e3, inst_f / 1e6,
                linewidth=0.5, alpha=0.8)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='DC (0 Hz)')
        ax.set_title(f'Ant {ant+1} — Instantaneous Frequency', fontsize=10)
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('Freq [MHz]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'DIAGNOSTIC: Is the spectrogram step real or an artifact?\n'
        'If col 1 (raw) has the step → it\'s physical (chirp crosses DC)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = 'tests/diagnose_step.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')

    # Also print summary
    for ant in range(N_ANT):
        inst_f = instantaneous_freq(raw[ant])
        crosses_dc = np.any(np.diff(np.sign(inst_f)) != 0)
        print(f'  Ant {ant+1}: freq range [{inst_f.min()/1e6:.2f}, {inst_f.max()/1e6:.2f}] MHz, '
              f'crosses DC: {crosses_dc}')


if __name__ == '__main__':
    main()
