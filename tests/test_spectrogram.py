"""
Test: does per-batch normalization cause the frequency step in spectrograms?

Shows the same sample 3 ways:
  col 1 — raw (no normalization)
  col 2 — per-sample z-score (normalize each sample independently)
  col 3 — per-batch z-score (what training currently does)

If col 1 & 2 are smooth and col 3 has the step → normalization is the cause.
If all three have the step → the step is real, in the signal itself.

Run with:  python tests/test_spectrogram.py
Saves:     tests/test_spectrogram.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import torch
from UniversalDataLoader import UniversalDataset
from torch.utils.data import DataLoader

_GNSS_FS = 40.5e6
NOVERLAP  = 120   # fixed at the value we confirmed works
NPERSEG   = 128
N_ANT     = 4
N_SAMPLES = 3     # dataset samples to show


def spectrogram_db(sig, fs=_GNSS_FS):
    f, t, Sxx = scipy.signal.spectrogram(
        sig, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def to_complex_np(inp_tensor):
    """[C, L] float → [L, C//2] complex  (same logic as _to_complex in visualizer)."""
    n = inp_tensor.shape[0] // 2
    I = inp_tensor[:n].T   # [L, n]
    Q = inp_tensor[n:].T   # [L, n]
    return I + 1j * Q


def build_batch(dataset, batch_size=16):
    """Return one batch as (inp_raw, inp_per_sample, inp_per_batch) each [B, 8, 1024]."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x_raw, _ = next(iter(loader))          # [B, 4, 1024] complex
    I = x_raw.real.float()
    Q = x_raw.imag.float()
    inp_raw = torch.cat([I, Q], dim=1)     # [B, 8, 1024], no norm

    # per-sample: normalize each sample independently
    inp_per_sample = torch.stack([
        (s - s.mean()) / (s.std() + 1e-8) for s in inp_raw
    ])

    # per-batch: current training approach
    inp_per_batch = (inp_raw - inp_raw.mean()) / (inp_raw.std() + 1e-8)

    return inp_raw, inp_per_sample, inp_per_batch


def main():
    ds = UniversalDataset(task_id=132, mode="test", angle_mode='sincos')
    inp_raw, inp_per_sample, inp_per_batch = build_batch(ds, batch_size=32)

    COLS = [
        ("Raw (no norm)",          inp_raw),
        ("Per-sample z-score",     inp_per_sample),
        ("Per-batch z-score\n(training)", inp_per_batch),
    ]

    n_rows = N_SAMPLES * N_ANT
    n_cols = len(COLS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    fig.suptitle(
        f"Does normalization cause the step?\n"
        f"(noverlap={NOVERLAP}, nperseg={NPERSEG})",
        fontsize=12
    )

    for s in range(N_SAMPLES):
        for i in range(N_ANT):
            row = s * N_ANT + i

            specs = []
            for label, batch in COLS:
                sig = to_complex_np(batch[s])[:, i]   # (1024,) complex
                f, t, S = spectrogram_db(sig)
                specs.append((f, t, S))

            vmin = min(S.min() for _, _, S in specs)
            vmax = max(S.max() for _, _, S in specs)

            for col, ((label, _), (f, t, S)) in enumerate(zip(COLS, specs)):
                ax = axes[row, col]
                extent = [t[0]*1e3, t[-1]*1e3, f[0], f[-1]]
                ax.imshow(S, aspect='auto', origin='lower', cmap='turbo',
                          vmin=vmin, vmax=vmax, extent=extent,
                          interpolation='nearest')
                ax.set_title(f"S{s+1} Ant{i+1} — {label}", fontsize=8)
                ax.set_xlabel("t [ms]", fontsize=7)
                ax.set_ylabel("f [Hz]", fontsize=7)

    plt.tight_layout()
    out = "tests/test_spectrogram.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
