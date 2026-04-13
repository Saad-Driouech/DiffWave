"""
Scan the full dataset and detect which samples have a frequency step.

For each sample, we compute the spectrogram of antenna 0, find the peak
frequency bin in each time frame, and flag the sample if the peak jumps
by more than STEP_THRESHOLD_HZ between any two consecutive frames.

Saves:
  - Console output with flagged sample IDs and jump statistics
  - tests/step_examples.png  — spectrograms of up to N_PLOT flagged samples

Run with:  python tests/scan_dataset_steps.py
"""

import numpy as np
import scipy.signal
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from UniversalDataLoader import UniversalDataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
FS             = 40.5e6   # sampling frequency (Hz)
NPERSEG        = 128
NOVERLAP       = 120
STEP_THRESHOLD = 0.5e6   # flag if peak jumps > 0.5 MHz between frames
BATCH_SIZE     = 64
MODE           = "test"   # change to "train" to scan training set
N_PLOT         = 6        # number of flagged samples to plot
# ─────────────────────────────────────────────────────────────────────────────


def spectrogram_db(sig):
    f, t, Sxx = scipy.signal.spectrogram(
        sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def peak_freq_per_frame(sig):
    f, _, Sxx = scipy.signal.spectrogram(
        sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    peak_bins = np.argmax(Sxx, axis=0)
    return np.fft.fftshift(f)[peak_bins]


def check_step(sig, threshold=STEP_THRESHOLD):
    peaks = peak_freq_per_frame(sig)
    jumps = np.abs(np.diff(peaks))
    return jumps.max() > threshold, jumps.max()


def normalize(inp):
    """Per-sample z-score for a single sample [C, L]."""
    return (inp - inp.mean()) / (inp.std() + 1e-8)


def to_complex(x_np):
    """[4, 1024] complex → per-antenna list of (1024,) complex."""
    return [x_np[i] for i in range(x_np.shape[0])]


def main():
    ds = UniversalDataset(task_id=132, mode=MODE, angle_mode='sincos')
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    n_total      = 0
    n_flagged    = 0
    flagged      = []   # list of (global_idx, max_jump, x_np)
    jump_sizes   = []

    for batch_idx, (x, _) in enumerate(tqdm(loader, desc="Scanning")):
        if x.dim() == 4:
            x = x.squeeze(1)
        x_np = x.numpy()   # [B, 4, 1024] complex64

        for i in range(x_np.shape[0]):
            global_idx = batch_idx * BATCH_SIZE + i
            sig = x_np[i, 0]   # antenna 0 for detection
            is_step, max_jump = check_step(sig)
            jump_sizes.append(max_jump)
            if is_step:
                n_flagged += 1
                flagged.append((global_idx, max_jump, x_np[i]))
            n_total += 1

    jump_sizes = np.array(jump_sizes)
    pct = 100.0 * n_flagged / n_total

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n── Results ({MODE} set) ──────────────────────────────")
    print(f"  Total samples     : {n_total}")
    print(f"  Flagged (step)    : {n_flagged}  ({pct:.1f}%)")
    print(f"  Threshold used    : {STEP_THRESHOLD/1e6:.2f} MHz")
    print(f"\n  Jump size stats (MHz):")
    print(f"    min    : {jump_sizes.min()/1e6:.3f}")
    print(f"    median : {np.median(jump_sizes)/1e6:.3f}")
    print(f"    mean   : {jump_sizes.mean()/1e6:.3f}")
    print(f"    p95    : {np.percentile(jump_sizes, 95)/1e6:.3f}")
    print(f"    max    : {jump_sizes.max()/1e6:.3f}")

    if flagged:
        # Sort by jump size descending so the worst examples plot first
        flagged.sort(key=lambda t: t[1], reverse=True)
        print(f"\n  Flagged sample IDs (sorted by jump size):")
        for idx, jump, _ in flagged:
            print(f"    id={idx:6d}  max_jump={jump/1e6:.3f} MHz")

    # ── Plot examples ─────────────────────────────────────────────────────────
    to_plot = flagged[:N_PLOT]
    if not to_plot:
        print("\nNo flagged samples — nothing to plot.")
        return

    n_ant = 4
    fig, axes = plt.subplots(len(to_plot), n_ant,
                             figsize=(n_ant * 4, len(to_plot) * 3))
    if len(to_plot) == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, jump, x_np) in enumerate(to_plot):
        for ant in range(n_ant):
            # normalize per-sample before plotting (best visual)
            I = torch.tensor(x_np.real).float()
            Q = torch.tensor(x_np.imag).float()
            inp = torch.cat([I, Q], dim=0)          # [8, 1024]
            inp_norm = normalize(inp).numpy()
            sig = inp_norm[:n_ant][ant] + 1j * inp_norm[n_ant:][ant]  # (1024,) complex

            f, t, S = spectrogram_db(sig)
            extent = [t[0]*1e3, t[-1]*1e3, f[0], f[-1]]
            ax = axes[row, ant]
            ax.imshow(S, aspect='auto', origin='lower', cmap='turbo',
                      extent=extent, interpolation='nearest')
            ax.set_title(f"id={idx}  jump={jump/1e6:.2f}MHz\nAnt{ant+1}", fontsize=8)
            ax.set_xlabel("t [ms]", fontsize=7)
            ax.set_ylabel("f [Hz]", fontsize=7)

    plt.suptitle(f"Flagged samples with frequency step > {STEP_THRESHOLD/1e6:.1f} MHz  ({MODE} set)",
                 fontsize=11)
    plt.tight_layout()
    out = "tests/step_examples.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\n  Plots saved → {out}")


if __name__ == "__main__":
    main()
