"""
Evaluate model generalization to masked azimuth bands.

During training, 5 azimuth bands of 5° width were held out:
  centers at 0°, +90°, -90°, +180°, -180°

This script:
  1. Finds real test samples within each masked band
  2. Generates signals conditioned on their exact az/el/pos
  3. Compares real vs generated across multiple metrics and plot types
  4. Uses 4 unmasked reference bands (at ±45°, ±135°) as a performance baseline

Outputs (all saved to --output_dir):
  spectrogram_band_X.png       — real vs generated spectrogram per band
  psd_band_X.png               — PSD mean ± std per band
  iq_constellation_band_X.png  — IQ constellation per band
  aoa_regression.png           — full ±180° regression with masked bands shaded
  aoa_consistency.png          — Δφ histograms at masked band centers
  phase_accuracy.png           — |measured - theoretical| Δφ error per band
  metrics_summary.png          — SNR / amp_ratio / freq_mse bar chart
  metrics_summary.txt          — same as plain text table
  az_coverage.png              — dataset azimuth histogram with masked bands shaded

Run:
  python tests/eval_masked_angles.py \\
      --weights /path/to/weights.h5 \\
      --output_dir ./masked_eval   \\
      [--n_samples 16]             \\
      [--device cuda]
"""

import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn.functional as F

from UniversalDataLoader import UniversalDataset
from models.DiffWave import DiffWaveRF, DiffusionEngine

# ── Constants ─────────────────────────────────────────────────────────────────
MASKED_BANDS_DEG = [
    (  -2.5,   2.5),
    (  87.5,  92.5),
    ( -92.5, -87.5),
    ( 177.5, 180.0),
    (-180.0,-177.5),
]
MASKED_BAND_LABELS = ['0°', '+90°', '-90°', '+180°', '-180°']
MASKED_BAND_CENTERS = [0.0, 90.0, -90.0, 180.0, -180.0]

UNMASKED_BANDS_DEG = [
    (  42.5,  47.5),
    ( 132.5, 137.5),
    ( -47.5, -42.5),
    (-137.5,-132.5),
]
UNMASKED_BAND_LABELS = ['+45°', '+135°', '-45°', '-135°']
UNMASKED_BAND_CENTERS = [45.0, 135.0, -45.0, -135.0]

# Position normalization (task 132)
_POS_MEAN = np.array([0.721, -0.034, -1.042], dtype=np.float32)
_POS_STD  = np.array([6.922,  4.555,  0.552], dtype=np.float32)

_GNSS_FS   = 40.5e6
_C_LIGHT   = 299_792_458.0

# Geometry (read from env, fall back to zeros)
def _aoa_geometry():
    def _floats(name, n, default):
        raw = os.environ.get(name)
        if not raw:
            return list(default)
        vals = [float(x) for x in raw.split(',')]
        return vals if len(vals) == n else list(default)
    ant_x    = _floats('DIFFWAVE_ANT_X', 4, [0.]*4)
    ant_y    = _floats('DIFFWAVE_ANT_Y', 4, [0.]*4)
    fc       = float(os.environ.get('DIFFWAVE_FC_HZ', '1.575e9'))
    el_ref   = float(os.environ.get('DIFFWAVE_EL_REF_DEG', '-30'))
    return np.array(ant_x), np.array(ant_y), fc, el_ref


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_pos(pos):
    return (np.asarray(pos, dtype=np.float32) - _POS_MEAN) / _POS_STD


def _preprocess(x_complex):
    """[4, 1024] complex → [8, 1024] float, per-sample z-score."""
    inp = torch.cat([x_complex.real.float(), x_complex.imag.float()], dim=0)
    inp = (inp - inp.mean()) / (inp.std() + 1e-8)
    return inp


def _to_complex_np(batch):
    """[B, 8, 1024] → [B, 4, 1024] complex numpy."""
    n = batch.shape[1] // 2
    return batch[:, :n, :].cpu().numpy() + 1j * batch[:, n:, :].cpu().numpy()


def _spectrogram_db(sig):
    f, t, Sxx = scipy.signal.spectrogram(
        sig, fs=_GNSS_FS, nperseg=128, noverlap=64,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    return np.fft.fftshift(f), t, 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)


def _dphi_baseline01(az_rad, el_rad, ant_x, ant_y, fc):
    k = 2 * np.pi / (_C_LIGHT / fc)
    dx, dy = ant_x[1] - ant_x[0], ant_y[1] - ant_y[0]
    return -k * np.cos(el_rad) * (dx * np.cos(az_rad) + dy * np.sin(az_rad))


def _snr_db(real, gen):
    sig_pwr   = np.mean(np.abs(real) ** 2)
    noise_pwr = np.mean(np.abs(real - gen) ** 2) + 1e-12
    return 10 * np.log10(sig_pwr / noise_pwr)


def _freq_mse(real, gen):
    fr = np.fft.fft(real, axis=-1)
    fg = np.fft.fft(gen,  axis=-1)
    return float(np.mean(np.abs(fr - fg) ** 2))


def _amp_ratio(real, gen):
    return float(np.mean(np.abs(gen)) / (np.mean(np.abs(real)) + 1e-12))


def _measured_dphi(x_complex):
    """Mean Δφ between antenna 0 and 1."""
    return float(np.angle(np.mean(x_complex[1] * np.conj(x_complex[0]))))


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _get_az_deg(dataset):
    az_arr = np.asarray(dataset.az_angles)
    return np.degrees(np.arctan2(az_arr[:, 0], az_arr[:, 1]))


def _samples_in_band(az_deg, lo, hi, max_n):
    idx = np.where((az_deg >= lo) & (az_deg <= hi))[0]
    return idx[:max_n]


def _load_band_samples(dataset, indices, device):
    """Returns real_batch [N,8,1024], cond_batch [N,7], az_list [N], el_list [N]."""
    signals, conds, azs, els = [], [], [], []
    for i in indices:
        x, (pos, az, el) = dataset[i]
        signals.append(_preprocess(x))
        pos_n = _norm_pos(pos.numpy())
        conds.append(np.concatenate([pos_n, az.numpy(), el.numpy()]))
        azs.append(float(np.degrees(np.arctan2(az[0].item(), az[1].item()))))
        els.append(float(np.degrees(np.arctan2(el[0].item(), el[1].item()))))
    real_batch = torch.stack(signals).to(device)
    cond_batch = torch.tensor(np.stack(conds), dtype=torch.float32).to(device)
    return real_batch, cond_batch, azs, els


# ── Per-band plots ────────────────────────────────────────────────────────────

def _plot_spectrogram(real_batch, gen_batch, label, outpath):
    real_c = _to_complex_np(real_batch)
    gen_c  = _to_complex_np(gen_batch)
    N, n_ant = real_c.shape[0], real_c.shape[1]
    mean_r = np.mean([_spectrogram_db(real_c[b, 0, :])[2] for b in range(N)], axis=0)
    mean_g = np.mean([_spectrogram_db(gen_c[b,  0, :])[2] for b in range(N)], axis=0)
    f, t, _ = _spectrogram_db(real_c[0, 0, :])
    extent  = [t[0]*1e3, t[-1]*1e3, f[0], f[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vmin = min(mean_r.min(), mean_g.min())
    vmax = max(mean_r.max(), mean_g.max())
    for ax, S, title in zip(axes, [mean_r, mean_g], ['Real', 'Generated']):
        im = ax.imshow(S, aspect='auto', origin='lower', cmap='turbo',
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f'{title} — Antenna 1 (mean, N={N})')
        ax.set_xlabel('t [ms]'); ax.set_ylabel('f [Hz]')
        fig.colorbar(im, ax=ax, format='%+.0f dB-Hz')
    plt.suptitle(f'Spectrogram — azimuth band {label}')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_psd(real_batch, gen_batch, label, outpath):
    real_c = _to_complex_np(real_batch)
    gen_c  = _to_complex_np(gen_batch)
    N      = real_c.shape[0]
    freqs  = np.fft.fftshift(np.fft.fftfreq(1024))
    psds_r = np.array([np.abs(np.fft.fftshift(np.fft.fft(real_c[b, 0, :])))**2 for b in range(N)])
    psds_g = np.array([np.abs(np.fft.fftshift(np.fft.fft(gen_c[b,  0, :])))**2 for b in range(N)])
    r_mean, r_std = psds_r.mean(0), psds_r.std(0)
    g_mean, g_std = psds_g.mean(0), psds_g.std(0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(freqs, r_mean, label='Real',      color='steelblue', lw=1.2)
    ax.fill_between(freqs, np.maximum(r_mean - r_std, 1e-20), r_mean + r_std,
                    alpha=0.2, color='steelblue')
    ax.semilogy(freqs, g_mean, label='Generated', color='crimson',   lw=1.2, linestyle='--')
    ax.fill_between(freqs, np.maximum(g_mean - g_std, 1e-20), g_mean + g_std,
                    alpha=0.2, color='crimson')
    ax.set_title(f'PSD — Antenna 1 (mean ± std, N={N}) — band {label}')
    ax.set_xlabel('Normalised Frequency'); ax.set_ylabel('Power')
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_iq(real_batch, gen_batch, label, outpath):
    real_c = _to_complex_np(real_batch)
    gen_c  = _to_complex_np(gen_batch)
    N, n_ant = real_c.shape[0], real_c.shape[1]
    fig, axes = plt.subplots(1, n_ant, figsize=(n_ant * 3.5, 3.5))
    for ant in range(n_ant):
        ax = axes[ant]
        for b in range(N):
            ax.scatter(real_c[b, ant, :].real, real_c[b, ant, :].imag,
                       s=1, alpha=0.15, color='steelblue', label='Real' if b == 0 else '')
            ax.scatter(gen_c[b,  ant, :].real, gen_c[b,  ant, :].imag,
                       s=1, alpha=0.15, color='crimson',   label='Generated' if b == 0 else '')
        ax.set_title(f'Ant {ant+1}'); ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        if ant == 0:
            ax.legend(fontsize=7, markerscale=6)
    plt.suptitle(f'IQ Constellation — band {label} (N={N})')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


# ── Global plots ──────────────────────────────────────────────────────────────

def _plot_aoa_regression(engine, device, outpath, n_gen=16, ddim_steps=20):
    ant_x, ant_y, fc, el_ref_deg = _aoa_geometry()
    el_ref_rad = np.deg2rad(el_ref_deg)
    angles_deg = np.linspace(-180, 180, 73)
    mean_phases, std_phases = [], []
    engine.model.eval()
    with torch.no_grad():
        for ad in angles_deg:
            ar = np.deg2rad(ad)
            cond = torch.zeros(n_gen, 7, device=device)
            cond[:, 3] = math.sin(ar); cond[:, 4] = math.cos(ar)
            cond[:, 5] = math.sin(el_ref_rad); cond[:, 6] = math.cos(el_ref_rad)
            gen, _ = engine.sample_ddim(n_gen, 1024, cond, steps=ddim_steps)
            n_ant  = gen.shape[1] // 2
            a1 = torch.complex(gen[:, 0], gen[:, n_ant])
            a2 = torch.complex(gen[:, 1], gen[:, n_ant+1])
            ph = torch.angle(torch.mean(a2 * a1.conj(), dim=1))
            mean_phases.append(ph.mean().item()); std_phases.append(ph.std().item())

    mean_phases = np.array(mean_phases); std_phases = np.array(std_phases)
    k   = 2 * np.pi / (_C_LIGHT / fc)
    dx  = ant_x[1] - ant_x[0]; dy = ant_y[1] - ant_y[0]
    theory = -k * np.cos(el_ref_rad) * (dx * np.cos(np.deg2rad(angles_deg))
                                        + dy * np.sin(np.deg2rad(angles_deg)))

    fig, ax = plt.subplots(figsize=(12, 5))
    # Shade masked bands
    for (lo, hi), lbl in zip(MASKED_BANDS_DEG, MASKED_BAND_LABELS):
        ax.axvspan(lo, hi, color='grey', alpha=0.3,
                   label='Masked' if lbl == '0°' else '')
    ax.plot(angles_deg, theory, 'r--', lw=1.5,
            label=f'Theory el={el_ref_deg:.0f}° (2×2 planar)')
    ax.errorbar(angles_deg, mean_phases, yerr=std_phases,
                fmt='o-', capsize=2, lw=1.2, ms=3, color='steelblue',
                label='Measured (mean ± std)')
    ax.set_xlabel('Conditioned AoA (°)'); ax.set_ylabel('Measured Δφ — baseline (0,1) (rad)')
    ax.set_title('AoA Regression — masked bands shaded (grey)')
    ax.set_xlim(-180, 180); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_aoa_consistency(engine, device, outpath, n_gen=32, ddim_steps=20):
    ant_x, ant_y, fc, el_ref_deg = _aoa_geometry()
    el_ref_rad = np.deg2rad(el_ref_deg)
    centers = [0.0, 90.0, -90.0, 180.0]
    labels  = ['0°', '+90°', '-90°', '±180°']
    fig, axes = plt.subplots(1, len(centers), figsize=(18, 3.5))
    engine.model.eval()
    with torch.no_grad():
        for ax, cd, lbl in zip(axes, centers, labels):
            ar   = np.deg2rad(cd)
            cond = torch.zeros(n_gen, 7, device=device)
            cond[:, 3] = math.sin(ar); cond[:, 4] = math.cos(ar)
            cond[:, 5] = math.sin(el_ref_rad); cond[:, 6] = math.cos(el_ref_rad)
            gen, _ = engine.sample_ddim(n_gen, 1024, cond, steps=ddim_steps)
            n_ant  = gen.shape[1] // 2
            a1 = torch.complex(gen[:, 0], gen[:, n_ant])
            a2 = torch.complex(gen[:, 1], gen[:, n_ant+1])
            diffs = torch.angle(torch.mean(a2 * a1.conj(), dim=1)).cpu().numpy()
            expected = _dphi_baseline01(ar, el_ref_rad, ant_x, ant_y, fc)
            ax.hist(diffs, bins=24, range=(-np.pi, np.pi),
                    color='orange', alpha=0.7, density=True)
            ax.axvline(expected, color='r', linestyle='--', lw=1.5,
                       label=f'Expected {expected:.2f} rad')
            ax.set_title(f'{lbl}', fontsize=10)
            ax.set_xlabel('Δφ (rad)'); ax.set_xlim(-np.pi, np.pi)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f'AoA Consistency at Masked Band Centers (el_ref={el_ref_deg:.0f}°)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_phase_accuracy(metrics_masked, metrics_unmasked, outpath):
    labels_m = MASKED_BAND_LABELS
    labels_u = UNMASKED_BAND_LABELS
    err_m = [m['dphi_err'] for m in metrics_masked]
    err_u = [m['dphi_err'] for m in metrics_unmasked]
    x_m = np.arange(len(labels_m))
    x_u = np.arange(len(labels_u))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(x_m, err_m, color='tomato', alpha=0.8)
    axes[0].set_xticks(x_m); axes[0].set_xticklabels(labels_m)
    axes[0].set_title('Masked bands — |Δφ error| (rad)'); axes[0].set_ylabel('rad')
    axes[0].grid(True, axis='y', alpha=0.4)
    axes[1].bar(x_u, err_u, color='steelblue', alpha=0.8)
    axes[1].set_xticks(x_u); axes[1].set_xticklabels(labels_u)
    axes[1].set_title('Unmasked bands — |Δφ error| (rad)'); axes[1].set_ylabel('rad')
    axes[1].grid(True, axis='y', alpha=0.4)
    plt.suptitle('Inter-antenna Phase Error vs Theoretical (baseline 0→1)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_metrics_summary(metrics_masked, metrics_unmasked, outpath):
    keys   = ['snr_db', 'amp_ratio', 'freq_mse']
    titles = ['SNR (dB)', 'Amplitude Ratio', 'Freq MSE']
    lm = MASKED_BAND_LABELS; lu = UNMASKED_BAND_LABELS

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, keys, titles):
        vals_m = [m[key] for m in metrics_masked]
        vals_u = [m[key] for m in metrics_unmasked]
        x = np.arange(max(len(lm), len(lu)))
        w = 0.35
        ax.bar(np.arange(len(lm)) - w/2, vals_m, w,
               label='Masked',   color='tomato',    alpha=0.8)
        ax.bar(np.arange(len(lu)) + w/2, vals_u, w,
               label='Unmasked', color='steelblue', alpha=0.8)
        ax.set_title(title)
        all_labels = lm + ['' for _ in range(max(0, len(lu)-len(lm)))]
        ax.set_xticks(np.arange(max(len(lm), len(lu))))
        ax.set_xticklabels(lm if len(lm) >= len(lu) else lu, rotation=30)
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.4)
    plt.suptitle('Quantitative Metrics — Masked vs Unmasked Bands')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_az_coverage(az_deg_test, outpath):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(az_deg_test, bins=180, range=(-180, 180), color='steelblue', alpha=0.6)
    for (lo, hi), lbl in zip(MASKED_BANDS_DEG, MASKED_BAND_LABELS):
        ax.axvspan(lo, hi, color='red', alpha=0.4,
                   label='Masked' if lbl == '0°' else '')
    ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Count')
    ax.set_title('Test set azimuth distribution — masked bands in red')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def _write_metrics_txt(metrics_masked, metrics_unmasked, outpath):
    lines = []
    header = f"{'Band':>10}  {'Type':>10}  {'SNR (dB)':>10}  {'AmpRatio':>10}  {'FreqMSE':>12}  {'DphiErr':>10}"
    lines.append(header)
    lines.append('-' * len(header))
    for m, lbl in zip(metrics_masked, MASKED_BAND_LABELS):
        lines.append(f"{lbl:>10}  {'masked':>10}  {m['snr_db']:10.3f}  "
                     f"{m['amp_ratio']:10.4f}  {m['freq_mse']:12.1f}  {m['dphi_err']:10.4f}")
    lines.append('')
    for m, lbl in zip(metrics_unmasked, UNMASKED_BAND_LABELS):
        lines.append(f"{lbl:>10}  {'unmasked':>10}  {m['snr_db']:10.3f}  "
                     f"{m['amp_ratio']:10.4f}  {m['freq_mse']:12.1f}  {m['dphi_err']:10.4f}")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_metrics(real_batch, gen_batch, az_list, el_list):
    """Compute scalar metrics for a batch."""
    real_c = _to_complex_np(real_batch)
    gen_c  = _to_complex_np(gen_batch)
    N      = real_c.shape[0]
    ant_x, ant_y, fc, _ = _aoa_geometry()

    snrs, amps, fmsqs, dphi_errs = [], [], [], []
    for b in range(N):
        snrs.append(_snr_db(real_c[b], gen_c[b]))
        amps.append(_amp_ratio(real_c[b], gen_c[b]))
        fmsqs.append(_freq_mse(real_c[b], gen_c[b]))
        meas    = _measured_dphi(gen_c[b])
        az_rad  = np.deg2rad(az_list[b])
        el_rad  = np.deg2rad(el_list[b])
        theory  = _dphi_baseline01(az_rad, el_rad, ant_x, ant_y, fc)
        dphi_errs.append(abs(meas - theory))

    return {
        'snr_db':    float(np.mean(snrs)),
        'amp_ratio': float(np.mean(amps)),
        'freq_mse':  float(np.mean(fmsqs)),
        'dphi_err':  float(np.mean(dphi_errs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./masked_eval')
    parser.add_argument('--n_samples',  type=int, default=16)
    parser.add_argument('--device',     type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ddim_steps', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model ──
    model = DiffWaveRF(input_channels=8, residual_channels=64, cond_dim=7).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    engine = DiffusionEngine(model=model, timesteps=1000)
    engine.model.eval()

    # ── Load test dataset ──
    ds = UniversalDataset(task_id=132, mode='test', angle_mode='sincos')
    az_deg_test = np.degrees(np.arctan2(
        np.asarray(ds.az_angles)[:, 0], np.asarray(ds.az_angles)[:, 1]))

    # ── Azimuth coverage plot ──
    _plot_az_coverage(az_deg_test, os.path.join(args.output_dir, 'az_coverage.png'))
    print('Saved: az_coverage.png')

    # ── Per-band evaluation ──
    def eval_bands(band_list, band_labels, tag):
        all_metrics = []
        for (lo, hi), lbl in zip(band_list, band_labels):
            print(f'\n── Band {lbl} [{lo:.1f}°, {hi:.1f}°] ──')
            idx = _samples_in_band(az_deg_test, lo, hi, args.n_samples)
            if len(idx) == 0:
                print(f'  No test samples found — skipping')
                all_metrics.append({'snr_db': float('nan'), 'amp_ratio': float('nan'),
                                    'freq_mse': float('nan'), 'dphi_err': float('nan')})
                continue
            print(f'  Found {len(idx)} samples')
            real_batch, cond_batch, az_list, el_list = _load_band_samples(ds, idx, device)

            with torch.no_grad():
                gen_batch, _ = engine.sample_ddim(
                    len(idx), 1024, cond_batch, steps=args.ddim_steps)

            slug = lbl.replace('°', 'deg').replace('+', 'p').replace('-', 'm')
            _plot_spectrogram(real_batch, gen_batch, lbl,
                              os.path.join(args.output_dir, f'spectrogram_{tag}_{slug}.png'))
            _plot_psd(real_batch, gen_batch, lbl,
                      os.path.join(args.output_dir, f'psd_{tag}_{slug}.png'))
            _plot_iq(real_batch, gen_batch, lbl,
                     os.path.join(args.output_dir, f'iq_constellation_{tag}_{slug}.png'))
            print(f'  Saved: spectrogram, psd, iq_constellation')

            m = compute_metrics(real_batch, gen_batch, az_list, el_list)
            all_metrics.append(m)
            print(f'  SNR={m["snr_db"]:.2f} dB  AmpRatio={m["amp_ratio"]:.4f}  '
                  f'FreqMSE={m["freq_mse"]:.1f}  DphiErr={m["dphi_err"]:.4f} rad')
        return all_metrics

    metrics_masked   = eval_bands(MASKED_BANDS_DEG,   MASKED_BAND_LABELS,   'masked')
    metrics_unmasked = eval_bands(UNMASKED_BANDS_DEG, UNMASKED_BAND_LABELS, 'unmasked')

    # ── Global plots ──
    print('\nGenerating global plots...')
    _plot_aoa_regression(engine, device,
                         os.path.join(args.output_dir, 'aoa_regression.png'),
                         ddim_steps=args.ddim_steps)
    print('Saved: aoa_regression.png')

    _plot_aoa_consistency(engine, device,
                          os.path.join(args.output_dir, 'aoa_consistency.png'),
                          ddim_steps=args.ddim_steps)
    print('Saved: aoa_consistency.png')

    _plot_phase_accuracy(metrics_masked, metrics_unmasked,
                         os.path.join(args.output_dir, 'phase_accuracy.png'))
    print('Saved: phase_accuracy.png')

    _plot_metrics_summary(metrics_masked, metrics_unmasked,
                          os.path.join(args.output_dir, 'metrics_summary.png'))
    print('Saved: metrics_summary.png')

    _write_metrics_txt(metrics_masked, metrics_unmasked,
                       os.path.join(args.output_dir, 'metrics_summary.txt'))
    print('Saved: metrics_summary.txt')
    print(f'\nAll outputs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
