"""
Holistic comparison of up to 4 trained DiffWaveRF models on in-band vs
out-of-band azimuth performance.

Two-phase workflow
------------------
1) Sampling (slow, ~1 hour for 4 models, 900 samples each):
     python tests/eval_full_vs_masked.py \\
         --weights_xyz_full   /path/weights.h5 \\
         --weights_xyz_mask   /path/weights.h5 \\
         --weights_noxyz_full /path/weights.h5 \\
         --weights_noxyz_mask /path/weights.h5 \\
         --output_dir ./compare_eval \\
         --mode eval

2) Plotting (seconds, reads raw.npz produced by step 1):
     python tests/eval_full_vs_masked.py \\
         --output_dir ./compare_eval \\
         --mode plot \\
         --compare all4    # or: xyz, noxyz, full, mask

All four `--weights_*` flags are optional in the eval phase; any missing
checkpoint just skips that model.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import spectrogram

from UniversalDataLoader import UniversalDataset
from models.DiffWave import DiffusionEngine, DiffWaveRF
from utils.chirp_phase import extract_phase_sincos
from utils.visualization import (_aoa_geometry, _dphi_baseline,
                                 _gnss_spectrogram_db)

# ── constants ─────────────────────────────────────────────────────────────────
MASKED_BANDS_DEG = [
    (  -2.5,   2.5),
    (  87.5,  92.5),
    ( -92.5, -87.5),
    ( 177.5, 180.0),
    (-180.0,-177.5),
]
BAND_CENTERS_DEG     = [0.0, 90.0, -90.0, 180.0]   # ±180 collapses to 180
REFERENCE_AZIMUTHS   = [-135.0, -45.0, 45.0, 135.0]
DEFAULT_SPEC_AZ      = [-90.0, -45.0, 0.0, 45.0]

POS_MEAN = np.array([0.721, -0.034, -1.042], dtype=np.float32)
POS_STD  = np.array([6.922,  4.555,  0.552], dtype=np.float32)

T_CHIRP_S = 33.19e-6
# 5 % of the chirp period.  Tighter than the original 10 % so the rate metric
# can actually discriminate between models; still above the spectrogram time
# resolution (≈ 1.6 µs) so the threshold is not measuring estimator noise.
PHASE_TOL_S = 0.05 * T_CHIRP_S

COND_DIM_BY_KEY = {
    'xyz_full':   9,
    'xyz_mask':   9,
    'noxyz_full': 6,
    'noxyz_mask': 6,
}

MODEL_STYLE = {
    'xyz_full':   dict(color='#1f77b4', linestyle='-',  label='XYZ + full data'),
    'xyz_mask':   dict(color='#1f77b4', linestyle='--', label='XYZ + masked'),
    'noxyz_full': dict(color='#ff7f0e', linestyle='-',  label='no XYZ + full data'),
    'noxyz_mask': dict(color='#ff7f0e', linestyle='--', label='no XYZ + masked'),
}
MODEL_BAR_COLOR = {
    'xyz_full':   '#1f77b4',
    'xyz_mask':   '#aec7e8',
    'noxyz_full': '#ff7f0e',
    'noxyz_mask': '#ffbb78',
}

COMPARE_SETS = {
    'all4':  ['xyz_full', 'xyz_mask', 'noxyz_full', 'noxyz_mask'],
    'xyz':   ['xyz_full', 'xyz_mask'],
    'noxyz': ['noxyz_full', 'noxyz_mask'],
    'full':  ['xyz_full', 'noxyz_full'],
    'mask':  ['xyz_mask', 'noxyz_mask'],
}


# ── helpers ───────────────────────────────────────────────────────────────────
def load_engine(weights_path, cond_dim, device):
    model = DiffWaveRF(input_channels=8, residual_channels=64,
                       cond_dim=cond_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return DiffusionEngine(model=model, timesteps=1000)


def build_cond(key, pos, az_sc, el_sc, phi_sc, device):
    pos_norm = (pos - torch.tensor(POS_MEAN, device=device)) / \
               torch.tensor(POS_STD, device=device)
    if key.startswith('xyz'):
        return torch.cat([pos_norm, az_sc, el_sc, phi_sc], dim=1)
    return torch.cat([az_sc, el_sc, phi_sc], dim=1)


def find_indices_for_azimuth(ds, target_deg, tol_deg, n, rng):
    az_arr = np.asarray(ds.az_angles)
    az_deg = np.degrees(np.arctan2(az_arr[:, 0], az_arr[:, 1]))
    delta = (az_deg - target_deg + 540.0) % 360.0 - 180.0     # wrap-safe
    mask = np.abs(delta) < tol_deg
    cand = np.where(mask)[0]
    if len(cand) == 0:
        return []
    chosen = rng.choice(cand, size=min(n, len(cand)), replace=False)
    return [int(i) for i in chosen]


def dphi_baseline_pair(gen):
    n_ant = gen.shape[1] // 2
    a1 = gen[:, 0] + 1j * gen[:, n_ant]
    a2 = gen[:, 1] + 1j * gen[:, n_ant + 1]
    return torch.angle(torch.mean(a2 * a1.conj(), dim=1)).cpu().numpy()


def dphi_baseline_pair_complex(iq):
    """Same estimator but for complex input [B, n_ant, L]."""
    a1 = iq[:, 0]
    a2 = iq[:, 1]
    return torch.angle(torch.mean(a2 * a1.conj(), dim=1)).cpu().numpy()


def phase_seconds_from_iq(iq_complex):
    sc = extract_phase_sincos(iq_complex.unsqueeze(0) if iq_complex.dim() == 2
                              else iq_complex)
    ang = float(torch.atan2(sc[0, 0], sc[0, 1]).item())
    if ang < 0:
        ang += 2 * math.pi
    return ang / (2 * math.pi) * T_CHIRP_S


def circular_diff(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


def scalar_metrics(real_iq, gen_real):
    """All three metrics computed in the per-sample z-score normalised domain
    the model was trained in, so the model output and the real reference are
    on the same scale.

    Returns
    -------
    amp_ratio : ⟨|gen|⟩ / ⟨|real_normalised|⟩.  ≈ 1 if the generated signal has
                the right scale in the training domain.
    snr_db    : reconstruction SNR  10·log10(‖real_normalised‖² / ‖real_normalised − gen‖²).
                Larger is better.  ≈ 0 dB for random output, → ∞ for perfect.
    fmse      : MSE between the normalised real PSD and the generated PSD.
                Now directly interpretable as spectral mismatch.
    """
    n_ant = gen_real.shape[1] // 2

    # Bring real_iq into the same domain as the model output
    real_stacked = torch.cat([real_iq.real, real_iq.imag],
                             dim=1).to(torch.float32)              # [B, 8, L]
    mean = real_stacked.mean(dim=(1, 2), keepdim=True)
    std  = real_stacked.std (dim=(1, 2), keepdim=True) + 1e-8
    real_norm = (real_stacked - mean) / std                        # [B, 8, L]

    # Complex magnitudes in the normalised domain
    real_c = real_norm[:, :n_ant] + 1j * real_norm[:, n_ant:]
    gen_c  = gen_real [:, :n_ant] + 1j * gen_real [:, n_ant:]
    amp_real = real_c.abs().mean().item()
    amp_gen  = gen_c .abs().mean().item()
    amp_ratio = amp_gen / max(amp_real, 1e-20)

    # Reconstruction SNR: signal power / error power, both in normalised domain
    sig_power = (real_norm ** 2).mean().item()
    err_power = ((real_norm - gen_real) ** 2).mean().item()
    snr_db = 10 * np.log10(max(sig_power, 1e-20) / max(err_power, 1e-20))

    # Spectral MSE on normalised PSDs
    def psd(x_c):
        X = torch.fft.fftshift(torch.fft.fft(x_c, dim=-1), dim=-1)
        return (X.abs() ** 2).mean(dim=1)
    fmse = ((psd(real_c) - psd(gen_c)) ** 2).mean().item()

    return amp_ratio, snr_db, fmse


# ── eval phase ────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    paths = {
        'xyz_full':   args.weights_xyz_full,
        'xyz_mask':   args.weights_xyz_mask,
        'noxyz_full': args.weights_noxyz_full,
        'noxyz_mask': args.weights_noxyz_mask,
    }
    available = {k: v for k, v in paths.items() if v}
    print(f'Loading {len(available)} models: {list(available.keys())}')
    engines = {k: load_engine(v, COND_DIM_BY_KEY[k], device)
               for k, v in available.items()}

    ds = UniversalDataset(task_id=args.task, mode='test', angle_mode='sincos')

    eval_points = ([('in_band',  c) for c in BAND_CENTERS_DEG] +
                   [('out_band', r) for r in REFERENCE_AZIMUTHS])

    ant_x, ant_y, fc, _, _, _ = _aoa_geometry()

    # per-sample records: lists keyed by model
    rec = {k: {'azimuth': [], 'region': [], 'dphi_measured': [],
               'dphi_real': [], 'dphi_theory': [],
               'phase_real': [], 'phase_gen': [],
               'amp_ratio': [], 'snr_db': [], 'freq_mse': []}
           for k in available}
    # representative spectrogram pairs for the grid
    spec_real = {}                           # spec_real[az] = complex [4, L]
    spec_gen  = {k: {} for k in available}   # spec_gen[k][az] = real [8, L]

    for region, target_deg in eval_points:
        n_target = (args.n_per_band if region == 'in_band'
                    else args.n_per_reference)
        idx = find_indices_for_azimuth(ds, target_deg, args.az_tolerance,
                                       n_target, rng)
        print(f'  {region:8s}  az={target_deg:+6.1f}°  found {len(idx)}/{n_target}')
        if not idx:
            continue
        # Stack the real batch + conditioning once, re-use for all models
        reals, poses, azs, els, phis = [], [], [], [], []
        for i in idx:
            x, (pos, az_sc, el_sc) = ds[i]
            reals.append(x)
            poses.append(pos.float())
            azs.append(az_sc.float())
            els.append(el_sc.float())
            phi = extract_phase_sincos(x.to(device).unsqueeze(0)).squeeze(0).cpu()
            phis.append(phi)
        real_iq = torch.stack(reals).to(device)      # [B, 4, L] complex
        pos_b   = torch.stack(poses).to(device)
        az_b    = torch.stack(azs).to(device)
        el_b    = torch.stack(els).to(device)
        phi_b   = torch.stack(phis).to(device)
        B       = real_iq.shape[0]

        # Per-sample theory using each conditioning sample's actual az/el.
        # Earlier versions used a single el_ref for all samples, which mixed in
        # a 0–60% elevation-driven offset on top of any model/data error.
        az_rad_per = torch.atan2(az_b[:, 0], az_b[:, 1]).cpu().numpy()
        el_rad_per = torch.atan2(el_b[:, 0], el_b[:, 1]).cpu().numpy()
        theory_dphi_per = _dphi_baseline((0, 1), az_rad_per, el_rad_per,
                                         ant_x, ant_y, fc)             # [B]
        dphi_real = dphi_baseline_pair_complex(real_iq)  # [B], computed once per az
        if target_deg in args.spec_grid_azimuths:
            spec_real[target_deg] = real_iq[0].cpu().numpy()

        for key, engine in engines.items():
            cond = build_cond(key, pos_b, az_b, el_b, phi_b, device)
            with torch.no_grad():
                gen, _ = engine.sample_ddim(B, 1024, cond,
                                            steps=args.ddim_steps)
            dphi_meas = dphi_baseline_pair(gen)
            amp_ratio, snr_db, fmse = scalar_metrics(real_iq, gen)
            for j in range(B):
                rec[key]['azimuth'].append(target_deg)
                rec[key]['region'].append(region)
                rec[key]['dphi_measured'].append(float(dphi_meas[j]))
                rec[key]['dphi_real'].append(float(dphi_real[j]))
                rec[key]['dphi_theory'].append(float(theory_dphi_per[j]))
                rec[key]['amp_ratio'].append(amp_ratio)
                rec[key]['snr_db'].append(snr_db)
                rec[key]['freq_mse'].append(fmse)
                phi_r = phase_seconds_from_iq(real_iq[j].cpu())
                gen_c = gen[j, :4].cpu() + 1j * gen[j, 4:].cpu()
                phi_g = phase_seconds_from_iq(gen_c)
                rec[key]['phase_real'].append(phi_r)
                rec[key]['phase_gen'].append(phi_g)
            if target_deg in args.spec_grid_azimuths:
                spec_gen[key][target_deg] = gen[0].cpu().numpy()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out = {}
    for key, data in rec.items():
        for f, v in data.items():
            out[f'{key}__{f}'] = np.array(v)
    for az, arr in spec_real.items():
        out[f'spec_real__az{az}'] = arr
    for key in spec_gen:
        for az, arr in spec_gen[key].items():
            out[f'spec_gen__{key}__az{az}'] = arr
    out['models'] = np.array(list(available.keys()))
    out['spec_azimuths'] = np.array(args.spec_grid_azimuths)
    raw = os.path.join(args.output_dir, 'raw.npz')
    np.savez(raw, **out)
    print(f'\nSaved {raw}')


# ── plot phase ────────────────────────────────────────────────────────────────
def _shade_masked_bands(ax):
    for lo, hi in MASKED_BANDS_DEG:
        ax.axvspan(lo, hi, color='lightgray', alpha=0.4, zorder=0)


def _band_for_az(az):
    for lo, hi in MASKED_BANDS_DEG:
        if lo <= az <= hi:
            return 'in_band'
    return 'out_band'


def _theory_curve(az_grid_deg):
    ant_x, ant_y, fc, el_ref_deg, _, _ = _aoa_geometry()
    el_ref_rad = np.deg2rad(el_ref_deg)
    return _dphi_baseline((0, 1), np.deg2rad(az_grid_deg), el_ref_rad,
                          ant_x, ant_y, fc)


def _aggregate_per_az(raw, key):
    az  = raw[f'{key}__azimuth']
    dm  = raw[f'{key}__dphi_measured']
    dt  = raw[f'{key}__dphi_theory']
    uniq = np.unique(az)
    mean_meas, std_meas, mean_theory = [], [], []
    for u in uniq:
        m = az == u
        mean_meas.append(dm[m].mean())
        std_meas.append(dm[m].std())
        mean_theory.append(dt[m].mean())
    return uniq, np.array(mean_meas), np.array(std_meas), np.array(mean_theory)


def _real_curve(raw, key):
    """Per-azimuth mean and std of real-data Δφ, from any one model's records
    (real data is shared across models so we just read from `key`)."""
    az = raw[f'{key}__azimuth']
    dr = raw[f'{key}__dphi_real']
    uniq = np.unique(az)
    mean_r = np.array([dr[az == u].mean() for u in uniq])
    std_r  = np.array([dr[az == u].std()  for u in uniq])
    return uniq, mean_r, std_r


def plot_aoa_regression(args, raw, keys, suffix):
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_masked_bands(ax)
    grid = np.linspace(-180, 180, 361)
    ax.plot(grid, _theory_curve(grid), 'r--', lw=1.2,
            label='Theory (planar)', zorder=2)
    if keys and f'{keys[0]}__dphi_real' in raw:
        ur, mr, sr = _real_curve(raw, keys[0])
        order = np.argsort(ur)
        ax.errorbar(ur[order], mr[order], yerr=sr[order],
                    capsize=2, marker='s', markersize=5,
                    color='black', linestyle=':', linewidth=1.5,
                    label='Real data (mean ± std)', zorder=2.5)
    for key in keys:
        uniq, mm, ms, _ = _aggregate_per_az(raw, key)
        order = np.argsort(uniq)
        ax.errorbar(uniq[order], mm[order], yerr=ms[order],
                    capsize=2, marker='o', markersize=4,
                    **MODEL_STYLE[key], zorder=3)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Measured Δφ baseline (0,1)  (rad)')
    ax.set_title(f'AoA Δφ vs azimuth — {suffix}')
    ax.set_xlim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    _save(fig, args.output_dir, f'aoa_regression_{suffix}')


def plot_aoa_error_model_vs_data(args, raw, keys, suffix):
    """|⟨gen⟩ − ⟨real⟩| per model — the model's bias from the data."""
    if not keys or f'{keys[0]}__dphi_real' not in raw:
        print('  skip aoa_error_model_vs_data: dphi_real not in raw.npz '
              '(re-run --mode eval)')
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_masked_bands(ax)
    for key in keys:
        az = raw[f'{key}__azimuth']
        dm = raw[f'{key}__dphi_measured']
        dr = raw[f'{key}__dphi_real']
        uniq = np.unique(az)
        err = np.array([abs(dm[az == u].mean() - dr[az == u].mean())
                        for u in uniq])
        order = np.argsort(uniq)
        ax.plot(uniq[order], np.rad2deg(err[order]),
                marker='o', markersize=5, **MODEL_STYLE[key])
    ax.set_xlim(-180, 180)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('|⟨Δφ_gen⟩ − ⟨Δφ_real⟩|  (°)')
    ax.set_title(f'Model bias from data — {suffix}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save(fig, args.output_dir, f'aoa_error_model_vs_data_{suffix}')


def plot_aoa_error_data_vs_theory(args, raw, keys, suffix):
    """|⟨real⟩ − theory| — the simulator's bias from the planar formula.
    Identical across all models, so plotted once per comparison."""
    if not keys or f'{keys[0]}__dphi_real' not in raw:
        print('  skip aoa_error_data_vs_theory: dphi_real not in raw.npz '
              '(re-run --mode eval)')
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_masked_bands(ax)
    az = raw[f'{keys[0]}__azimuth']
    dr = raw[f'{keys[0]}__dphi_real']
    dt = raw[f'{keys[0]}__dphi_theory']
    uniq = np.unique(az)
    err = np.array([abs(dr[az == u].mean() - dt[az == u].mean())
                    for u in uniq])
    order = np.argsort(uniq)
    ax.plot(uniq[order], np.rad2deg(err[order]),
            marker='s', markersize=5, color='black', linestyle=':',
            label='Real vs theory')
    ax.set_xlim(-180, 180)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('|⟨Δφ_real⟩ − Δφ_theory|  (°)')
    ax.set_title(f'Simulator bias from theory — {suffix}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save(fig, args.output_dir, f'aoa_error_data_vs_theory_{suffix}')


def plot_aoa_error(args, raw, keys, suffix):
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_masked_bands(ax)
    for key in keys:
        uniq, mm, _, mt = _aggregate_per_az(raw, key)
        err_deg = np.rad2deg(np.abs(mm - mt))
        order = np.argsort(uniq)
        ax.plot(uniq[order], err_deg[order], marker='o', markersize=5,
                **MODEL_STYLE[key])
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('|Δφ measured − theory|  (°)')
    ax.set_title(f'AoA error magnitude vs azimuth — {suffix}')
    ax.set_xlim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    _save(fig, args.output_dir, f'aoa_error_{suffix}')


def _normalize_real_iq(iq):
    """Per-sample z-score on [n_ant, L] complex, matching train.py's
    `_prepare_batch`.  Puts the real reference on the same absolute scale as
    the model's z-score-domain output, so the colormap shows comparable
    noise floors and chirp ridges."""
    real = np.real(iq).astype(np.float32)
    imag = np.imag(iq).astype(np.float32)
    stacked = np.concatenate([real, imag], axis=0)
    norm = (stacked - stacked.mean()) / (stacked.std() + 1e-8)
    n_ant = iq.shape[0]
    return norm[:n_ant] + 1j * norm[n_ant:]


def _plot_spec_panel(ax, sig_complex):
    f, t, S_db = _gnss_spectrogram_db(sig_complex)
    im = ax.pcolormesh(t * 1e3, f, S_db, shading='auto', cmap='turbo',
                       vmin=-140, vmax=-60)
    ax.set_xlabel('t [ms]', fontsize=8)
    ax.set_ylabel('f [Hz]', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def plot_spec_grid_antenna1(args, raw, keys, suffix):
    azs = list(raw['spec_azimuths'])
    n_az = len(azs)
    n_col = 1 + len(keys)
    fig, axes = plt.subplots(n_az, n_col, figsize=(3.2 * n_col, 3.0 * n_az),
                             squeeze=False)
    for i, az in enumerate(azs):
        real_arr = _normalize_real_iq(raw[f'spec_real__az{az}'])   # [4, L] complex
        _plot_spec_panel(axes[i, 0], real_arr[0])
        axes[i, 0].set_title(f'Real — az={az:+.0f}°', fontsize=9)
        for j, key in enumerate(keys, start=1):
            gen_arr = raw[f'spec_gen__{key}__az{az}']  # [8, L] real
            n_ant = gen_arr.shape[0] // 2
            iq = gen_arr[0] + 1j * gen_arr[n_ant]
            _plot_spec_panel(axes[i, j], iq)
            axes[i, j].set_title(
                f'{MODEL_STYLE[key]["label"]}\naz={az:+.0f}°', fontsize=9)
    fig.suptitle(f'Spectrogram comparison (antenna 1) — {suffix}', y=1.005)
    fig.tight_layout()
    _save(fig, args.output_dir, f'spectrogram_grid_antenna1_{suffix}')


def plot_spec_grid_allants(args, raw, keys, suffix):
    sub = os.path.join(args.output_dir, f'spectrogram_grid_allants_{suffix}')
    os.makedirs(sub, exist_ok=True)
    for az in raw['spec_azimuths']:
        real_arr = _normalize_real_iq(raw[f'spec_real__az{az}'])   # [4, L] complex
        n_col = 1 + len(keys)
        fig, axes = plt.subplots(4, n_col, figsize=(3.2 * n_col, 11),
                                 squeeze=False)
        for a in range(4):
            _plot_spec_panel(axes[a, 0], real_arr[a])
            axes[a, 0].set_ylabel(f'Antenna {a+1}\nf [Hz]', fontsize=8)
            if a == 0:
                axes[a, 0].set_title(f'Real — az={az:+.0f}°', fontsize=9)
        for j, key in enumerate(keys, start=1):
            gen_arr = raw[f'spec_gen__{key}__az{az}']
            n_ant = gen_arr.shape[0] // 2
            for a in range(4):
                iq = gen_arr[a] + 1j * gen_arr[a + n_ant]
                _plot_spec_panel(axes[a, j], iq)
                if a == 0:
                    axes[a, j].set_title(
                        f'{MODEL_STYLE[key]["label"]}\naz={az:+.0f}°',
                        fontsize=9)
        fig.tight_layout()
        # save into the subdir
        path = os.path.join(sub, f'az_{az:+.0f}')
        fig.savefig(f'{path}.pdf', bbox_inches='tight')
        fig.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_scalar_metrics(args, raw, keys, suffix):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    metric_keys = ['snr_db', 'amp_ratio', 'freq_mse']
    metric_labels = ['SNR (dB)', 'Amplitude ratio', 'Frequency MSE']
    regions = ['in_band', 'out_band']
    region_labels = ['Inside masked bands', 'Outside masked bands']
    for row, (region, rlabel) in enumerate(zip(regions, region_labels)):
        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            means, stds, colors = [], [], []
            for key in keys:
                vals = raw[f'{key}__{mkey}']
                regs = raw[f'{key}__region']
                m = regs == region
                if m.sum() == 0:
                    means.append(np.nan); stds.append(0); colors.append('gray')
                    continue
                means.append(np.mean(vals[m]))
                stds.append(np.std(vals[m]))
                colors.append(MODEL_BAR_COLOR[key])
            xs = np.arange(len(keys))
            ax.bar(xs, means, yerr=stds, color=colors, capsize=3,
                   edgecolor='black', linewidth=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels([MODEL_STYLE[k]['label'] for k in keys],
                               rotation=20, ha='right', fontsize=8)
            ax.set_ylabel(mlabel)
            if col == 0:
                ax.text(-0.25, 0.5, rlabel, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10,
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'Scalar metrics — in-band vs out-of-band — {suffix}')
    fig.tight_layout()
    _save(fig, args.output_dir, f'scalar_metrics_{suffix}')


def plot_phase_match(args, raw, keys, suffix):
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ['in_band', 'out_band']
    region_labels = ['Inside masked bands', 'Outside masked bands']
    width = 0.8 / len(keys)
    xs = np.arange(len(regions))
    for j, key in enumerate(keys):
        rates = []
        for region in regions:
            pr = raw[f'{key}__phase_real']
            pg = raw[f'{key}__phase_gen']
            rg = raw[f'{key}__region']
            m = rg == region
            if m.sum() == 0:
                rates.append(0); continue
            diffs = np.array([circular_diff(a, b, T_CHIRP_S)
                              for a, b in zip(pr[m], pg[m])])
            rates.append(100.0 * np.mean(diffs < PHASE_TOL_S))
        offset = (j - (len(keys) - 1) / 2) * width
        ax.bar(xs + offset, rates, width=width,
               color=MODEL_BAR_COLOR[key], edgecolor='black', linewidth=0.6,
               label=MODEL_STYLE[key]['label'])
    ax.set_xticks(xs)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel(f'% generated within ±{PHASE_TOL_S*1e6:.1f} µs of real')
    ax.set_title(f'Chirp phase match rate — {suffix}')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save(fig, args.output_dir, f'phase_match_{suffix}')


def plot_phase_error_median(args, raw, keys, suffix):
    """Continuous companion to the rate plot: median |Δφ_gen − Δφ_real| per
    region per model.  Error bars show the 25th / 75th percentiles.
    Spectrogram-based phase extraction has ~1.6 µs intrinsic noise; values
    below that line are at the resolution floor of the estimator."""
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ['in_band', 'out_band']
    region_labels = ['Inside masked bands', 'Outside masked bands']
    width = 0.8 / len(keys)
    xs = np.arange(len(regions))
    for j, key in enumerate(keys):
        meds, lo_err, hi_err = [], [], []
        for region in regions:
            pr = raw[f'{key}__phase_real']
            pg = raw[f'{key}__phase_gen']
            rg = raw[f'{key}__region']
            m = rg == region
            if m.sum() == 0:
                meds.append(0); lo_err.append(0); hi_err.append(0); continue
            diffs_us = np.array([circular_diff(a, b, T_CHIRP_S) * 1e6
                                 for a, b in zip(pr[m], pg[m])])
            med = np.median(diffs_us)
            q25, q75 = np.percentile(diffs_us, [25, 75])
            meds.append(med); lo_err.append(med - q25); hi_err.append(q75 - med)
        offset = (j - (len(keys) - 1) / 2) * width
        ax.bar(xs + offset, meds, width=width,
               yerr=[lo_err, hi_err], capsize=3,
               color=MODEL_BAR_COLOR[key], edgecolor='black', linewidth=0.6,
               label=MODEL_STYLE[key]['label'])
    # Estimator noise floor reference line
    ax.axhline(1.58, color='gray', linestyle=':', linewidth=1.2,
               label='Estimator resolution (~1.6 µs)')
    ax.set_xticks(xs)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel('Median |Δφ_gen − Δφ_real|  (µs)')
    ax.set_title(f'Chirp phase error (median ± IQR) — {suffix}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save(fig, args.output_dir, f'phase_error_median_{suffix}')


def plot_degradation_profile(args, raw, keys, suffix):
    """Per-azimuth AoA error vs distance from nearest unmasked azimuth.
    Only meaningful for azimuths inside masked bands."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        az = raw[f'{key}__azimuth']
        dm = raw[f'{key}__dphi_measured']
        dt = raw[f'{key}__dphi_theory']
        rg = raw[f'{key}__region']
        m = rg == 'in_band'
        if m.sum() == 0:
            continue
        az_m = az[m]; err = np.rad2deg(np.abs(dm[m] - dt[m]))
        # distance from band edge
        def dist_to_edge(a):
            for lo, hi in MASKED_BANDS_DEG:
                if lo <= a <= hi:
                    return min(abs(a - lo), abs(a - hi))
            return 0.0
        dists = np.array([dist_to_edge(a) for a in az_m])
        order = np.argsort(dists)
        ax.plot(dists[order], err[order], marker='o', markersize=4,
                **MODEL_STYLE[key])
    ax.set_xlabel('Distance from nearest band edge (°)  — inside masked bands')
    ax.set_ylabel('AoA error (°)')
    ax.set_title(f'Degradation profile inside masked bands — {suffix}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save(fig, args.output_dir, f'degradation_profile_{suffix}')


def _save(fig, output_dir, basename):
    p = os.path.join(output_dir, basename)
    fig.savefig(f'{p}.pdf', bbox_inches='tight')
    fig.savefig(f'{p}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {p}.{{pdf,png}}')


def write_summary(args, raw, keys, suffix):
    path = os.path.join(args.output_dir, f'summary_{suffix}.txt')
    with open(path, 'w') as fh:
        fh.write(f'Comparison: {suffix}  ({", ".join(keys)})\n')
        fh.write('=' * 70 + '\n\n')
        for key in keys:
            fh.write(f'[{key}]\n')
            for region in ['in_band', 'out_band']:
                regs = raw[f'{key}__region']
                m = regs == region
                if m.sum() == 0:
                    continue
                dm = raw[f'{key}__dphi_measured'][m]
                dt = raw[f'{key}__dphi_theory'][m]
                err = np.rad2deg(np.abs(dm - dt))
                snr = raw[f'{key}__snr_db'][m]
                amp = raw[f'{key}__amp_ratio'][m]
                fmse = raw[f'{key}__freq_mse'][m]
                pr  = raw[f'{key}__phase_real'][m]
                pg  = raw[f'{key}__phase_gen'][m]
                pmatch = 100.0 * np.mean(
                    [circular_diff(a, b, T_CHIRP_S) < PHASE_TOL_S
                     for a, b in zip(pr, pg)])
                fh.write(f'  {region:8s}  N={m.sum():4d}  '
                         f'AoA_err={err.mean():5.2f}°±{err.std():4.2f}  '
                         f'SNR={snr.mean():+5.2f} dB  '
                         f'amp_ratio={amp.mean():.3f}  '
                         f'freq_mse={fmse.mean():.2e}  '
                         f'phase_match={pmatch:.1f}%\n')
            fh.write('\n')
    print(f'  wrote {path}')


def plot_phase(args):
    raw_path = os.path.join(args.output_dir, 'raw.npz')
    raw = np.load(raw_path, allow_pickle=True)
    available = list(raw['models'])
    requested = COMPARE_SETS[args.compare]
    keys = [k for k in requested if k in available]
    if not keys:
        raise RuntimeError(f'No models from {requested} found in {raw_path}. '
                           f'Available: {available}')
    print(f'Plotting comparison {args.compare!r}: {keys}')
    plot_aoa_regression(args, raw, keys, args.compare)
    plot_aoa_error(args, raw, keys, args.compare)
    plot_aoa_error_model_vs_data(args, raw, keys, args.compare)
    plot_aoa_error_data_vs_theory(args, raw, keys, args.compare)
    plot_spec_grid_antenna1(args, raw, keys, args.compare)
    plot_spec_grid_allants(args, raw, keys, args.compare)
    plot_scalar_metrics(args, raw, keys, args.compare)
    plot_phase_match(args, raw, keys, args.compare)
    plot_phase_error_median(args, raw, keys, args.compare)
    plot_degradation_profile(args, raw, keys, args.compare)
    write_summary(args, raw, keys, args.compare)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['eval', 'plot'], required=True)
    ap.add_argument('--output_dir', required=True)
    # eval-only
    ap.add_argument('--weights_xyz_full',   default=None)
    ap.add_argument('--weights_xyz_mask',   default=None)
    ap.add_argument('--weights_noxyz_full', default=None)
    ap.add_argument('--weights_noxyz_mask', default=None)
    ap.add_argument('--n_per_band',      type=int, default=100)
    ap.add_argument('--n_per_reference', type=int, default=100)
    ap.add_argument('--ddim_steps',      type=int, default=50)
    ap.add_argument('--task',            type=int, default=132)
    ap.add_argument('--seed',            type=int, default=0)
    ap.add_argument('--az_tolerance',    type=float, default=2.5,
                    help='Match azimuth within ±tol degrees of target')
    ap.add_argument('--spec_grid_azimuths', type=float, nargs='+',
                    default=DEFAULT_SPEC_AZ)
    # plot-only
    ap.add_argument('--compare', choices=list(COMPARE_SETS.keys()),
                    default='all4')
    args = ap.parse_args()

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'savefig.dpi': 300})

    if args.mode == 'eval':
        evaluate(args)
    else:
        plot_phase(args)


if __name__ == '__main__':
    main()
