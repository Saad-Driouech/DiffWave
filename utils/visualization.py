import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import math

_GNSS_FS = 40.5e6  # GNSS sampling frequency (Hz)
_C_LIGHT = 299_792_458.0

# Fixed position normalization — same constants as train.py (task 132)
_POS_MEAN = np.array([0.721, -0.034, -1.042], dtype=np.float32)
_POS_STD  = np.array([6.922,  4.555,  0.552], dtype=np.float32)


def _norm_pos(pos_np):
    return (pos_np - _POS_MEAN) / _POS_STD


def _env_floats(name, n, default):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    try:
        vals = [float(x) for x in raw.split(',')]
    except ValueError:
        print(f"[visualization] could not parse {name}; using defaults")
        return list(default)
    if len(vals) != n:
        print(f"[visualization] {name} expected {n} values, got {len(vals)}; using defaults")
        return list(default)
    return vals


def _aoa_geometry():
    """Antenna geometry / carrier / elevation reference for AoA plots.

    Set on the cluster via environment variables; defaults are zeros so
    no real values are committed to git:
        DIFFWAVE_ANT_X="x0,x1,x2,x3"        # metres
        DIFFWAVE_ANT_Y="y0,y1,y2,y3"
        DIFFWAVE_FC_HZ="1.575e9"
        DIFFWAVE_EL_REF_DEG="-30"           # central reference elevation
        DIFFWAVE_EL_RANGE_DEG="-67,-1"      # min,max for shaded band
    """
    ant_x = _env_floats('DIFFWAVE_ANT_X', 4, [0.0, 0.0, 0.0, 0.0])
    ant_y = _env_floats('DIFFWAVE_ANT_Y', 4, [0.0, 0.0, 0.0, 0.0])
    fc    = float(os.environ.get('DIFFWAVE_FC_HZ', '1.575e9'))
    el_ref = float(os.environ.get('DIFFWAVE_EL_REF_DEG', '0'))
    el_min, el_max = _env_floats('DIFFWAVE_EL_RANGE_DEG', 2, [0.0, 0.0])
    return np.array(ant_x), np.array(ant_y), fc, el_ref, el_min, el_max


def _dphi_baseline(pair, angles_rad, el_rad, ant_x, ant_y, fc_hz):
    """Theoretical Δφ between antennas pair=(i, j) for far-field source
    at (azimuth, elevation). Sign convention matches the dataset's, as
    verified empirically by tests/aoa_geometry_test.py (slope −1 vs
    physics-textbook k·r convention)."""
    i, j = pair
    dx, dy = ant_x[j] - ant_x[i], ant_y[j] - ant_y[i]
    k_mag = 2 * np.pi / (_C_LIGHT / fc_hz)
    return -k_mag * np.cos(el_rad) * (dx * np.cos(angles_rad) + dy * np.sin(angles_rad))


def _gnss_spectrogram_db(x, fs=_GNSS_FS, noverlap=64):
    """Return (f, t, Sxx_dB) using scipy's spectrogram (Blackman window, two-sided PSD)."""
    f, t, Sxx = scipy.signal.spectrogram(
        x, fs=fs, nperseg=128, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


def cossin_to_angle_deg(cossin):
    """Convert sin/cos representation to angle in degrees"""
    if type(cossin) == list:
        cossin = torch.tensor(cossin)
    if type(cossin) == np.ndarray:
        if len(cossin.shape) == 2:
            cossin = cossin[0]
        return np.arctan2(cossin[0], cossin[1]) * 180.0 / math.pi
    return torch.atan2(cossin[:, 0], cossin[:, 1]) * 180.0 / math.pi


def angle_deg_to_cossin(angle):
    """Convert angle in degrees to sin/cos representation"""
    if type(angle) == list:
        angle = torch.tensor(angle)
    angle_rad = angle * math.pi / 180.0
    if type(angle) == np.ndarray:
        return np.stack([np.sin(angle_rad), np.cos(angle_rad)], axis=1)
    if isinstance(angle, (float, np.floating)):
        return np.array([np.sin(angle_rad), np.cos(angle_rad)])
    return torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=1)


class DiffusionVisualizer:
    def __init__(self, writer, engine, device, n_eval=16):
        self.writer = writer
        self.engine = engine
        self.device = device
        self.n_eval = n_eval
        # Load n_eval fixed test samples once at init so every epoch visualizes
        # the exact same samples — enables meaningful epoch-over-epoch comparison.
        from UniversalDataLoader import UniversalDataset
        from utils.chirp_phase import extract_phase_sincos
        ds = UniversalDataset(task_id=132, mode='test', angle_mode='sincos')
        signals, geom_conds = [], []
        for i in range(n_eval):
            x, (pos, az, el) = ds[i]
            signals.append(x)
            pos_norm = torch.tensor(_norm_pos(pos.numpy()), dtype=torch.float32)
            geom_conds.append(torch.cat([pos_norm, az.float(), el.float()]))  # [7]
        x = torch.stack(signals)                          # [n_eval, 4, 1024] complex64
        phase_sc = extract_phase_sincos(x.to(device))     # [n_eval, 2]
        inp = torch.cat([x.real.float(), x.imag.float()], dim=1)  # [n_eval, 8, 1024]
        mean = inp.mean(dim=(1, 2), keepdim=True)
        std  = inp.std(dim=(1, 2), keepdim=True) + 1e-8
        self.fixed_batch     = ((inp - mean) / std).to(device)
        geom = torch.stack(geom_conds).to(device)         # [n_eval, 7]
        self.fixed_condition = torch.cat([geom, phase_sc], dim=1)  # [n_eval, 9]

    def _to_complex(self, batch):
        """Convert DiffWave (B, 2*n_ant, L) → complex numpy (B, L, n_ant)."""
        n = batch.shape[1] // 2
        return (batch[:, :n, :].permute(0, 2, 1) + 1j * batch[:, n:, :].permute(0, 2, 1)).cpu().numpy()

    def log_all(self, epoch):
        real_batch = self.fixed_batch     # [N, 8, 1024]
        condition  = self.fixed_condition # [N, 9]
        N = real_batch.shape[0]
        print(f"[DiffusionVisualizer] log_all: fixed_batch={real_batch.shape}, "
              f"condition={condition.shape}, epoch={epoch}")

        # Pre-generate all N samples once — reused by every RF and Samples method.
        self.engine.model.eval()
        with torch.no_grad():
            gen_batch, _ = self.engine.sample_ddim(N, 1024, condition, steps=50)

        methods = {
            'log_noise_schedule':               lambda: self.log_noise_schedule(epoch),
            'log_denoising_chain':              lambda: self.log_denoising_chain(epoch),
            'log_aoa_verification':             lambda: self.log_aoa_verification(epoch),
            'log_aoa_regression':               lambda: self.log_aoa_regression(epoch),
            'log_aoa_phase_emergence':          lambda: self.log_aoa_phase_emergence(epoch),
            'log_spectral_fidelity':            lambda: self.log_spectral_fidelity(real_batch, gen_batch, epoch),
            'log_weight_histograms':            lambda: self.log_weight_histograms(self.engine.model, epoch),
            'log_multi_antenna_comparison':     lambda: self.log_multi_antenna_comparison(real_batch, gen_batch, epoch),
            'log_constellation_grid':           lambda: self.log_constellation_grid(real_batch, gen_batch, epoch),
            'log_cross_antenna_correlation':    lambda: self.log_cross_antenna_correlation(real_batch, condition, epoch),
            'log_prediction_error_vs_timestep': lambda: self.log_prediction_error_vs_timestep(real_batch, condition, epoch),
            'log_aoa_sweep':                    lambda: self.log_aoa_sweep(epoch),
            'log_skip_norms':                   lambda: self.log_skip_norms(self.engine.model, real_batch, condition, epoch),
            'log_psd_semilogy':                 lambda: self.log_psd_semilogy(real_batch, gen_batch, epoch),
            'log_spectrogram_comparison':       lambda: self.log_spectrogram_comparison(real_batch, gen_batch, epoch),
            'log_time_amplitude_rf':            lambda: self.log_time_amplitude_rf(real_batch, gen_batch, epoch),
            'log_iq_time_series_rf':            lambda: self.log_iq_time_series_rf(real_batch, gen_batch, epoch),
            'log_iq_constellation_rf':          lambda: self.log_iq_constellation_rf(real_batch, gen_batch, epoch),
            'log_degradation_steps':            lambda: self.log_degradation_steps(real_batch, epoch),
            'log_stft_spectrogram':             lambda: self.log_stft_spectrogram(real_batch, gen_batch, epoch),
            'log_rf_scalars':                   lambda: self.log_rf_scalars(real_batch, condition, epoch),
            'log_samples_panel':                lambda: self.log_samples_panel(real_batch, gen_batch, epoch),
        }
        for name, fn in methods.items():
            try:
                fn()
            except Exception as e:
                print(f"[DiffusionVisualizer] {name} failed: {e}")

    # ------------------------------------------------------------------
    # EXISTING METHODS (no real_batch change)
    # ------------------------------------------------------------------

    def log_denoising_chain(self, epoch):
        """Visualizes the reverse process: Noise -> Signal"""
        cond = torch.zeros(1, 9).to(self.device)
        cond[:, 8] = 1.0   # cos(2π·0/T) = 1, phase fixed at 0

        self.engine.model.eval()
        with torch.no_grad():
            _, sigs = self.engine.sample_ddim(1, 1024, cond, steps=50)

        for i in (1, 10, 20, 30, 40, 50):
            i = min(i, len(sigs) - 1)
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            sig_np = sigs[i - 1][0, 0, :]
            ax.plot(sig_np)
            ax.set_title(f"Denoising Step {i}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            self.writer.add_figure(f'Diffusion/Denoising_Step_{i}', fig, epoch)
            plt.close(fig)

    def log_noise_schedule(self, epoch):
        """Plots the cosine noise schedule and derived SNR curve. Log once at epoch 0."""
        if epoch != 0:
            return
        alphas = self.engine.alphas_cumprod.cpu().numpy()
        t = np.arange(len(alphas))
        snr = alphas / (1 - alphas + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(t, alphas)
        axes[0].set_title("Alpha Bar (Cumulative Product)")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("ᾱ_t")
        axes[0].grid(True)

        axes[1].semilogy(t, snr)
        axes[1].set_title("Signal-to-Noise Ratio vs Timestep")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("SNR = ᾱ / (1 - ᾱ)")
        axes[1].grid(True)

        plt.tight_layout()
        self.writer.add_figure('Diffusion/Noise_Schedule', fig, epoch)
        plt.close(fig)

    def log_weight_histograms(self, model, epoch):
        """Logs weight distributions for key layers."""
        layers = {
            'input_projection': model.input_projection,
            'cond_mlp_0': model.cond_mlp[0],
            'cond_mlp_2': model.cond_mlp[2],
            'output_proj_0': model.output_projection[0],
            'output_proj_2': model.output_projection[2],
        }
        for name, layer in layers.items():
            if hasattr(layer, 'weight') and layer.weight is not None:
                self.writer.add_histogram(f'Weights/{name}', layer.weight.detach().cpu(), epoch)
            if hasattr(layer, 'bias') and layer.bias is not None:
                self.writer.add_histogram(f'Weights/{name}_bias', layer.bias.detach().cpu(), epoch)

    def log_cross_antenna_correlation(self, real_batch, condition, epoch):
        """Spatial correlation matrix |R| for real and generated signals."""
        B = min(real_batch.shape[0], 16)
        cond_b = condition[:B].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen_batch, _ = self.engine.sample_ddim(B, 1024, cond_b, steps=50)

        n_ant = real_batch.shape[1] // 2

        def corr_matrix(batch_np):
            mats = []
            for b in range(batch_np.shape[0]):
                X = batch_np[b, :n_ant] + 1j * batch_np[b, n_ant:]
                R = np.abs(X @ X.conj().T) / X.shape[1]
                mats.append(R)
            return np.mean(mats, axis=0)

        R_real = corr_matrix(real_batch[:B].cpu().numpy())
        R_gen  = corr_matrix(gen_batch.cpu().numpy())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmax = max(R_real.max(), R_gen.max())
        for ax, R, title in zip(axes, [R_real, R_gen], ['Real', 'Generated']):
            im = ax.imshow(R, vmin=0, vmax=vmax, cmap='viridis')
            ax.set_title(f"Cross-Antenna Correlation — {title}")
            ax.set_xlabel("Antenna")
            ax.set_ylabel("Antenna")
            ax.set_xticks(range(n_ant))
            ax.set_yticks(range(n_ant))
            ax.set_xticklabels([f"Ant{i+1}" for i in range(n_ant)], rotation=45, fontsize=7)
            ax.set_yticklabels([f"Ant{i+1}" for i in range(n_ant)], fontsize=7)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        self.writer.add_figure('Signals/Cross_Antenna_Correlation', fig, epoch)
        plt.close(fig)

    def log_prediction_error_vs_timestep(self, real_batch, condition, epoch):
        """Plots model prediction MSE as a function of noise timestep t."""
        n_probe = 20
        probe_ts = torch.linspace(0, self.engine.timesteps - 1, n_probe).long().to(self.device)
        B = min(real_batch.shape[0], 8)
        x = real_batch[:B].to(self.device)
        cond = condition[:B].to(self.device)

        errors = []
        self.engine.model.eval()
        with torch.no_grad():
            for t_val in probe_ts:
                t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
                noise = torch.randn_like(x)
                x_t, _ = self.engine.add_noise(x, t_batch, noise)
                noise_pred = self.engine.model(x_t, t_batch, cond)
                mse = torch.nn.functional.mse_loss(noise_pred, noise).item()
                errors.append(mse)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(probe_ts.cpu().numpy(), errors, marker='o', linewidth=1.5)
        ax.set_title("Noise Prediction MSE vs Timestep")
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        self.writer.add_figure('Diffusion/Prediction_Error_vs_Timestep', fig, epoch)
        plt.close(fig)

    def log_aoa_sweep(self, epoch):
        """Generates signals at 6 AoA angles spanning the dataset's full
        azimuth range, and compares measured Δφ to the planar geometry."""
        angles_deg = np.linspace(-150, 150, 6)
        angles_rad = angles_deg * math.pi / 180.0

        _, _, _, el_ref_deg, _, _ = _aoa_geometry()
        el_ref_rad = np.deg2rad(el_ref_deg)
        conds = torch.tensor(
            [[0.0, 0.0, 0.0, math.sin(a), math.cos(a), math.sin(el_ref_rad), math.cos(el_ref_rad)]
             for a in angles_rad],
            dtype=torch.float32, device=self.device
        )

        self.engine.model.eval()
        gen_signals = []
        with torch.no_grad():
            for i in range(len(angles_deg)):
                cond_single = conds[i:i+1]
                sig, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=20)
                gen_signals.append(sig[0].cpu().numpy())

        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        axes = axes.flatten()
        for i, (sig, angle) in enumerate(zip(gen_signals, angles_deg)):
            axes[i].plot(sig[0, :256], linewidth=0.8)
            axes[i].set_title(f"AoA = {angle:.0f}°")
            axes[i].set_xlabel("Sample")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)
        plt.suptitle("Generated Signal (Ant 1, I-channel) vs AoA", fontsize=12)
        plt.tight_layout()
        self.writer.add_figure('Conditioning/AoA_Sweep_TimeDomain', fig, epoch)
        plt.close(fig)

        n_ant = gen_signals[0].shape[0] // 2
        measured_phases = []
        for sig in gen_signals:
            ant1 = sig[0] + 1j * sig[n_ant]
            ant2 = sig[1] + 1j * sig[n_ant + 1]
            phase = np.angle(np.mean(ant2 * ant1.conj()))
            measured_phases.append(phase)

        ant_x, ant_y, fc, el_ref_deg, _, _ = _aoa_geometry()
        theory = _dphi_baseline(
            (0, 1), angles_rad, np.deg2rad(el_ref_deg), ant_x, ant_y, fc)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(angles_deg, np.array(measured_phases), marker='o',
                label='Measured (ant1→ant2)')
        ax.plot(angles_deg, theory, 'r--',
                label=f'Theory el={el_ref_deg:.0f}° (2×2 planar)')
        ax.set_title("Inter-Antenna Phase Difference vs AoA Condition")
        ax.set_xlabel("Conditioned AoA (degrees)")
        ax.set_ylabel("Measured Phase Diff (rad)")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        self.writer.add_figure('Conditioning/AoA_Phase_vs_Angle', fig, epoch)
        plt.close(fig)

    def log_skip_norms(self, model, real_batch, condition, epoch):
        """Per-block skip connection L2 norm via forward hooks."""
        skip_norms = []
        hooks = []

        def make_hook(idx):
            def hook_fn(_module, _input, output):
                _, skip = output
                norm = skip.detach().norm(dim=1).mean().item()
                skip_norms.append((idx, norm))
            return hook_fn

        for idx, block in enumerate(model.blocks):
            h = block.register_forward_hook(make_hook(idx))
            hooks.append(h)

        model.eval()
        B = min(real_batch.shape[0], 4)
        x = real_batch[:B].to(self.device)
        cond = condition[:B].to(self.device)
        t = torch.zeros(B, dtype=torch.long, device=self.device)

        with torch.no_grad():
            model(x, t, cond)

        for h in hooks:
            h.remove()

        skip_norms.sort(key=lambda v: v[0])
        indices = [v[0] for v in skip_norms]
        norms   = [v[1] for v in skip_norms]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(indices, norms, color='steelblue', alpha=0.8)
        ax.set_title("Skip Connection L2 Norm per Block")
        ax.set_xlabel("Block Index")
        ax.set_ylabel("Mean L2 Norm")
        ax.set_xticks(indices)
        ax.grid(True, axis='y', alpha=0.4)
        plt.tight_layout()
        self.writer.add_figure('Model/Skip_Connection_Norms', fig, epoch)
        plt.close(fig)

    # ------------------------------------------------------------------
    # PHYSICS / AoA  (fixed target, new regression plot)
    # ------------------------------------------------------------------

    def log_aoa_verification(self, epoch):
        """Phase consistency histogram for baseline (0,1) at fixed AoA angles
        spanning the dataset's full azimuth range. Theoretical reference uses
        the actual 2×2 planar geometry at a representative elevation."""
        fixed_angles_deg = [-150, -75, 0, 75, 150]
        n_gen = 32
        ant_x, ant_y, fc, el_ref_deg, _, _ = _aoa_geometry()
        el_ref_rad = np.deg2rad(el_ref_deg)

        self.engine.model.eval()
        fig, axes = plt.subplots(1, len(fixed_angles_deg), figsize=(18, 3.5))

        with torch.no_grad():
            for ax, angle_deg in zip(axes, fixed_angles_deg):
                angle_rad = angle_deg * math.pi / 180.0
                cond = torch.zeros(n_gen, 9, device=self.device)
                cond[:, 3] = math.sin(angle_rad)
                cond[:, 4] = math.cos(angle_rad)
                cond[:, 5] = math.sin(el_ref_rad)
                cond[:, 6] = math.cos(el_ref_rad)
                cond[:, 8] = 1.0   # phase fixed at 0

                gen_data, _ = self.engine.sample_ddim(n_gen, 1024, cond, steps=20)
                n_ant = gen_data.shape[1] // 2
                ant1 = torch.complex(gen_data[:, 0], gen_data[:, n_ant])
                ant2 = torch.complex(gen_data[:, 1], gen_data[:, n_ant + 1])
                phase_diffs = torch.angle(
                    torch.mean(ant2 * ant1.conj(), dim=1)).cpu().numpy()

                expected = float(_dphi_baseline(
                    (0, 1), angle_rad, el_ref_rad, ant_x, ant_y, fc))

                ax.hist(phase_diffs, bins=20, color='orange', alpha=0.7, density=True)
                ax.axvline(expected, color='r', linestyle='--', linewidth=1.5,
                           label=f'Expected {expected:.2f} rad')
                ax.set_title(f'{angle_deg}°', fontsize=10)
                ax.set_xlabel('Δφ (rad)')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'AoA Phase Consistency — baseline (0,1), 2×2 planar array, '
            f'el_ref={el_ref_deg:.0f}° — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('Physics/AoA_Consistency', fig, epoch)
        plt.close(fig)

    def log_aoa_phase_emergence(self, epoch):
        """For several denoising timesteps t, build AoA-Consistency-style
        histograms of inter-antenna Δφ measured on the partially-denoised
        sample x_t. Reveals at which point in the reverse diffusion process
        the model commits to the correct phase relationship.

        One figure per snapshot t, logged under Physics/AoA_Emergence/tNNNN.
        """
        fixed_angles_deg = [-150, -75, 0, 75, 150]
        n_gen = 32
        ddim_steps = 50
        target_ts = [999, 800, 600, 400, 200, 0]

        T = self.engine.timesteps
        # DDIM linspace timesteps, descending (start of each reverse step):
        ddim_t_schedule = np.linspace(0, T - 1, ddim_steps).astype(int)[::-1]

        ant_x, ant_y, fc, el_ref_deg, _, _ = _aoa_geometry()
        el_ref_rad = np.deg2rad(el_ref_deg)

        # Generate per-angle and capture all intermediates.
        self.engine.model.eval()
        intermediates = {}
        with torch.no_grad():
            for angle_deg in fixed_angles_deg:
                angle_rad = angle_deg * math.pi / 180.0
                cond = torch.zeros(n_gen, 9, device=self.device)
                cond[:, 3] = math.sin(angle_rad)
                cond[:, 4] = math.cos(angle_rad)
                cond[:, 5] = math.sin(el_ref_rad)
                cond[:, 6] = math.cos(el_ref_rad)
                cond[:, 8] = 1.0   # phase fixed at 0
                _, sigs = self.engine.sample_ddim(
                    n_gen, 1024, cond, steps=ddim_steps)
                intermediates[angle_deg] = sigs

        # Build one figure per snapshot timestep.
        for target_t in target_ts:
            idx = int(np.argmin(np.abs(ddim_t_schedule - target_t)))
            idx = min(idx, len(intermediates[fixed_angles_deg[0]]) - 1)
            actual_t = int(ddim_t_schedule[idx])

            fig, axes = plt.subplots(1, len(fixed_angles_deg), figsize=(18, 3.5))
            for ax, angle_deg in zip(axes, fixed_angles_deg):
                x_t = intermediates[angle_deg][idx]
                if not torch.is_tensor(x_t):
                    x_t = torch.as_tensor(x_t)
                n_ant = x_t.shape[1] // 2
                ant1 = torch.complex(x_t[:, 0], x_t[:, n_ant])
                ant2 = torch.complex(x_t[:, 1], x_t[:, n_ant + 1])
                phase_diffs = torch.angle(
                    torch.mean(ant2 * ant1.conj(), dim=1)).cpu().numpy()

                angle_rad = angle_deg * math.pi / 180.0
                expected = float(_dphi_baseline(
                    (0, 1), angle_rad, el_ref_rad, ant_x, ant_y, fc))

                ax.hist(phase_diffs, bins=24, range=(-np.pi, np.pi),
                        color='orange', alpha=0.7, density=True)
                ax.axvline(expected, color='r', linestyle='--', linewidth=1.5,
                           label=f'Expected {expected:.2f} rad')
                ax.set_title(f'{angle_deg}°', fontsize=10)
                ax.set_xlabel('Δφ (rad)')
                ax.set_xlim(-np.pi, np.pi)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.suptitle(
                f'AoA Phase Emergence — t≈{actual_t} '
                f'(DDIM step {idx + 1}/{ddim_steps}) — epoch {epoch}')
            plt.tight_layout()
            self.writer.add_figure(
                f'Physics/AoA_Emergence/t{actual_t:04d}', fig, epoch)
            plt.close(fig)

    def log_aoa_regression(self, epoch):
        """Conditioned AoA vs measured Δφ for baseline (0,1), with theoretical
        reference computed from the actual 2×2 planar geometry at the
        dataset's mean elevation, plus a shaded band over the dataset's
        elevation range."""
        angles_deg = np.linspace(-180, 180, 37)
        n_gen = 16
        ant_x, ant_y, fc, el_ref_deg, el_min_deg, el_max_deg = _aoa_geometry()

        el_ref_rad = np.deg2rad(el_ref_deg)
        self.engine.model.eval()
        mean_phases, std_phases = [], []

        with torch.no_grad():
            for angle_deg in angles_deg:
                angle_rad = angle_deg * math.pi / 180.0
                cond = torch.zeros(n_gen, 9, device=self.device)
                cond[:, 3] = math.sin(angle_rad)
                cond[:, 4] = math.cos(angle_rad)
                cond[:, 5] = math.sin(el_ref_rad)
                cond[:, 6] = math.cos(el_ref_rad)
                cond[:, 8] = 1.0   # phase fixed at 0

                gen_data, _ = self.engine.sample_ddim(n_gen, 1024, cond, steps=20)
                n_ant = gen_data.shape[1] // 2
                ant1 = torch.complex(gen_data[:, 0], gen_data[:, n_ant])
                ant2 = torch.complex(gen_data[:, 1], gen_data[:, n_ant + 1])
                phases = torch.angle(torch.mean(ant2 * ant1.conj(), dim=1))
                mean_phases.append(phases.mean().item())
                std_phases.append(phases.std().item())

        mean_phases = np.array(mean_phases)
        std_phases  = np.array(std_phases)
        ang_rad     = np.deg2rad(angles_deg)
        theoretical = _dphi_baseline(
            (0, 1), ang_rad, np.deg2rad(el_ref_deg), ant_x, ant_y, fc)
        theory_min  = _dphi_baseline(
            (0, 1), ang_rad, np.deg2rad(el_min_deg), ant_x, ant_y, fc)
        theory_max  = _dphi_baseline(
            (0, 1), ang_rad, np.deg2rad(el_max_deg), ant_x, ant_y, fc)
        band_lo = np.minimum(theory_min, theory_max)
        band_hi = np.maximum(theory_min, theory_max)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(angles_deg, band_lo, band_hi, color='red', alpha=0.15,
                        label=f'Theory band  el ∈ [{el_min_deg:.0f}, {el_max_deg:.0f}]°')
        ax.plot(angles_deg, theoretical, 'r--', linewidth=1.5,
                label=f'Theory  el={el_ref_deg:.0f}° (2×2 planar)')
        ax.errorbar(angles_deg, mean_phases, yerr=std_phases, fmt='o-',
                    capsize=3, linewidth=1.2, markersize=4,
                    label='Measured (mean ± std)', color='steelblue')
        ax.set_xlabel('Conditioned AoA (°)')
        ax.set_ylabel('Measured Δφ — baseline (0,1) (rad)')
        ax.set_title(f'AoA Conditioning Regression — epoch {epoch}')
        ax.set_xlim(-180, 180)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.writer.add_figure('Physics/AoA_Regression', fig, epoch)
        plt.close(fig)

    # ------------------------------------------------------------------
    # RF PANEL  (aggregate: mean / mean±std over all N fixed samples)
    # ------------------------------------------------------------------

    def log_spectral_fidelity(self, real_batch, gen_batch, epoch):
        """PSD overlay for all N samples, antenna 1."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)  # (N, 1024, n_ant)
        g_all = self._to_complex(gen_batch)

        fig, ax = plt.subplots()
        for b in range(N):
            ax.psd(x_all[b, :, 0], Fs=1.0, NFFT=512,
                   color='steelblue', alpha=0.4, label='Real' if b == 0 else '')
            ax.psd(g_all[b, :, 0], Fs=1.0, NFFT=512,
                   color='crimson', alpha=0.4, linestyle='--', label='Generated' if b == 0 else '')
        ax.legend()
        ax.set_title(f"Power Spectral Density — Antenna 1 (N={N})")
        self.writer.add_figure('Fidelity/PSD', fig, epoch)
        plt.close(fig)

    def log_multi_antenna_comparison(self, real_batch, gen_batch, epoch):
        """Per-antenna figure: real I-channel vs generated I-channel (sample 0)."""
        n_ant = real_batch.shape[1] // 2
        real_np = real_batch[0].cpu().numpy()
        gen_np  = gen_batch[0].cpu().numpy()

        n_cols = min(n_ant, 4)
        n_rows = math.ceil(n_ant / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
        axes = np.array(axes).flatten()

        for ant in range(n_ant):
            ax = axes[ant]
            ax.plot(real_np[ant, :256], label='Real',      alpha=0.8, linewidth=0.8)
            ax.plot(gen_np[ant,  :256], label='Generated', alpha=0.8, linewidth=0.8, linestyle='--')
            ax.set_title(f"Antenna {ant + 1} — I channel")
            ax.set_xlabel("Sample")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure('Signals/Multi_Antenna_TimeDomain', fig, epoch)
        plt.close(fig)

    def log_constellation_grid(self, real_batch, gen_batch, epoch):
        """I/Q constellation plots (real=blue, generated=red) for all antennas (sample 0)."""
        n_ant = real_batch.shape[1] // 2
        real_np = real_batch[0].cpu().numpy()
        gen_np  = gen_batch[0].cpu().numpy()

        n_cols = min(n_ant, 4)
        n_rows = math.ceil(n_ant / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        axes = np.array(axes).flatten()

        for ant in range(n_ant):
            ax = axes[ant]
            ax.scatter(real_np[ant], real_np[ant + n_ant], alpha=0.3, s=1, c='steelblue', label='Real')
            ax.scatter(gen_np[ant],  gen_np[ant + n_ant],  alpha=0.3, s=1, c='crimson',   label='Generated')
            ax.set_title(f"Ant {ant + 1}", fontsize=9)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.3)
            if ant == 0:
                ax.legend(fontsize=7, markerscale=5)

        plt.suptitle("Constellation Grid — Real vs Generated", fontsize=12)
        plt.tight_layout()
        self.writer.add_figure('Signals/Constellation_Grid', fig, epoch)
        plt.close(fig)

    def log_psd_semilogy(self, real_batch, gen_batch, epoch):
        """PSD — mean ± std over all N samples, per antenna."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)  # (N, 1024, n_ant)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]

        freqs  = np.fft.fftshift(np.fft.fftfreq(1024))
        n_cols = min(n_ant, 4)
        n_rows = math.ceil(n_ant / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
        for i, ax in enumerate(np.array(axes).flatten()):
            psds_r = np.array([np.abs(np.fft.fftshift(np.fft.fft(x_all[b, :, i])))**2 for b in range(N)])
            psds_g = np.array([np.abs(np.fft.fftshift(np.fft.fft(g_all[b, :, i])))**2 for b in range(N)])

            r_mean = psds_r.mean(0); r_std = psds_r.std(0)
            g_mean = psds_g.mean(0); g_std = psds_g.std(0)

            ax.semilogy(freqs, r_mean, label='Real',      alpha=0.9, lw=1.2, color='steelblue')
            ax.fill_between(freqs,
                            np.maximum(r_mean - r_std, 1e-20),
                            r_mean + r_std, alpha=0.2, color='steelblue')
            ax.semilogy(freqs, g_mean, label='Generated', alpha=0.9, lw=1.2,
                        linestyle='--', color='crimson')
            ax.fill_between(freqs,
                            np.maximum(g_mean - g_std, 1e-20),
                            g_mean + g_std, alpha=0.2, color='crimson')
            ax.set_title(f'Antenna {i+1} PSD')
            ax.set_xlabel('Normalised Frequency')
            ax.set_ylabel('Power')
            ax.legend(fontsize=7)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.suptitle(f'PSD (mean ± std, N={N}) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/psd_per_antenna', fig, epoch)
        plt.close(fig)

    def log_spectrogram_comparison(self, real_batch, gen_batch, epoch):
        """Spectrogram — mean over all N samples, per antenna (n_ant × 2 grid)."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)  # (N, 1024, n_ant)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]

        fig, axes = plt.subplots(n_ant, 2, figsize=(12, n_ant * 4))
        for i in range(n_ant):
            Sxx_reals, Sxx_gens = [], []
            for b in range(N):
                f_ax, t_ax_s, S = _gnss_spectrogram_db(x_all[b, :, i])
                Sxx_reals.append(S)
                _, _, S = _gnss_spectrogram_db(g_all[b, :, i])
                Sxx_gens.append(S)
            Sxx_real = np.mean(Sxx_reals, axis=0)
            Sxx_gen  = np.mean(Sxx_gens,  axis=0)
            vmin = min(Sxx_real.min(), Sxx_gen.min())
            vmax = max(Sxx_real.max(), Sxx_gen.max())
            extent = [t_ax_s[0] * 1e3, t_ax_s[-1] * 1e3, f_ax[0], f_ax[-1]]
            for ax, Sxx, title in zip(axes[i], [Sxx_real, Sxx_gen], ['Real', 'Generated']):
                im = ax.imshow(Sxx, aspect='auto', origin='lower', cmap='turbo',
                               vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
                ax.set_title(f'Antenna {i+1} — {title} (mean, N={N})')
                ax.set_xlabel('t [ms]')
                ax.set_ylabel('f [Hz]')
                fig.colorbar(im, ax=ax, format='%+.0f dB-Hz')
        plt.suptitle(f'Spectrogram (averaged over {N} samples) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/spectrogram', fig, epoch)
        plt.close(fig)

    def log_time_amplitude_rf(self, real_batch, gen_batch, epoch):
        """Time-domain amplitude |IQ| — mean ± std over all N samples."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]

        t_ax   = np.arange(256)
        n_cols = min(n_ant, 4)
        n_rows = math.ceil(n_ant / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
        for i, ax in enumerate(np.array(axes).flatten()):
            amps_r = np.array([np.abs(x_all[b, :256, i]) for b in range(N)])
            amps_g = np.array([np.abs(g_all[b, :256, i]) for b in range(N)])
            r_mean = amps_r.mean(0); r_std = amps_r.std(0)
            g_mean = amps_g.mean(0); g_std = amps_g.std(0)
            ax.plot(t_ax, r_mean, label='Real',      alpha=0.85, lw=1.2, color='steelblue')
            ax.fill_between(t_ax, r_mean - r_std, r_mean + r_std, alpha=0.2, color='steelblue')
            ax.plot(t_ax, g_mean, label='Generated', alpha=0.85, lw=1.2,
                    linestyle='--', color='crimson')
            ax.fill_between(t_ax, g_mean - g_std, g_mean + g_std, alpha=0.2, color='crimson')
            ax.set_title(f'Antenna {i+1}  |IQ| (mean ± std)')
            ax.set_xlabel('Sample')
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', linewidth=0.4)
        plt.suptitle(f'Time-domain Amplitude — first 256 samples (N={N}) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/time_amplitude', fig, epoch)
        plt.close(fig)

    def log_iq_time_series_rf(self, real_batch, gen_batch, epoch):
        """I(t) and Q(t) — mean ± std over all N samples, per antenna."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]
        t_iq  = np.arange(256)

        fig, axes = plt.subplots(n_ant, 2, figsize=(14, n_ant * 3))
        for i in range(n_ant):
            I_r = np.array([x_all[b, :256, i].real for b in range(N)])
            Q_r = np.array([x_all[b, :256, i].imag for b in range(N)])
            I_g = np.array([g_all[b, :256, i].real for b in range(N)])
            Q_g = np.array([g_all[b, :256, i].imag for b in range(N)])
            for ax, R, G, label in zip(
                [axes[i, 0], axes[i, 1]], [I_r, Q_r], [I_g, Q_g], ['I(t)', 'Q(t)']
            ):
                r_mean = R.mean(0); r_std = R.std(0)
                g_mean = G.mean(0); g_std = G.std(0)
                ax.plot(t_iq, r_mean, label='Real',      lw=1.0, alpha=0.85, color='steelblue')
                ax.fill_between(t_iq, r_mean - r_std, r_mean + r_std, alpha=0.2, color='steelblue')
                ax.plot(t_iq, g_mean, label='Generated', lw=1.0, alpha=0.85,
                        linestyle='--', color='crimson')
                ax.fill_between(t_iq, g_mean - g_std, g_mean + g_std, alpha=0.2, color='crimson')
                ax.set_title(f'Antenna {i+1} — {label} (mean ± std)')
                ax.set_xlabel('Sample')
                ax.legend(fontsize=7)
                ax.grid(True, linestyle='--', linewidth=0.4)
        plt.suptitle(f'IQ Time Series (first 256 samples, N={N}) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/iq_time_series', fig, epoch)
        plt.close(fig)

    def log_iq_constellation_rf(self, real_batch, gen_batch, epoch):
        """IQ Constellation — all N samples overlaid per antenna."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]

        n_cols = min(n_ant, 4)
        n_rows = math.ceil(n_ant / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 5))
        for i, ax in enumerate(np.array(axes).flatten()):
            for b in range(N):
                ax.scatter(x_all[b, :, i].real, x_all[b, :, i].imag,
                           s=1, alpha=max(0.1, 0.5 / N),
                           color='steelblue', label='Real'      if b == 0 else '')
                ax.scatter(g_all[b, :, i].real, g_all[b, :, i].imag,
                           s=1, alpha=max(0.1, 0.5 / N),
                           color='crimson',   label='Generated' if b == 0 else '')
            ax.set_title(f'IQ Constellation — Antenna {i+1} (N={N})')
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            ax.legend(fontsize=7, markerscale=6)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', linewidth=0.4)
        plt.suptitle(f'IQ Constellation — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/iq_constellation', fig, epoch)
        plt.close(fig)

    def log_degradation_steps(self, real_batch, epoch):
        """Forward degradation spectrogram — Antenna 1, 5 timesteps (mean over N samples)."""
        N = real_batch.shape[0]
        steps_to_show = [0, 25, 50, 75, 99]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for ax, step in zip(axes, steps_to_show):
            t_s   = torch.full((N,), step, dtype=torch.long, device=self.device)
            noise = torch.randn_like(real_batch)
            x_deg, _ = self.engine.add_noise(real_batch.to(self.device), t_s, noise.to(self.device))
            n_ant = x_deg.shape[1] // 2
            Sxx_list = []
            for b in range(N):
                sig = x_deg[b, 0].cpu().numpy() + 1j * x_deg[b, n_ant].cpu().numpy()
                _, _, Sxx = _gnss_spectrogram_db(sig)
                Sxx_list.append(Sxx)
            mean_Sxx = np.mean(Sxx_list, axis=0)
            ax.imshow(mean_Sxx, aspect='auto', origin='lower', cmap='turbo',
                      interpolation='nearest')
            ax.set_title(f't = {step}')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('f [Hz]' if step == 0 else '')
        plt.suptitle(f'Forward Degradation — Antenna 1 (mean, N={N}) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/degradation_steps', fig, epoch)
        plt.close(fig)

    def log_stft_spectrogram(self, real_batch, gen_batch, epoch):
        """STFT spectrogram — mean over all N samples, Antenna 1."""
        N = real_batch.shape[0]
        n_fft = 24
        hop_length = 17

        specs_real, specs_gen = [], []
        for b in range(N):
            data_sig = real_batch[b, 0, :512].cpu()
            pred_sig = gen_batch[b,  0, :512].cpu()
            specs_real.append(
                torch.abs(torch.stft(data_sig, n_fft=n_fft, hop_length=hop_length,
                                     return_complex=True)).numpy())
            specs_gen.append(
                torch.abs(torch.stft(pred_sig, n_fft=n_fft, hop_length=hop_length,
                                     return_complex=True)).numpy())

        mean_real_db = 20 * np.log10(np.mean(specs_real, axis=0) + 1e-6)
        mean_gen_db  = 20 * np.log10(np.mean(specs_gen,  axis=0) + 1e-6)

        fig = plt.figure(figsize=(6, 3))
        ax1 = plt.subplot(1, 2, 1)
        im1 = ax1.matshow(mean_real_db, cmap='viridis', origin='lower')
        ax1.set_title(f'Data STFT (mean, N={N})')
        plt.colorbar(im1, format='%+2.0f dB', ax=ax1, orientation='horizontal', pad=0.05)
        ax2 = plt.subplot(1, 2, 2)
        im2 = ax2.matshow(mean_gen_db, cmap='viridis', origin='lower')
        ax2.set_title(f'Generated STFT (mean, N={N})')
        plt.colorbar(im2, format='%+2.0f dB', ax=ax2, orientation='horizontal', pad=0.05)

        plt.suptitle(f'STFT Spectrogram — Antenna 1 — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/stft_spectrogram', fig, epoch)
        plt.close(fig)

    def log_rf_scalars(self, real_batch, condition, epoch):
        """Scalar metrics: freq_mse, snr_db, amp_ratio, cond_norm."""
        B = min(real_batch.shape[0], 8)
        x    = real_batch[:B].to(self.device)
        cond = condition[:B].to(self.device)

        self.engine.model.eval()
        with torch.no_grad():
            t_max = torch.full((B,), self.engine.timesteps - 1, dtype=torch.long, device=self.device)
            noise = torch.randn_like(x)
            x_T, _ = self.engine.add_noise(x, t_max, noise)
            x_hat  = self.engine.model(x_T, t_max, cond)

            n_ant = x.shape[1] // 2
            x0_c  = torch.complex(x[:, :n_ant,    :].permute(0, 2, 1),
                                   x[:, n_ant:,    :].permute(0, 2, 1))
            xh_c  = torch.complex(x_hat[:, :n_ant, :].permute(0, 2, 1),
                                   x_hat[:, n_ant:, :].permute(0, 2, 1))

            fft_real = torch.fft.fft(x0_c, dim=1)
            fft_pred = torch.fft.fft(xh_c, dim=1)
            freq_mse = torch.mean(torch.abs(fft_real - fft_pred) ** 2).item()
            self.writer.add_scalar('RF/freq_mse', freq_mse, epoch)

            sig_pwr   = torch.mean(torch.abs(x0_c) ** 2).item()
            noise_pwr = torch.mean(torch.abs(x0_c - xh_c) ** 2).item() + 1e-12
            self.writer.add_scalar('RF/snr_db', 10 * np.log10(sig_pwr / noise_pwr), epoch)

            output_amp = torch.abs(xh_c).mean().item()
            target_amp = torch.abs(x0_c).mean().item()
            self.writer.add_scalar('RF/output_amp', output_amp, epoch)
            self.writer.add_scalar('RF/target_amp', target_amp, epoch)
            self.writer.add_scalar('RF/amp_ratio',  output_amp / (target_amp + 1e-12), epoch)
            self.writer.add_scalar('RF/cond_norm',  cond.norm(dim=-1).mean().item(), epoch)

            for step in [0, 25, 50, 75, 99]:
                t_s     = torch.full((B,), step, dtype=torch.long, device=self.device)
                noise_s = torch.randn_like(x)
                x_s, _  = self.engine.add_noise(x, t_s, noise_s)
                x_s_hat = self.engine.model(x_s, t_s, cond)
                step_loss = torch.nn.functional.mse_loss(x_s_hat, x).item()
                self.writer.add_scalar(f'RF/recon_loss_t{step}', step_loss, epoch)
                x_s_c = torch.complex(x_s[:, :n_ant, :].permute(0, 2, 1),
                                      x_s[:, n_ant:, :].permute(0, 2, 1))
                self.writer.add_scalar(f'RF/noisy_amp_t{step}', torch.abs(x_s_c).mean().item(), epoch)

    # ------------------------------------------------------------------
    # SAMPLES PANEL  (per-sample figures under Samples/)
    # ------------------------------------------------------------------

    def log_samples_panel(self, real_batch, gen_batch, epoch):
        """Per-sample spectrogram, PSD, and IQ constellation under the Samples/ panel."""
        N = real_batch.shape[0]
        x_all = self._to_complex(real_batch)  # (N, 1024, n_ant)
        g_all = self._to_complex(gen_batch)
        n_ant = x_all.shape[2]
        freqs = np.fft.fftshift(np.fft.fftfreq(1024))

        for b in range(N):
            # --- Spectrogram ---
            fig, axes = plt.subplots(n_ant, 2, figsize=(12, n_ant * 4))
            for i in range(n_ant):
                f_ax, t_ax_s, Sxx_real = _gnss_spectrogram_db(x_all[b, :, i])
                _, _, Sxx_gen = _gnss_spectrogram_db(g_all[b, :, i])
                vmin = min(Sxx_real.min(), Sxx_gen.min())
                vmax = max(Sxx_real.max(), Sxx_gen.max())
                extent = [t_ax_s[0] * 1e3, t_ax_s[-1] * 1e3, f_ax[0], f_ax[-1]]
                for ax, Sxx, title in zip(axes[i], [Sxx_real, Sxx_gen], ['Real', 'Generated']):
                    im = ax.imshow(Sxx, aspect='auto', origin='lower', cmap='turbo',
                                   vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
                    ax.set_title(f'Antenna {i+1} — {title}')
                    ax.set_xlabel('t [ms]')
                    ax.set_ylabel('f [Hz]')
                    fig.colorbar(im, ax=ax, format='%+.0f dB-Hz')
            plt.suptitle(f'Sample {b} — Spectrogram — epoch {epoch}')
            plt.tight_layout()
            self.writer.add_figure(f'Samples/spectrogram/sample_{b}', fig, epoch)
            plt.close(fig)

            # --- PSD ---
            n_cols = min(n_ant, 4)
            n_rows = math.ceil(n_ant / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
            for i, ax in enumerate(np.array(axes).flatten()):
                psd_r = np.abs(np.fft.fftshift(np.fft.fft(x_all[b, :, i])))**2
                psd_g = np.abs(np.fft.fftshift(np.fft.fft(g_all[b, :, i])))**2
                ax.semilogy(freqs, psd_r, label='Real',      alpha=0.85, lw=1.2)
                ax.semilogy(freqs, psd_g, label='Generated', alpha=0.85, lw=1.2, linestyle='--')
                ax.set_title(f'Antenna {i+1} PSD')
                ax.set_xlabel('Normalised Frequency')
                ax.legend(fontsize=7)
                ax.grid(True, which='both', linestyle='--', linewidth=0.4)
            plt.suptitle(f'Sample {b} — PSD — epoch {epoch}')
            plt.tight_layout()
            self.writer.add_figure(f'Samples/psd/sample_{b}', fig, epoch)
            plt.close(fig)

            # --- IQ Constellation ---
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 5))
            for i, ax in enumerate(np.array(axes).flatten()):
                ax.scatter(x_all[b, :, i].real, x_all[b, :, i].imag,
                           s=1, alpha=0.25, label='Real')
                ax.scatter(g_all[b, :, i].real, g_all[b, :, i].imag,
                           s=1, alpha=0.25, label='Generated')
                ax.set_title(f'IQ — Ant {i+1}')
                ax.set_xlabel('I')
                ax.set_ylabel('Q')
                ax.legend(fontsize=7, markerscale=6)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', linewidth=0.4)
            plt.suptitle(f'Sample {b} — IQ Constellation — epoch {epoch}')
            plt.tight_layout()
            self.writer.add_figure(f'Samples/iq_constellation/sample_{b}', fig, epoch)
            plt.close(fig)


# ------------------------------------------------------------------
# Legacy standalone functions (kept for backward compatibility)
# ------------------------------------------------------------------

def visualize_latent_traversal(writer, model, epoch, device,
                               num_steps=5,
                               angle_range=(0, 180)):
    model.eval()
    z_fixed = torch.randn(1, model.latent_dim).to(device)
    z_fixed = z_fixed.repeat(num_steps, 1)

    angle_values = torch.linspace(angle_range[0], angle_range[1], num_steps).to(device)
    snr_values = angle_deg_to_cossin(angle_values)

    with torch.no_grad():
        z_cond = torch.cat([z_fixed, snr_values], dim=1)
        generated_signals = model.decode(z_cond)

    fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
    for i in range(num_steps):
        sig = generated_signals[i].cpu().numpy()
        I, Q = sig[0], sig[1]
        ax = axes[i]
        ax.scatter(I, Q, alpha=0.3, s=2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Angle: {angle_values[i]:.1f} dB")
        ax.axis('off')

    plt.tight_layout()
    writer.add_figure('Visuals/Latent_Traversal_SNR', fig, epoch)
    plt.close(fig)


def log_visualizations(writer, model, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        input_real = x.real.to(device, dtype=torch.float32)
        input_imag = x.imag.to(device, dtype=torch.float32)

        input = torch.cat([input_real, input_imag], dim=1)
        condition = y[1].to(next(model.parameters()).device, dtype=torch.float32)
        recon, _, _ = model(input, condition)

        fig_const, ax = plt.subplots(1, 2, figsize=(10, 5))
        real_I = input[0, 0, :].cpu().numpy()
        real_Q = input[0, 1, :].cpu().numpy()
        ax[0].scatter(real_I, real_Q, alpha=0.5, s=1)
        ax[0].set_title("Real Constellation")
        ax[0].grid(True)
