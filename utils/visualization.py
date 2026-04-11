import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import math

_GNSS_FS = 40.5e6  # GNSS sampling frequency (Hz)


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
    def __init__(self, writer, engine, device):
        self.writer = writer
        self.engine = engine
        self.device = device

    def _to_complex(self, batch):
        """Convert DiffWave (B, 16, 1024) → complex numpy (B, 1024, 8)."""
        return (batch[:, :8, :].permute(0, 2, 1) + 1j * batch[:, 8:, :].permute(0, 2, 1)).cpu().numpy()

    def log_all(self, real_batch, condition, epoch):
        methods = [
            lambda: self.log_noise_schedule(epoch),
            lambda: self.log_denoising_chain(epoch),
            lambda: self.log_aoa_verification(epoch),
            lambda: self.log_spectral_fidelity(real_batch, epoch),
            lambda: self.log_weight_histograms(self.engine.model, epoch),
            lambda: self.log_multi_antenna_comparison(real_batch, condition, epoch),
            lambda: self.log_constellation_grid(real_batch, condition, epoch),
            lambda: self.log_cross_antenna_correlation(real_batch, condition, epoch),
            lambda: self.log_prediction_error_vs_timestep(real_batch, condition, epoch),
            lambda: self.log_aoa_sweep(epoch),
            lambda: self.log_skip_norms(self.engine.model, real_batch, condition, epoch),
            # ── Additional plots ──────────────────────────────────────────────────
            lambda: self.log_psd_semilogy(real_batch, condition, epoch),
            lambda: self.log_spectrogram_comparison(real_batch, condition, epoch),
            lambda: self.log_time_amplitude_rf(real_batch, condition, epoch),
            lambda: self.log_iq_time_series_rf(real_batch, condition, epoch),
            lambda: self.log_iq_constellation_rf(real_batch, condition, epoch),
            lambda: self.log_degradation_steps(real_batch, epoch),
            lambda: self.log_stft_spectrogram(real_batch, condition, epoch),
            lambda: self.log_rf_scalars(real_batch, condition, epoch),
        ]
        for fn in methods:
            try:
                fn()
            except Exception as e:
                print(f"[DiffusionVisualizer] {fn.__name__ if hasattr(fn, '__name__') else 'method'} failed: {e}")

    # ------------------------------------------------------------------
    # EXISTING METHODS
    # ------------------------------------------------------------------

    def log_denoising_chain(self, epoch):
        """Visualizes the reverse process: Noise -> Signal"""
        cond = torch.zeros(1, 2).to(self.device)
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

    def log_aoa_verification(self, epoch):
        """Checks if the model obeys the Angle of Arrival condition."""
        batch_size = 64
        cond = torch.zeros(batch_size, 2).to(self.device)
        cond[:, 0] = np.random.random()
        cond[:, 1] = np.random.random()

        angle = torch.atan2(cond[:, 0], cond[:, 1])

        gen_data, _ = self.engine.sample_ddim(batch_size, 1024, cond, steps=20)

        ant1 = torch.complex(gen_data[:, 0], gen_data[:, 8])
        ant2 = torch.complex(gen_data[:, 1], gen_data[:, 9])
        phase_diffs = torch.angle(torch.mean(ant2 * ant1.conj(), dim=1))

        fig, ax = plt.subplots()
        ax.hist(phase_diffs.cpu().numpy(), bins=30, color='orange', alpha=0.7)
        ax.axvline(x=angle[0].cpu().numpy(), color='r', linestyle='--', label='Target (Approx)')
        ax.set_title(f"Phase Difference Distribution (Target Cond={angle[0]:.2f} rad)")
        ax.set_xlabel("Measured Phase Difference (Rad)")
        ax.legend()
        self.writer.add_figure('Physics/AoA_Consistency', fig, epoch)
        plt.close(fig)

    def log_spectral_fidelity(self, real_batch, epoch):
        """Compares PSD of Real vs Generated."""
        B = min(real_batch.shape[0], 16)
        dummy_cond = torch.zeros(B, 2).to(self.device)
        gen_batch, _ = self.engine.sample_ddim(B, 1024, dummy_cond, steps=50)

        real_c = real_batch[0, 0].cpu().numpy() + 1j * real_batch[0, 8].cpu().numpy()
        gen_c = gen_batch[0, 0].cpu().numpy() + 1j * gen_batch[0, 8].cpu().numpy()

        fig, ax = plt.subplots()
        ax.psd(real_c, Fs=1.0, NFFT=512, label='Real')
        ax.psd(gen_c, Fs=1.0, NFFT=512, label='Generated')
        ax.legend()
        ax.set_title("Power Spectral Density (Antenna 1)")
        self.writer.add_figure('Fidelity/PSD', fig, epoch)
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

    def log_multi_antenna_comparison(self, real_batch, condition, epoch):
        """8-panel figure: real I-channel vs generated I-channel per antenna."""
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        axes = axes.flatten()
        real_np = real_batch[0].cpu().numpy()
        gen_np = gen[0].cpu().numpy()

        for ant in range(8):
            ax = axes[ant]
            ax.plot(real_np[ant, :256], label='Real', alpha=0.8, linewidth=0.8)
            ax.plot(gen_np[ant, :256], label='Generated', alpha=0.8, linewidth=0.8, linestyle='--')
            ax.set_title(f"Antenna {ant + 1} — I channel")
            ax.set_xlabel("Sample")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure('Signals/Multi_Antenna_TimeDomain', fig, epoch)
        plt.close(fig)

    def log_constellation_grid(self, real_batch, condition, epoch):
        """2x4 grid of I/Q constellation plots (real=blue, generated=red) for all 8 antennas."""
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        real_np = real_batch[0].cpu().numpy()
        gen_np = gen[0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()

        for ant in range(8):
            ax = axes[ant]
            ax.scatter(real_np[ant], real_np[ant + 8], alpha=0.3, s=1, c='steelblue', label='Real')
            ax.scatter(gen_np[ant], gen_np[ant + 8], alpha=0.3, s=1, c='crimson', label='Generated')
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

    def log_cross_antenna_correlation(self, real_batch, condition, epoch):
        """Spatial correlation matrix |R| for real and generated signals."""
        B = min(real_batch.shape[0], 16)
        cond_b = condition[:B].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen_batch, _ = self.engine.sample_ddim(B, 1024, cond_b, steps=50)

        def corr_matrix(batch_np):
            mats = []
            for b in range(batch_np.shape[0]):
                X = batch_np[b, :8] + 1j * batch_np[b, 8:]   # (8, 1024)
                R = np.abs(X @ X.conj().T) / X.shape[1]
                mats.append(R)
            return np.mean(mats, axis=0)

        R_real = corr_matrix(real_batch[:B].cpu().numpy())
        R_gen = corr_matrix(gen_batch.cpu().numpy())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmax = max(R_real.max(), R_gen.max())
        for ax, R, title in zip(axes, [R_real, R_gen], ['Real', 'Generated']):
            im = ax.imshow(R, vmin=0, vmax=vmax, cmap='viridis')
            ax.set_title(f"Cross-Antenna Correlation — {title}")
            ax.set_xlabel("Antenna")
            ax.set_ylabel("Antenna")
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels([f"Ant{i+1}" for i in range(8)], rotation=45, fontsize=7)
            ax.set_yticklabels([f"Ant{i+1}" for i in range(8)], fontsize=7)
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
        """Generates signals at 6 evenly-spaced AoA angles and shows time-domain and phase."""
        angles_deg = np.linspace(-90, 90, 6)
        angles_rad = angles_deg * math.pi / 180.0

        conds = torch.tensor(
            [[math.sin(a), math.cos(a)] for a in angles_rad],
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

        measured_phases = []
        for sig in gen_signals:
            ant1 = sig[0] + 1j * sig[8]
            ant2 = sig[1] + 1j * sig[9]
            phase = np.angle(np.mean(ant2 * ant1.conj()))
            measured_phases.append(phase)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(angles_deg, np.array(measured_phases), marker='o', label='Measured phase diff (ant1→ant2)')
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
        norms = [v[1] for v in skip_norms]

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
    # ADDITIONAL SIGNAL PLOTS
    # ------------------------------------------------------------------

    def log_psd_semilogy(self, real_batch, condition, epoch):
        """
        PSD overlay (semilogy) — one subplot per antenna.
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        x0 = self._to_complex(real_batch[:1])[0]   # (1024, 8) complex
        xh = self._to_complex(gen)[0]               # (1024, 8) complex

        freqs = np.fft.fftshift(np.fft.fftfreq(1024))
        fig, axes = plt.subplots(4, 2, figsize=(12, 14))
        for i, ax in enumerate(axes.flat):
            psd_real = np.abs(np.fft.fftshift(np.fft.fft(x0[:, i]))) ** 2
            psd_pred = np.abs(np.fft.fftshift(np.fft.fft(xh[:, i]))) ** 2
            ax.semilogy(freqs, psd_real, label='Real',      alpha=0.85, lw=1.2)
            ax.semilogy(freqs, psd_pred, label='Generated', alpha=0.85, lw=1.2, linestyle='--')
            ax.set_title(f'Antenna {i+1} PSD')
            ax.set_xlabel('Normalised Frequency')
            ax.set_ylabel('Power')
            ax.legend(fontsize=7)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.suptitle(f'Power Spectral Density — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/psd_per_antenna', fig, epoch)
        plt.close(fig)

    def log_spectrogram_comparison(self, real_batch, condition, epoch):
        """
        Spectrogram comparison — all 8 antennas (8×2 grid, Real | Generated).
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        x0 = self._to_complex(real_batch[:1])[0]   # (1024, 8) complex
        xh = self._to_complex(gen)[0]               # (1024, 8) complex

        fig, axes = plt.subplots(8, 2, figsize=(12, 32))
        for i in range(8):
            f_ax, t_ax_s, Sxx_real = _gnss_spectrogram_db(x0[:, i])
            _,    _,      Sxx_gen  = _gnss_spectrogram_db(xh[:, i])
            vmin = min(Sxx_real.min(), Sxx_gen.min())
            vmax = max(Sxx_real.max(), Sxx_gen.max())
            extent = [t_ax_s[0] * 1e3, t_ax_s[-1] * 1e3, f_ax[0], f_ax[-1]]
            for ax, Sxx, title in zip(axes[i], [Sxx_real, Sxx_gen], ['Real', 'Generated']):
                im = ax.imshow(Sxx, aspect='auto', origin='lower', cmap='turbo',
                               vmin=vmin, vmax=vmax, extent=extent,
                               interpolation='nearest')
                ax.set_title(f'Antenna {i+1} — {title}')
                ax.set_xlabel('t [ms]')
                ax.set_ylabel('f [Hz]')
                fig.colorbar(im, ax=ax, format='%+.0f dB-Hz')
        plt.suptitle(f'Spectrogram — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/spectrogram', fig, epoch)
        plt.close(fig)

    def log_time_amplitude_rf(self, real_batch, condition, epoch):
        """
        Time-domain amplitude |IQ| — first 256 samples per antenna (4×2 grid).
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        x0 = self._to_complex(real_batch[:1])[0]   # (1024, 8)
        xh = self._to_complex(gen)[0]               # (1024, 8)

        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        t_ax = np.arange(256)
        for i, ax in enumerate(axes.flat):
            ax.plot(t_ax, np.abs(x0[:256, i]), label='Real',      alpha=0.85, lw=1.2)
            ax.plot(t_ax, np.abs(xh[:256, i]), label='Generated', alpha=0.85, lw=1.2, linestyle='--')
            ax.set_title(f'Antenna {i+1}  |IQ|')
            ax.set_xlabel('Sample')
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', linewidth=0.4)
        plt.suptitle(f'Time-domain Amplitude (first 256 samples) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/time_amplitude', fig, epoch)
        plt.close(fig)

    def log_iq_time_series_rf(self, real_batch, condition, epoch):
        """
        I(t) and Q(t) per antenna — 8 rows × 2 cols (I | Q), first 256 samples.
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        x0 = self._to_complex(real_batch[:1])[0]   # (1024, 8)
        xh = self._to_complex(gen)[0]               # (1024, 8)

        fig, axes = plt.subplots(8, 2, figsize=(14, 24))
        t_iq = np.arange(256)
        for i in range(8):
            ax_i, ax_q = axes[i, 0], axes[i, 1]
            # I component
            ax_i.plot(t_iq, x0[:256, i].real, label='Real',      alpha=0.85, lw=1.0)
            ax_i.plot(t_iq, xh[:256, i].real, label='Generated', alpha=0.85, lw=1.0, linestyle='--')
            ax_i.set_title(f'Antenna {i+1} — I(t)')
            ax_i.set_xlabel('Sample')
            ax_i.set_ylabel('I')
            ax_i.legend(fontsize=7)
            ax_i.grid(True, linestyle='--', linewidth=0.4)
            # Q component
            ax_q.plot(t_iq, x0[:256, i].imag, label='Real',      alpha=0.85, lw=1.0)
            ax_q.plot(t_iq, xh[:256, i].imag, label='Generated', alpha=0.85, lw=1.0, linestyle='--')
            ax_q.set_title(f'Antenna {i+1} — Q(t)')
            ax_q.set_xlabel('Sample')
            ax_q.set_ylabel('Q')
            ax_q.legend(fontsize=7)
            ax_q.grid(True, linestyle='--', linewidth=0.4)
        plt.suptitle(f'IQ Time Series (first 256 samples) — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/iq_time_series', fig, epoch)
        plt.close(fig)

    def log_iq_constellation_rf(self, real_batch, condition, epoch):
        """
        IQ Constellation scatter — all 8 antennas (4×2 grid).
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        x0 = self._to_complex(real_batch[:1])[0]   # (1024, 8)
        xh = self._to_complex(gen)[0]               # (1024, 8)

        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        for i, ax in enumerate(axes.flat):
            ax.scatter(x0[:, i].real, x0[:, i].imag, s=1, alpha=0.25, label='Real')
            ax.scatter(xh[:, i].real, xh[:, i].imag, s=1, alpha=0.25, label='Generated')
            ax.set_title(f'IQ Constellation — Antenna {i+1}')
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
        """
        Forward degradation spectrogram — Antenna 1, 5 timesteps (t=0,25,50,75,99).
        """
        steps_to_show = [0, 25, 50, 75, 99]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for ax, step in zip(axes, steps_to_show):
            t_s = torch.full((1,), step, dtype=torch.long, device=self.device)
            noise = torch.randn_like(real_batch[:1])
            x_deg, _ = self.engine.add_noise(real_batch[:1].to(self.device), t_s, noise.to(self.device))
            # Antenna 1: channel 0 (I) + channel 8 (Q) → complex
            sig_deg = x_deg[0, 0].cpu().numpy() + 1j * x_deg[0, 8].cpu().numpy()  # (1024,)
            _, _, Sxx_db = _gnss_spectrogram_db(sig_deg)
            ax.imshow(Sxx_db, aspect='auto', origin='lower', cmap='turbo',
                      interpolation='nearest')
            ax.set_title(f't = {step}')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('f [Hz]' if step == 0 else '')
        plt.suptitle(f'Forward Degradation — Antenna 1 — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/degradation_steps', fig, epoch)
        plt.close(fig)

    def log_stft_spectrogram(self, real_batch, condition, epoch):
        """
        STFT spectrogram (dB) — Real vs Generated, Antenna 1.
        STFT spectrogram logged to TensorBoard.
        """
        cond_single = condition[:1].to(self.device)
        self.engine.model.eval()
        with torch.no_grad():
            gen, _ = self.engine.sample_ddim(1, 1024, cond_single, steps=50)

        # Antenna 1 I-channel, shape (512,) — use first 512 samples to match save_wifi
        data_sig = real_batch[0, 0, :512].cpu()
        pred_sig = gen[0, 0, :512].cpu()

        n_fft = 24
        hop_length = 17
        data_spec = torch.stft(data_sig, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        pred_spec = torch.stft(pred_sig, n_fft=n_fft, hop_length=hop_length, return_complex=True)

        data_spec_mag = torch.abs(data_spec)
        pred_spec_mag = torch.abs(pred_spec)
        data_spec_dB = 20 * np.log10(data_spec_mag.numpy() + 1e-6)
        pred_spec_dB = 20 * np.log10(pred_spec_mag.numpy() + 1e-6)

        fig = plt.figure(figsize=(6, 3))
        ax1 = plt.subplot(1, 2, 1)
        im1 = ax1.matshow(data_spec_dB, cmap='viridis', origin='lower')
        ax1.set_title('Data Spectrogram (dB)')
        plt.colorbar(im1, format='%+2.0f dB', ax=ax1, orientation='horizontal', pad=0.05)

        ax2 = plt.subplot(1, 2, 2)
        im2 = ax2.matshow(pred_spec_dB, cmap='viridis', origin='lower')
        ax2.set_title('Prediction Spectrogram (dB)')
        plt.colorbar(im2, format='%+2.0f dB', ax=ax2, orientation='horizontal', pad=0.05)

        plt.suptitle(f'STFT Spectrogram — Antenna 1 — epoch {epoch}')
        plt.tight_layout()
        self.writer.add_figure('RF/stft_spectrogram', fig, epoch)
        plt.close(fig)

    def log_rf_scalars(self, real_batch, condition, epoch):
        """
        Scalar metrics: freq_mse, snr_db, amp_ratio, cond_norm.
        Uses a single model forward at t_max (full noise) for speed.
        """
        B = min(real_batch.shape[0], 8)
        x = real_batch[:B].to(self.device)
        cond = condition[:B].to(self.device)

        self.engine.model.eval()
        with torch.no_grad():
            t_max = torch.full((B,), self.engine.timesteps - 1, dtype=torch.long, device=self.device)
            noise = torch.randn_like(x)
            x_T, _ = self.engine.add_noise(x, t_max, noise)
            x_hat = self.engine.model(x_T, t_max, cond)

            # Convert to complex: (B, 1024, 8)
            x0_c = torch.complex(x[:, :8, :].permute(0, 2, 1),
                                  x[:, 8:, :].permute(0, 2, 1))   # (B, 1024, 8)
            xh_c = torch.complex(x_hat[:, :8, :].permute(0, 2, 1),
                                  x_hat[:, 8:, :].permute(0, 2, 1))

            # Spectral MSE
            fft_real = torch.fft.fft(x0_c, dim=1)
            fft_pred = torch.fft.fft(xh_c, dim=1)
            freq_mse = torch.mean(torch.abs(fft_real - fft_pred) ** 2).item()
            self.writer.add_scalar('RF/freq_mse', freq_mse, epoch)

            # Reconstruction SNR (dB)
            sig_pwr   = torch.mean(torch.abs(x0_c) ** 2).item()
            noise_pwr = torch.mean(torch.abs(x0_c - xh_c) ** 2).item() + 1e-12
            self.writer.add_scalar('RF/snr_db', 10 * np.log10(sig_pwr / noise_pwr), epoch)

            # Output amplitude vs target amplitude — zero-collapse detector
            output_amp = torch.abs(xh_c).mean().item()
            target_amp = torch.abs(x0_c).mean().item()
            self.writer.add_scalar('RF/output_amp',  output_amp, epoch)
            self.writer.add_scalar('RF/target_amp',  target_amp, epoch)
            self.writer.add_scalar('RF/amp_ratio',   output_amp / (target_amp + 1e-12), epoch)

            # Condition vector norm — verify conditioning is active
            self.writer.add_scalar('RF/cond_norm', cond.norm(dim=-1).mean().item(), epoch)

            # Per-step reconstruction loss at t=0,25,50,75,99
            for step in [0, 25, 50, 75, 99]:
                t_s = torch.full((B,), step, dtype=torch.long, device=self.device)
                noise_s = torch.randn_like(x)
                x_s, _ = self.engine.add_noise(x, t_s, noise_s)
                x_s_hat = self.engine.model(x_s, t_s, cond)
                step_loss = torch.nn.functional.mse_loss(x_s_hat, x).item()
                self.writer.add_scalar(f'RF/recon_loss_t{step}', step_loss, epoch)
                x_s_c = torch.complex(x_s[:, :8, :].permute(0, 2, 1),
                                      x_s[:, 8:, :].permute(0, 2, 1))
                self.writer.add_scalar(f'RF/noisy_amp_t{step}', torch.abs(x_s_c).mean().item(), epoch)


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

        gen_I = recon[0, 0, :].cpu().numpy()
        gen_Q = recon[0, 1, :].cpu().numpy()
        ax[1].scatter(gen_I, gen_Q, alpha=0.5, s=1, c='r')
        ax[1].set_title("Generated Constellation")
        ax[1].grid(True)

        writer.add_figure('Visuals/Constellation', fig_const, epoch)

        fig_psd, ax_psd = plt.subplots(figsize=(10, 5))
        ax_psd.psd(real_I + 1j * real_Q, NFFT=512, Fs=1.0, label='Real')
        ax_psd.psd(gen_I + 1j * gen_Q, NFFT=512, Fs=1.0, label='Generated', color='r', alpha=0.7)
        ax_psd.legend()
        ax_psd.set_title("Power Spectral Density")

        writer.add_figure('Visuals/PSD', fig_psd, epoch)
        plt.close('all')


def log_phase_diff(writer, real, recon, epoch):
    real_ant1 = torch.complex(real[:, 0, :], real[:, 2, :])
    real_ant2 = torch.complex(real[:, 1, :], real[:, 3, :])

    recon_ant1 = torch.complex(recon[:, 0, :], recon[:, 2, :])
    recon_ant2 = torch.complex(recon[:, 1, :], recon[:, 3, :])

    real_diff = torch.angle(real_ant2 * real_ant1.conj())
    recon_diff = torch.angle(recon_ant2 * recon_ant1.conj())

    fig, ax = plt.subplots()
    ax.hist(real_diff.flatten().cpu().numpy(), bins=50, alpha=0.5, label='Real')
    ax.hist(recon_diff.flatten().cpu().numpy(), bins=50, alpha=0.5, color='r', label='Gen')
    ax.legend()
    ax.set_title("Phase Difference Distribution (Ant1 vs Ant2)")

    writer.add_figure('Visuals/PhaseDiff', fig, epoch)
    plt.close(fig)
