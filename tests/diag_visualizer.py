#!/usr/bin/env python3
"""
DiffusionVisualizer diagnostics.
Run from the DiffWave project root:
    python tests/diag_visualizer.py
Prints shapes at every stage that could cause the two known runtime errors.
"""

import torch
import numpy as np
import yaml

# ── Load config & model ──────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

from models.DiffWave import DiffWaveRF, DiffusionEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

model = DiffWaveRF(input_channels=16, residual_channels=64, cond_dim=2).to(device)
engine = DiffusionEngine(model=model, timesteps=1000)

print(f"[model] input_channels = {model.input_channels}")
print(f"[engine] device        = {engine.device}")

# ── Simulate one real batch as evaluation_epoch builds it ───────────────────
# Dataset returns complex (B, 8, 1024); evaluation_epoch cats I/Q → (B, 16, 1024)
B = 4
fake_complex = torch.randn(B, 8, 1024, dtype=torch.complex64).to(device)
input_real   = fake_complex.real.to(dtype=torch.float32)
input_imag   = fake_complex.imag.to(dtype=torch.float32)
real_batch   = torch.cat([input_real, input_imag], dim=1)
condition    = torch.randn(B, 2).to(device)

print(f"\n[eval] real_batch.shape  = {real_batch.shape}   <- should be (B, 16, 1024)")
print(f"[eval] condition.shape   = {condition.shape}     <- should be (B,  2)")

# ── DDIM output shape ────────────────────────────────────────────────────────
print("\n--- sample_ddim (1 sample, 10 steps) ---")
cond1 = condition[:1]
gen, pics = engine.sample_ddim(1, 1024, cond1, steps=10)
print(f"[ddim] gen.shape = {gen.shape}   <- should be (1, 16, 1024)")
print(f"[ddim] len(pics) = {len(pics)},  pics[0].shape = {np.array(pics[0]).shape}")

# ── Test every indexing pattern that the visualizer uses ────────────────────
print("\n--- Indexing tests ---")

# log_spectral_fidelity
try:
    v = real_batch[0, 8]
    print(f"[OK] real_batch[0, 8].shape = {v.shape}")
except Exception as e:
    print(f"[FAIL] real_batch[0, 8]: {e}")

try:
    v = gen[0, 8]
    print(f"[OK] gen[0, 8].shape = {v.shape}")
except Exception as e:
    print(f"[FAIL] gen[0, 8]: {e}")

# log_aoa_verification
try:
    ant1 = torch.complex(gen[:, 0], gen[:, 8])
    print(f"[OK] torch.complex(gen[:,0], gen[:,8]).shape = {ant1.shape}")
except Exception as e:
    print(f"[FAIL] torch.complex(gen[:,0], gen[:,8]): {e}")

# _to_complex helper on real_batch
try:
    r = real_batch[:, :8, :].permute(0, 2, 1)
    im = real_batch[:, 8:, :].permute(0, 2, 1)
    print(f"[OK] _to_complex(real_batch): r={r.shape}, im={im.shape}")
    c = r + 1j * im
    print(f"[OK] r + 1j*im shape = {c.shape}")
except Exception as e:
    print(f"[FAIL] _to_complex(real_batch): {e}")

# _to_complex on gen
try:
    r = gen[:, :8, :].permute(0, 2, 1)
    im = gen[:, 8:, :].permute(0, 2, 1)
    print(f"[OK] _to_complex(gen): r={r.shape}, im={im.shape}")
    c = r + 1j * im
    print(f"[OK] r + 1j*im shape = {c.shape}")
except Exception as e:
    print(f"[FAIL] _to_complex(gen): {e}")

# log_rf_scalars: torch.complex path
try:
    x_hat = model(real_batch, torch.zeros(B, dtype=torch.long, device=device), condition)
    print(f"[OK] model output shape = {x_hat.shape}   <- should be (B, 16, 1024)")
    x0_c = torch.complex(real_batch[:, :8, :].permute(0, 2, 1),
                          real_batch[:, 8:, :].permute(0, 2, 1))
    xh_c = torch.complex(x_hat[:, :8, :].permute(0, 2, 1),
                          x_hat[:, 8:, :].permute(0, 2, 1))
    print(f"[OK] x0_c.shape = {x0_c.shape}, xh_c.shape = {xh_c.shape}")
except Exception as e:
    print(f"[FAIL] log_rf_scalars complex path: {e}")

# log_constellation_grid: real_np[ant + 8] indexing
try:
    real_np = real_batch[0].cpu().numpy()   # (16, 1024)
    print(f"[OK] real_np.shape = {real_np.shape}")
    for ant in range(8):
        _ = real_np[ant];  _ = real_np[ant + 8]
    print("[OK] real_np[ant] and real_np[ant+8] for ant in 0..7")
except Exception as e:
    print(f"[FAIL] real_np[ant+8]: {e}")

# log_aoa_sweep: gen_signals[i][8] indexing
try:
    sig_np = gen[0].cpu().numpy()   # (16, 1024)
    print(f"[OK] sig_np.shape = {sig_np.shape}")
    _ = sig_np[8];  _ = sig_np[9]
    print("[OK] sig_np[8] and sig_np[9]")
except Exception as e:
    print(f"[FAIL] sig_np[8]: {e}")

# stft
try:
    data_sig = real_batch[0, 0, :512].cpu()
    spec = torch.stft(data_sig, n_fft=24, hop_length=17, return_complex=True)
    print(f"[OK] torch.stft output shape = {spec.shape}")
except Exception as e:
    print(f"[FAIL] torch.stft: {e}")

print("\n=== Done. Paste the full output here. ===")
