#!/usr/bin/env python3
"""
DiffusionVisualizer diagnostics — real data end-to-end.
Run from the DiffWave project root:
    python tests/diag_visualizer.py

Phase 1: shape checks with synthetic data (fast).
Phase 2: load one real batch from StationaryEttus and call log_all
         with DDIM steps=5 so it finishes in seconds.
"""

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

from models.DiffWave import DiffWaveRF, DiffusionEngine
from utils.StationaryEttus import StationaryEttus
from utils.visualization import DiffusionVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

model  = DiffWaveRF(input_channels=16, residual_channels=64, cond_dim=2).to(device)
engine = DiffusionEngine(model=model, timesteps=1000)

print(f"[model]  input_channels = {model.input_channels}")
print(f"[engine] device         = {engine.device}")

# ── Phase 1: synthetic shape checks ──────────────────────────────────────────
print("\n=== Phase 1: synthetic data ===")
B = 4
fake_complex = torch.randn(B, 8, 1024, dtype=torch.complex64).to(device)
real_batch   = torch.cat([fake_complex.real, fake_complex.imag], dim=1).to(torch.float32)
condition    = torch.randn(B, 2).to(device)

print(f"[synth] real_batch.shape = {real_batch.shape}  <- should be (B, 16, 1024)")
print(f"[synth] condition.shape  = {condition.shape}   <- should be (B,  2)")

gen, _ = engine.sample_ddim(1, 1024, condition[:1], steps=5)
print(f"[ddim]  gen.shape        = {gen.shape}  <- should be (1, 16, 1024)")

for label, t in [("real_batch[0,8]", lambda: real_batch[0, 8]),
                 ("gen[0,8]",         lambda: gen[0, 8]),
                 ("complex(gen[:,0],gen[:,8])", lambda: torch.complex(gen[:, 0], gen[:, 8])),
                 ("_to_complex real", lambda: real_batch[:, :8, :].permute(0,2,1) + 1j*real_batch[:, 8:, :].permute(0,2,1)),
                 ("_to_complex gen",  lambda: gen[:, :8, :].permute(0,2,1) + 1j*gen[:, 8:, :].permute(0,2,1)),
                 ("stft",             lambda: torch.stft(real_batch[0,0,:512].cpu(), n_fft=24, hop_length=17, return_complex=True))]:
    try:
        v = t()
        print(f"[OK]   {label} -> shape {v.shape}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")

# ── Phase 2: real data end-to-end ────────────────────────────────────────────
print("\n=== Phase 2: real data from StationaryEttus ===")
dataset = StationaryEttus(mode="test")
loader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
x, y   = next(iter(loader))

print(f"[data] x.dtype={x.dtype}, x.shape={x.shape}")
print(f"[data] y[1]={y[1]}, y[1].shape={y[1].shape if hasattr(y[1], 'shape') else type(y[1])}")

x          = x.to(device)
real_batch = torch.cat([x.real.to(torch.float32), x.imag.to(torch.float32)], dim=1)
condition  = y[1].to(device, dtype=torch.float32)

print(f"[eval] real_batch.shape = {real_batch.shape}  <- should be (2, 16, 1024)")
print(f"[eval] condition.shape  = {condition.shape}   <- should be (2,  2)")

# Monkey-patch DDIM to use 5 steps so log_all finishes fast
_orig_ddim = engine.sample_ddim
engine.sample_ddim = lambda *a, **kw: _orig_ddim(*a, **{**kw, 'steps': 5})

writer     = SummaryWriter(log_dir="/tmp/diag_tb")
visualizer = DiffusionVisualizer(engine=engine, writer=writer, device=device)

print("\n--- Calling log_all (DDIM steps=5 for speed) ---")
visualizer.log_all(real_batch, condition, epoch=0)

engine.sample_ddim = _orig_ddim
writer.close()
print("\n=== Done. ===")
