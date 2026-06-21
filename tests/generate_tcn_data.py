"""
Generate a synthetic CRPA dataset from a trained DiffWaveRF checkpoint, for
use as TCN training augmentation or as a paired synthetic test set.

For each real label in the chosen UniversalDataset split, one DDIM sample is
generated conditioned on that label's (pos, az, el, chirp_phase). Output is
saved to a single .npz file with the same field layout as UniversalDataset's
attributes, so downstream loaders need minimal changes.

The saved signals are kept in DiffWave's z-score domain (per-sample mean/std
applied at training time). The TCN training script must apply the matching
per-sample z-score on real signals before feeding the model, so real and
synthetic samples live in the same scale.

Example (Track A — generate from existing ±2.5° models):

  # Training augmentation pool, labels matched to full real train set
  python tests/generate_tcn_data.py \\
      --weights /path/xyz_full.h5 \\
      --label_source train \\
      --output ../generated_data/aug_train_xyz_full.npz

  # Paired synthetic test set, labels matched to real test set
  python tests/generate_tcn_data.py \\
      --weights /path/xyz_full.h5 \\
      --label_source test \\
      --output ../generated_data/test_xyz_full.npz
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from UniversalDataLoader import UniversalDataset
from models.DiffWave import DiffusionEngine, DiffWaveRF
from utils.chirp_phase import extract_phase_sincos


POS_MEAN = np.array([0.721, -0.034, -1.042], dtype=np.float32)
POS_STD  = np.array([6.922,  4.555,  0.552], dtype=np.float32)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',      type=str, required=True,
                   help='Path to DiffWaveRF .h5 weights.')
    p.add_argument('--label_source', choices=['train', 'test'], required=True,
                   help='UniversalDataset split whose labels drive generation.')
    p.add_argument('--output',       type=str, required=True,
                   help='Output .npz path.')
    p.add_argument('--task',         type=int, default=132)
    p.add_argument('--no_xyz',       action='store_true',
                   help='Set if the checkpoint was trained with --no_xyz.')
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--ddim_steps',   type=int, default=50)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--max_samples',  type=int, default=-1,
                   help='Cap on number of samples (for quick smoke tests). '
                        '-1 = use the full split.')
    return p.parse_args()


def _build_condition(pos, az_sc, el_sc, phi_sc, no_xyz):
    if no_xyz:
        return torch.cat([az_sc, el_sc, phi_sc], dim=1)
    pos_norm = (pos - torch.tensor(POS_MEAN, device=pos.device)) / \
               torch.tensor(POS_STD, device=pos.device)
    return torch.cat([pos_norm, az_sc, el_sc, phi_sc], dim=1)


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cond_dim = 6 if args.no_xyz else 9
    model = DiffWaveRF(input_channels=8, residual_channels=64,
                       cond_dim=cond_dim).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    engine = DiffusionEngine(model=model, timesteps=1000)

    ds = UniversalDataset(task_id=args.task, mode=args.label_source,
                          angle_mode='sincos')
    n_total = len(ds) if args.max_samples < 0 else min(args.max_samples, len(ds))
    print(f'[gen] split={args.label_source}  N={n_total}  cond_dim={cond_dim}')

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    signals_out = np.empty((n_total, 1024, 4), dtype=np.complex64)
    positions   = np.empty((n_total, 3),       dtype=np.float32)
    az_angles   = np.empty((n_total, 2),       dtype=np.float32)
    el_angles   = np.empty((n_total, 2),       dtype=np.float32)

    written = 0
    with torch.no_grad():
        for x, (pos, az_sc, el_sc) in loader:
            if written >= n_total:
                break
            B = min(x.shape[0], n_total - written)
            x   = x[:B].to(device)
            pos = pos[:B].to(device, dtype=torch.float32)
            az_sc = az_sc[:B].to(device, dtype=torch.float32)
            el_sc = el_sc[:B].to(device, dtype=torch.float32)
            phi_sc = extract_phase_sincos(x)                # [B, 2]

            cond = _build_condition(pos, az_sc, el_sc, phi_sc, args.no_xyz)
            gen, _ = engine.sample_ddim(B, 1024, cond, steps=args.ddim_steps)
            # gen: [B, 8, 1024] real, z-score domain.
            # Repack to [B, 1024, 4] complex by recombining I+Q channels.
            gen_real = gen[:, :4].cpu().numpy()              # [B, 4, 1024]
            gen_imag = gen[:, 4:].cpu().numpy()              # [B, 4, 1024]
            gen_complex = (gen_real + 1j * gen_imag).astype(np.complex64)
            gen_complex = np.transpose(gen_complex, (0, 2, 1))  # [B, 1024, 4]

            signals_out[written:written + B] = gen_complex
            positions[written:written + B]   = pos.cpu().numpy()
            az_angles[written:written + B]   = az_sc.cpu().numpy()
            el_angles[written:written + B]   = el_sc.cpu().numpy()

            written += B
            print(f'  [{written}/{n_total}]', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        signals=signals_out,
        positions=positions,
        az_angles=az_angles,
        el_angles=el_angles,
        weights=args.weights,
        label_source=args.label_source,
        task=args.task,
        no_xyz=args.no_xyz,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
    )
    print(f'[gen] wrote {written} samples to {args.output}')


if __name__ == '__main__':
    main()
