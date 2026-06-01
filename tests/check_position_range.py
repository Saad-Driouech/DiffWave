"""
Check the range of XYZ positions in the dataset.

Run:  python tests/check_position_range.py
"""

import torch
import numpy as np
from UniversalDataLoader import UniversalDataset

for split in ['train', 'test']:
    ds = UniversalDataset(task_id=132, mode=split, angle_mode='sincos')
    positions = []
    for i in range(len(ds)):
        _, (pos, _, _) = ds[i]
        positions.append(pos.numpy())

    pos = np.stack(positions)   # [N, 3]
    print(f'\n{split} ({len(ds)} samples)')
    print(f'  x: min={pos[:,0].min():.3f}  max={pos[:,0].max():.3f}  mean={pos[:,0].mean():.3f}  std={pos[:,0].std():.3f}')
    print(f'  y: min={pos[:,1].min():.3f}  max={pos[:,1].max():.3f}  mean={pos[:,1].mean():.3f}  std={pos[:,1].std():.3f}')
    print(f'  z: min={pos[:,2].min():.3f}  max={pos[:,2].max():.3f}  mean={pos[:,2].mean():.3f}  std={pos[:,2].std():.3f}')
    print(f'  L2 norm: min={np.linalg.norm(pos, axis=1).min():.3f}  max={np.linalg.norm(pos, axis=1).max():.3f}')
