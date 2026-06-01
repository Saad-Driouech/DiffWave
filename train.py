import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import mlflow
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import os

from models.DiffWave import DiffWaveRF, DiffusionEngine
from utils.visualization import DiffusionVisualizer

# ── Masked azimuth bands ───────────────────────────────────────────────────────
# 5 bands of 5° width, centers at 0°, ±90°, ±180°
MASKED_BANDS_DEG = [
    (  -2.5,   2.5),
    (  87.5,  92.5),
    ( -92.5, -87.5),
    ( 177.5, 180.0),
    (-180.0,-177.5),
]


def _filter_masked_bands(dataset):
    """Return a Subset with all samples in masked azimuth bands removed."""
    az_arr = np.asarray(dataset.az_angles)        # [N, 2] sincos
    az_deg = np.degrees(np.arctan2(az_arr[:, 0], az_arr[:, 1]))
    keep   = np.ones(len(az_deg), dtype=bool)
    for lo, hi in MASKED_BANDS_DEG:
        keep &= ~((az_deg >= lo) & (az_deg <= hi))
    n_removed = (~keep).sum()
    print(f'[mask] removed {n_removed} / {len(az_deg)} training samples '
          f'({n_removed / len(az_deg) * 100:.2f}%) in masked azimuth bands')
    return Subset(dataset, np.where(keep)[0])

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, required=True, help='MLflow experiment name')
parser.add_argument('--log_dir',    type=str, required=True, help='TensorBoard log directory')
parser.add_argument('--output_dir', type=str, required=True, help='Directory for weights and artifacts')
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
with open("/data/beegfs/home/driouech/darcy/IQ_Diffusion/DiffWave/config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=args.log_dir)

mlflow.set_experiment(args.experiment)

# ── Model ─────────────────────────────────────────────────────────────────────
model = DiffWaveRF(input_channels=8, residual_channels=64, cond_dim=7).to(device)
model.train()

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

# ── Data ──────────────────────────────────────────────────────────────────────
from UniversalDataLoader import UniversalDataset
dataset_train = UniversalDataset(task_id=132, mode="train", angle_mode='sincos')
dataset_train = _filter_masked_bands(dataset_train)
dataloader_train = DataLoader(dataset_train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
dataset_test = UniversalDataset(task_id=132, mode="test", angle_mode='sincos')
dataloader_test = DataLoader(dataset_test, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

engine = DiffusionEngine(model=model, timesteps=1000)
diffusion_localizer = DiffusionVisualizer(engine=engine, writer=writer, device=device)
min_loss = float('inf')


# Fixed position normalization constants (computed from full dataset, task 132)
_POS_MEAN = torch.tensor([0.721, -0.034, -1.042], dtype=torch.float32)
_POS_STD  = torch.tensor([6.922,  4.555,  0.552], dtype=torch.float32)


def _normalize_position(pos):
    """Normalize XYZ position using fixed dataset statistics → zero mean, unit std."""
    mean = _POS_MEAN.to(pos.device)
    std  = _POS_STD.to(pos.device)
    return (pos - mean) / std


def _prepare_batch(x, y):
    """Convert raw complex batch to normalized float I/Q and extract condition.

    Per-sample z-score: each sample is normalized independently over all its
    channels and time steps, avoiding batch-composition effects on scale.
    Condition: [pos_x, pos_y, pos_z, sin(az), cos(az), sin(el), cos(el)] — shape [B, 7]
    """
    x = x.to(device)
    input_real = x.real.to(dtype=torch.float32)
    input_imag = x.imag.to(dtype=torch.float32)
    inp = torch.cat([input_real, input_imag], dim=1)   # [B, 8, 1024]
    mean = inp.mean(dim=(1, 2), keepdim=True)
    std  = inp.std(dim=(1, 2), keepdim=True) + 1e-8
    inp  = (inp - mean) / std
    pos = _normalize_position(y[0].to(device, dtype=torch.float32))  # [B, 3]
    az  = y[1].to(device, dtype=torch.float32)                        # [B, 2]
    el  = y[2].to(device, dtype=torch.float32)                        # [B, 2]
    condition = torch.cat([pos, az, el], dim=1)                       # [B, 7]
    return inp, condition


def train_step(x, labels):
    model.train()
    optimizer.zero_grad()

    t = torch.randint(0, engine.timesteps, (x.shape[0],), device=device)
    noise = torch.randn_like(x)
    x_t, _ = engine.add_noise(x, t, noise)
    noise_pred = model(x_t, t, labels)

    loss = F.mse_loss(noise_pred, noise)

    with torch.no_grad():
        per_channel_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(0, 2))

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    optimizer.step()

    return loss.item(), grad_norm.item(), per_channel_loss, t


def training_epoch(model, dataloader, optimizer, epoch):
    all_t_values = []
    total_loss   = 0.0
    max_grad     = 0.0
    ema_loss     = None
    ema_alpha    = 0.98

    for step, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inp, condition = _prepare_batch(x, y)

        loss, grad_norm, per_channel_loss, t = train_step(inp, condition)
        global_step = epoch * len(dataloader) + step

        total_loss += loss
        max_grad    = max(max_grad, grad_norm)
        ema_loss    = loss if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss

        writer.add_scalar('Loss/Train_Step',     loss,     global_step)
        writer.add_scalar('Loss/Train_Step_EMA', ema_loss, global_step)
        writer.add_scalar('Training/Gradient_Norm', grad_norm, global_step)
        mlflow.log_metrics({'train_loss': loss, 'grad_norm': grad_norm}, step=global_step)
        all_t_values.append(t.cpu())

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}, GradNorm: {grad_norm:.4f}")
            for ch in range(per_channel_loss.shape[0]):
                writer.add_scalar(f'Training/PerChannel_Loss_ch{ch}', per_channel_loss[ch].item(), global_step)

    avg_loss = total_loss / len(dataloader)

    # Per-epoch summaries
    writer.add_scalar('Training/Max_Grad_Norm_Epoch', max_grad, epoch)
    writer.add_histogram('Training/Timestep_Distribution', torch.cat(all_t_values).float(), epoch)

    # Per-layer gradient norm histogram (gradients still live from last backward)
    grad_norms = [p.grad.norm().item()
                  for p in model.parameters() if p.grad is not None]
    if grad_norms:
        writer.add_histogram('Training/Per_Layer_Grad_Norms', torch.tensor(grad_norms), epoch)

    return avg_loss


def evaluation_epoch(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inp, condition = _prepare_batch(x, y)

            t = torch.randint(0, engine.timesteps, (inp.shape[0],), device=device)
            noise = torch.randn_like(inp)
            x_t, noise = engine.add_noise(inp, t, noise)
            noise_pred = model(x_t, t, condition)

            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            writer.add_scalar('Loss/Eval_Step', loss.item(), epoch * len(dataloader) + step)
            mlflow.log_metric('eval_loss', loss.item(), step=epoch * len(dataloader) + step)

        diffusion_localizer.log_all(epoch)

    return total_loss / len(dataloader)


# ── Training loop ─────────────────────────────────────────────────────────────
early_stopping_counter = cfg['early_stopping_patience']
os.makedirs(args.output_dir, exist_ok=True)
with mlflow.start_run():
    mlflow.log_params({
        'batch_size': cfg['batch_size'],
        'learning_rate': cfg['learning_rate'],
        'epochs': cfg['epochs'],
        'early_stopping_patience': cfg['early_stopping_patience'],
        'model': 'DiffWaveRF',
        'input_channels': 8,
        'residual_channels': 64,
        'cond_dim': 7,
        'timesteps': 1000,
        'ddim_steps': 50,
        'experiment': args.experiment,
        'log_dir': args.log_dir,
        'output_dir': args.output_dir,
    })
    for i in range(cfg['epochs']):
        train_loss = training_epoch(model, dataloader_train, optimizer, i)
        writer.add_scalar('Loss/Epoch_Train', train_loss, i)
        print(f"Epoch {i}, Train_Loss: {train_loss}")
        eval_loss = evaluation_epoch(model, dataloader_test, epoch=i)
        writer.add_scalar('Loss/Epoch_Eval', eval_loss, i)
        print(f"Epoch {i}, Evaluation Loss: {eval_loss}")
        mlflow.log_metrics({'epoch_train_loss': train_loss, 'epoch_eval_loss': eval_loss}, step=i)

        if train_loss < min_loss:
            weights_path = f'{args.output_dir}/weights_{i}.h5'
            torch.save(model.state_dict(), weights_path)
            mlflow.log_artifact(weights_path, artifact_path='checkpoints')
            min_loss = train_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter > cfg['early_stopping_patience']:
            print("Early stopping triggered.")
            mlflow.set_tag('stopped_early', True)
            mlflow.log_metric('stopped_at_epoch', i)
            break
