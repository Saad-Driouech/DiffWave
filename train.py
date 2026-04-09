import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from utils.StationaryEttus import StationaryEttus
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.DiffWave import DiffWaveRF, DiffusionEngine
from utils.visualization import DiffusionVisualizer

with open("/data/beegfs/home/driouech/darcy/IQ_Diffusion/DiffWave/config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='/data/beegfs/home/driouech/darcy/IQ_Diffusion/DiffWave/NormalizedData3/logs')

# Model
#model = UNetSignalCVAE(seq_len=1024, input_channel=16, latent_dim=64, num_conditions=2).to(device)
model = DiffWaveRF(input_channels=16, residual_channels=64, cond_dim=2).to(device)

model.train()

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

# Data
dataset_train = StationaryEttus(mode = "train")
dataloader_train = DataLoader(dataset_train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
dataset_test = StationaryEttus(mode = "test")
dataloader_test = DataLoader(dataset_test, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)


engine = DiffusionEngine(model=model, timesteps=1000)
diffusion_localizer = DiffusionVisualizer(engine=engine, writer=writer, device=device)
min_loss = float('inf')
def train_step(x, labels, beta=0.1):
    """
    beta: Weight for KL Divergence (Disentanglement)
    lambda_wavelet: Weight for Wavelet Loss (Feature fidelity)
    """
    model.train()
    optimizer.zero_grad()
    # 1. Sample Noise & Time
    t = torch.randint(0, engine.timesteps, (x.shape[0],), device=device)
    noise = torch.randn_like(x)

    # 2. Create Noisy Input (x_t)
    x_t, _ = engine.add_noise(x, t, noise)

    # 3. Forward Pass (Predict Noise)
    noise_pred = model(x_t, t, labels)

    loss = F.mse_loss(noise_pred, noise)

    # Per-channel loss for debugging (no extra backward pass needed)
    with torch.no_grad():
        per_channel_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(0, 2))  # (16,)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    optimizer.step()

    #### LEGACY CODE FOR CVAE ####
    # recon_x, mu, logvar = model(x, labels)
    #
    # # 1. Standard Reconstruction Loss (MSE or L1)
    # loss_recon = recon_loss(recon_x, x)
    # # 2. Wavelet Loss (Time-Frequency structure)
    # loss_wavelet = wavelet_criterion(recon_x, x)
    # # 3. Spatial Consistency Loss (across antennas)
    # loss_spatial = spatial_loss(recon_x, x)
    #
    # # 3. KL Divergence
    # loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #
    #
    # # Composite Loss
    # # We heavily weight the wavelet loss to force the model to respect signal transients
    # # print(f"Recon Loss: {loss_recon.item()}, Wavelet Loss: {loss_wavelet.item()}, KL Loss: {loss_kld.item()}")
    # total_loss = loss_recon + 10*loss_spatial + loss_wavelet + beta * loss_kld
    #
    # total_loss.backward()
    # optimizer.step()
    #### LEGACY CODE FOR CVAE ####

    return loss.item(), grad_norm.item(), per_channel_loss, t

def training_epoch(model, dataloader, optimizer, epoch):
    all_t_values = []
    for step, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device)
        optimizer.zero_grad()
        input_real = x.real.to(dtype=torch.float32)
        input_imag = x.imag.to(dtype=torch.float32)

        input = torch.cat([input_real, input_imag], dim=1)
        condition = y[1].to(device, dtype=torch.float32)

        loss, grad_norm, per_channel_loss, t = train_step(input, condition, beta=0.1)
        global_step = epoch * len(dataloader) + step

        writer.add_scalar('Loss/Train_Total_Loss', loss, global_step)
        writer.add_scalar('Training/Gradient_Norm', grad_norm, global_step)
        all_t_values.append(t.cpu())

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}, GradNorm: {grad_norm:.4f}")
            for ch in range(per_channel_loss.shape[0]):
                writer.add_scalar(f'Training/PerChannel_Loss_ch{ch}', per_channel_loss[ch].item(), global_step)

    # Timestep distribution histogram once per epoch
    writer.add_histogram('Training/Timestep_Distribution', torch.cat(all_t_values).float(), epoch)
    return loss

def evaluation_epoch(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, (x, y) in tqdm(enumerate(dataloader), total = len(dataloader)):
            x = x.to(device)
            input_real = x.real.to(dtype=torch.float32)
            input_imag = x.imag.to(dtype=torch.float32)

            input = torch.cat([input_real, input_imag], dim=1)
            condition = y[1].to(device, dtype = torch.float32)

            t = torch.randint(0, engine.timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(input)
            x_t, noise= engine.add_noise(input, t, noise)
            noise_pred = model(x_t, t, condition)

            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            #log_visualizations(writer, model, dataloader, epoch=epoch, device=device)
            #visualize_latent_traversal(writer, model, epoch=epoch, device=device)
            #log_phase_diff(writer, input[:, ::4, :], output[:, ::4, :], epoch=epoch)
            writer.add_scalar('Loss/Eval_Total_Loss', loss.item(), epoch*len(dataloader) + step)
        diffusion_localizer.log_all(input, condition, epoch)
        
    return total_loss / len(dataloader)

early_stopping_counter = cfg['early_stopping_patience']
for i in range(cfg['epochs']):
    train_loss = training_epoch(model, dataloader_train, optimizer, i)
    print(f"Epoch {i}, Train_Loss: {train_loss}")
    eval_loss = evaluation_epoch(model, dataloader_test, epoch=i)
    print(f"Epoch {i}, Evaluation Loss: {eval_loss}")
    if train_loss < min_loss:
        torch.save(model.state_dict(), f'/data/beegfs/home/driouech/darcy/IQ_Diffusion/DiffWave/NormalizedData3/weights_{i}.h5')
        min_loss = train_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    if early_stopping_counter > 20:
        print("Early stopping triggered.")
        break

