import math
from torch import nn
import torch

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionEngine(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = next(model.parameters()).device

        # --- 1. Define the Schedule (The "Fuel") ---
        # We calculate Alpha Bars (Cumulative Signal Strength) directly
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod.to(self.device)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod = alphas_cumprod[1:]  # Remove initial 1.0 entry
        alphas_cumprod = torch.clamp(alphas_cumprod, min=0.001)

        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(self.device)

    def add_noise(self, x_start, t, noise=None):
        """
        Forward Process: Adds noise to clean data for training.
        """
        
        # Extract coeff for the specific timesteps t
        # (Batch, 1, 1) for broadcasting
        s_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        s_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        # The mixture: Signal * Signal_Strength + Noise * Noise_Strength
        x_noisy = s_alpha * x_start + s_one_minus_alpha * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample_ddim(self, batch_size, seq_len, conditions, steps=50, eta=0.0):
        """
        DDIM Sampler: The 'Fast' Engine.
        Generates data in 'steps' (e.g. 50) instead of 'timesteps' (1000).
        
        eta: 0.0 = Deterministic (DDIM), 1.0 = Stochastic (DDPM)
        """
        self.model.eval()
        B = batch_size
        C = self.model.input_channels
        L = seq_len
        
        # 1. Start with pure Gaussian Noise
        x = torch.randn(B, C, L, device=self.device)
        
        # FIX 1: Correct 0-based Indexing
        # If T=1000, we want indices from 999 down to 0.
        # We stop at 0. 'steps' defines how many jumps we take.
        times = torch.linspace(self.timesteps - 1, 0, steps).long().to(self.device)
        
        pics = []
        
        for i, t in enumerate(times):
            # Calculate next timestep (t_prev)
            # If we are at the last step, t_prev must be -1 (conceptually) or handled explicitly
            # Standard practice: use indices, so the "next" index is simply the next in the list.
            if i < len(times) - 1:
                t_prev = times[i + 1]
            else:
                t_prev = torch.tensor(-1, device=self.device) # Special marker for final step

            # A. Model Prediction
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch, conditions)
            
            # B. Get alphas
            alpha_bar_t = self.alphas_cumprod[t]
            print(f"Step {i+1}/{steps}, t={t.item()}, alpha_bar_t={alpha_bar_t.item()}")
            
            # Handle the final step (t_prev < 0) where alpha_bar_prev should be 1.0 (no noise)
            if t_prev >= 0:
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                print(f"    t_prev={t_prev.item()}, alpha_bar_prev={alpha_bar_prev.item()}")
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)
                print()

            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            assert sigma == sigma, "Sigma is NaN, check calculations."
            
            # Predict x0 (Clean signal)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
            
            # Noise (only if eta > 0)
            noise = torch.randn_like(x) * sigma
            
            # Update x
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

            pics.append(x.cpu().numpy())
            
        return x, pics

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation, cond_dim):
        super().__init__()
        self.dilation = dilation
        
        # 1. Dilated Convolution (The "Receptive Field" expander)
        self.dilated_conv = nn.Conv1d(
            channels, 
            2 * channels, # Double channels for Gated Activation unit
            kernel_size=3, 
            dilation=dilation, 
            padding=dilation # Keep length consistent
        )
        
        # 2. Conditioning Projection (Time + AoA + SNR)
        # We map the condition vector to modify the features
        self.cond_proj = nn.Conv1d(cond_dim, 2 * channels, 1)
        
        # 3. Output projections
        self.output_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x, conditions):
        """
        x: (Batch, Channels, Length)
        conditions: (Batch, Cond_Dim, 1) - expanded conditioning vector
        """
        # A. Dilated Conv
        h = self.dilated_conv(x)
        
        # B. Inject Condition (Add to the conv output)
        # conditions is already projected to correct size before loop or here? 
        # For efficiency, we usually project inside the block or pre-project. 
        # Here we project inside to match the 2*channels.
        cond = self.cond_proj(conditions)
        h = h + cond
        
        # C. Gated Activation Unit (WaveNet style)
        # Split into Filter and Gate
        filter_gate, gate_gate = h.chunk(2, dim=1)
        h = torch.tanh(filter_gate) * torch.sigmoid(gate_gate)
        
        # D. Skip Connection & Residual
        out = self.output_proj(h)
        skip = self.skip_proj(h)
        
        # Residual connection (input + processed)
        return (x + out), skip
class DiffWaveRF(nn.Module):
    def __init__(self, 
                 input_channels=4,      # 2 Antennas (I,Q)
                 residual_channels=64,  # Internal width
                 num_layers=30,         # Deep network
                 cond_dim=2):           # SNR + Angle
        super().__init__()
        
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        
        # 1. Input Projection
        self.input_projection = nn.Conv1d(input_channels, residual_channels, 1)
        
        # 2. Time & Condition Embedding
        self.diffusion_embedding = DiffusionEmbedding(128)
        
        # We combine Time (128) + Condition (cond_dim) -> Hidden Size
        # Then project that to 'residual_channels' size for the blocks
        self.cond_mlp = nn.Sequential(
            nn.Linear(128 + cond_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU()
        )
        
        # 3. Stack of Dilated Blocks
        self.blocks = nn.ModuleList()
        # Cycle dilation: 1, 2, 4, ..., 512, 1, 2, 4, ...
        # This resets the dilation loop multiple times to capture different scales
        cycle_layers = 10 
        for i in range(num_layers):
            dilation = 2 ** (i % cycle_layers)
            self.blocks.append(
                DilatedResidualBlock(
                    channels=residual_channels, 
                    dilation=dilation,
                    cond_dim=512 # Size coming out of cond_mlp
                )
            )
            
        # 4. Output Projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(residual_channels, residual_channels, 1),
            nn.SiLU(),
            nn.Conv1d(residual_channels, input_channels, 1)
        )

    def forward(self, x, t, conditions):
        # x: (Batch, 4, Length)
        # t: (Batch)
        # conditions: (Batch, 2)
        
        # 1. Embed Time and Conditions
        t_emb = self.diffusion_embedding(t) # (Batch, 128)
        
        # Concatenate Time + Signal Conditions
        # (Batch, 128 + 2)
        combined_cond = torch.cat([t_emb, conditions], dim=1)
        
        # Project to embedding dimension
        cond_feats = self.cond_mlp(combined_cond) # (Batch, 512)
        
        # Unsqueeze for Conv1d broadcasting: (Batch, 512, 1)
        cond_feats = cond_feats.unsqueeze(-1) 
        
        # 2. Initial Conv
        x = self.input_projection(x)
        
        # 3. Iterate Blocks
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x, cond_feats)
            skip_connections = skip_connections + skip
            
        # 4. Final Output (Predict Noise)
        # We normalize by sqrt(num_layers) to keep magnitude stable
        return self.output_projection(skip_connections / math.sqrt(len(self.blocks)))