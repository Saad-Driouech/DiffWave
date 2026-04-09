from torch import nn
import torch
import torch.nn.functional as F

class HaarWaveletLoss(nn.Module):
    def __init__(self, levels=3, device='cpu'):
        super(HaarWaveletLoss, self).__init__()
        self.levels = levels
        self.device = device
        
        # Haar Wavelet Filters (Low-pass and High-pass)
        # Normalized by 1/sqrt(2) to preserve energy
        self.scale = 0.70710678
        
        # We create filters for 1D Convolution
        # Shape: (Out_Channels, In_Channels/Groups, Kernel_Size)
        # We apply this independently to I and Q channels (Groups=2)
        self.dec_hi = torch.tensor([-self.scale, self.scale], device=device).view(1, 1, 2)
        self.dec_lo = torch.tensor([self.scale, self.scale], device=device).view(1, 1, 2)
        self.dec_hi = self.dec_hi.repeat(16, 1, 8)
        self.dec_lo = self.dec_lo.repeat(16, 1, 8)
        

    def forward(self, input_sig, target_sig):
        """
        Calculates L1 loss between Wavelet Coefficients of Input and Target
        """
        loss = 0.0
        
        # Inputs need to be (Batch, 2, Length)
        curr_input = input_sig
        curr_target = target_sig
        
        for i in range(self.levels):
            # Apply Filters (High Pass & Low Pass)
            # Padding=0, Stride=2 (Downsampling)
            
            # --- INPUT ---
            in_hi = F.conv1d(curr_input, self.dec_hi, stride=2, groups=16)
            in_lo = F.conv1d(curr_input, self.dec_lo, stride=2, groups=16)
            
            # --- TARGET ---
            tar_hi = F.conv1d(curr_target, self.dec_hi, stride=2, groups=16)
            tar_lo = F.conv1d(curr_target, self.dec_lo, stride=2, groups=16)
            
            # Accumulate L1 Loss on Details (High Freq) and Approximation (Low Freq)
            # We weight lower levels (finer details) higher? Or equal?
            # Usually equal weighting across scales works well for signals.
            loss += F.l1_loss(in_hi, tar_hi) + F.l1_loss(in_lo, tar_lo)
            
            # Prepare next level (process the Low Pass version)
            curr_input = in_lo
            curr_target = tar_lo
        return loss


class RFSignalLoss(nn.Module):
    def __init__(self, fft_size=512, alpha_time=1.0, alpha_freq=1.0):
        super(RFSignalLoss, self).__init__()
        self.fft_size = fft_size
        self.alpha_time = alpha_time
        self.alpha_freq = alpha_freq

    def forward(self, recon_x, x):
        """
        recon_x: (Batch, 2, Seq_Len) - Generated
        x:       (Batch, 2, Seq_Len) - Target
        """
        loss_time = F.mse_loss(recon_x, x)

        # --- 2. Frequency Domain Loss (Spectral Magnitude) ---
        # We treat the (2, Seq_Len) input as complex data for FFT
        # PyTorch expects the last dimension to be the complex one for some older versions,
        # but modern torch.fft expects complex tensors.
        
        # Convert (B, 2, L) -> (B, L) complex tensor
        x_complex = torch.complex(x[:, ::2, :], x[:, 1::2, :])
        recon_complex = torch.complex(recon_x[:, ::2, :], recon_x[:, 1::2, :])

        # Compute FFT (Periodogram)
        # We normalize by length to keep loss scale manageable
        x_fft = torch.fft.fft(x_complex, dim=-1)
        recon_fft = torch.fft.fft(recon_complex, dim=-1)

        # Compute Magnitude (ignore phase alignment issues here)
        x_mag = torch.abs(x_fft)
        recon_mag = torch.abs(recon_fft)

        # Log-Magnitude Distance (approximates dB difference)
        # Add epsilon to avoid log(0)
        loss_freq = F.mse_loss(torch.log(x_mag + 1e-6), torch.log(recon_mag + 1e-6))

        # --- Combine ---
        total_loss = (self.alpha_time * loss_time) + (self.alpha_freq * loss_freq)
        
        return total_loss

class SpatialConsistencyLoss(nn.Module):
    def __init__(self, num_antennas=2):
        super(SpatialConsistencyLoss, self).__init__()
        self.num_antennas = num_antennas

    def forward(self, recon_x, x):
        """
        recon_x, x: Shape (Batch, Input_Channels, Length)
        Input_Channels must be 2 * Num_Antennas (e.g., 4 channels for 2 antennas: I1, Q1, I2, Q2)
        """
        B, C, L = x.shape
        
        # 1. Reshape to Complex: (Batch, Num_Antennas, Length)
        
        x_c = torch.complex(x[:, :self.num_antennas, :], x[:, self.num_antennas:, :])         # (B, Ant, L)
        recon_c = torch.complex(recon_x[:, :self.num_antennas, :], recon_x[:, self.num_antennas:, :]) # (B, Ant, L)

        # 2. Compute Spatial Covariance Matrix for each item in batch
        # We want R = X * X^H. 
        # Result shape: (Batch, Ant, Ant)
        # We average over the Time dimension (L) to get the statistical covariance
        
        # Real Covariance
        # bmm: Batch Matrix Multiplication
        # (B, Ant, L) x (B, L, Ant) -> (B, Ant, Ant)
        R_real = torch.matmul(x_c, x_c.transpose(1, 2).conj()) / L
        
        # Generated Covariance
        R_gen = torch.matmul(recon_c, recon_c.transpose(1, 2).conj()) / L
        
        # 3. Calculate Loss
        # We want the generated covariance matrix to match the real one.
        # This aligns both Power (diagonal) and Phase Differences (off-diagonal).
        
        loss = torch.norm(R_gen - R_real, p='fro') # Frobenius norm
        
        return loss