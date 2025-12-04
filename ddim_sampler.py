import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

from model import UNet1D_Final

class DDIMSampler:
    def __init__(self, model, device, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.device = device
        self.timesteps = timesteps
        
        # Define Noise Schedule (Same as training)
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def sample(self, n_samples, ddim_steps=50, shape=(3, 6000), mag=None):
        self.model.eval()
        
        # Subsequence selection
        # We want ddim_steps evenly spaced from timesteps-1 to 0
        # e.g. if T=1000, steps=50 -> [999, 979, ..., 19, -1] (approx)
        # We use a simple linspace
        c = self.timesteps // ddim_steps
        time_seq = list(range(0, self.timesteps, c))
        time_seq = [t for t in time_seq if t < self.timesteps]
        time_seq = reversed(time_seq)
        # Convert to tensor list for iteration
        time_seq = list(time_seq)
        
        # Start with random noise
        x = torch.randn((n_samples, *shape)).to(self.device)
        
        print(f"Sampling {n_samples} waveforms with {len(time_seq)} DDIM steps...")
        
        for i, t in enumerate(tqdm(time_seq)):
            # Current timestep tensor
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.model(x, t_tensor, y=mag)
                
            # Get alpha_hat for current t and prev t
            alpha_hat_t = self.alpha_hat[t]
            
            # Previous timestep in the sequence
            if i < len(time_seq) - 1:
                t_prev = time_seq[i+1]
                alpha_hat_t_prev = self.alpha_hat[t_prev]
            else:
                # Last step, t_prev = -1 (conceptually), alpha_hat_0 = 1 (conceptually, but actually alpha_hat[0] is not 1)
                # Ideally we want to go to x0.
                # If t=0, alpha_hat_t is close to 1? No, alpha_hat[0] = 1 - beta[0] ~ 0.9999
                # We assume alpha_hat_prev = 1.0 for the final step to x0
                alpha_hat_t_prev = torch.tensor(1.0).to(self.device)
                
            # DDIM Update
            # 1. Predict x0
            # x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * eps
            # x0 = (x_t - sqrt(1 - alpha_hat_t) * eps) / sqrt(alpha_hat_t)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
            
            # Clip x0 to [-1, 1] for stability (optional but recommended)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # 2. Direction pointing to x_t
            # dir_xt = sqrt(1 - alpha_hat_t_prev) * eps
            dir_xt = torch.sqrt(1 - alpha_hat_t_prev) * predicted_noise
            
            # 3. Random noise (0 for deterministic DDIM)
            noise = 0
            
            # 4. Update x
            x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + noise
            
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = UNet1D_Final(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    sampler = DDIMSampler(model, device)
    
    # Batch generation to avoid OOM
    batch_size = 100
    n_batches = int(np.ceil(args.n_samples / batch_size))
    
    all_samples = []
    all_mags = []
    
    print(f"Generating {args.n_samples} samples in {n_batches} batches...")
    
    for i in range(n_batches):
        current_batch_size = min(batch_size, args.n_samples - i * batch_size)
        
        # Sample magnitudes
        mags = torch.rand(current_batch_size, device=device) * 2.0 + 3.0
        
        samples = sampler.sample(current_batch_size, ddim_steps=args.steps, mag=mags)
        
        all_samples.append(samples.cpu().numpy())
        all_mags.append(mags.cpu().numpy())
        
    all_samples = np.concatenate(all_samples, axis=0)
    all_mags = np.concatenate(all_mags, axis=0)
    
    # Save results
    os.makedirs("results/generated", exist_ok=True)
    
    # Save as NPY for classifier training
    np.save("results/generated/synthetic_data.npy", all_samples)
    np.save("results/generated/synthetic_labels.npy", all_mags)
    print(f"Saved {args.n_samples} samples to results/generated/")
    
    # Plot first sample
    sample = all_samples[0]
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(sample[0], label='E')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(sample[1], label='N')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(sample[2], label='Z')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/generated_sample.png")
    print("Saved sample plot to results/generated_sample.png")
