import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from model import UNet1D_Final
from data_loader import get_dataloader

class DiffusionTrainer:
    def __init__(self, model, device, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.device = device
        self.timesteps = timesteps
        
        # Define Noise Schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(self.device)

    def train_step(self, x, mag=None):
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_images(x, t)
        
        # Predict noise
        # If model supports conditional generation, pass mag
        # UNet1D_Final supports 'y' for magnitude
        predicted_noise = self.model(x_t, t, y=mag)
        
        loss = nn.MSELoss()(noise, predicted_noise)
        return loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = UNet1D_Final(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Data Loader
    csv_path = "STEAD/merge.csv"
    hdf5_path = "STEAD/merge.hdf5"
    
    # Fallback for testing if merge not found
    if not os.path.exists(csv_path) and os.path.exists("STEAD/chunk2.csv"):
        print("Using chunk2 data for training...")
        csv_path = "STEAD/chunk2.csv"
        hdf5_path = "STEAD/chunk2.hdf5"
        
    dataloader = get_dataloader(
        csv_path, 
        hdf5_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        mode='train',
        limit=args.limit
    )
    
    # Trainer
    trainer = DiffusionTrainer(model, device, timesteps=args.timesteps)
    
    # Resume logic
    start_epoch = 0
    if args.resume:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir("results") if f.startswith("diffusion_ckpt_") and f.endswith(".pt")]
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_ckpt = checkpoints[-1]
            ckpt_path = os.path.join("results", latest_ckpt)
            
            print(f"Resuming from checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            
            # Parse epoch
            start_epoch = int(latest_ckpt.split("_")[-1].split(".")[0])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoints found to resume from. Starting from scratch.")

    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        total_loss = 0
        
        for i, (images, mags) in enumerate(pbar):
            images = images.to(device)
            mags = mags.to(device)
            
            optimizer.zero_grad()
            loss = trainer.train_step(images, mag=mags)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            
            if args.debug and i >= 50:
                print("Debug mode: stopping after 50 batches.")
                break
        
        avg_loss = total_loss / (i + 1)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or args.debug:
            save_path = os.path.join("results", f"diffusion_ckpt_{epoch+1}.pt")
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            if args.debug:
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0) # Set to 0 for debugging/windows safety
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (limit batches)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    train(args)
