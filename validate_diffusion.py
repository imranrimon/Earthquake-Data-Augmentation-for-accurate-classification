import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse

from model import UNet1D_Final
from ddim_sampler import DDIMSampler
from data_loader import get_dataloader

def plot_waveform(ax, waveform, title):
    # waveform: (3, 6000)
    # Channels: E, N, Z
    labels = ['E', 'N', 'Z']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        ax.plot(waveform[i], label=labels[i], color=colors[i], alpha=0.7)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_spectrogram(ax, waveform, title, fs=100):
    # waveform: (6000,) - use Z component (index 2) usually, or sum
    # Let's plot Z component spectrogram
    f, t, Sxx = signal.spectrogram(waveform, fs)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)

def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Real Data
    print("Loading real data...")
    csv_path = "STEAD/merge.csv"
    hdf5_path = "STEAD/merge.hdf5"
    if not os.path.exists(csv_path) and os.path.exists("STEAD/chunk2.csv"):
        csv_path = "STEAD/chunk2.csv"
        hdf5_path = "STEAD/chunk2.hdf5"
        
    loader = get_dataloader(csv_path, hdf5_path, batch_size=1, mode='validation')
    real_batch, _ = next(iter(loader))
    real_waveform = real_batch[0].numpy() # (3, 6000)
    
    # 2. Generate Synthetic Data
    print("Generating synthetic data...")
    model = UNet1D_Final(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ).to(device)
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint {args.ckpt} not found. Skipping generation.")
        return

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    sampler = DDIMSampler(model, device)
    
    # Generate 1 sample
    synthetic_tensor = sampler.sample(1, ddim_steps=50)
    synthetic_waveform = synthetic_tensor[0].cpu().numpy() # (3, 6000)
    
    # 3. Visualization
    print("Creating plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time Domain
    plot_waveform(axs[0, 0], real_waveform, "Real Earthquake (Time Domain)")
    plot_waveform(axs[0, 1], synthetic_waveform, "Synthetic Earthquake (Time Domain)")
    
    # Frequency Domain (Spectrogram of Z component)
    plot_spectrogram(axs[1, 0], real_waveform[2], "Real Spectrogram (Z channel)")
    plot_spectrogram(axs[1, 1], synthetic_waveform[2], "Synthetic Spectrogram (Z channel)")
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/validation_comparison.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    validate(args)
