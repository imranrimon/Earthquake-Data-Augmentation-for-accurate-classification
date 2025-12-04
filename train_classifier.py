import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import STEADDataset
from magnet import MagNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

def train_magnet(use_synthetic=False):
    print(f"\n--- Training MagNet (Synthetic Augmented: {use_synthetic}) ---")
    csv_path = "STEAD/merge.csv"
    hdf5_path = "STEAD/merge.hdf5"
    
    if not os.path.exists(csv_path) and os.path.exists("STEAD/chunk2.csv"):
        csv_path = "STEAD/chunk2.csv"
        hdf5_path = "STEAD/chunk2.hdf5"
    
    if not os.path.exists(hdf5_path):
        print("Real data not found.")
        return

    # We need a custom way to get data + labels (magnitude)
    # The current STEADDataset only returns waveforms.
    print("Loading Real Data...")
    full_dataset = STEADDataset(csv_path, hdf5_path)
    
    # Use DataLoader for faster loading
    temp_loader = DataLoader(full_dataset, batch_size=100, num_workers=4, shuffle=False)
    
    real_waveforms = []
    real_labels = []
    
    print("Loading Real Samples with DataLoader...")
    for wf, mag in tqdm(temp_loader):
        real_waveforms.append(wf.numpy())
        real_labels.append(mag.numpy())
        
    real_X = torch.tensor(np.concatenate(real_waveforms), dtype=torch.float32)
    real_y = torch.tensor(np.concatenate(real_labels), dtype=torch.float32).unsqueeze(1)
        
    real_X = torch.tensor(np.concatenate(real_waveforms), dtype=torch.float32)
    real_y = torch.tensor(np.concatenate(real_labels), dtype=torch.float32).unsqueeze(1)
    
    print(f"Real Data: {real_X.shape}, Labels: {real_y.shape}")
    
    # Split Real into Train/Test
    train_size = int(0.8 * len(real_X))
    test_size = len(real_X) - train_size
    
    real_train_X, real_test_X = torch.split(real_X, [train_size, test_size])
    real_train_y, real_test_y = torch.split(real_y, [train_size, test_size])
    
    # 2. Load Synthetic Data (if enabled)
    if use_synthetic:
        syn_path = "results/generated/synthetic_data.npy"
        syn_label_path = "results/generated/synthetic_labels.npy"
        if os.path.exists(syn_path) and os.path.exists(syn_label_path):
            print("Loading Synthetic Data and Labels...")
            syn_data = np.load(syn_path) # (N, 3, 6000)
            syn_labels = np.load(syn_label_path) # (N,)
            
            syn_X = torch.tensor(syn_data, dtype=torch.float32)
            syn_y = torch.tensor(syn_labels, dtype=torch.float32).unsqueeze(1)
            
            print(f"Synthetic Data: {syn_X.shape}, Labels: {syn_y.shape}")
            
            # Combine Real Train + Synthetic
            train_X = torch.cat((real_train_X, syn_X), dim=0)
            train_y = torch.cat((real_train_y, syn_y), dim=0)
        else:
            print("Synthetic data or labels not found. Training on Real only.")
            train_X = real_train_X
            train_y = real_train_y
    else:
        train_X = real_train_X
        train_y = real_train_y
        
    print(f"Training Set: {train_X.shape}")
    print(f"Test Set: {real_test_X.shape} (Always Real)")
    
    # Dataloaders
    train_ds = TensorDataset(train_X, train_y)
    test_ds = TensorDataset(real_test_X, real_test_y)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model
    model = MagNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Train
    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train MSE={avg_train_loss:.4f}, Test MSE={avg_test_loss:.4f}, Test RMSE={np.sqrt(avg_test_loss):.4f}")
            
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # Final Evaluation on Test Set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mse = np.mean((all_preds - all_targets)**2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"Final Test Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Save results
    suffix = "augmented" if use_synthetic else "baseline"
    torch.save(model.state_dict(), f"results/magnet_{suffix}.pth")
    
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.title(f"MagNet Training ({suffix})")
    plt.savefig(f"results/magnet_loss_{suffix}.png")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true", help="Train on Real + Synthetic data")
    args = parser.parse_args()
    
    train_magnet(use_synthetic=args.augmented)
