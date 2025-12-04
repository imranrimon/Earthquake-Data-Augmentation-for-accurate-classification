import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from data_loader import STEADDataset
from magnet import MagNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

def evaluate_model(model_path, test_loader, description):
    print(f"\nEvaluating {description}...")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None, None
        
    model = MagNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    
    print(f"--- {description} Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    return rmse, mae, r2

def main():
    # Load Real Data for Testing
    hdf5_path = "STEAD/merge.hdf5"
    csv_path = "STEAD/merge.csv"
    
    if not os.path.exists(csv_path) and os.path.exists("STEAD/chunk2.csv"):
        csv_path = "STEAD/chunk2.csv"
        hdf5_path = "STEAD/chunk2.hdf5"
    
    if not os.path.exists(hdf5_path):
        print("Data not found.")
        return

    print("Loading Test Data (Real)...")
    full_dataset = STEADDataset(csv_path, hdf5_path)
    
    # Use DataLoader
    # num_workers=0 to avoid Windows deadlock
    loader = DataLoader(full_dataset, batch_size=100, num_workers=0, shuffle=False)
    
    real_waveforms = []
    real_labels = []
    
    print("Loading Real Samples...")
    for wf, mag in tqdm(loader):
        real_waveforms.append(wf.numpy())
        real_labels.append(mag.numpy())
        
    real_X = torch.tensor(np.concatenate(real_waveforms), dtype=torch.float32)
    real_y = torch.tensor(np.concatenate(real_labels), dtype=torch.float32).unsqueeze(1)
    
    # Split (Same as training: 70/15/15)
    total_len = len(real_X)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    _, _, real_test_X = torch.split(real_X, [train_len, val_len, test_len])
    _, _, real_test_y = torch.split(real_y, [train_len, val_len, test_len])
    
    print(f"Test Set: {real_test_X.shape}")
    
    test_ds = TensorDataset(real_test_X, real_test_y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Evaluate Baseline
    evaluate_model("results/magnet_baseline_best.pth", test_loader, "Baseline (Real Only)")
    
    # Evaluate Augmented
    evaluate_model("results/magnet_augmented_best.pth", test_loader, "Augmented (Real + Synthetic)")

if __name__ == "__main__":
    main()
