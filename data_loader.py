import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal

class STEADDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, mode='train', validation_split=0.1, limit=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            hdf5_file (string): Path to the hdf5 file with waveforms.
            mode (string): 'train' or 'validation'.
            validation_split (float): Fraction of data to use for validation.
            limit (int): Limit number of samples.
        """
        self.hdf5_file = hdf5_file
        
        print(f"Loading metadata from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        
        # Filter Data
        print("Filtering data...")
        initial_len = len(self.df)
        
        # 1. Trace Category: earthquake_local
        self.df = self.df[self.df['trace_category'] == 'earthquake_local']
        
        # 2. Magnitude: 3.0 <= M <= 5.0
        self.df = self.df[(self.df['source_magnitude'] >= 3.0) & (self.df['source_magnitude'] <= 5.0)]
        
        # 3. Distance: < 100 km
        if 'source_distance_km' in self.df.columns:
            self.df = self.df[self.df['source_distance_km'] < 100]
            
        print(f"Filtered {initial_len} -> {len(self.df)} samples.")
        
        if limit is not None:
            print(f"Limiting to {limit} samples.")
            self.df = self.df.iloc[:limit]
        
        # Train/Validation Split
        total_samples = len(self.df)
        val_size = int(total_samples * validation_split)
        
        if mode == 'train':
            self.df = self.df.iloc[:-val_size]
        elif mode == 'validation':
            self.df = self.df.iloc[-val_size:]
            
        self.indices = self.df.index.tolist()
        self.trace_names = self.df['trace_name'].tolist()
        self.h5 = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_file, 'r')
            
        waveform = self.h5.get(f'data/{trace_name}')
        if waveform is None:
            raise ValueError(f"Trace {trace_name} not found in HDF5.")
        waveform = np.array(waveform)
            
        if waveform.shape != (6000, 3):
            pass

        waveform = signal.detrend(waveform, axis=0, type='linear')
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
            
        waveform = waveform.transpose(1, 0)
        waveform_tensor = torch.from_numpy(waveform).float()
        mag = self.df.iloc[idx]['source_magnitude']
        mag_tensor = torch.tensor(mag, dtype=torch.float)
        
        return waveform_tensor, mag_tensor

def get_dataloader(csv_file, hdf5_file, batch_size=32, num_workers=4, mode='train', limit=None):
    dataset = STEADDataset(csv_file, hdf5_file, mode=mode, limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Test the loader
    csv_path = "STEAD/merge.csv"
    hdf5_path = "STEAD/merge.hdf5"
    
    # Use chunk2 if merge not available (for testing)
    import os
    if not os.path.exists(csv_path) and os.path.exists("STEAD/chunk2.csv"):
        csv_path = "STEAD/chunk2.csv"
        hdf5_path = "STEAD/chunk2.hdf5"
        
    print(f"Testing with {csv_path}...")
    
    loader = get_dataloader(csv_path, hdf5_path, batch_size=4, num_workers=0, mode='train')
    
    for batch_idx, (data, mag) in enumerate(loader):
        print(f"Batch {batch_idx}: Data Shape {data.shape}, Mag Shape {mag.shape}")
        print(f"Data Range: [{data.min():.4f}, {data.max():.4f}]")
        break
