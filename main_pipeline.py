import os
import subprocess
import sys

def run_step(script_cmd, description):
    print(f"\n{'='*20} {description} {'='*20}")
    try:
        # Use the current python executable
        cmd = [sys.executable]
        if isinstance(script_cmd, list):
            cmd.extend(script_cmd)
        else:
            cmd.append(script_cmd)
            
        subprocess.check_call(cmd)
        print(f"{description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_cmd}: {e}")
        sys.exit(1)

def main():
    print("Starting Seismic DDPM Pipeline...") 
    
    # 1. Extraction
    if os.path.exists("STEAD/merge.hdf5"):
        print("Data found (merge.hdf5). Skipping extraction.")
    elif not os.path.exists("STEAD/chunk2.hdf5"):
        run_step("extract_data.py", "Extracting Data")
    else:
        print("Data already extracted.")
        
    # 2. Train VAE (Fast)
    # print("\n=== Step 2: Training VAE ===")
    # subprocess.run([sys.executable, "train_vae.py"], check=True)
    
    # 3. Train LDM (Fast)
    # print("\n=== Step 3: Training LDM ===")
    # subprocess.run([sys.executable, "train_ldm.py"], check=True)
    
    # 4. Generate Synthetic Data (LDM)
    # print("\n=== Step 4: Generating Synthetic Data (LDM) ===")
    # subprocess.run([sys.executable, "generate.py"], check=True)
    
    # --- NEW: Signal-Space Diffusion ---
    
    # 4a. Train Signal-Space Diffusion
    print("\n=== Step 4a: Training Signal-Space Diffusion ===")
    # Run for 50 epochs for research-quality results
    subprocess.run([sys.executable, "train_diffusion.py", "--epochs", "50", "--save_interval", "10", "--resume"], check=True)
    
    # 4b. Generate Synthetic Data (DDIM)
    print("\n=== Step 4b: Generating Synthetic Data (DDIM) ===")
    # Generate 50000 samples for augmentation (approx 1:1 with real data)
    # Use the last checkpoint (e.g., epoch 50)
    ckpt_path = "results/diffusion_ckpt_50.pt"
    syn_data_path = "results/generated/synthetic_data.npy"
    
    if os.path.exists(syn_data_path):
        print(f"Synthetic data found at {syn_data_path}. Skipping generation.")
    elif os.path.exists(ckpt_path):
        subprocess.run([sys.executable, "ddim_sampler.py", "--ckpt", ckpt_path, "--n_samples", "50000", "--steps", "50"], check=True)
    else:
        print(f"Checkpoint {ckpt_path} not found. Skipping generation.")
    
    # 5. Train Baseline Classifier (Real Data Only)
    print("\n=== Step 5: Training Baseline Classifier ===")
    subprocess.run([sys.executable, "train_classifier.py"], check=True)
    
    # 6. Train Augmented Classifier (Real + Synthetic)
    print("\n=== Step 6: Training Augmented Classifier ===")
    subprocess.run([sys.executable, "train_classifier.py", "--augmented"], check=True)
    
    print("\nPipeline finished. Check 'results/' directory for outputs.")

if __name__ == "__main__":
    main()
