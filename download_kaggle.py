import kagglehub
import shutil
import os

def download_stead_kaggle():
    print("Downloading STEAD dataset via KaggleHub...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("isevilla/stanford-earthquake-dataset-stead")
        print("Path to dataset files:", path)
        
        # Move files to our project directory
        target_dir = "STEAD"
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Moving files from {path} to {target_dir}...")
        
        for filename in os.listdir(path):
            src = os.path.join(path, filename)
            dst = os.path.join(target_dir, filename)
            
            # Handle potential collisions
            if os.path.exists(dst):
                print(f"File {filename} already exists in target. Overwriting/Skipping...")
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
                
        print("Files moved successfully.")
        
        # List files in STEAD to confirm
        print("Contents of STEAD directory:")
        for f in os.listdir(target_dir):
            print(f)
            
    except Exception as e:
        print(f"Kaggle download failed: {e}")

if __name__ == "__main__":
    download_stead_kaggle()
