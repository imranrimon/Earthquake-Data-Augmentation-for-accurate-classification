# Seismic DDPM: Generative Diffusion for Earthquake Waveforms

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate synthetic seismic waveforms. It uses a 1D UNet architecture and is trained on the **STEAD** (Stanford Earthquake Dataset).

The goal is to augment training data for geohazard classifiers (e.g., MagNet) to improve their performance and robustness.

## Project Structure

*   **`main_pipeline.py`**: The master script. Runs the entire workflow: Extraction -> Training -> Generation -> Evaluation.
*   **`train_ddpm.py`**: Trains the Diffusion Model.
*   **`generate.py`**: Generates synthetic waveforms using the trained model.
*   **`train_classifier.py`**: Trains the MagNet classifier (Baseline vs. Augmented).
*   **`model.py`**: Defines the 1D UNet architecture.
*   **`dataset.py`**: Handles data loading and normalization.
*   **`magnet.py`**: Defines the MagNet classifier architecture.
*   **`download_kaggle.py`**: Downloads the full STEAD dataset (~80GB) from Kaggle.

## Setup

1.  **Environment**: Ensure you are in the `pytorch_env` (or equivalent with PyTorch installed).
    ```bash
    conda activate pytorch_env
    ```
2.  **Dependencies**:
    ```bash
    pip install torch numpy pandas matplotlib h5py tqdm kagglehub
    ```

## Usage

### 1. Download Data
Run the Kaggle downloader to get the full dataset:
```bash
python download_kaggle.py
```
*Note: This downloads ~80GB of data to the `STEAD/` directory.*

### 2. Run Full Pipeline
Once the data is downloaded, run the main pipeline:
```bash
python main_pipeline.py
```
This will:
1.  Verify the data exists.
2.  **Train** the DDPM (saves to `results/ddpm_final.pth`).
3.  **Generate** synthetic samples (saved to `results/generated/`).
4.  **Train** the Classifier (Baseline) and report RMSE.

### 3. Experiments
*   **Baseline**: The default run trains MagNet on **Real Data Only**.
*   **Augmentation**: To train on **Real + Synthetic**, edit `train_classifier.py` or `main_pipeline.py` to enable the augmented experiment.

## Results
Check the `results/` directory for:
*   `training_loss.png`: DDPM training curve.
*   `generated/*.png`: Plots of synthetic waveforms.
*   `magnet_loss_baseline.png`: Classifier training curve.
