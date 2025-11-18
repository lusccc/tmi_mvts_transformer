# Enhancing Applicable Travel Mode Identification Under Real-World Noise

> Lu, S., Su, Y., Chai, J., & Yu, L. (2026). *Enhancing applicable travel mode identification under real-world noise: A transformer-based framework with hybrid data and behavior-indication masks.* Transportation Research Part C: Emerging Technologies, 182, 105388.

This repository accompanies the paper above. It implements a transformer-based multi-view time-series (MVTS) framework that fuses trajectory signals, motion-derived features, and behavior-indication masks to improve travel mode identification when faced with heterogeneous and noisy data sources.

---

## Repository Layout

```
.
├── main.py                     # End-to-end training / evaluation entry point
├── requirements.txt            # Python dependencies
├── tmi/
│   ├── data_preprocess/        # Data extraction, augmentation, feature engineering, noise simulation
│   ├── datasets/               # Dataset abstractions, normalization utilities, data split helpers
│   ├── models/                 # Transformer encoders, dual-branch classifiers, custom losses
│   ├── utils/                  # Logging, metrics aggregation, visualization helpers
│   ├── options.py              # Command-line argument definitions
│   ├── runner.py               # Training / validation executors for different tasks
│   ├── training_tools.py       # Early stopping and other generic training utilities
│   ├── ml_classification.py    # Non-deep-learning benchmarks (random forest, etc.)
│   └── optimizers.py           # Wrapper for optimizer selection (Adam / RAdam)
```

Key modules:

- `DualTSTransformerEncoderClassifier`: dual-branch transformer that separately encodes trajectory and feature streams, with mask-aware attention.
- `DenoisingDataset` / `ImputationDataset`: self-supervised tasks used during pretraining.
- `TrainingPipeline`: orchestrates data preparation, model setup, training loops, evaluation, and export of metrics.

---

## Environment Setup

1. **Create environment (recommended)**
   ```bash
   conda create -n tmi python=3.10
   conda activate tmi
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **GPU support**
   - The codebase targets PyTorch 1.13.1 with CUDA 11.7 (`torch~=1.13.1+cu117`).
   - Modify `requirements.txt` if you need a different CUDA toolkit.

---

## Data Preparation

The framework supports trajectory-only, feature-only, and hybrid inputs. Because raw datasets may be proprietary, the repository expects preprocessed artifacts in `./data/`.

1. **Download raw datasets.**
   - **SHL Daily Activity (2018)**: smartphone inertial signals with transportation labels.
   - **Microsoft Geolife**: GPS trajectories with annotated travel modes.
2. **Organize the raw files** according to the scripts in `tmi/data_preprocess/` (see comments inside each script for expected folder layout).
3. **Run the preprocessing pipeline** to reproduce the hybrid data used in the paper:
   - `s1_trajectory_extraction_*.py`: extract raw trajectories and align timestamps.
   - `s2_dataset_split.py`: build train/val/test partitions with stratification.
   - `s3_data_augmentation.py`: augment trajectories via time warping, rotation, etc.
   - `s4_trajectory_feature_calculation_with_CPD.py`: compute motion statistics and change-point-based behavior indicators.
   - `s5_realistic_diverse_noise_simulation.py`: inject sensor and trajectory noise across different levels/types.
4. **Generated artifacts** will be stored under `./data/{DATASET_NAME}_features/{split}/` (or `..._features_sim_noise/` when noise simulation is enabled). The training pipeline loads `.pkl` and `.npy` files produced there.

To accelerate preprocessing on multi-core machines, most scripts support multiprocessing; adjust parameters inside each script as needed.

---

## Running Experiments

All experiments are launched from `main.py`. You can configure runs entirely via command-line flags (see `tmi/options.py`) or by supplying a JSON config with `--config`.

### Common Arguments

- `--task`: selects the objective (e.g., `dual_branch_classification`, `feature_branch_classification`, `ml_classification`, `denoising_pretrain`, `imputation_pretrain`).
- `--data_name`: dataset identifier (e.g., `SHL`, `geolife`).
- `--data_class`: `feature`, `trajectory`, or `trajectory_with_feature`.
- `--output_dir`: destination for checkpoints, predictions, tensorboard logs.
- `--gpu`: GPU index (`-1` for CPU).
- `--test_only testset`: skip training and evaluate a saved model on the test set.
- `--load_model`: path to a `.pth` checkpoint to resume or fine-tune.
- `--noise_level_sweep`: iterate through noise levels, aggregating results (paper Table 5).
- `--sim_noise_sweep`: evaluate across synthetic noise types.

### Example: Dual-branch Transformer (SHL)

```bash
python main.py \
  --task dual_branch_classification \
  --data_name SHL \
  --data_class trajectory_with_feature \
  --output_dir ./output/shl_dual_branch \
  --epochs 200 \
  --batch_size 64 \
  --lr 1e-3 \
  --gpu 0
```

### Example: Machine-Learning Baselines

```bash
python main.py \
  --task ml_classification \
  --data_name geolife \
  --data_class feature \
  --output_dir ./output/geolife_ml
```

This path executes the pipeline defined in `tmi/ml_classification.py`, which computes handcrafted descriptors and trains traditional classifiers (Random Forest, Gradient Boosting, etc.) for comparison.

---

## Outputs & Logging

- **`output_dir/checkpoints/`**: latest and best model weights.
- **`output_dir/predictions/`**: serialized predictions for validation/test sets.
- **`output_dir/tb_summaries/`**: TensorBoard logs (`tensorboard --logdir output_dir/tb_summaries`).
- **`output_dir/output.log`**: detailed logzero trace, including configuration snapshots.
- **`output_dir/configuration.json`**: final merged configuration (CLI + JSON).

Early stopping and key-metric tracking (accuracy or loss) are handled in `tmi/training_tools.py` and `tmi/runner.py`.


