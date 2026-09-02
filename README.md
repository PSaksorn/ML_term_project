# Effect of Pose-Diverse Training on Cross-Pose Face Recognition and Learned CNN Representations

## Overview
This project studies whether pose-diverse CNN training improves cross-pose face recognition and reduces pose-induced representation shift compared with frontal-only training.

- **Model A (Baseline):** trained using frontal images only (`0°`)
- **Model B (Pose-diverse):** trained using `0°`, `±30°`, and `±60°`
- Both models use the same CNN architecture, training-image budget, preprocessing, optimizer, validation set, test set, and training protocol.

## Dataset
A filtered subset of the **Multi-PIE** face dataset is used.

- 153 identities
- 13 yaw conditions: `0°`, `±15°`, `±30°`, `±45°`, `±60°`, `±75°`, `±90°`
- 20 illumination conditions per pose
- 39,780 images
- Image size: `128 × 128`
- Eyeglass samples excluded
- Task: closed-set 153-class identification

The dataset itself is not included and must be obtained separately according to its access and usage conditions.

## Fixed Data Split
The split is performed by illumination condition and remains fixed across all experiments.

- **Train:** 1, 4, 6, 7, 8, 10, 11, 13, 15, 16, 17, 20
- **Validation:** 3, 5, 12, 19
- **Test:** 2, 9, 14, 18
- **Split seed:** 42

Repeated training seeds (`42`, `123`, `2026`) change only training randomness such as initialization, shuffling, and augmentation.

## Training Setup
Each model uses 12 training images per identity (`1,836` total).

**Model A**
- `0°`: 12 images / identity

**Model B**
- `0°`: 4 images / identity
- `-30°`, `+30°`, `-60°`, `+60°`: 2 images each / identity

Common training settings:

- Input: `128 × 128 × 1`
- Loss: Cross-Entropy
- Optimizer: Adam
- Learning rate: `3e-4`
- Weight decay: `1e-5`
- Batch size: 32
- Epochs: 40
- Gradient clipping: max norm = 5.0
- Embedding dimension: 128
- Best checkpoint: highest validation Macro-F1, tie broken by lower validation loss

Training augmentation uses mild brightness and contrast changes only. No geometric augmentation is used because head pose is the experimental variable.

## Running the Project
Run scripts in numerical order:

```powershell
python .\01_dataset_audit.py
python .\02_create_split.py
python .\03_preprocessing_check.py
python .\04_train_models_v3.py
python .\05_final_evaluation.py
python .\06_error_analysis.py
python .\07_repeated_seed_runs.py
python .\08_aggregate_repeated_seeds.py
python .\09_appendix_analysis.py
python .\10_make_appendix_pose_figure.py
```

### Script Summary
| Script | Purpose |
|---|---|
| `01_dataset_audit.py` | Audit dataset completeness and image properties |
| `02_create_split.py` | Create the fixed train/validation/test split |
| `03_preprocessing_check.py` | Verify preprocessing and pose progression |
| `04_train_models_v3.py` | Train the valid seed-42 Model A and Model B |
| `05_final_evaluation.py` | Evaluate accuracy, Macro-F1, yaw performance, and embeddings |
| `06_error_analysis.py` | Analyze prediction outcomes and representative errors |
| `07_repeated_seed_runs.py` | Repeat final training for multiple seeds |
| `08_aggregate_repeated_seeds.py` | Aggregate mean ± SD results |
| `09_appendix_analysis.py` | Generate appendix-level error and confusion analyses |
| `10_make_appendix_pose_figure.py` | Generate enlarged pose visualization for the appendix |

## Final Results
Across seeds `42`, `123`, and `2026`:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Model A — Frontal-only | 19.17 ± 1.33% | 23.40 ± 0.88% |
| Model B — Pose-diverse | 86.45 ± 0.51% | 87.03 ± 0.25% |

Model B pose-category accuracy:

- Seen poses: `99.90 ± 0.03%`
- Unseen interpolation poses: `94.99 ± 0.67%`
- Extreme extrapolation poses: `61.08 ± 1.10%`

## Output Folders
Important results are stored under `outputs/`, including:

- `splits/`
- `final_training_v3/`
- `final_analysis/`
- `repeated_seed_summary/`
- `appendix_analysis/`

For exact reproduction, reuse the saved split files rather than generating a new split.

## Software and Hardware
Fill these values using the actual experiment environment:

```text
Python  : 3.10.11
PyTorch : 2.7.1+cu118
CUDA    : 11.8
GPU     : NVIDIA GeForce GTX 1650
OS      : Window 11
Additional Python package versions are listed in requirements.txt
```

Check with:

```powershell
python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## AI and External Assistance Statement
Generative AI tools were used as a supporting aid for code debugging, implementation suggestions, and language refinement. The research question, experimental design, model comparisons, evaluation strategy, execution of experiments, and interpretation of results were developed and performed by the author. All AI-assisted suggestions were reviewed and adapted before being used in the project.
