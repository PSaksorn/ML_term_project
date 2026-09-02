from pathlib import Path
import json
import multiprocessing as mp
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# STEP 4 (REVISED) — FINAL CONTROLLED CNN TRAINING
# ============================================================
#
# Why this version exists
# -----------------------
# The pilot CNN underfit the 153-identity task:
# - Model A best validation accuracy was very low.
# - Model B was better, but both models had low training capacity.
#
# This revised version changes the PILOT model capacity while
# preserving the A-vs-B experimental comparison:
#
# Model A:
#   Frontal-only training
#
# Model B:
#   Pose-diverse training: 0°, ±30°, ±60°
#
# Controlled variables:
#   - Same architecture
#   - Same number of original training images
#   - Same optimizer / LR / batch size / epochs
#   - Same validation set
#   - Same preprocessing / augmentation
#   - Same initialization seed
#
# Training budget:
#   FIXED 40 epochs for BOTH models.
#   No early stopping in the final controlled run.
#
# Model selection:
#   Best checkpoint = highest validation Macro-F1.
#   Tie-breaker       = lower validation loss.
#
# GPU:
#   CUDA is REQUIRED by default.
#
# Windows:
#   All executable work is inside main(), so DataLoader workers
#   do not repeatedly execute the training script.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4

NUM_EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EMBEDDING_DIM = 128

REQUIRE_CUDA = True
USE_AUGMENTATION = True

# Photometric augmentation only.
# No geometric augmentation because yaw is the research variable.
BRIGHTNESS_JITTER = 0.10
CONTRAST_JITTER = 0.10

SPLIT_DIR = Path("outputs") / "splits"

TRAIN_A_CSV = SPLIT_DIR / "train_model_A.csv"
TRAIN_B_CSV = SPLIT_DIR / "train_model_B.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"

RUN_DIR = Path("outputs") / "final_training"
MODEL_DIR = RUN_DIR / "models"
HISTORY_DIR = RUN_DIR / "history"
FIGURE_DIR = RUN_DIR / "figures"
CONFIG_DIR = RUN_DIR / "config"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 3. DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(self, dataframe, person_to_label, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.person_to_label = person_to_label
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.person_to_label[row["person"]]

        return image, label


# ============================================================
# 4. REVISED CNN
# ============================================================
#
# Pilot:
#   Conv32 -> Conv64 -> Conv128 -> GAP(1x1) -> Embedding
#
# Revised:
#   Two convolutions per block
#   32 -> 64 -> 128 -> 256 channels
#   AdaptiveAvgPool(4x4) retains more spatial information
#   Dense 4096 -> 256 -> 128 embedding
#
# forward_features() is intentionally exposed because Step 5
# will use the 128-D embeddings for cosine-similarity analysis.
# ============================================================

class ImprovedFaceCNN(nn.Module):

    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Keep 4x4 spatial organization instead of collapsing to 1x1.
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.spatial_pool(x)
        embedding = self.embedding_head(x)
        return embedding

    def forward(self, x):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        return logits


# ============================================================
# 5. AMP HELPERS
# ============================================================

def create_grad_scaler(use_amp):
    # Modern PyTorch API.
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        # Fallback for older PyTorch.
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp):
    # Modern PyTorch API.
    try:
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    except AttributeError:
        # Fallback for older PyTorch.
        return torch.cuda.amp.autocast(enabled=use_amp)


# ============================================================
# 6. DATALOADER
# ============================================================

def make_loader(dataframe, person_to_label, transform, shuffle, device):

    dataset = FaceDataset(
        dataframe=dataframe,
        person_to_label=person_to_label,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    return loader


# ============================================================
# 7. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# 8. CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(
    checkpoint_path,
    experiment_name,
    epoch,
    model,
    optimizer,
    val_loss,
    val_accuracy,
    val_macro_f1,
    num_classes,
    person_to_label,
):

    torch.save({
        "experiment": experiment_name,
        "architecture": "ImprovedFaceCNN_v2",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "num_classes": num_classes,
        "embedding_dim": EMBEDDING_DIM,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "person_to_label": person_to_label,
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, device):

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device
        )


# ============================================================
# 9. TRAIN ONE EXPERIMENT
# ============================================================

def train_experiment(
    experiment_name,
    train_df,
    val_df,
    person_to_label,
    num_classes,
    train_transform,
    eval_transform,
    device,
    use_amp,
):

    print("\n" + "#" * 80)
    print(f"TRAINING {experiment_name}")
    print("#" * 80)

    # Ensures A and B start from the same random initialization.
    set_seed(SEED)

    train_loader = make_loader(
        dataframe=train_df,
        person_to_label=person_to_label,
        transform=train_transform,
        shuffle=True,
        device=device,
    )

    val_loader = make_loader(
        dataframe=val_df,
        person_to_label=person_to_label,
        transform=eval_transform,
        shuffle=False,
        device=device,
    )

    model = ImprovedFaceCNN(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = create_grad_scaler(use_amp)

    history = []

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    checkpoint_path = MODEL_DIR / f"{experiment_name}_best.pt"

    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.perf_counter()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        train_labels = []
        train_predictions = []

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)

            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            train_labels.extend(
                labels.detach().cpu().numpy()
            )

            train_predictions.extend(
                predictions.detach().cpu().numpy()
            )

        train_loss = train_loss_sum / train_samples

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions
        )

        train_macro_f1 = f1_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy, val_macro_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        if device.type == "cuda":
            peak_memory_gb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 3)
            )
            torch.cuda.reset_peak_memory_stats(device)
        else:
            peak_memory_gb = np.nan

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_memory_gb,
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy*100:6.2f}% | "
            f"Train F1 {train_macro_f1:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy*100:6.2f}% | "
            f"Val F1 {val_macro_f1:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # Primary criterion: highest Validation Macro-F1
        # Tie-breaker: lower Validation Loss
        # ----------------------------------------------------

        improved = (
            val_macro_f1 > best_val_macro_f1 + 1e-12
            or (
                abs(val_macro_f1 - best_val_macro_f1) <= 1e-12
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                experiment_name=experiment_name,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_macro_f1=val_macro_f1,
                num_classes=num_classes,
                person_to_label=person_to_label,
            )

            print(
                f"  -> Saved new best checkpoint "
                f"(Val Macro-F1 = {val_macro_f1:.4f})"
            )

    total_seconds = time.perf_counter() - total_start

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = HISTORY_DIR / f"{experiment_name}_history.csv"

    history_df.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # RELOAD BEST CHECKPOINT
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        checkpoint_path,
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_val_loss_eval, best_val_accuracy_eval, best_val_macro_f1_eval = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    print("\n" + "-" * 80)
    print(f"{experiment_name} BEST MODEL")
    print("-" * 80)

    print("Best epoch     :", best_epoch)
    print(f"Best Val Loss  : {best_val_loss_eval:.4f}")
    print(f"Best Val Acc   : {best_val_accuracy_eval*100:.2f}%")
    print(f"Best Val F1    : {best_val_macro_f1_eval:.4f}")
    print(f"Training time  : {total_seconds/60:.2f} minutes")

    result = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss_eval,
        "best_val_accuracy": best_val_accuracy_eval,
        "best_val_macro_f1": best_val_macro_f1_eval,
        "training_seconds": total_seconds,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    del train_loader
    del val_loader
    del model
    del optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 10. LEARNING CURVES
# ============================================================

def plot_learning_curves(history_a_path, history_b_path):

    history_a = pd.read_csv(history_a_path)
    history_b = pd.read_csv(history_b_path)

    # Loss
    plt.figure(figsize=(9, 6))

    plt.plot(
        history_a["epoch"],
        history_a["train_loss"],
        label="Model A Train"
    )
    plt.plot(
        history_a["epoch"],
        history_a["val_loss"],
        label="Model A Validation"
    )
    plt.plot(
        history_b["epoch"],
        history_b["train_loss"],
        label="Model B Train"
    )
    plt.plot(
        history_b["epoch"],
        history_b["val_loss"],
        label="Model B Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    loss_path = FIGURE_DIR / "learning_curves_loss.png"
    plt.savefig(loss_path, dpi=170, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure(figsize=(9, 6))

    plt.plot(
        history_a["epoch"],
        history_a["train_accuracy"],
        label="Model A Train"
    )
    plt.plot(
        history_a["epoch"],
        history_a["val_accuracy"],
        label="Model A Validation"
    )
    plt.plot(
        history_b["epoch"],
        history_b["train_accuracy"],
        label="Model B Train"
    )
    plt.plot(
        history_b["epoch"],
        history_b["val_accuracy"],
        label="Model B Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    accuracy_path = FIGURE_DIR / "learning_curves_accuracy.png"
    plt.savefig(accuracy_path, dpi=170, bbox_inches="tight")
    plt.close()

    # Macro-F1
    plt.figure(figsize=(9, 6))

    plt.plot(
        history_a["epoch"],
        history_a["train_macro_f1"],
        label="Model A Train"
    )
    plt.plot(
        history_a["epoch"],
        history_a["val_macro_f1"],
        label="Model A Validation"
    )
    plt.plot(
        history_b["epoch"],
        history_b["train_macro_f1"],
        label="Model B Train"
    )
    plt.plot(
        history_b["epoch"],
        history_b["val_macro_f1"],
        label="Model B Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and Validation Macro-F1")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    f1_path = FIGURE_DIR / "learning_curves_macro_f1.png"
    plt.savefig(f1_path, dpi=170, bbox_inches="tight")
    plt.close()

    print("\nLearning curves saved:")
    print(loss_path.resolve())
    print(accuracy_path.resolve())
    print(f1_path.resolve())


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    for directory in [
        MODEL_DIR,
        HISTORY_DIR,
        FIGURE_DIR,
        CONFIG_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "\nCUDA GPU is required for this training run, "
            "but PyTorch cannot access CUDA.\n\n"
            f"PyTorch version : {torch.__version__}\n"
            f"PyTorch CUDA    : {torch.version.cuda}\n\n"
            "Install a CUDA-enabled PyTorch build before running "
            "this script."
        )

    use_amp = device.type == "cuda"

    print("=" * 80)
    print("STEP 4 REVISED — FINAL CONTROLLED CNN TRAINING")
    print("=" * 80)

    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("PyTorch CUDA    :", torch.version.cuda)
    print("Device          :", device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print("GPU             :", gpu_name)
        print(f"GPU memory      : {gpu_memory:.2f} GB")
        print("Mixed precision :", use_amp)

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # LOAD CSV FILES
    # --------------------------------------------------------

    for csv_path in [
        TRAIN_A_CSV,
        TRAIN_B_CSV,
        VAL_CSV,
    ]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{csv_path.resolve()}\n\n"
                "Run Step 2 first."
            )

    train_a_df = pd.read_csv(
        TRAIN_A_CSV
    )

    train_b_df = pd.read_csv(
        TRAIN_B_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    print("\n" + "=" * 80)
    print("DATASET SIZES")
    print("=" * 80)

    print(f"Model A train : {len(train_a_df):,}")
    print(f"Model B train : {len(train_b_df):,}")
    print(f"Validation    : {len(val_df):,}")

    # --------------------------------------------------------
    # LABEL MAPPING
    # --------------------------------------------------------

    persons_a = set(
        train_a_df["person"]
    )

    persons_b = set(
        train_b_df["person"]
    )

    persons_val = set(
        val_df["person"]
    )

    if not (
        persons_a
        == persons_b
        == persons_val
    ):
        raise ValueError(
            "Identity sets differ between "
            "Model A, Model B, and validation."
        )

    persons = sorted(persons_a)

    person_to_label = {
        person: idx
        for idx, person
        in enumerate(persons)
    }

    label_to_person = {
        idx: person
        for person, idx
        in person_to_label.items()
    }

    num_classes = len(persons)

    print("Number of classes:", num_classes)

    label_map_path = (
        CONFIG_DIR
        / "label_mapping.csv"
    )

    pd.DataFrame({
        "label": list(
            range(num_classes)
        ),
        "person": [
            label_to_person[i]
            for i
            in range(num_classes)
        ],
    }).to_csv(
        label_map_path,
        index=False
    )

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    if USE_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.Grayscale(
                num_output_channels=1
            ),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ColorJitter(
                brightness=BRIGHTNESS_JITTER,
                contrast=CONTRAST_JITTER,
            ),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(
                num_output_channels=1
            ),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
        ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=1
        ),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
    ])

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    example_model = ImprovedFaceCNN(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM
    )

    total_params = sum(
        p.numel()
        for p
        in example_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p
        in example_model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 80)
    print("REVISED MODEL ARCHITECTURE")
    print("=" * 80)

    print(example_model)

    print(
        f"\nTotal parameters    : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    del example_model

    # --------------------------------------------------------
    # SAVE CONFIG
    # --------------------------------------------------------

    training_config = {
        "run_type": "final_controlled_training",
        "architecture": "ImprovedFaceCNN_v2",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": NUM_EPOCHS,
        "early_stopping": False,
        "best_model_criterion": "highest validation Macro-F1; tie-breaker lower validation loss",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": num_classes,
        "augmentation_enabled": USE_AUGMENTATION,
        "brightness_jitter": BRIGHTNESS_JITTER,
        "contrast_jitter": CONTRAST_JITTER,
        "horizontal_flip": False,
        "rotation_augmentation": False,
        "translation_augmentation": False,
        "perspective_augmentation": False,
        "synthetic_yaw": False,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_enabled": use_amp,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    if device.type == "cuda":
        training_config[
            "gpu_name"
        ] = torch.cuda.get_device_name(0)

    config_path = (
        CONFIG_DIR
        / "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            training_config,
            f,
            indent=2
        )

    print("\nTraining configuration:")
    print(
        json.dumps(
            training_config,
            indent=2
        )
    )

    # --------------------------------------------------------
    # MODEL A
    # --------------------------------------------------------

    result_a = train_experiment(
        experiment_name="model_A",
        train_df=train_a_df,
        val_df=val_df,
        person_to_label=person_to_label,
        num_classes=num_classes,
        train_transform=train_transform,
        eval_transform=eval_transform,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # MODEL B
    # --------------------------------------------------------

    result_b = train_experiment(
        experiment_name="model_B",
        train_df=train_b_df,
        val_df=val_df,
        person_to_label=person_to_label,
        num_classes=num_classes,
        train_transform=train_transform,
        eval_transform=eval_transform,
        device=device,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame([
        result_a,
        result_b,
    ])

    summary_path = (
        HISTORY_DIR
        / "training_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # LEARNING CURVES
    # --------------------------------------------------------

    plot_learning_curves(
        result_a["history"],
        result_b["history"],
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 80)

    print(
        summary_df[
            [
                "experiment",
                "best_epoch",
                "best_val_loss",
                "best_val_accuracy",
                "best_val_macro_f1",
                "training_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nFiles:")
    print(
        "Config  :",
        config_path.resolve()
    )

    print(
        "Summary :",
        summary_path.resolve()
    )

    print(
        "Models  :",
        MODEL_DIR.resolve()
    )

    print(
        "\nStep 4 revised training "
        "completed successfully."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
