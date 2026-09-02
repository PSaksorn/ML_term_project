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
# STEP 4 v3 — STABLE GPU TRAINING
# ============================================================
#
# Fixes from v2:
# 1) Disable AMP/FP16 by default because v2 produced NaN losses.
# 2) Lower learning rate from 1e-3 to 3e-4.
# 3) Add gradient clipping.
# 4) Fail immediately if loss/logits/gradients become NaN/Inf.
# 5) Keep GPU execution (CUDA) in float32.
# 6) Keep the SAME architecture and SAME A/B controlled design.
#
# IMPORTANT:
# The v2 run with NaN loss is invalid and must not be used for
# final evaluation.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4

NUM_EPOCHS = 40

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 5.0

EMBEDDING_DIM = 128

REQUIRE_CUDA = True

# Stability first. Still uses CUDA GPU, but computations are FP32.
USE_AMP = False

USE_AUGMENTATION = True
BRIGHTNESS_JITTER = 0.10
CONTRAST_JITTER = 0.10

SPLIT_DIR = Path("outputs") / "splits"
TRAIN_A_CSV = SPLIT_DIR / "train_model_A.csv"
TRAIN_B_CSV = SPLIT_DIR / "train_model_B.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"

RUN_DIR = Path("outputs") / "final_training_v3"
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

        if not torch.isfinite(image).all():
            raise RuntimeError(f"Non-finite input image tensor: {image_path}")

        label = self.person_to_label[row["person"]]

        return image, label


# ============================================================
# 4. CNN
# ============================================================

class ImprovedFaceCNN(nn.Module):

    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

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
        return self.embedding_head(x)

    def forward(self, x):
        embedding = self.forward_features(x)
        return self.classifier(embedding)


# ============================================================
# 5. DATALOADER
# ============================================================

def make_loader(dataframe, person_to_label, transform, shuffle, device):
    dataset = FaceDataset(dataframe, person_to_label, transform)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
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


# ============================================================
# 6. NUMERICAL SAFETY
# ============================================================

def assert_finite_tensor(tensor, name, experiment_name, epoch, batch_idx):
    if not torch.isfinite(tensor).all():
        finite_ratio = torch.isfinite(tensor).float().mean().item()

        raise FloatingPointError(
            f"\nNon-finite values detected.\n"
            f"Experiment   : {experiment_name}\n"
            f"Epoch        : {epoch}\n"
            f"Batch        : {batch_idx}\n"
            f"Tensor       : {name}\n"
            f"Finite ratio : {finite_ratio:.6f}\n\n"
            "Training stopped to prevent invalid checkpoints."
        )


def gradients_are_finite(model):
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                return False
    return True


# ============================================================
# 7. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"Validation logits became NaN/Inf at batch {batch_idx}."
            )

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Validation loss became NaN/Inf at batch {batch_idx}."
            )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# 8. TRAIN ONE MODEL
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
):

    print("\n" + "#" * 80)
    print(f"TRAINING {experiment_name}")
    print("#" * 80)

    set_seed(SEED)

    train_loader = make_loader(
        train_df,
        person_to_label,
        train_transform,
        True,
        device,
    )

    val_loader = make_loader(
        val_df,
        person_to_label,
        eval_transform,
        False,
        device,
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

    history = []

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    checkpoint_path = MODEL_DIR / f"{experiment_name}_best.pt"

    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.perf_counter()

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        train_labels = []
        train_predictions = []

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP intentionally disabled in v3 for numerical stability.
            logits = model(images)

            assert_finite_tensor(
                logits,
                "training logits",
                experiment_name,
                epoch,
                batch_idx
            )

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"\nTraining loss became NaN/Inf.\n"
                    f"Experiment : {experiment_name}\n"
                    f"Epoch      : {epoch}\n"
                    f"Batch      : {batch_idx}\n"
                    f"Loss       : {loss.item()}\n"
                )

            loss.backward()

            if not gradients_are_finite(model):
                raise FloatingPointError(
                    f"\nGradient became NaN/Inf.\n"
                    f"Experiment : {experiment_name}\n"
                    f"Epoch      : {epoch}\n"
                    f"Batch      : {batch_idx}\n"
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"\nGradient norm became NaN/Inf.\n"
                    f"Experiment : {experiment_name}\n"
                    f"Epoch      : {epoch}\n"
                    f"Batch      : {batch_idx}\n"
                )

            optimizer.step()

            batch_size = images.size(0)

            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

            predictions = logits.argmax(dim=1)

            train_labels.extend(labels.detach().cpu().numpy())
            train_predictions.extend(predictions.detach().cpu().numpy())

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

        val_loss, val_accuracy, val_macro_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_memory_gb,
        }

        history.append(row)

        # Save history every epoch so progress survives interruption.
        pd.DataFrame(history).to_csv(
            HISTORY_DIR / f"{experiment_name}_history.csv",
            index=False
        )

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

            torch.save({
                "experiment": experiment_name,
                "architecture": "ImprovedFaceCNN_v3",
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

            print(
                f"  -> Saved best checkpoint "
                f"(Val F1 {val_macro_f1:.4f}, "
                f"Val Loss {val_loss:.4f})"
            )

    total_seconds = time.perf_counter() - total_start

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_loss, final_val_accuracy, final_val_macro_f1 = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print("\n" + "-" * 80)
    print(f"{experiment_name} BEST MODEL")
    print("-" * 80)
    print("Best epoch     :", best_epoch)
    print(f"Best Val Loss  : {final_val_loss:.4f}")
    print(f"Best Val Acc   : {final_val_accuracy*100:.2f}%")
    print(f"Best Val F1    : {final_val_macro_f1:.4f}")
    print(f"Training time  : {total_seconds/60:.2f} minutes")

    del model, optimizer, train_loader, val_loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_loss": final_val_loss,
        "best_val_accuracy": final_val_accuracy,
        "best_val_macro_f1": final_val_macro_f1,
        "training_seconds": total_seconds,
        "checkpoint": str(checkpoint_path),
        "history": str(HISTORY_DIR / f"{experiment_name}_history.csv"),
    }


# ============================================================
# 9. PLOTS
# ============================================================

def plot_learning_curves(history_a_path, history_b_path):

    a = pd.read_csv(history_a_path)
    b = pd.read_csv(history_b_path)

    # Fail clearly instead of creating misleading blank graphs.
    required_numeric = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
    ]

    for name, df in [("Model A", a), ("Model B", b)]:
        if df[required_numeric].isna().any().any():
            raise RuntimeError(
                f"{name} history contains NaN values. "
                "Learning curves were not created."
            )

    plt.figure(figsize=(9, 6))
    plt.plot(a["epoch"], a["train_loss"], label="Model A Train")
    plt.plot(a["epoch"], a["val_loss"], label="Model A Validation")
    plt.plot(b["epoch"], b["train_loss"], label="Model B Train")
    plt.plot(b["epoch"], b["val_loss"], label="Model B Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    loss_path = FIGURE_DIR / "learning_curves_loss.png"
    plt.savefig(loss_path, dpi=170, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(a["epoch"], a["train_accuracy"], label="Model A Train")
    plt.plot(a["epoch"], a["val_accuracy"], label="Model A Validation")
    plt.plot(b["epoch"], b["train_accuracy"], label="Model B Train")
    plt.plot(b["epoch"], b["val_accuracy"], label="Model B Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    acc_path = FIGURE_DIR / "learning_curves_accuracy.png"
    plt.savefig(acc_path, dpi=170, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(a["epoch"], a["train_macro_f1"], label="Model A Train")
    plt.plot(a["epoch"], a["val_macro_f1"], label="Model A Validation")
    plt.plot(b["epoch"], b["train_macro_f1"], label="Model B Train")
    plt.plot(b["epoch"], b["val_macro_f1"], label="Model B Validation")
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
    print(acc_path.resolve())
    print(f1_path.resolve())


# ============================================================
# 10. MAIN
# ============================================================

def main():

    for directory in [MODEL_DIR, HISTORY_DIR, FIGURE_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "\nCUDA GPU is required, but PyTorch cannot access CUDA.\n"
            f"PyTorch version : {torch.__version__}\n"
            f"PyTorch CUDA    : {torch.version.cuda}\n"
        )

    print("=" * 80)
    print("STEP 4 v3 — STABLE GPU TRAINING")
    print("=" * 80)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("PyTorch CUDA    :", torch.version.cuda)
    print("Device          :", device)
    print("AMP enabled     :", USE_AMP)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print("GPU             :", gpu_name)
        print(f"GPU memory      : {gpu_memory:.2f} GB")
        print("Compute dtype   : float32")

        torch.backends.cudnn.benchmark = True

    for path in [TRAIN_A_CSV, TRAIN_B_CSV, VAL_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split CSV: {path.resolve()}"
            )

    train_a_df = pd.read_csv(TRAIN_A_CSV)
    train_b_df = pd.read_csv(TRAIN_B_CSV)
    val_df = pd.read_csv(VAL_CSV)

    persons_a = set(train_a_df["person"])
    persons_b = set(train_b_df["person"])
    persons_val = set(val_df["person"])

    if not (persons_a == persons_b == persons_val):
        raise ValueError("Identity sets differ across splits.")

    persons = sorted(persons_a)

    person_to_label = {
        person: index
        for index, person in enumerate(persons)
    }

    num_classes = len(persons)

    print("\nDataset:")
    print(f"Model A train : {len(train_a_df):,}")
    print(f"Model B train : {len(train_b_df):,}")
    print(f"Validation    : {len(val_df):,}")
    print(f"Classes       : {num_classes}")

    if USE_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(
                brightness=BRIGHTNESS_JITTER,
                contrast=CONTRAST_JITTER
            ),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    example_model = ImprovedFaceCNN(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM
    )

    total_params = sum(
        p.numel() for p in example_model.parameters()
    )

    print(f"Total parameters: {total_params:,}")

    del example_model

    config = {
        "version": "v3_stable",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": num_classes,
        "use_amp": USE_AMP,
        "compute_dtype": "float32",
        "augmentation": {
            "brightness": BRIGHTNESS_JITTER,
            "contrast": CONTRAST_JITTER,
            "horizontal_flip": False,
            "rotation": False,
            "translation": False,
            "synthetic_yaw": False,
        },
        "best_model_criterion": (
            "highest validation Macro-F1; "
            "tie-breaker lower validation loss"
        ),
        "total_parameters": total_params,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }

    if device.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(0)

    config_path = CONFIG_DIR / "training_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nConfiguration:")
    print(json.dumps(config, indent=2))

    result_a = train_experiment(
        "model_A",
        train_a_df,
        val_df,
        person_to_label,
        num_classes,
        train_transform,
        eval_transform,
        device,
    )

    result_b = train_experiment(
        "model_B",
        train_b_df,
        val_df,
        person_to_label,
        num_classes,
        train_transform,
        eval_transform,
        device,
    )

    summary = pd.DataFrame([result_a, result_b])

    summary_path = HISTORY_DIR / "training_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_learning_curves(
        result_a["history"],
        result_b["history"],
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 80)

    print(
        summary[
            [
                "experiment",
                "best_epoch",
                "best_val_loss",
                "best_val_accuracy",
                "best_val_macro_f1",
                "training_seconds",
            ]
        ].to_string(index=False)
    )

    print("\nStep 4 v3 completed successfully.")
    print("Output:", RUN_DIR.resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()
