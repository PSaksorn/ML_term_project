from pathlib import Path
import json
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
# STEP 4 — TRAIN MODEL A AND MODEL B
# GPU-FIRST PYTORCH TRAINING PIPELINE
# ============================================================
#
# Model A:
#   Frontal-only training
#
# Model B:
#   Pose-diverse training: 0°, ±30°, ±60°
#
# Controlled variables:
#   - Same CNN architecture
#   - Same training-image budget
#   - Same optimizer / LR / batch size / epochs
#   - Same validation set
#   - Same preprocessing
#   - Same augmentation policy
#
# Model selection:
#   Lowest validation loss
#
# Augmentation:
#   Photometric only (brightness + contrast).
#
# Why no rotation / translation / flip?
#   Head pose is the main experimental variable.
#   Geometric augmentation can create unwanted artifacts or
#   interfere with the left/right yaw structure.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 128

USE_AUGMENTATION = True

# Photometric augmentation only.
BRIGHTNESS_JITTER = 0.10
CONTRAST_JITTER = 0.10

SPLIT_DIR = Path("outputs") / "splits"

TRAIN_A_CSV = SPLIT_DIR / "train_model_A.csv"
TRAIN_B_CSV = SPLIT_DIR / "train_model_B.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"

MODEL_DIR = Path("outputs") / "models"
HISTORY_DIR = Path("outputs") / "training_history"
FIGURE_DIR = Path("outputs") / "figures"
CONFIG_DIR = Path("outputs") / "config"

for directory in [MODEL_DIR, HISTORY_DIR, FIGURE_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


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


set_seed(SEED)


# ============================================================
# 3. DEVICE / GPU CHECK
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = device.type == "cuda"

print("=" * 80)
print("STEP 4 — CNN TRAINING")
print("=" * 80)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("PyTorch CUDA    :", torch.version.cuda)
print("Device          :", device)

if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("GPU             :", gpu_name)
    print(f"GPU memory      : {gpu_memory:.2f} GB")
    print("Mixed precision :", USE_AMP)

    # Fixed input size. Benchmark can improve speed.
    torch.backends.cudnn.benchmark = True
else:
    print("GPU             : Not available")
    print("Mixed precision : False")


# ============================================================
# 4. LOAD CSV FILES
# ============================================================

for csv_path in [TRAIN_A_CSV, TRAIN_B_CSV, VAL_CSV]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{csv_path.resolve()}\n\n"
            "Run Step 2 first."
        )

train_a_df = pd.read_csv(TRAIN_A_CSV)
train_b_df = pd.read_csv(TRAIN_B_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\n" + "=" * 80)
print("DATASET SIZES")
print("=" * 80)

print(f"Model A train : {len(train_a_df):,}")
print(f"Model B train : {len(train_b_df):,}")
print(f"Validation    : {len(val_df):,}")


# ============================================================
# 5. LABEL MAPPING
# ============================================================

persons_a = set(train_a_df["person"])
persons_b = set(train_b_df["person"])
persons_val = set(val_df["person"])

if not (persons_a == persons_b == persons_val):
    raise ValueError(
        "Identity sets differ between Model A, Model B, and validation."
    )

persons = sorted(persons_a)

person_to_label = {
    person: idx
    for idx, person in enumerate(persons)
}

label_to_person = {
    idx: person
    for person, idx in person_to_label.items()
}

NUM_CLASSES = len(persons)

print("Number of classes:", NUM_CLASSES)

label_map_path = CONFIG_DIR / "label_mapping.csv"

pd.DataFrame({
    "label": list(range(NUM_CLASSES)),
    "person": [label_to_person[i] for i in range(NUM_CLASSES)],
}).to_csv(label_map_path, index=False)


# ============================================================
# 6. TRANSFORMS
# ============================================================

# IMPORTANT:
# All images are converted to grayscale and resized to 128x128.
# ToTensor automatically scales uint8 pixels from [0,255] -> [0,1].

if USE_AUGMENTATION:
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(
            brightness=BRIGHTNESS_JITTER,
            contrast=CONTRAST_JITTER,
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


# ============================================================
# 7. DATASET CLASS
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

            if self.transform is not None:
                image = self.transform(image)

        label = self.person_to_label[row["person"]]

        return image, label


# ============================================================
# 8. DATALOADER FACTORY
# ============================================================

def make_loader(dataframe, transform, shuffle):

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
        generator=generator,
    )

    return loader


# ============================================================
# 9. CNN MODEL
# ============================================================

class SmallFaceCNN(nn.Module):

    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward_features(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.embedding(x)

        return x

    def forward(self, x):
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)

        return logits


# ============================================================
# 10. MODEL INFO
# ============================================================

example_model = SmallFaceCNN(
    num_classes=NUM_CLASSES,
    embedding_dim=EMBEDDING_DIM
)

total_params = sum(
    p.numel()
    for p in example_model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in example_model.parameters()
    if p.requires_grad
)

print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

print(example_model)

print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

del example_model


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

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
# 12. TRAIN ONE MODEL
# ============================================================

def train_experiment(
    experiment_name,
    train_df,
):

    print("\n" + "#" * 80)
    print(f"TRAINING {experiment_name}")
    print("#" * 80)

    # Reset seed so A and B start from the same initial
    # parameter initialization and random state.
    set_seed(SEED)

    train_loader = make_loader(
        dataframe=train_df,
        transform=train_transform,
        shuffle=True,
    )

    val_loader = make_loader(
        dataframe=val_df,
        transform=eval_transform,
        shuffle=False,
    )

    model = SmallFaceCNN(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        MODEL_DIR
        / f"{experiment_name}_best.pt"
    )

    total_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):

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

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=USE_AMP
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)

            train_loss_sum += (
                loss.item() * batch_size
            )

            train_samples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            train_labels.extend(
                labels.detach().cpu().numpy()
            )

            train_predictions.extend(
                predictions.detach().cpu().numpy()
            )

        train_loss = (
            train_loss_sum / train_samples
        )

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
            criterion=criterion
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy*100:6.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy*100:6.2f}% | "
            f"Val F1 {val_macro_f1:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT / EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "experiment": experiment_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "num_classes": NUM_CLASSES,
                "embedding_dim": EMBEDDING_DIM,
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "person_to_label": person_to_label,
            }, checkpoint_path)

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered. "
                f"Best epoch = {best_epoch}"
            )
            break

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = (
        HISTORY_DIR
        / f"{experiment_name}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # RELOAD BEST MODEL
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_val_loss, best_val_accuracy, best_val_macro_f1 = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion
    )

    print("\n" + "-" * 80)

    print(
        f"{experiment_name} BEST MODEL"
    )

    print("-" * 80)

    print("Best epoch     :", best_epoch)
    print(f"Best Val Loss  : {best_val_loss:.4f}")
    print(
        f"Best Val Acc   : "
        f"{best_val_accuracy*100:.2f}%"
    )
    print(
        f"Best Val F1    : "
        f"{best_val_macro_f1:.4f}"
    )
    print(
        f"Training time  : "
        f"{total_seconds/60:.2f} minutes"
    )

    # --------------------------------------------------------
    # RELEASE MEMORY
    # --------------------------------------------------------

    del train_loader
    del val_loader
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "training_seconds": total_seconds,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


# ============================================================
# 13. PLOT LEARNING CURVES
# ============================================================

def plot_learning_curves(
    history_a_path,
    history_b_path,
):

    history_a = pd.read_csv(
        history_a_path
    )

    history_b = pd.read_csv(
        history_b_path
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

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
    plt.title(
        "Training and Validation Loss"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    loss_path = (
        FIGURE_DIR
        / "learning_curves_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=170,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

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
    plt.title(
        "Training and Validation Accuracy"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    accuracy_path = (
        FIGURE_DIR
        / "learning_curves_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=170,
        bbox_inches="tight"
    )

    plt.close()

    print("\nLearning curve figures saved:")
    print(loss_path.resolve())
    print(accuracy_path.resolve())


# ============================================================
# 14. SAVE CONFIG
# ============================================================

training_config = {
    "seed": SEED,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "embedding_dim": EMBEDDING_DIM,
    "num_classes": NUM_CLASSES,
    "augmentation_enabled": USE_AUGMENTATION,
    "brightness_jitter": BRIGHTNESS_JITTER,
    "contrast_jitter": CONTRAST_JITTER,
    "horizontal_flip": False,
    "rotation_augmentation": False,
    "translation_augmentation": False,
    "synthetic_yaw": False,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss",
    "model_selection": "lowest validation loss",
    "device": str(device),
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "amp_enabled": USE_AMP,
}

if device.type == "cuda":
    training_config["gpu_name"] = torch.cuda.get_device_name(0)

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


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("\nTraining configuration:")
    print(json.dumps(
        training_config,
        indent=2
    ))

    # --------------------------------------------------------
    # MODEL A
    # --------------------------------------------------------

    result_a = train_experiment(
        experiment_name="model_A",
        train_df=train_a_df
    )

    # --------------------------------------------------------
    # MODEL B
    # --------------------------------------------------------

    result_b = train_experiment(
        experiment_name="model_B",
        train_df=train_b_df
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame([
        result_a,
        result_b
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
        result_b["history"]
    )

    # --------------------------------------------------------
    # PRINT FINAL COMPARISON
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
    print("Config  :", config_path.resolve())
    print("Summary :", summary_path.resolve())
    print("Models  :", MODEL_DIR.resolve())

    print("\nStep 4 completed successfully.")


if __name__ == "__main__":
    main()
