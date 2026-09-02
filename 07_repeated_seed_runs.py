from pathlib import Path
import json
import multiprocessing as mp
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# STEP 7 — REPEATED-SEED TRAINING + TEST EVALUATION
# ============================================================
#
# Existing final run:
#   Seed 42  -> already completed in Step 4 v3 + Step 5
#
# This script adds:
#   Seed 123
#   Seed 2026
#
# IMPORTANT:
# - Train/Validation/Test split is NOT recreated.
# - Architecture is identical to Step 4 v3.
# - Hyperparameters are identical to Step 4 v3.
# - Training budget is identical: 40 epochs.
# - Augmentation policy is identical.
# - Only the RANDOM SEED changes.
#
# Each seed trains:
#   Model A — frontal-only
#   Model B — pose-diverse
#
# Then each best checkpoint is evaluated on the SAME test.csv.
# ============================================================


# ============================================================
# 1. CONFIG — MUST MATCH STEP 4 v3
# ============================================================

SEEDS = [123, 2026]

IMAGE_SIZE = 128
BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
NUM_WORKERS = 4

NUM_EPOCHS = 40

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 5.0

EMBEDDING_DIM = 128

REQUIRE_CUDA = True

USE_AUGMENTATION = True
BRIGHTNESS_JITTER = 0.10
CONTRAST_JITTER = 0.10

# If a seed has already produced its final evaluation summary,
# it will be skipped when this is True.
SKIP_COMPLETED = True

SPLIT_DIR = Path("outputs") / "splits"

TRAIN_A_CSV = SPLIT_DIR / "train_model_A.csv"
TRAIN_B_CSV = SPLIT_DIR / "train_model_B.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

ROOT_OUTPUT = Path("outputs") / "repeated_seeds"

SEEN_YAWS = {0, -30, 30, -60, 60}
INTERPOLATION_YAWS = {-15, 15, -45, 45}
EXTREME_YAWS = {-75, 75, -90, 90}


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

    def __init__(self, dataframe, person_to_label, transform, return_index=False):
        self.df = dataframe.reset_index(drop=True).copy()
        self.person_to_label = person_to_label
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.person_to_label[row["person"]]

        if self.return_index:
            return image, label, index

        return image, label


# ============================================================
# 4. MODEL — IDENTICAL TO STEP 4 v3
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

def make_loader(
    dataframe,
    person_to_label,
    transform,
    shuffle,
    device,
    seed,
    batch_size,
    return_index=False,
):
    dataset = FaceDataset(
        dataframe=dataframe,
        person_to_label=person_to_label,
        transform=transform,
        return_index=return_index,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
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

def gradients_are_finite(model):
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


# ============================================================
# 7. VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_classification(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Validation logits contain NaN/Inf.")

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss contains NaN/Inf.")

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    return (
        total_loss / total_samples,
        accuracy_score(all_labels, all_predictions),
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        ),
    )


# ============================================================
# 8. TRAIN ONE MODEL
# ============================================================

def train_one_model(
    seed,
    experiment_name,
    train_df,
    val_df,
    person_to_label,
    num_classes,
    train_transform,
    eval_transform,
    device,
    model_dir,
    history_dir,
):
    print("\n" + "#" * 80)
    print(f"SEED {seed} — TRAINING {experiment_name}")
    print("#" * 80)

    # Same initialization seed for A and B within this repeated run.
    set_seed(seed)

    train_loader = make_loader(
        train_df,
        person_to_label,
        train_transform,
        True,
        device,
        seed,
        BATCH_SIZE,
    )

    val_loader = make_loader(
        val_df,
        person_to_label,
        eval_transform,
        False,
        device,
        seed,
        BATCH_SIZE,
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

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    checkpoint_path = model_dir / f"{experiment_name}_best.pt"
    history_path = history_dir / f"{experiment_name}_history.csv"

    history = []
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

            logits = model(images)

            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"Seed {seed} {experiment_name}: "
                    f"NaN/Inf logits at epoch {epoch}, batch {batch_idx}"
                )

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Seed {seed} {experiment_name}: "
                    f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}"
                )

            loss.backward()

            if not gradients_are_finite(model):
                raise FloatingPointError(
                    f"Seed {seed} {experiment_name}: "
                    f"NaN/Inf gradients at epoch {epoch}, batch {batch_idx}"
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Seed {seed} {experiment_name}: non-finite gradient norm."
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

        val_loss, val_accuracy, val_macro_f1 = evaluate_classification(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_seconds = time.perf_counter() - epoch_start

        history.append({
            "seed": seed,
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_seconds,
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy*100:6.2f}% | "
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
                "seed": seed,
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
                "person_to_label": person_to_label,
            }, checkpoint_path)

    total_seconds = time.perf_counter() - total_start

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_val_loss, final_val_accuracy, final_val_macro_f1 = (
        evaluate_classification(
            model,
            val_loader,
            criterion,
            device
        )
    )

    print("\nBest checkpoint:")
    print("Seed           :", seed)
    print("Model          :", experiment_name)
    print("Best epoch     :", best_epoch)
    print(f"Best Val Loss  : {final_val_loss:.4f}")
    print(f"Best Val Acc   : {final_val_accuracy*100:.2f}%")
    print(f"Best Val F1    : {final_val_macro_f1:.4f}")

    result = {
        "seed": seed,
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_loss": final_val_loss,
        "best_val_accuracy": final_val_accuracy,
        "best_val_macro_f1": final_val_macro_f1,
        "training_seconds": total_seconds,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    del train_loader, val_loader, optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model, result


# ============================================================
# 9. TEST INFERENCE
# ============================================================

@torch.no_grad()
def test_inference(
    model,
    test_df,
    person_to_label,
    eval_transform,
    device,
    seed,
):

    test_loader = make_loader(
        test_df,
        person_to_label,
        eval_transform,
        False,
        device,
        seed,
        TEST_BATCH_SIZE,
        return_index=True,
    )

    n = len(test_df)

    embeddings = np.zeros(
        (n, EMBEDDING_DIM),
        dtype=np.float32
    )

    true_labels = np.zeros(n, dtype=np.int64)
    pred_labels = np.zeros(n, dtype=np.int64)
    confidences = np.zeros(n, dtype=np.float32)

    model.eval()

    for images, labels, indices in test_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z = model.forward_features(images)
        logits = model.classifier(z)

        if not torch.isfinite(z).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Test inference produced NaN/Inf.")

        probs = F.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)

        idx = indices.numpy()

        embeddings[idx] = z.cpu().numpy().astype(np.float32)
        true_labels[idx] = labels.cpu().numpy()
        pred_labels[idx] = predictions.cpu().numpy()
        confidences[idx] = confidence.cpu().numpy()

    return embeddings, true_labels, pred_labels, confidences


# ============================================================
# 10. METRIC HELPERS
# ============================================================

def overall_metrics(true_labels, pred_labels):
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0
        ),
    }


def grouped_metrics(test_df, true_labels, pred_labels, group_col):

    temp = test_df.copy().reset_index(drop=True)
    temp["true_label"] = true_labels
    temp["pred_label"] = pred_labels

    rows = []

    for group_value, group in temp.groupby(group_col):
        rows.append({
            group_col: group_value,
            "n_images": len(group),
            "accuracy": accuracy_score(
                group["true_label"],
                group["pred_label"]
            ),
            "macro_f1": f1_score(
                group["true_label"],
                group["pred_label"],
                average="macro",
                zero_division=0
            ),
        })

    return pd.DataFrame(rows)


def pose_category(yaw):
    yaw = int(yaw)

    if yaw in SEEN_YAWS:
        return "seen_training_pose"

    if yaw in INTERPOLATION_YAWS:
        return "unseen_interpolation"

    if yaw in EXTREME_YAWS:
        return "extreme_extrapolation"

    return "other"


# ============================================================
# 11. EMBEDDING SIMILARITY
# ============================================================

def embedding_similarity_by_abs_yaw(test_df, embeddings):

    metadata = test_df.reset_index(drop=True).copy()

    lookup = {}

    for idx, row in metadata.iterrows():
        lookup[
            (
                row["person"],
                int(row["illumination"]),
                int(row["yaw"])
            )
        ] = idx

    rows = []

    for idx, row in metadata.iterrows():

        person = row["person"]
        illumination = int(row["illumination"])
        yaw = int(row["yaw"])

        frontal_idx = lookup[
            (person, illumination, 0)
        ]

        z_pose = embeddings[idx]
        z_frontal = embeddings[frontal_idx]

        denom = (
            np.linalg.norm(z_pose)
            * np.linalg.norm(z_frontal)
        )

        cosine = (
            float(np.dot(z_pose, z_frontal) / denom)
            if denom > 0
            else np.nan
        )

        rows.append({
            "person": person,
            "illumination": illumination,
            "yaw": yaw,
            "abs_yaw": abs(yaw),
            "cosine_similarity_to_frontal": cosine,
        })

    pairs = pd.DataFrame(rows)

    summary = (
        pairs
        .groupby("abs_yaw")[
            "cosine_similarity_to_frontal"
        ]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={
            "count": "n_pairs",
            "mean": "mean_cosine_similarity",
            "std": "std_cosine_similarity_within_images",
            "median": "median_cosine_similarity",
        })
    )

    return summary


# ============================================================
# 12. SAVE TEST RESULTS FOR ONE SEED
# ============================================================

def evaluate_seed_models(
    seed,
    model_a,
    model_b,
    test_df,
    person_to_label,
    eval_transform,
    device,
    result_dir,
):

    test_df = test_df.copy().reset_index(drop=True)

    test_df["yaw"] = test_df["yaw"].astype(int)
    test_df["abs_yaw"] = test_df["yaw"].abs()
    test_df["pose_category"] = test_df["yaw"].map(pose_category)

    (
        emb_a,
        true_a,
        pred_a,
        conf_a,
    ) = test_inference(
        model_a,
        test_df,
        person_to_label,
        eval_transform,
        device,
        seed,
    )

    (
        emb_b,
        true_b,
        pred_b,
        conf_b,
    ) = test_inference(
        model_b,
        test_df,
        person_to_label,
        eval_transform,
        device,
        seed,
    )

    if not np.array_equal(true_a, true_b):
        raise RuntimeError("A/B test label order differs.")

    # ---------------- overall ----------------

    overall_a = overall_metrics(true_a, pred_a)
    overall_b = overall_metrics(true_b, pred_b)

    overall_df = pd.DataFrame([
        {
            "seed": seed,
            "model": "Model_A",
            **overall_a,
        },
        {
            "seed": seed,
            "model": "Model_B",
            **overall_b,
        },
    ])

    overall_df.to_csv(
        result_dir / "overall_test_metrics.csv",
        index=False
    )

    # ---------------- signed yaw ----------------

    signed_a = grouped_metrics(
        test_df,
        true_a,
        pred_a,
        "yaw"
    ).rename(columns={
        "accuracy": "model_A_accuracy",
        "macro_f1": "model_A_macro_f1",
    })

    signed_b = grouped_metrics(
        test_df,
        true_b,
        pred_b,
        "yaw"
    ).rename(columns={
        "accuracy": "model_B_accuracy",
        "macro_f1": "model_B_macro_f1",
    })

    signed = pd.merge(
        signed_a[
            ["yaw", "n_images", "model_A_accuracy", "model_A_macro_f1"]
        ],
        signed_b[
            ["yaw", "model_B_accuracy", "model_B_macro_f1"]
        ],
        on="yaw"
    )

    signed.insert(0, "seed", seed)

    signed["accuracy_gain_B_minus_A"] = (
        signed["model_B_accuracy"]
        - signed["model_A_accuracy"]
    )

    signed["macro_f1_gain_B_minus_A"] = (
        signed["model_B_macro_f1"]
        - signed["model_A_macro_f1"]
    )

    signed.to_csv(
        result_dir / "metrics_by_signed_yaw.csv",
        index=False
    )

    # ---------------- absolute yaw ----------------

    abs_a = grouped_metrics(
        test_df,
        true_a,
        pred_a,
        "abs_yaw"
    ).rename(columns={
        "accuracy": "model_A_accuracy",
        "macro_f1": "model_A_macro_f1",
    })

    abs_b = grouped_metrics(
        test_df,
        true_b,
        pred_b,
        "abs_yaw"
    ).rename(columns={
        "accuracy": "model_B_accuracy",
        "macro_f1": "model_B_macro_f1",
    })

    abs_metrics = pd.merge(
        abs_a[
            ["abs_yaw", "n_images", "model_A_accuracy", "model_A_macro_f1"]
        ],
        abs_b[
            ["abs_yaw", "model_B_accuracy", "model_B_macro_f1"]
        ],
        on="abs_yaw"
    )

    abs_metrics.insert(0, "seed", seed)

    abs_metrics["accuracy_gain_B_minus_A"] = (
        abs_metrics["model_B_accuracy"]
        - abs_metrics["model_A_accuracy"]
    )

    abs_metrics["macro_f1_gain_B_minus_A"] = (
        abs_metrics["model_B_macro_f1"]
        - abs_metrics["model_A_macro_f1"]
    )

    abs_metrics.to_csv(
        result_dir / "metrics_by_absolute_yaw.csv",
        index=False
    )

    # ---------------- pose category ----------------

    category_a = grouped_metrics(
        test_df,
        true_a,
        pred_a,
        "pose_category"
    ).rename(columns={
        "accuracy": "model_A_accuracy",
        "macro_f1": "model_A_macro_f1",
    })

    category_b = grouped_metrics(
        test_df,
        true_b,
        pred_b,
        "pose_category"
    ).rename(columns={
        "accuracy": "model_B_accuracy",
        "macro_f1": "model_B_macro_f1",
    })

    category = pd.merge(
        category_a[
            [
                "pose_category",
                "n_images",
                "model_A_accuracy",
                "model_A_macro_f1"
            ]
        ],
        category_b[
            [
                "pose_category",
                "model_B_accuracy",
                "model_B_macro_f1"
            ]
        ],
        on="pose_category"
    )

    category.insert(0, "seed", seed)

    category["accuracy_gain_B_minus_A"] = (
        category["model_B_accuracy"]
        - category["model_A_accuracy"]
    )

    category["macro_f1_gain_B_minus_A"] = (
        category["model_B_macro_f1"]
        - category["model_A_macro_f1"]
    )

    category.to_csv(
        result_dir / "metrics_by_pose_category.csv",
        index=False
    )

    # ---------------- embedding ----------------

    embedding_a = embedding_similarity_by_abs_yaw(
        test_df,
        emb_a
    ).rename(columns={
        "mean_cosine_similarity": "model_A_mean_cosine",
        "std_cosine_similarity_within_images": "model_A_within_image_std",
        "median_cosine_similarity": "model_A_median_cosine",
    })

    embedding_b = embedding_similarity_by_abs_yaw(
        test_df,
        emb_b
    ).rename(columns={
        "mean_cosine_similarity": "model_B_mean_cosine",
        "std_cosine_similarity_within_images": "model_B_within_image_std",
        "median_cosine_similarity": "model_B_median_cosine",
    })

    embedding = pd.merge(
        embedding_a[
            [
                "abs_yaw",
                "n_pairs",
                "model_A_mean_cosine",
                "model_A_within_image_std",
                "model_A_median_cosine",
            ]
        ],
        embedding_b[
            [
                "abs_yaw",
                "model_B_mean_cosine",
                "model_B_within_image_std",
                "model_B_median_cosine",
            ]
        ],
        on="abs_yaw"
    )

    embedding.insert(0, "seed", seed)

    embedding["cosine_gain_B_minus_A"] = (
        embedding["model_B_mean_cosine"]
        - embedding["model_A_mean_cosine"]
    )

    embedding.to_csv(
        result_dir / "embedding_similarity_by_absolute_yaw.csv",
        index=False
    )

    # Compact JSON summary.
    summary = {
        "seed": seed,
        "test_images": int(len(test_df)),
        "model_A": overall_a,
        "model_B": overall_b,
        "overall_accuracy_gain_B_minus_A": (
            overall_b["accuracy"]
            - overall_a["accuracy"]
        ),
    }

    with open(
        result_dir / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SEED {seed} — TEST RESULTS")
    print("=" * 80)

    print(
        overall_df[
            ["model", "accuracy", "macro_f1"]
        ].to_string(index=False)
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "\nCUDA GPU is required.\n"
            f"PyTorch version: {torch.__version__}\n"
            f"PyTorch CUDA   : {torch.version.cuda}\n"
        )

    print("=" * 80)
    print("STEP 7 — REPEATED-SEED RUNS")
    print("=" * 80)

    print("Seeds           :", SEEDS)
    print("Device          :", device)
    print("CUDA available  :", torch.cuda.is_available())

    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    for path in [
        TRAIN_A_CSV,
        TRAIN_B_CSV,
        VAL_CSV,
        TEST_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split CSV: {path.resolve()}"
            )

    train_a_df = pd.read_csv(TRAIN_A_CSV)
    train_b_df = pd.read_csv(TRAIN_B_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    persons_a = set(train_a_df["person"])
    persons_b = set(train_b_df["person"])
    persons_val = set(val_df["person"])
    persons_test = set(test_df["person"])

    if not (
        persons_a == persons_b == persons_val == persons_test
    ):
        raise ValueError("Identity sets differ across splits.")

    persons = sorted(persons_a)

    person_to_label = {
        person: index
        for index, person in enumerate(persons)
    }

    num_classes = len(persons)

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(
            brightness=BRIGHTNESS_JITTER,
            contrast=CONTRAST_JITTER
        ),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    for seed in SEEDS:

        seed_dir = ROOT_OUTPUT / f"seed_{seed}"
        model_dir = seed_dir / "models"
        history_dir = seed_dir / "history"
        result_dir = seed_dir / "results"
        config_dir = seed_dir / "config"

        for d in [
            model_dir,
            history_dir,
            result_dir,
            config_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        completion_file = result_dir / "evaluation_summary.json"

        if SKIP_COMPLETED and completion_file.exists():
            print(
                f"\nSeed {seed} already completed. "
                f"Skipping: {completion_file}"
            )
            continue

        config = {
            "seed": seed,
            "architecture": "ImprovedFaceCNN_v3",
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "embedding_dim": EMBEDDING_DIM,
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
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        }

        if device.type == "cuda":
            config["gpu_name"] = torch.cuda.get_device_name(0)

        with open(
            config_dir / "training_config.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)

        # ---------------- Train A ----------------

        model_a, result_a = train_one_model(
            seed,
            "model_A",
            train_a_df,
            val_df,
            person_to_label,
            num_classes,
            train_transform,
            eval_transform,
            device,
            model_dir,
            history_dir,
        )

        # ---------------- Train B ----------------

        model_b, result_b = train_one_model(
            seed,
            "model_B",
            train_b_df,
            val_df,
            person_to_label,
            num_classes,
            train_transform,
            eval_transform,
            device,
            model_dir,
            history_dir,
        )

        training_summary = pd.DataFrame([
            result_a,
            result_b,
        ])

        training_summary.to_csv(
            history_dir / "training_summary.csv",
            index=False
        )

        # ---------------- Test A/B ----------------

        evaluate_seed_models(
            seed,
            model_a,
            model_b,
            test_df,
            person_to_label,
            eval_transform,
            device,
            result_dir,
        )

        del model_a, model_b

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nStep 7 completed successfully.")
    print("Output:", ROOT_OUTPUT.resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()
