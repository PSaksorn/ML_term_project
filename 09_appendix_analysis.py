from pathlib import Path
import json
import math
import multiprocessing as mp
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# STEP 9 — APPENDIX / SUPPLEMENTARY ANALYSIS
# ============================================================
#
# Purpose:
#   Create detailed supplementary results after the final
#   three-seed experiment.
#
# This script DOES NOT TRAIN.
#
# It loads the best checkpoints for:
#   Seed 42
#   Seed 123
#   Seed 2026
#
# and evaluates them on the SAME fixed test set.
#
# Appendix outputs:
# 1) Per-seed predictions for Model A and Model B
# 2) Per-seed confusion matrices
# 3) Aggregate confusion matrices across all seeds
# 4) Row-normalized aggregate confusion matrices
# 5) Full confusion-matrix figures
# 6) Per-identity recall, mean ± SD across seeds
# 7) Worst identities for each model
# 8) Aggregate top confusion pairs
# 9) Cross-seed consensus correctness per test image
# 10) Persistent failures (wrong in all 3 seeds)
# 11) Persistent Model-B improvements over Model A
# 12) Confidence / calibration-style summary
# 13) Optional representative consensus-failure contact sheet
#
# Main report should still use Step 8 mean ± SD results.
# These outputs are intended mainly for Appendix / supplementary use.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEEDS = [42, 123, 2026]

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4
EMBEDDING_DIM = 128
REQUIRE_CUDA = True
RANDOM_SEED = 42

TEST_CSV = Path("outputs") / "splits" / "test.csv"

CHECKPOINTS = {
    42: {
        "Model_A": (
            Path("outputs")
            / "final_training_v3"
            / "models"
            / "model_A_best.pt"
        ),
        "Model_B": (
            Path("outputs")
            / "final_training_v3"
            / "models"
            / "model_B_best.pt"
        ),
    },
    123: {
        "Model_A": (
            Path("outputs")
            / "repeated_seeds"
            / "seed_123"
            / "models"
            / "model_A_best.pt"
        ),
        "Model_B": (
            Path("outputs")
            / "repeated_seeds"
            / "seed_123"
            / "models"
            / "model_B_best.pt"
        ),
    },
    2026: {
        "Model_A": (
            Path("outputs")
            / "repeated_seeds"
            / "seed_2026"
            / "models"
            / "model_A_best.pt"
        ),
        "Model_B": (
            Path("outputs")
            / "repeated_seeds"
            / "seed_2026"
            / "models"
            / "model_B_best.pt"
        ),
    },
}

OUTPUT_DIR = Path("outputs") / "appendix_analysis"
PRED_DIR = OUTPUT_DIR / "predictions"
CM_DIR = OUTPUT_DIR / "confusion_matrices"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXAMPLE_DIR = OUTPUT_DIR / "examples"

N_WORST_IDENTITIES = 20
N_TOP_CONFUSIONS = 30
N_CONTACT_SHEET = 16

CONFIDENCE_BINS = np.linspace(0.0, 1.0, 11)


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
# 3. MODEL — SAME ARCHITECTURE AS STEP 4 v3 / STEP 7
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

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward_features(self, x):
        x = self.features(x)
        x = self.spatial_pool(x)
        return self.embedding_head(x)

    def forward(self, x):
        z = self.forward_features(x)
        return self.classifier(z)


# ============================================================
# 4. DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, df, person_to_label, transform):
        self.df = df.reset_index(drop=True).copy()
        self.person_to_label = person_to_label
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(Path(row["path"])) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.person_to_label[row["person"]]

        return image, label, index


# ============================================================
# 5. LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path, device):

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device
        )

    return checkpoint


def build_model(checkpoint, device):

    model = ImprovedFaceCNN(
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint.get(
            "embedding_dim",
            EMBEDDING_DIM
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 6. INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    test_df,
    label_to_person,
    device,
    seed,
    model_name,
):

    n = len(test_df)

    true_labels = np.zeros(n, dtype=np.int64)
    pred_labels = np.zeros(n, dtype=np.int64)
    confidences = np.zeros(n, dtype=np.float32)

    for images, labels, indices in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"{model_name}, seed {seed}: non-finite logits."
            )

        probs = F.softmax(
            logits,
            dim=1
        )

        confidence, prediction = probs.max(
            dim=1
        )

        idx = indices.numpy()

        true_labels[idx] = labels.cpu().numpy()
        pred_labels[idx] = prediction.cpu().numpy()
        confidences[idx] = confidence.cpu().numpy()

    output = test_df.reset_index(drop=True).copy()

    output["seed"] = seed
    output["model"] = model_name
    output["true_label"] = true_labels
    output["predicted_label"] = pred_labels
    output["predicted_person"] = [
        label_to_person[int(x)]
        for x in pred_labels
    ]
    output["correct"] = (
        true_labels == pred_labels
    ).astype(int)
    output["confidence"] = confidences
    output["abs_yaw"] = output["yaw"].abs()

    return output


# ============================================================
# 7. PER-IDENTITY RECALL
# ============================================================

def per_identity_recall(pred_df):

    rows = []

    for person, group in pred_df.groupby("person"):

        rows.append({
            "person": person,
            "n_images": len(group),
            "recall": group["correct"].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. CONFIDENCE BIN SUMMARY
# ============================================================

def confidence_bins(pred_df, seed, model_name):

    rows = []

    for low, high in zip(
        CONFIDENCE_BINS[:-1],
        CONFIDENCE_BINS[1:]
    ):

        if high == 1.0:
            mask = (
                (pred_df["confidence"] >= low)
                & (pred_df["confidence"] <= high)
            )
        else:
            mask = (
                (pred_df["confidence"] >= low)
                & (pred_df["confidence"] < high)
            )

        group = pred_df[mask]

        if len(group) == 0:
            rows.append({
                "seed": seed,
                "model": model_name,
                "confidence_bin_low": low,
                "confidence_bin_high": high,
                "n_images": 0,
                "mean_confidence": np.nan,
                "accuracy": np.nan,
            })
            continue

        rows.append({
            "seed": seed,
            "model": model_name,
            "confidence_bin_low": low,
            "confidence_bin_high": high,
            "n_images": len(group),
            "mean_confidence": group["confidence"].mean(),
            "accuracy": group["correct"].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 9. CONTACT SHEET
# ============================================================

def make_contact_sheet(
    df,
    output_path,
    title,
    n=N_CONTACT_SHEET,
    columns=4
):
    if len(df) == 0:
        return None

    examples = (
        df
        .sort_values(
            ["abs_yaw", "B_correct_seeds"],
            ascending=[False, True]
        )
        .head(n)
        .reset_index(drop=True)
    )

    thumb_w = 260
    thumb_h = 245
    image_box_h = 165
    title_h = 45

    rows = math.ceil(
        len(examples) / columns
    )

    canvas = Image.new(
        "RGB",
        (
            columns * thumb_w,
            title_h + rows * thumb_h
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        title,
        fill="black"
    )

    for i, row in examples.iterrows():

        r = i // columns
        c = i % columns

        x0 = c * thumb_w
        y0 = title_h + r * thumb_h

        try:
            with Image.open(Path(row["path"])) as image:
                image = image.convert("L")
                image = ImageOps.contain(
                    image,
                    (165, image_box_h)
                ).convert("RGB")
        except Exception:
            image = Image.new(
                "RGB",
                (165, image_box_h),
                "white"
            )

        image_x = (
            x0 + (thumb_w - image.width) // 2
        )

        canvas.paste(
            image,
            (image_x, y0)
        )

        text_y = y0 + image_box_h + 5

        lines = [
            f"{row['person']} | yaw {int(row['yaw']):+d}",
            f"A correct seeds: {int(row['A_correct_seeds'])}/3",
            f"B correct seeds: {int(row['B_correct_seeds'])}/3",
        ]

        for j, line in enumerate(lines):
            draw.text(
                (x0 + 6, text_y + 17 * j),
                line,
                fill="black"
            )

    canvas.save(output_path)

    return output_path


# ============================================================
# 10. MAIN
# ============================================================

def main():

    for directory in [
        PRED_DIR,
        CM_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        EXAMPLE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            "\nCUDA GPU is required for Step 9.\n"
            f"PyTorch version: {torch.__version__}\n"
            f"PyTorch CUDA   : {torch.version.cuda}\n"
        )

    print("=" * 80)
    print("STEP 9 — APPENDIX / SUPPLEMENTARY ANALYSIS")
    print("=" * 80)
    print("Device :", device)

    if device.type == "cuda":
        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

        torch.backends.cudnn.benchmark = True

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Missing test CSV:\n{TEST_CSV.resolve()}"
        )

    test_df = pd.read_csv(TEST_CSV)

    test_df["yaw"] = (
        test_df["yaw"].astype(int)
    )

    test_df["illumination"] = (
        test_df["illumination"].astype(int)
    )

    # --------------------------------------------------------
    # Use seed 42 mapping as canonical identity mapping.
    # --------------------------------------------------------

    seed42_ckpt = load_checkpoint(
        CHECKPOINTS[42]["Model_A"],
        device
    )

    person_to_label = seed42_ckpt[
        "person_to_label"
    ]

    label_to_person = {
        int(label): person
        for person, label
        in person_to_label.items()
    }

    labels = list(
        range(
            len(person_to_label)
        )
    )

    transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=1
        ),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
    ])

    dataset = TestDataset(
        test_df,
        person_to_label,
        transform
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    all_predictions = []
    identity_rows = []
    confidence_rows = []

    cms = {
        "Model_A": [],
        "Model_B": [],
    }

    # --------------------------------------------------------
    # RUN ALL 6 BEST CHECKPOINTS
    # --------------------------------------------------------

    for seed in SEEDS:

        for model_name in [
            "Model_A",
            "Model_B",
        ]:

            checkpoint = load_checkpoint(
                CHECKPOINTS[seed][model_name],
                device
            )

            if (
                checkpoint["person_to_label"]
                != person_to_label
            ):
                raise ValueError(
                    f"Label mapping mismatch: "
                    f"seed={seed}, model={model_name}"
                )

            model = build_model(
                checkpoint,
                device
            )

            print(
                f"\nInference: "
                f"seed={seed}, {model_name}"
            )

            pred_df = run_inference(
                model,
                loader,
                test_df,
                label_to_person,
                device,
                seed,
                model_name,
            )

            pred_path = (
                PRED_DIR
                / f"seed_{seed}_{model_name}_predictions.csv"
            )

            pred_df.to_csv(
                pred_path,
                index=False
            )

            all_predictions.append(
                pred_df
            )

            # Per-identity recall.
            identity_df = per_identity_recall(
                pred_df
            )

            identity_df.insert(
                0,
                "seed",
                seed
            )

            identity_df.insert(
                1,
                "model",
                model_name
            )

            identity_rows.append(
                identity_df
            )

            # Confidence bins.
            confidence_rows.append(
                confidence_bins(
                    pred_df,
                    seed,
                    model_name
                )
            )

            # Confusion matrix.
            cm = confusion_matrix(
                pred_df["true_label"],
                pred_df["predicted_label"],
                labels=labels
            )

            cms[model_name].append(
                cm
            )

            np.save(
                CM_DIR
                / f"seed_{seed}_{model_name}_confusion_matrix.npy",
                cm
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_pred_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    all_pred_df.to_csv(
        TABLE_DIR
        / "all_seeds_all_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # AGGREGATE CONFUSION MATRICES
    # --------------------------------------------------------

    aggregate_cms = {}

    for model_name in [
        "Model_A",
        "Model_B",
    ]:

        aggregate_cm = np.sum(
            np.stack(cms[model_name]),
            axis=0
        )

        aggregate_cms[
            model_name
        ] = aggregate_cm

        np.save(
            CM_DIR
            / f"aggregate_{model_name}_confusion_matrix.npy",
            aggregate_cm
        )

        row_sum = aggregate_cm.sum(
            axis=1,
            keepdims=True
        )

        normalized_cm = np.divide(
            aggregate_cm,
            row_sum,
            out=np.zeros_like(
                aggregate_cm,
                dtype=np.float64
            ),
            where=row_sum != 0
        )

        np.save(
            CM_DIR
            / f"aggregate_{model_name}_confusion_matrix_normalized.npy",
            normalized_cm
        )

        # Full 153x153 figure.
        plt.figure(
            figsize=(12, 10)
        )

        image = plt.imshow(
            normalized_cm,
            aspect="auto",
            vmin=0,
            vmax=1
        )

        plt.colorbar(
            image,
            label="Row-normalized proportion"
        )

        plt.xlabel(
            "Predicted identity index"
        )

        plt.ylabel(
            "True identity index"
        )

        plt.title(
            f"{model_name} Aggregate Confusion Matrix "
            f"Across 3 Seeds"
        )

        # 153 labels would be unreadable.
        # Show sparse identity indices only.
        ticks = np.arange(
            0,
            len(labels),
            10
        )

        plt.xticks(
            ticks,
            ticks,
            rotation=90
        )

        plt.yticks(
            ticks,
            ticks
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / f"aggregate_{model_name}_confusion_matrix.png",
            dpi=220,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # PER-IDENTITY RECALL — MEAN ± SD ACROSS SEEDS
    # --------------------------------------------------------

    identity_all = pd.concat(
        identity_rows,
        ignore_index=True
    )

    identity_all.to_csv(
        TABLE_DIR
        / "per_seed_per_identity_recall.csv",
        index=False
    )

    identity_agg = (
        identity_all
        .groupby(
            ["model", "person"]
        )
        .agg(
            n_seeds=("seed", "nunique"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
        .reset_index()
    )

    identity_agg.to_csv(
        TABLE_DIR
        / "aggregate_per_identity_recall.csv",
        index=False
    )

    for model_name in [
        "Model_A",
        "Model_B",
    ]:

        worst = (
            identity_agg[
                identity_agg["model"]
                == model_name
            ]
            .sort_values(
                [
                    "recall_mean",
                    "recall_sd",
                ],
                ascending=[
                    True,
                    False,
                ]
            )
            .head(
                N_WORST_IDENTITIES
            )
        )

        worst.to_csv(
            TABLE_DIR
            / f"{model_name}_worst_{N_WORST_IDENTITIES}_identities.csv",
            index=False
        )

    # --------------------------------------------------------
    # TOP CONFUSION PAIRS ACROSS ALL SEEDS
    # --------------------------------------------------------

    confusion_pair_tables = []

    for model_name in [
        "Model_A",
        "Model_B",
    ]:

        wrong = all_pred_df[
            (all_pred_df["model"] == model_name)
            & (all_pred_df["correct"] == 0)
        ]

        pairs = (
            wrong
            .groupby(
                [
                    "person",
                    "predicted_person",
                ]
            )
            .agg(
                count=(
                    "path",
                    "size"
                ),
                n_seeds=(
                    "seed",
                    "nunique"
                ),
                mean_confidence=(
                    "confidence",
                    "mean"
                ),
                mean_abs_yaw=(
                    "abs_yaw",
                    "mean"
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "count",
                    "n_seeds",
                    "mean_confidence",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ]
            )
        )

        pairs.insert(
            0,
            "model",
            model_name
        )

        pairs.head(
            N_TOP_CONFUSIONS
        ).to_csv(
            TABLE_DIR
            / f"{model_name}_top_{N_TOP_CONFUSIONS}_confusion_pairs.csv",
            index=False
        )

        confusion_pair_tables.append(
            pairs
        )

    pd.concat(
        confusion_pair_tables,
        ignore_index=True
    ).to_csv(
        TABLE_DIR
        / "all_aggregate_confusion_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # CROSS-SEED CONSENSUS PER TEST IMAGE
    # --------------------------------------------------------

    sample_keys = [
        "person",
        "yaw",
        "illumination",
        "path",
    ]

    a_all = all_pred_df[
        all_pred_df["model"]
        == "Model_A"
    ]

    b_all = all_pred_df[
        all_pred_df["model"]
        == "Model_B"
    ]

    a_consensus = (
        a_all
        .groupby(sample_keys)
        .agg(
            A_correct_seeds=(
                "correct",
                "sum"
            ),
            A_mean_confidence=(
                "confidence",
                "mean"
            ),
        )
        .reset_index()
    )

    b_consensus = (
        b_all
        .groupby(sample_keys)
        .agg(
            B_correct_seeds=(
                "correct",
                "sum"
            ),
            B_mean_confidence=(
                "confidence",
                "mean"
            ),
        )
        .reset_index()
    )

    consensus = pd.merge(
        a_consensus,
        b_consensus,
        on=sample_keys,
        how="inner",
        validate="one_to_one",
    )

    consensus["abs_yaw"] = (
        consensus["yaw"].abs()
    )

    consensus["A_always_correct"] = (
        consensus["A_correct_seeds"]
        == len(SEEDS)
    ).astype(int)

    consensus["B_always_correct"] = (
        consensus["B_correct_seeds"]
        == len(SEEDS)
    ).astype(int)

    consensus["A_always_wrong"] = (
        consensus["A_correct_seeds"]
        == 0
    ).astype(int)

    consensus["B_always_wrong"] = (
        consensus["B_correct_seeds"]
        == 0
    ).astype(int)

    consensus.to_csv(
        TABLE_DIR
        / "cross_seed_consensus_per_test_image.csv",
        index=False
    )

    # Persistent Model-B rescue:
    # Model A wrong in all three seeds,
    # Model B correct in all three seeds.
    persistent_rescue = consensus[
        (consensus["A_correct_seeds"] == 0)
        & (consensus["B_correct_seeds"] == 3)
    ].copy()

    persistent_rescue.to_csv(
        TABLE_DIR
        / "persistent_A_wrong_B_right_all_3_seeds.csv",
        index=False
    )

    # Persistent failure:
    # Both models wrong in all three seeds.
    persistent_both_wrong = consensus[
        (consensus["A_correct_seeds"] == 0)
        & (consensus["B_correct_seeds"] == 0)
    ].copy()

    persistent_both_wrong.to_csv(
        TABLE_DIR
        / "persistent_both_wrong_all_3_seeds.csv",
        index=False
    )

    # Model B persistent failure only.
    persistent_b_wrong = consensus[
        consensus["B_correct_seeds"]
        == 0
    ].copy()

    persistent_b_wrong.to_csv(
        TABLE_DIR
        / "persistent_model_B_failures_all_3_seeds.csv",
        index=False
    )

    # Consensus summary by yaw.
    consensus_yaw = (
        consensus
        .groupby("abs_yaw")
        .agg(
            n_images=(
                "path",
                "size"
            ),
            A_correct_all_3=(
                "A_always_correct",
                "sum"
            ),
            B_correct_all_3=(
                "B_always_correct",
                "sum"
            ),
            A_wrong_all_3=(
                "A_always_wrong",
                "sum"
            ),
            B_wrong_all_3=(
                "B_always_wrong",
                "sum"
            ),
        )
        .reset_index()
    )

    for column in [
        "A_correct_all_3",
        "B_correct_all_3",
        "A_wrong_all_3",
        "B_wrong_all_3",
    ]:
        consensus_yaw[
            column + "_percent"
        ] = (
            consensus_yaw[column]
            / consensus_yaw["n_images"]
            * 100
        )

    consensus_yaw.to_csv(
        TABLE_DIR
        / "cross_seed_consensus_by_absolute_yaw.csv",
        index=False
    )

    # --------------------------------------------------------
    # CONFIDENCE / CALIBRATION-STYLE TABLE
    # --------------------------------------------------------

    confidence_all = pd.concat(
        confidence_rows,
        ignore_index=True
    )

    confidence_all.to_csv(
        TABLE_DIR
        / "confidence_bins_per_seed.csv",
        index=False
    )

    confidence_agg = (
        confidence_all
        .groupby(
            [
                "model",
                "confidence_bin_low",
                "confidence_bin_high",
            ]
        )
        .agg(
            n_images_mean=(
                "n_images",
                "mean"
            ),
            mean_confidence_mean=(
                "mean_confidence",
                "mean"
            ),
            mean_confidence_sd=(
                "mean_confidence",
                "std"
            ),
            accuracy_mean=(
                "accuracy",
                "mean"
            ),
            accuracy_sd=(
                "accuracy",
                "std"
            ),
        )
        .reset_index()
    )

    confidence_agg.to_csv(
        TABLE_DIR
        / "aggregate_confidence_bins.csv",
        index=False
    )

    # --------------------------------------------------------
    # CONTACT SHEET OF PERSISTENT BOTH-WRONG CASES
    # --------------------------------------------------------

    contact_path = make_contact_sheet(
        persistent_both_wrong,
        EXAMPLE_DIR
        / "persistent_both_wrong_all_3_seeds.png",
        "Persistent failures: both models wrong in all 3 seeds"
    )

    # --------------------------------------------------------
    # FIGURE — PER-IDENTITY RECALL DISTRIBUTION
    # --------------------------------------------------------

    a_identity = identity_agg[
        identity_agg["model"]
        == "Model_A"
    ]["recall_mean"].to_numpy()

    b_identity = identity_agg[
        identity_agg["model"]
        == "Model_B"
    ]["recall_mean"].to_numpy()

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        a_identity,
        bins=20,
        alpha=0.55,
        label="Model A"
    )

    plt.hist(
        b_identity,
        bins=20,
        alpha=0.55,
        label="Model B"
    )

    plt.xlabel(
        "Mean per-identity recall across seeds"
    )

    plt.ylabel(
        "Number of identities"
    )

    plt.title(
        "Distribution of Per-Identity Recall Across 153 Identities"
    )

    plt.legend()
    plt.tight_layout()

    recall_hist_path = (
        FIGURE_DIR
        / "per_identity_recall_distribution.png"
    )

    plt.savefig(
        recall_hist_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "seeds": SEEDS,
        "n_test_images": int(
            len(test_df)
        ),
        "n_identities": int(
            test_df["person"].nunique()
        ),
        "persistent_A_wrong_B_right_all_3_seeds": int(
            len(persistent_rescue)
        ),
        "persistent_both_wrong_all_3_seeds": int(
            len(persistent_both_wrong)
        ),
        "persistent_model_B_failures_all_3_seeds": int(
            len(persistent_b_wrong)
        ),
    }

    with open(
        TABLE_DIR
        / "appendix_analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n" + "=" * 80)
    print("APPENDIX ANALYSIS SUMMARY")
    print("=" * 80)

    print(
        "Persistent A wrong / B right in all 3 seeds :",
        f"{len(persistent_rescue):,}"
    )

    print(
        "Persistent both wrong in all 3 seeds        :",
        f"{len(persistent_both_wrong):,}"
    )

    print(
        "Persistent Model B failures in all 3 seeds  :",
        f"{len(persistent_b_wrong):,}"
    )

    print("\nOutputs:")
    print("Predictions :", PRED_DIR.resolve())
    print("Matrices    :", CM_DIR.resolve())
    print("Tables      :", TABLE_DIR.resolve())
    print("Figures     :", FIGURE_DIR.resolve())
    print("Examples    :", EXAMPLE_DIR.resolve())

    if contact_path is not None:
        print(
            "Contact sheet:",
            contact_path.resolve()
        )

    print("\nStep 9 completed successfully.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
