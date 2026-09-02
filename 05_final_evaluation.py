from pathlib import Path
import json
import multiprocessing as mp
import random

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# STEP 5 — FINAL TEST EVALUATION + REPRESENTATION ANALYSIS
# ============================================================
#
# This script does NOT train.
#
# It loads the best Model A and Model B checkpoints from Step 4 v3
# and evaluates them on the COMMON held-out test set:
#
#   -90, -75, -60, -45, -30, -15, 0,
#   +15, +30, +45, +60, +75, +90 degrees
#
# Outputs:
# 1) Overall Accuracy and Macro-F1
# 2) Accuracy / Macro-F1 by signed yaw
# 3) Accuracy / Macro-F1 by absolute yaw
# 4) Model B gain over Model A
# 5) Seen / interpolation / extreme-pose performance
# 6) 128-D embeddings for every test image
# 7) Same-identity cosine similarity to frontal image
#    using the SAME held-out illumination condition
# 8) Confusion matrices
# 9) Prediction-level CSV files
# 10) Initial failure-case tables for later error analysis
#
# No augmentation is used during test.
# CUDA GPU is used when available.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4
EMBEDDING_DIM = 128

REQUIRE_CUDA = True

TEST_CSV = Path("outputs") / "splits" / "test.csv"

MODEL_A_PATH = (
    Path("outputs")
    / "final_training_v3"
    / "models"
    / "model_A_best.pt"
)

MODEL_B_PATH = (
    Path("outputs")
    / "final_training_v3"
    / "models"
    / "model_B_best.pt"
)

OUTPUT_DIR = Path("outputs") / "final_evaluation"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
ERROR_DIR = OUTPUT_DIR / "error_analysis"

EXPECTED_YAWS = [
    -90, -75, -60, -45, -30, -15,
    0,
    15, 30, 45, 60, 75, 90
]

# Interpretation for Model B
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
# 3. MODEL — MUST MATCH STEP 4 v3
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
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        return logits


# ============================================================
# 4. DATASET
# ============================================================

class TestFaceDataset(Dataset):

    def __init__(
        self,
        dataframe,
        person_to_label,
        transform
    ):
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

        return {
            "image": image,
            "label": label,
            "index": index,
        }


# ============================================================
# 5. CHECKPOINT LOADING
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


def build_model_from_checkpoint(
    checkpoint,
    device
):

    num_classes = checkpoint["num_classes"]

    embedding_dim = checkpoint.get(
        "embedding_dim",
        EMBEDDING_DIM
    )

    model = ImprovedFaceCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 6. TEST INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    dataframe,
    label_to_person,
    device,
    model_name,
):

    prediction_rows = []

    all_embeddings = np.zeros(
        (
            len(dataframe),
            EMBEDDING_DIM
        ),
        dtype=np.float32
    )

    all_true = np.zeros(
        len(dataframe),
        dtype=np.int64
    )

    all_pred = np.zeros(
        len(dataframe),
        dtype=np.int64
    )

    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        indices = batch["index"].numpy()

        embeddings = model.forward_features(
            images
        )

        logits = model.classifier(
            embeddings
        )

        if not torch.isfinite(
            embeddings
        ).all():
            raise FloatingPointError(
                f"{model_name}: non-finite embedding detected."
            )

        if not torch.isfinite(
            logits
        ).all():
            raise FloatingPointError(
                f"{model_name}: non-finite logits detected."
            )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        embeddings_np = (
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        labels_np = (
            labels
            .cpu()
            .numpy()
        )

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        confidence_np = (
            confidence
            .cpu()
            .numpy()
        )

        all_embeddings[
            indices
        ] = embeddings_np

        all_true[
            indices
        ] = labels_np

        all_pred[
            indices
        ] = predictions_np

        for local_index, dataframe_index in enumerate(
            indices
        ):

            row = dataframe.iloc[
                dataframe_index
            ]

            true_label = int(
                labels_np[local_index]
            )

            pred_label = int(
                predictions_np[local_index]
            )

            prediction_rows.append({
                "dataset_index": int(dataframe_index),
                "person": row["person"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "predicted_person": label_to_person[pred_label],
                "correct": int(
                    true_label == pred_label
                ),
                "confidence": float(
                    confidence_np[local_index]
                ),
                "yaw": int(row["yaw"]),
                "abs_yaw": abs(
                    int(row["yaw"])
                ),
                "direction": row["direction"],
                "illumination": int(
                    row["illumination"]
                ),
                "filename": row["filename"],
                "path": row["path"],
            })

        if batch_number % 20 == 0:
            processed = min(
                batch_number * BATCH_SIZE,
                len(dataframe)
            )

            print(
                f"{model_name}: "
                f"{processed:,}/{len(dataframe):,}"
            )

    predictions_df = (
        pd.DataFrame(prediction_rows)
        .sort_values("dataset_index")
        .reset_index(drop=True)
    )

    return (
        predictions_df,
        all_embeddings,
        all_true,
        all_pred,
    )


# ============================================================
# 7. METRICS
# ============================================================

def calculate_overall_metrics(
    predictions_df,
    model_name
):

    accuracy = accuracy_score(
        predictions_df["true_label"],
        predictions_df["predicted_label"]
    )

    macro_f1 = f1_score(
        predictions_df["true_label"],
        predictions_df["predicted_label"],
        average="macro",
        zero_division=0
    )

    return {
        "model": model_name,
        "n_images": len(
            predictions_df
        ),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def calculate_group_metrics(
    predictions_df,
    group_column,
    model_name
):

    rows = []

    for group_value, group in (
        predictions_df
        .groupby(group_column)
    ):

        accuracy = accuracy_score(
            group["true_label"],
            group["predicted_label"]
        )

        macro_f1 = f1_score(
            group["true_label"],
            group["predicted_label"],
            average="macro",
            zero_division=0
        )

        rows.append({
            "model": model_name,
            group_column: group_value,
            "n_images": len(group),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

    return pd.DataFrame(rows)


def add_pose_category(
    predictions_df
):

    df = predictions_df.copy()

    def category(yaw):

        yaw = int(yaw)

        if yaw in SEEN_YAWS:
            return "seen_training_pose"

        if yaw in INTERPOLATION_YAWS:
            return "unseen_interpolation"

        if yaw in EXTREME_YAWS:
            return "extreme_extrapolation"

        return "other"

    df[
        "pose_category"
    ] = df["yaw"].map(category)

    return df


# ============================================================
# 8. EMBEDDING SIMILARITY
# ============================================================
#
# Pair each non-frontal test image with:
#
#   SAME identity
#   SAME held-out illumination
#   frontal (yaw = 0)
#
# This isolates pose change while controlling identity and lighting.
#
# Similarity:
#
#   cos(z_0, z_yaw)
#
# ============================================================

def embedding_similarity_to_frontal(
    test_df,
    embeddings,
    model_name
):

    metadata = (
        test_df
        .reset_index(drop=True)
        .copy()
    )

    lookup = {}

    for index, row in metadata.iterrows():

        key = (
            row["person"],
            int(row["illumination"]),
            int(row["yaw"]),
        )

        if key in lookup:
            raise ValueError(
                f"Duplicate test key found: {key}"
            )

        lookup[key] = index

    pair_rows = []

    for index, row in metadata.iterrows():

        person = row["person"]
        illumination = int(
            row["illumination"]
        )
        yaw = int(row["yaw"])

        frontal_key = (
            person,
            illumination,
            0
        )

        if frontal_key not in lookup:
            raise ValueError(
                "Missing matched frontal image for "
                f"{person}, illumination {illumination}"
            )

        frontal_index = lookup[
            frontal_key
        ]

        z_pose = embeddings[index]
        z_frontal = embeddings[
            frontal_index
        ]

        denominator = (
            np.linalg.norm(z_pose)
            * np.linalg.norm(z_frontal)
        )

        if denominator == 0:
            cosine_similarity = np.nan
        else:
            cosine_similarity = float(
                np.dot(
                    z_pose,
                    z_frontal
                )
                / denominator
            )

        pair_rows.append({
            "model": model_name,
            "person": person,
            "illumination": illumination,
            "yaw": yaw,
            "abs_yaw": abs(yaw),
            "cosine_similarity_to_frontal": cosine_similarity,
            "pose_path": row["path"],
            "frontal_path": metadata.iloc[
                frontal_index
            ]["path"],
        })

    pairs_df = pd.DataFrame(
        pair_rows
    )

    signed_summary = (
        pairs_df
        .groupby("yaw")[
            "cosine_similarity_to_frontal"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
            ]
        )
        .reset_index()
        .rename(columns={
            "count": "n_pairs",
            "mean": "mean_cosine_similarity",
            "std": "std_cosine_similarity",
            "median": "median_cosine_similarity",
        })
    )

    signed_summary.insert(
        0,
        "model",
        model_name
    )

    abs_summary = (
        pairs_df
        .groupby("abs_yaw")[
            "cosine_similarity_to_frontal"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
            ]
        )
        .reset_index()
        .rename(columns={
            "count": "n_pairs",
            "mean": "mean_cosine_similarity",
            "std": "std_cosine_similarity",
            "median": "median_cosine_similarity",
        })
    )

    abs_summary.insert(
        0,
        "model",
        model_name
    )

    return (
        pairs_df,
        signed_summary,
        abs_summary,
    )


# ============================================================
# 9. COMPARISON TABLES
# ============================================================

def compare_models(
    metrics_a,
    metrics_b,
    key
):

    a = metrics_a.copy()
    b = metrics_b.copy()

    a = a.rename(columns={
        "accuracy": "model_A_accuracy",
        "macro_f1": "model_A_macro_f1",
        "n_images": "n_images_A",
    })

    b = b.rename(columns={
        "accuracy": "model_B_accuracy",
        "macro_f1": "model_B_macro_f1",
        "n_images": "n_images_B",
    })

    comparison = pd.merge(
        a[
            [
                key,
                "n_images_A",
                "model_A_accuracy",
                "model_A_macro_f1",
            ]
        ],
        b[
            [
                key,
                "n_images_B",
                "model_B_accuracy",
                "model_B_macro_f1",
            ]
        ],
        on=key,
        how="outer"
    )

    comparison[
        "accuracy_gain_B_minus_A"
    ] = (
        comparison[
            "model_B_accuracy"
        ]
        - comparison[
            "model_A_accuracy"
        ]
    )

    comparison[
        "macro_f1_gain_B_minus_A"
    ] = (
        comparison[
            "model_B_macro_f1"
        ]
        - comparison[
            "model_A_macro_f1"
        ]
    )

    return comparison.sort_values(
        key
    )


def compare_embedding_summaries(
    summary_a,
    summary_b,
    key
):

    a = summary_a.rename(columns={
        "mean_cosine_similarity":
            "model_A_mean_cosine",
        "std_cosine_similarity":
            "model_A_std_cosine",
        "median_cosine_similarity":
            "model_A_median_cosine",
    })

    b = summary_b.rename(columns={
        "mean_cosine_similarity":
            "model_B_mean_cosine",
        "std_cosine_similarity":
            "model_B_std_cosine",
        "median_cosine_similarity":
            "model_B_median_cosine",
    })

    comparison = pd.merge(
        a[
            [
                key,
                "n_pairs",
                "model_A_mean_cosine",
                "model_A_std_cosine",
                "model_A_median_cosine",
            ]
        ],
        b[
            [
                key,
                "model_B_mean_cosine",
                "model_B_std_cosine",
                "model_B_median_cosine",
            ]
        ],
        on=key,
        how="outer"
    )

    comparison[
        "cosine_gain_B_minus_A"
    ] = (
        comparison[
            "model_B_mean_cosine"
        ]
        - comparison[
            "model_A_mean_cosine"
        ]
    )

    return comparison.sort_values(
        key
    )


# ============================================================
# 10. FIGURES
# ============================================================

def plot_accuracy_by_abs_yaw(
    comparison
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        comparison["abs_yaw"],
        comparison["model_A_accuracy"],
        marker="o",
        label="Model A — Frontal-only"
    )

    plt.plot(
        comparison["abs_yaw"],
        comparison["model_B_accuracy"],
        marker="o",
        label="Model B — Pose-diverse"
    )

    plt.xlabel(
        "Absolute yaw angle (degrees)"
    )

    plt.ylabel(
        "Top-1 accuracy"
    )

    plt.title(
        "Cross-Pose Face Recognition Accuracy"
    )

    plt.xticks(
        sorted(
            comparison["abs_yaw"]
            .unique()
        )
    )

    plt.ylim(
        0,
        1.05
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "accuracy_by_absolute_yaw.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_accuracy_gain(
    comparison
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        comparison["abs_yaw"],
        comparison[
            "accuracy_gain_B_minus_A"
        ],
        marker="o"
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Absolute yaw angle (degrees)"
    )

    plt.ylabel(
        "Accuracy gain (Model B - Model A)"
    )

    plt.title(
        "Benefit of Pose-Diverse Training by Yaw"
    )

    plt.xticks(
        sorted(
            comparison["abs_yaw"]
            .unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "accuracy_gain_by_absolute_yaw.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_embedding_similarity(
    comparison
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        comparison["abs_yaw"],
        comparison[
            "model_A_mean_cosine"
        ],
        marker="o",
        label="Model A — Frontal-only"
    )

    plt.plot(
        comparison["abs_yaw"],
        comparison[
            "model_B_mean_cosine"
        ],
        marker="o",
        label="Model B — Pose-diverse"
    )

    plt.xlabel(
        "Absolute yaw angle (degrees)"
    )

    plt.ylabel(
        "Mean same-identity cosine similarity to frontal"
    )

    plt.title(
        "Pose-Induced Shift in Learned Identity Representation"
    )

    plt.xticks(
        sorted(
            comparison["abs_yaw"]
            .unique()
        )
    )

    plt.ylim(
        -0.1,
        1.05
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "embedding_similarity_by_absolute_yaw.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    return path


# ============================================================
# 11. ERROR ANALYSIS TABLES
# ============================================================

def create_error_tables(
    predictions_a,
    predictions_b
):

    key_columns = [
        "person",
        "yaw",
        "illumination",
        "path",
    ]

    a = predictions_a[
        key_columns
        + [
            "correct",
            "predicted_person",
            "confidence",
        ]
    ].rename(columns={
        "correct": "A_correct",
        "predicted_person":
            "A_predicted_person",
        "confidence":
            "A_confidence",
    })

    b = predictions_b[
        key_columns
        + [
            "correct",
            "predicted_person",
            "confidence",
        ]
    ].rename(columns={
        "correct": "B_correct",
        "predicted_person":
            "B_predicted_person",
        "confidence":
            "B_confidence",
    })

    merged = pd.merge(
        a,
        b,
        on=key_columns,
        how="inner"
    )

    merged[
        "abs_yaw"
    ] = merged[
        "yaw"
    ].abs()

    # Model B fixes Model A error.
    a_wrong_b_right = (
        merged[
            (merged["A_correct"] == 0)
            & (merged["B_correct"] == 1)
        ]
        .sort_values(
            [
                "abs_yaw",
                "B_confidence",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    # Model A succeeds where Model B fails.
    a_right_b_wrong = (
        merged[
            (merged["A_correct"] == 1)
            & (merged["B_correct"] == 0)
        ]
        .sort_values(
            [
                "abs_yaw",
                "A_confidence",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    # Both fail, prioritize extreme yaw.
    both_wrong = (
        merged[
            (merged["A_correct"] == 0)
            & (merged["B_correct"] == 0)
        ]
        .sort_values(
            [
                "abs_yaw",
                "B_confidence",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    return (
        merged,
        a_wrong_b_right,
        a_right_b_wrong,
        both_wrong,
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    for directory in [
        RESULT_DIR,
        FIGURE_DIR,
        EMBEDDING_DIR,
        ERROR_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if (
        REQUIRE_CUDA
        and device.type != "cuda"
    ):
        raise RuntimeError(
            "\nCUDA GPU is required for Step 5, "
            "but PyTorch cannot access CUDA.\n"
            f"PyTorch version: {torch.__version__}\n"
            f"PyTorch CUDA   : {torch.version.cuda}\n"
        )

    print("=" * 80)
    print("STEP 5 — FINAL TEST EVALUATION")
    print("=" * 80)

    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("PyTorch CUDA    :", torch.version.cuda)
    print("Device          :", device)

    if device.type == "cuda":
        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # LOAD TEST METADATA
    # --------------------------------------------------------

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n"
            f"{TEST_CSV.resolve()}"
        )

    test_df = pd.read_csv(
        TEST_CSV
    )

    test_df["yaw"] = (
        test_df["yaw"]
        .astype(int)
    )

    test_df[
        "illumination"
    ] = (
        test_df[
            "illumination"
        ]
        .astype(int)
    )

    actual_yaws = sorted(
        test_df["yaw"].unique()
    )

    if (
        actual_yaws
        != EXPECTED_YAWS
    ):
        raise ValueError(
            "Unexpected yaw set.\n"
            f"Expected: {EXPECTED_YAWS}\n"
            f"Found   : {actual_yaws}"
        )

    print("\nTest images :", f"{len(test_df):,}")
    print(
        "Identities  :",
        test_df["person"].nunique()
    )
    print(
        "Yaw values  :",
        actual_yaws
    )
    print(
        "Illuminations:",
        sorted(
            test_df[
                "illumination"
            ].unique()
        )
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINTS
    # --------------------------------------------------------

    checkpoint_a = load_checkpoint(
        MODEL_A_PATH,
        device
    )

    checkpoint_b = load_checkpoint(
        MODEL_B_PATH,
        device
    )

    mapping_a = checkpoint_a[
        "person_to_label"
    ]

    mapping_b = checkpoint_b[
        "person_to_label"
    ]

    if mapping_a != mapping_b:
        raise ValueError(
            "Model A and Model B label mappings differ."
        )

    person_to_label = mapping_a

    label_to_person = {
        int(label): person
        for person, label
        in person_to_label.items()
    }

    expected_persons = set(
        person_to_label.keys()
    )

    test_persons = set(
        test_df["person"]
    )

    if (
        expected_persons
        != test_persons
    ):
        raise ValueError(
            "Test identity set does not match training identity set."
        )

    print("\nModel A checkpoint:")
    print(
        "Best epoch:",
        checkpoint_a["epoch"]
    )
    print(
        "Val accuracy:",
        checkpoint_a[
            "val_accuracy"
        ]
    )
    print(
        "Val Macro-F1:",
        checkpoint_a[
            "val_macro_f1"
        ]
    )

    print("\nModel B checkpoint:")
    print(
        "Best epoch:",
        checkpoint_b["epoch"]
    )
    print(
        "Val accuracy:",
        checkpoint_b[
            "val_accuracy"
        ]
    )
    print(
        "Val Macro-F1:",
        checkpoint_b[
            "val_macro_f1"
        ]
    )

    # --------------------------------------------------------
    # TEST TRANSFORM
    # --------------------------------------------------------

    test_transform = (
        transforms.Compose([
            transforms.Grayscale(
                num_output_channels=1
            ),
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),
            transforms.ToTensor(),
        ])
    )

    test_dataset = TestFaceDataset(
        dataframe=test_df,
        person_to_label=person_to_label,
        transform=test_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    test_loader = DataLoader(
        test_dataset,
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

    # --------------------------------------------------------
    # MODEL A INFERENCE
    # --------------------------------------------------------

    print("\n" + "#" * 80)
    print("MODEL A TEST INFERENCE")
    print("#" * 80)

    model_a = build_model_from_checkpoint(
        checkpoint_a,
        device
    )

    (
        pred_a,
        embedding_a,
        true_a,
        predicted_a,
    ) = run_inference(
        model=model_a,
        loader=test_loader,
        dataframe=test_df,
        label_to_person=label_to_person,
        device=device,
        model_name="Model A",
    )

    del model_a

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # MODEL B INFERENCE
    # --------------------------------------------------------

    print("\n" + "#" * 80)
    print("MODEL B TEST INFERENCE")
    print("#" * 80)

    model_b = build_model_from_checkpoint(
        checkpoint_b,
        device
    )

    (
        pred_b,
        embedding_b,
        true_b,
        predicted_b,
    ) = run_inference(
        model=model_b,
        loader=test_loader,
        dataframe=test_df,
        label_to_person=label_to_person,
        device=device,
        model_name="Model B",
    )

    del model_b

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # VERIFY ORDER
    # --------------------------------------------------------

    if not np.array_equal(
        true_a,
        true_b
    ):
        raise RuntimeError(
            "Model A/B test label order differs."
        )

    # --------------------------------------------------------
    # ADD POSE CATEGORIES
    # --------------------------------------------------------

    pred_a = add_pose_category(
        pred_a
    )

    pred_b = add_pose_category(
        pred_b
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    pred_a_path = (
        RESULT_DIR
        / "model_A_test_predictions.csv"
    )

    pred_b_path = (
        RESULT_DIR
        / "model_B_test_predictions.csv"
    )

    pred_a.to_csv(
        pred_a_path,
        index=False
    )

    pred_b.to_csv(
        pred_b_path,
        index=False
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall = pd.DataFrame([
        calculate_overall_metrics(
            pred_a,
            "Model_A"
        ),
        calculate_overall_metrics(
            pred_b,
            "Model_B"
        ),
    ])

    overall[
        "accuracy_percent"
    ] = (
        overall["accuracy"]
        * 100
    )

    overall[
        "macro_f1_percent"
    ] = (
        overall["macro_f1"]
        * 100
    )

    overall_path = (
        RESULT_DIR
        / "overall_test_metrics.csv"
    )

    overall.to_csv(
        overall_path,
        index=False
    )

    # --------------------------------------------------------
    # SIGNED YAW METRICS
    # --------------------------------------------------------

    signed_a = calculate_group_metrics(
        pred_a,
        "yaw",
        "Model_A"
    )

    signed_b = calculate_group_metrics(
        pred_b,
        "yaw",
        "Model_B"
    )

    signed_comparison = compare_models(
        signed_a,
        signed_b,
        "yaw"
    )

    signed_path = (
        RESULT_DIR
        / "metrics_by_signed_yaw.csv"
    )

    signed_comparison.to_csv(
        signed_path,
        index=False
    )

    # --------------------------------------------------------
    # ABSOLUTE YAW METRICS
    # --------------------------------------------------------

    abs_a = calculate_group_metrics(
        pred_a,
        "abs_yaw",
        "Model_A"
    )

    abs_b = calculate_group_metrics(
        pred_b,
        "abs_yaw",
        "Model_B"
    )

    abs_comparison = compare_models(
        abs_a,
        abs_b,
        "abs_yaw"
    )

    abs_path = (
        RESULT_DIR
        / "metrics_by_absolute_yaw.csv"
    )

    abs_comparison.to_csv(
        abs_path,
        index=False
    )

    # --------------------------------------------------------
    # POSE CATEGORY METRICS
    # --------------------------------------------------------

    category_a = calculate_group_metrics(
        pred_a,
        "pose_category",
        "Model_A"
    )

    category_b = calculate_group_metrics(
        pred_b,
        "pose_category",
        "Model_B"
    )

    category_comparison = (
        compare_models(
            category_a,
            category_b,
            "pose_category"
        )
    )

    category_path = (
        RESULT_DIR
        / "metrics_by_pose_category.csv"
    )

    category_comparison.to_csv(
        category_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE EMBEDDINGS
    # --------------------------------------------------------

    np.save(
        EMBEDDING_DIR
        / "model_A_test_embeddings.npy",
        embedding_a
    )

    np.save(
        EMBEDDING_DIR
        / "model_B_test_embeddings.npy",
        embedding_b
    )

    # --------------------------------------------------------
    # EMBEDDING SIMILARITY
    # --------------------------------------------------------

    (
        embedding_pairs_a,
        embedding_signed_a,
        embedding_abs_a,
    ) = embedding_similarity_to_frontal(
        test_df,
        embedding_a,
        "Model_A"
    )

    (
        embedding_pairs_b,
        embedding_signed_b,
        embedding_abs_b,
    ) = embedding_similarity_to_frontal(
        test_df,
        embedding_b,
        "Model_B"
    )

    embedding_pairs_a.to_csv(
        EMBEDDING_DIR
        / "model_A_embedding_pairs.csv",
        index=False
    )

    embedding_pairs_b.to_csv(
        EMBEDDING_DIR
        / "model_B_embedding_pairs.csv",
        index=False
    )

    embedding_signed_comparison = (
        compare_embedding_summaries(
            embedding_signed_a,
            embedding_signed_b,
            "yaw"
        )
    )

    embedding_abs_comparison = (
        compare_embedding_summaries(
            embedding_abs_a,
            embedding_abs_b,
            "abs_yaw"
        )
    )

    embedding_signed_path = (
        RESULT_DIR
        / "embedding_similarity_by_signed_yaw.csv"
    )

    embedding_abs_path = (
        RESULT_DIR
        / "embedding_similarity_by_absolute_yaw.csv"
    )

    embedding_signed_comparison.to_csv(
        embedding_signed_path,
        index=False
    )

    embedding_abs_comparison.to_csv(
        embedding_abs_path,
        index=False
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    label_order = list(
        range(
            len(
                person_to_label
            )
        )
    )

    cm_a = confusion_matrix(
        true_a,
        predicted_a,
        labels=label_order
    )

    cm_b = confusion_matrix(
        true_b,
        predicted_b,
        labels=label_order
    )

    np.save(
        RESULT_DIR
        / "confusion_matrix_model_A.npy",
        cm_a
    )

    np.save(
        RESULT_DIR
        / "confusion_matrix_model_B.npy",
        cm_b
    )

    # --------------------------------------------------------
    # ERROR ANALYSIS CANDIDATES
    # --------------------------------------------------------

    (
        merged_errors,
        a_wrong_b_right,
        a_right_b_wrong,
        both_wrong,
    ) = create_error_tables(
        pred_a,
        pred_b
    )

    merged_errors.to_csv(
        ERROR_DIR
        / "all_A_B_prediction_comparison.csv",
        index=False
    )

    a_wrong_b_right.head(
        100
    ).to_csv(
        ERROR_DIR
        / "A_wrong_B_right_top100.csv",
        index=False
    )

    a_right_b_wrong.head(
        100
    ).to_csv(
        ERROR_DIR
        / "A_right_B_wrong_top100.csv",
        index=False
    )

    both_wrong.head(
        100
    ).to_csv(
        ERROR_DIR
        / "both_wrong_top100.csv",
        index=False
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    accuracy_path = (
        plot_accuracy_by_abs_yaw(
            abs_comparison
        )
    )

    gain_path = (
        plot_accuracy_gain(
            abs_comparison
        )
    )

    embedding_figure_path = (
        plot_embedding_similarity(
            embedding_abs_comparison
        )
    )

    # --------------------------------------------------------
    # SUMMARY TEXT
    # --------------------------------------------------------

    summary = {
        "test_images": int(
            len(test_df)
        ),
        "identities": int(
            test_df["person"].nunique()
        ),
        "test_yaws": EXPECTED_YAWS,
        "model_A": {
            "checkpoint_epoch": int(
                checkpoint_a["epoch"]
            ),
            "overall_accuracy": float(
                overall.loc[
                    overall["model"] == "Model_A",
                    "accuracy"
                ].iloc[0]
            ),
            "overall_macro_f1": float(
                overall.loc[
                    overall["model"] == "Model_A",
                    "macro_f1"
                ].iloc[0]
            ),
        },
        "model_B": {
            "checkpoint_epoch": int(
                checkpoint_b["epoch"]
            ),
            "overall_accuracy": float(
                overall.loc[
                    overall["model"] == "Model_B",
                    "accuracy"
                ].iloc[0]
            ),
            "overall_macro_f1": float(
                overall.loc[
                    overall["model"] == "Model_B",
                    "macro_f1"
                ].iloc[0]
            ),
        },
    }

    summary[
        "overall_accuracy_gain_B_minus_A"
    ] = (
        summary["model_B"][
            "overall_accuracy"
        ]
        - summary["model_A"][
            "overall_accuracy"
        ]
    )

    with open(
        RESULT_DIR
        / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL TEST RESULTS")
    print("=" * 80)

    print(
        overall[
            [
                "model",
                "n_images",
                "accuracy_percent",
                "macro_f1_percent",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("RESULTS BY ABSOLUTE YAW")
    print("=" * 80)

    printable_abs = (
        abs_comparison.copy()
    )

    for column in [
        "model_A_accuracy",
        "model_B_accuracy",
        "accuracy_gain_B_minus_A",
        "model_A_macro_f1",
        "model_B_macro_f1",
    ]:
        printable_abs[column] = (
            printable_abs[column]
            * 100
        )

    print(
        printable_abs.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("POSE CATEGORY RESULTS")
    print("=" * 80)

    printable_category = (
        category_comparison.copy()
    )

    for column in [
        "model_A_accuracy",
        "model_B_accuracy",
        "accuracy_gain_B_minus_A",
        "model_A_macro_f1",
        "model_B_macro_f1",
    ]:
        printable_category[column] = (
            printable_category[column]
            * 100
        )

    print(
        printable_category.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("EMBEDDING SIMILARITY BY ABSOLUTE YAW")
    print("=" * 80)

    print(
        embedding_abs_comparison.to_string(
            index=False
        )
    )

    print("\nFigures:")
    print(
        accuracy_path.resolve()
    )
    print(
        gain_path.resolve()
    )
    print(
        embedding_figure_path.resolve()
    )

    print("\nStep 5 completed successfully.")
    print(
        "Output:",
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
