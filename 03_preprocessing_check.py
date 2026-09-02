from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# ============================================================
# STEP 3 — PREPROCESSING CHECK + AUGMENTATION PREVIEW
# ============================================================
#
# Purpose
# -------
# 1) Verify that all split CSV paths exist.
# 2) Audit real image size / mode / intensity statistics.
# 3) Confirm preprocessing: grayscale -> 128x128 -> [0,1].
# 4) Visualize pose progression using the SAME identity and
#    SAME held-out illumination across yaw angles.
# 5) Preview mild training augmentation WITHOUT changing the
#    original dataset.
#
# Important
# ---------
# - Validation and test images are NEVER augmented.
# - Horizontal flip is intentionally NOT used because left/right
#   yaw is part of the research variable.
# - No perspective or synthetic-yaw augmentation is used.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
IMAGE_SIZE = (128, 128)

SPLIT_DIR = Path("outputs") / "splits"
OUTPUT_DIR = Path("outputs") / "preprocessing"
FIGURE_DIR = Path("outputs") / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_A_CSV = SPLIT_DIR / "train_model_A.csv"
TRAIN_B_CSV = SPLIT_DIR / "train_model_B.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

# Mild augmentation ranges for preview.
# These are NOT applied permanently to disk.
AUG_BRIGHTNESS = (0.90, 1.10)
AUG_CONTRAST = (0.90, 1.10)
AUG_ROTATION_DEG = (-5.0, 5.0)
AUG_TRANSLATION_FRAC = (-0.04, 0.04)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. LOAD SPLITS
# ============================================================

required_files = [TRAIN_A_CSV, TRAIN_B_CSV, VAL_CSV, TEST_CSV]

for csv_path in required_files:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing split file:\n{csv_path.resolve()}\n\n"
            "Run 02_create_split.py first."
        )

train_a = pd.read_csv(TRAIN_A_CSV)
train_b = pd.read_csv(TRAIN_B_CSV)
validation = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)

datasets = {
    "train_model_A": train_a,
    "train_model_B": train_b,
    "validation": validation,
    "test": test,
}

print("=" * 80)
print("STEP 3 — PREPROCESSING CHECK")
print("=" * 80)

for name, frame in datasets.items():
    print(f"{name:15s}: {len(frame):,} images")


# ============================================================
# 3. CHECK IMAGE PATHS
# ============================================================

print("\n" + "=" * 80)
print("FILE PATH CHECK")
print("=" * 80)

missing_rows = []

for split_name, frame in datasets.items():
    missing_count = 0

    for _, row in frame.iterrows():
        image_path = Path(row["path"])

        if not image_path.exists():
            missing_count += 1
            missing_rows.append({
                "split": split_name,
                "person": row["person"],
                "yaw": row["yaw"],
                "illumination": row["illumination"],
                "path": row["path"],
            })

    if missing_count == 0:
        print(f"PASS — {split_name}: all paths exist")
    else:
        print(f"FAIL — {split_name}: {missing_count} missing paths")

missing_df = pd.DataFrame(missing_rows)

if len(missing_df) > 0:
    missing_path = OUTPUT_DIR / "missing_files.csv"
    missing_df.to_csv(missing_path, index=False)

    raise FileNotFoundError(
        f"\nMissing image files detected.\n"
        f"See: {missing_path.resolve()}"
    )


# ============================================================
# 4. BUILD UNIQUE IMAGE AUDIT TABLE
# ============================================================
#
# A and B may share some training images, so audit each unique
# path only once.
# ============================================================

all_rows = []

for split_name, frame in datasets.items():
    temp = frame.copy()
    temp["source_split"] = split_name
    all_rows.append(temp)

all_df = pd.concat(all_rows, ignore_index=True)

unique_images = (
    all_df.sort_values(["path", "source_split"])
          .drop_duplicates(subset=["path"])
          .reset_index(drop=True)
)

print("\nUnique images to audit:", f"{len(unique_images):,}")


# ============================================================
# 5. IMAGE AUDIT
# ============================================================

print("\n" + "=" * 80)
print("IMAGE FORMAT / INTENSITY AUDIT")
print("=" * 80)

audit_rows = []

for idx, row in unique_images.iterrows():
    image_path = Path(row["path"])

    with Image.open(image_path) as img:
        original_mode = img.mode
        width, height = img.size

        # Convert only for intensity statistics.
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

    audit_rows.append({
        "person": row["person"],
        "yaw": int(row["yaw"]),
        "illumination": int(row["illumination"]),
        "path": row["path"],
        "original_mode": original_mode,
        "width": width,
        "height": height,
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "min_intensity": float(arr.min()),
        "max_intensity": float(arr.max()),
    })

    if (idx + 1) % 2000 == 0:
        print(f"Audited {idx + 1:,} / {len(unique_images):,} images...")

audit_df = pd.DataFrame(audit_rows)

audit_csv = OUTPUT_DIR / "image_audit.csv"
audit_df.to_csv(audit_csv, index=False)

print("\nImage dimensions:")
print(
    audit_df.groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
            .to_string()
)

print("\nImage modes:")
print(audit_df["original_mode"].value_counts().to_string())

print("\nIntensity summary:")
print(
    audit_df[
        ["mean_intensity", "std_intensity", "min_intensity", "max_intensity"]
    ].describe().round(3).to_string()
)


# ============================================================
# 6. PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image_path):
    """
    Fixed preprocessing used for BOTH Model A and Model B.

    1. Read image
    2. Convert to grayscale
    3. Resize to 128x128
    4. Convert to float32
    5. Normalize to [0, 1]

    Returns
    -------
    np.ndarray with shape (128, 128, 1)
    """
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr[..., np.newaxis]


# Quick preprocessing validation.
example_path = Path(train_a.iloc[0]["path"])
example_arr = preprocess_image(example_path)

print("\n" + "=" * 80)
print("PREPROCESSING VALIDATION")
print("=" * 80)

print("Example path :", example_path)
print("Output shape :", example_arr.shape)
print("Output dtype :", example_arr.dtype)
print("Output min   :", float(example_arr.min()))
print("Output max   :", float(example_arr.max()))

assert example_arr.shape == (128, 128, 1)
assert example_arr.dtype == np.float32
assert 0.0 <= example_arr.min() <= 1.0
assert 0.0 <= example_arr.max() <= 1.0

print("Preprocessing validation: PASS")


# ============================================================
# 7. POSE PROGRESSION FIGURE
# ============================================================
#
# Show the same identity + same test illumination from -90 to +90.
# This isolates pose visually while keeping identity and lighting fixed.
# ============================================================

print("\n" + "=" * 80)
print("CREATE POSE PROGRESSION FIGURE")
print("=" * 80)

test_persons = sorted(test["person"].unique())
selected_person = test_persons[0]

test_illums = sorted(test["illumination"].unique())
selected_illum = test_illums[0]

yaw_order = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]

pose_rows = test[
    (test["person"] == selected_person)
    & (test["illumination"] == selected_illum)
].copy()

fig, axes = plt.subplots(1, len(yaw_order), figsize=(22, 3))

for ax, yaw in zip(axes, yaw_order):
    match = pose_rows[pose_rows["yaw"] == yaw]

    if len(match) != 1:
        ax.axis("off")
        ax.set_title(f"{yaw}°\nmissing")
        continue

    arr = preprocess_image(match.iloc[0]["path"])[..., 0]

    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{yaw:+d}°" if yaw != 0 else "0°")
    ax.axis("off")

fig.suptitle(
    f"Pose progression — {selected_person}, illumination {selected_illum}",
    fontsize=14
)
fig.tight_layout()

pose_figure_path = FIGURE_DIR / "pose_progression_same_identity.png"
fig.savefig(pose_figure_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", pose_figure_path.resolve())


# ============================================================
# 8. MODEL A vs MODEL B TRAINING SAMPLE FIGURE
# ============================================================

print("\n" + "=" * 80)
print("CREATE TRAINING SAMPLE FIGURE")
print("=" * 80)

selected_person = sorted(train_a["person"].unique())[0]

a_person = train_a[train_a["person"] == selected_person].copy()
b_person = train_b[train_b["person"] == selected_person].copy()

# Show up to 12 images from each model.
a_person = a_person.sort_values(["yaw", "illumination"]).head(12)
b_person = b_person.sort_values(["yaw", "illumination"]).head(12)

fig, axes = plt.subplots(2, 12, figsize=(18, 4.8))

for col in range(12):
    ax = axes[0, col]

    if col < len(a_person):
        row = a_person.iloc[col]
        arr = preprocess_image(row["path"])[..., 0]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"0°\nI{int(row['illumination']):02d}", fontsize=8)

    ax.axis("off")

for col in range(12):
    ax = axes[1, col]

    if col < len(b_person):
        row = b_person.iloc[col]
        arr = preprocess_image(row["path"])[..., 0]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{int(row['yaw']):+d}°\nI{int(row['illumination']):02d}",
            fontsize=8
        )

    ax.axis("off")

axes[0, 0].set_ylabel("Model A", fontsize=11)
axes[1, 0].set_ylabel("Model B", fontsize=11)

fig.suptitle(
    f"Training samples — {selected_person}\n"
    "Model A: frontal-only | Model B: pose-diverse",
    fontsize=13
)
fig.tight_layout()

train_compare_path = FIGURE_DIR / "model_A_vs_B_training_samples.png"
fig.savefig(train_compare_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", train_compare_path.resolve())


# ============================================================
# 9. MILD AUGMENTATION PREVIEW
# ============================================================

def mild_augment_pil(image, rng):
    """
    Mild appearance/2D acquisition augmentation.

    Does NOT synthesize head yaw.
    Does NOT horizontally flip.
    Does NOT use perspective warping.
    """

    brightness_factor = rng.uniform(*AUG_BRIGHTNESS)
    contrast_factor = rng.uniform(*AUG_CONTRAST)
    rotation_deg = rng.uniform(*AUG_ROTATION_DEG)

    max_dx = IMAGE_SIZE[0] * max(abs(AUG_TRANSLATION_FRAC[0]), abs(AUG_TRANSLATION_FRAC[1]))
    max_dy = IMAGE_SIZE[1] * max(abs(AUG_TRANSLATION_FRAC[0]), abs(AUG_TRANSLATION_FRAC[1]))

    dx = rng.uniform(-max_dx, max_dx)
    dy = rng.uniform(-max_dy, max_dy)

    image = image.convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    image = image.rotate(
        rotation_deg,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0
    )

    # PIL affine mapping uses inverse coordinates.
    image = image.transform(
        IMAGE_SIZE,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0
    )

    return image, {
        "brightness": brightness_factor,
        "contrast": contrast_factor,
        "rotation": rotation_deg,
        "dx": dx,
        "dy": dy,
    }


preview_row = train_b.iloc[0]
preview_path = Path(preview_row["path"])

with Image.open(preview_path) as img:
    original = img.convert("L").resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

rng = np.random.default_rng(SEED)

fig, axes = plt.subplots(2, 4, figsize=(11, 6))

axes = axes.flatten()

axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(
    f"Original\nYaw {int(preview_row['yaw']):+d}°"
)
axes[0].axis("off")

for idx in range(1, 8):
    aug_img, params = mild_augment_pil(original, rng)

    axes[idx].imshow(aug_img, cmap="gray", vmin=0, vmax=255)
    axes[idx].set_title(
        f"Aug {idx}\n"
        f"R={params['rotation']:.1f}°, "
        f"B={params['brightness']:.2f}"
    )
    axes[idx].axis("off")

fig.suptitle(
    "Mild augmentation preview — training only",
    fontsize=14
)
fig.tight_layout()

augmentation_path = FIGURE_DIR / "augmentation_preview.png"
fig.savefig(augmentation_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", augmentation_path.resolve())


# ============================================================
# 10. INTENSITY SUMMARY BY SPLIT
# ============================================================

path_to_stats = audit_df.set_index("path")[
    ["mean_intensity", "std_intensity"]
]

split_stats_rows = []

for split_name, frame in datasets.items():
    temp = frame[["path", "yaw", "illumination"]].copy()

    temp = temp.join(
        path_to_stats,
        on="path"
    )

    split_stats_rows.append({
        "split": split_name,
        "n_images": len(temp),
        "mean_of_image_means": temp["mean_intensity"].mean(),
        "std_of_image_means": temp["mean_intensity"].std(),
        "mean_image_std": temp["std_intensity"].mean(),
    })

split_stats_df = pd.DataFrame(split_stats_rows)

split_stats_csv = OUTPUT_DIR / "split_intensity_summary.csv"
split_stats_df.to_csv(split_stats_csv, index=False)

print("\n" + "=" * 80)
print("INTENSITY SUMMARY BY SPLIT")
print("=" * 80)
print(split_stats_df.round(3).to_string(index=False))


# ============================================================
# 11. SAVE PREPROCESSING MANIFEST
# ============================================================

summary_path = OUTPUT_DIR / "preprocessing_manifest.txt"

dimension_counts = (
    audit_df.groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
)

mode_counts = audit_df["original_mode"].value_counts()

summary_text = f"""ML Term Project — Preprocessing Manifest

Random seed:
{SEED}

Fixed preprocessing for Model A and Model B:
1. Read original image.
2. Convert to grayscale.
3. Resize to {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} pixels.
4. Convert to float32.
5. Normalize pixel intensity to [0, 1].

Output tensor shape:
({IMAGE_SIZE[1]}, {IMAGE_SIZE[0]}, 1)

Augmentation policy under consideration:
- Training only.
- Same augmentation policy for Model A and Model B.
- Brightness factor: {AUG_BRIGHTNESS}
- Contrast factor: {AUG_CONTRAST}
- In-plane rotation: {AUG_ROTATION_DEG} degrees
- Translation fraction: {AUG_TRANSLATION_FRAC}
- Horizontal flip: DISABLED
- Perspective warp: DISABLED
- Synthetic yaw augmentation: DISABLED

Reason:
Head yaw is the main experimental variable. Augmentation should reduce
minor acquisition/appearance overfitting without artificially changing
the left/right yaw distribution or synthesizing pose.

Validation:
No augmentation.

Test:
No augmentation.

Unique images audited:
{len(audit_df):,}

Image dimension counts:
{dimension_counts.to_string()}

Original image mode counts:
{mode_counts.to_string()}

Files produced:
- {audit_csv}
- {split_stats_csv}
- {pose_figure_path}
- {train_compare_path}
- {augmentation_path}
"""

summary_path.write_text(summary_text, encoding="utf-8")


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("Image audit          :", audit_csv.resolve())
print("Intensity summary    :", split_stats_csv.resolve())
print("Pose progression     :", pose_figure_path.resolve())
print("A vs B train samples :", train_compare_path.resolve())
print("Augmentation preview :", augmentation_path.resolve())
print("Manifest             :", summary_path.resolve())

print("\nStep 3 completed successfully.")
