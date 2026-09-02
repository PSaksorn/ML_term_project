from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# STEP 10 — ENLARGED POSE PROGRESSION FOR APPENDIX
# ============================================================
#
# This script follows the SAME data flow used in:
#   01_dataset_audit.py
#   02_create_split.py
#   03_preprocessing_check.py
#
# Fig. A8 is an enlarged version related to Fig. 1.
#
# Important:
# - Uses the SAME test split created in Step 02.
# - Uses the SAME identity-selection rule as Step 03.
# - Uses the SAME held-out test illumination.
# - Shows the SAME yaw range: -90° ... +90°.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"
TEST_CSV = SPLIT_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "appendix_analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "appendix_pose_progression_enlarged.png"

IMAGE_SIZE = (128, 128)

LEFT_YAWS = [-90, -75, -60, -45, -30, -15, 0]
RIGHT_YAWS = [0, 15, 30, 45, 60, 75, 90]

# Keep these as None to use exactly the same automatic
# selection rule as 03_preprocessing_check.py.
#
# Or manually specify:
# FORCE_PERSON = "person001"
# FORCE_ILLUMINATION = 2

FORCE_PERSON = None
FORCE_ILLUMINATION = None


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Same preprocessing used in Step 03:
        grayscale -> 128x128 -> float32 [0,1]
    """
    image_path = Path(image_path)

    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr


# ============================================================
# 3. LOAD TEST SPLIT
# ============================================================

print("=" * 80)
print("STEP 10 — APPENDIX POSE PROGRESSION")
print("=" * 80)

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test split not found:\n{TEST_CSV}\n\n"
        "Run 02_create_split.py first."
    )

test = pd.read_csv(TEST_CSV)

required_columns = {
    "person",
    "yaw",
    "illumination",
    "path",
}

missing_columns = required_columns - set(test.columns)

if missing_columns:
    raise ValueError(
        f"test.csv is missing columns: {sorted(missing_columns)}"
    )

test["yaw"] = test["yaw"].astype(int)
test["illumination"] = test["illumination"].astype(int)

print(f"Test images : {len(test):,}")
print(f"Identities  : {test['person'].nunique()}")
print(f"Yaw values  : {sorted(test['yaw'].unique())}")


# ============================================================
# 4. SELECT SAME CASE AS FIG. 1
# ============================================================

if FORCE_PERSON is None:
    # Same rule as 03_preprocessing_check.py
    selected_person = sorted(test["person"].unique())[0]
else:
    selected_person = FORCE_PERSON

if FORCE_ILLUMINATION is None:
    # Same rule as 03_preprocessing_check.py
    selected_illum = sorted(test["illumination"].unique())[0]
else:
    selected_illum = int(FORCE_ILLUMINATION)

pose_rows = test[
    (test["person"] == selected_person)
    & (test["illumination"] == selected_illum)
].copy()

expected_yaws = set(LEFT_YAWS + RIGHT_YAWS)
available_yaws = set(pose_rows["yaw"].unique())

missing_yaws = expected_yaws - available_yaws

if missing_yaws:
    raise ValueError(
        f"Selected case does not contain all required yaw angles.\n"
        f"Person       : {selected_person}\n"
        f"Illumination : {selected_illum}\n"
        f"Missing yaw  : {sorted(missing_yaws)}"
    )

print("\nSelected case")
print("-" * 80)
print("Person       :", selected_person)
print("Illumination :", selected_illum)
print("Yaw values   :", sorted(available_yaws))


# ============================================================
# 5. HELPER — DRAW ONE IMAGE
# ============================================================

def draw_pose(ax, yaw):
    match = pose_rows[pose_rows["yaw"] == yaw]

    if len(match) != 1:
        ax.axis("off")
        ax.set_title(f"{yaw:+d}°\nMissing", fontsize=12)
        return

    image_path = Path(match.iloc[0]["path"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    arr = preprocess_image(image_path)

    ax.imshow(
        arr,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    if yaw == 0:
        title = "0°"
    else:
        title = f"{yaw:+d}°"

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.axis("off")


# ============================================================
# 6. CREATE ENLARGED APPENDIX FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    7,
    figsize=(15, 6.6)
)

# ------------------------------------------------------------
# Top row: left profile -> frontal
# ------------------------------------------------------------

for ax, yaw in zip(axes[0], LEFT_YAWS):
    draw_pose(ax, yaw)

# ------------------------------------------------------------
# Bottom row: frontal -> right profile
# ------------------------------------------------------------

for ax, yaw in zip(axes[1], RIGHT_YAWS):
    draw_pose(ax, yaw)

# Row labels
fig.text(
    0.015,
    0.69,
    "Left rotation",
    rotation=90,
    va="center",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

fig.text(
    0.015,
    0.28,
    "Right rotation",
    rotation=90,
    va="center",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

fig.suptitle(
    "Detailed Pose Progression of the Same Identity",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.925,
    f"{selected_person} | held-out illumination {selected_illum}",
    ha="center",
    fontsize=11
)

plt.subplots_adjust(
    left=0.055,
    right=0.99,
    top=0.86,
    bottom=0.04,
    wspace=0.08,
    hspace=0.22
)

fig.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 7. DONE
# ============================================================

print("\n" + "=" * 80)
print("FIGURE SAVED")
print("=" * 80)
print(OUTPUT_IMAGE.resolve())

print("\nThis figure uses:")
print(f"  Person       : {selected_person}")
print(f"  Illumination : {selected_illum}")
print("\nIt should correspond to the same selection rule used")
print("for Fig. 1 in 03_preprocessing_check.py.")