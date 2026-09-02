from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# STEP 2 — CREATE CONTROLLED TRAIN / VAL / TEST SPLITS
# ============================================================
#
# Experimental design
# -------------------
# Model A:
#   Frontal-only training (0°)
#
# Model B:
#   Pose-diverse training using 0°, ±30°, ±60°
#
# Important:
#   - Model A and Model B use the SAME number of original
#     training images per identity.
#   - Both models are evaluated on the SAME validation/test sets.
#   - Validation/Test illuminations are held out from training.
#   - Test covers all poses from -90° to +90°.
#
# Expected dataset:
#   153 identities
#   13 signed yaw conditions
#   20 illumination conditions per identity/pose
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42

INPUT_CSV = Path("outputs") / "dataset_index.csv"

SPLIT_DIR = Path("outputs") / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ILLUM = list(range(1, 21))

# Global illumination split
N_TRAIN_ILLUM = 12
N_VAL_ILLUM = 4
N_TEST_ILLUM = 4

# Model B training pose allocation per identity.
#
# Total = 4 + 2 + 2 + 2 + 2 = 12 images/person
#
# Absolute yaw balance:
#   0°  -> 4 images
#   30° -> 4 images total (+30 and -30)
#   60° -> 4 images total (+60 and -60)
#
MODEL_B_POSE_COUNTS = {
    0: 4,
    -30: 2,
    30: 2,
    -60: 2,
    60: 2,
}

# Common validation poses
VAL_YAWS = [0, -30, 30, -60, 60]

# Common final test poses
TEST_YAWS = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]


# ============================================================
# 2. LOAD DATASET INDEX
# ============================================================

print("=" * 80)
print("STEP 2 — CREATE DATA SPLITS")
print("=" * 80)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Dataset index not found:\n{INPUT_CSV.resolve()}\n\n"
        "Run 01_dataset_audit.py first."
    )

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "person",
    "direction",
    "angle",
    "yaw",
    "illumination",
    "filename",
    "path",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"dataset_index.csv is missing columns: {sorted(missing_columns)}"
    )

df["illumination"] = df["illumination"].astype(int)
df["yaw"] = df["yaw"].astype(int)

persons = sorted(df["person"].unique())

print(f"Images loaded : {len(df):,}")
print(f"Identities    : {len(persons)}")
print(f"Illuminations : {sorted(df['illumination'].unique())}")
print(f"Yaw values    : {sorted(df['yaw'].unique())}")


# ============================================================
# 3. VALIDATE DATASET BEFORE SPLITTING
# ============================================================

actual_illum = sorted(df["illumination"].unique())

if actual_illum != EXPECTED_ILLUM:
    raise ValueError(
        "Expected illumination IDs 1-20, but found:\n"
        f"{actual_illum}"
    )

expected_yaws = set(TEST_YAWS)
actual_yaws = set(df["yaw"].unique())

missing_yaws = expected_yaws - actual_yaws

if missing_yaws:
    raise ValueError(
        f"Dataset is missing required yaw conditions: {sorted(missing_yaws)}"
    )

# Each person × yaw should have exactly 20 illumination images.
group_counts = df.groupby(["person", "yaw"]).size()

if not (group_counts == 20).all():
    bad = group_counts[group_counts != 20]

    raise ValueError(
        "Some person/yaw groups do not contain exactly 20 images.\n"
        f"{bad.head(20)}"
    )

print("\nDataset validation: PASS")


# ============================================================
# 4. CREATE GLOBAL ILLUMINATION SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

illum_ids = np.array(EXPECTED_ILLUM)
shuffled_illum = rng.permutation(illum_ids)

train_illum = sorted(shuffled_illum[:N_TRAIN_ILLUM].tolist())
val_illum = sorted(
    shuffled_illum[
        N_TRAIN_ILLUM:
        N_TRAIN_ILLUM + N_VAL_ILLUM
    ].tolist()
)
test_illum = sorted(
    shuffled_illum[
        N_TRAIN_ILLUM + N_VAL_ILLUM:
    ].tolist()
)

print("\n" + "=" * 80)
print("ILLUMINATION SPLIT")
print("=" * 80)

print("Random seed       :", SEED)
print("Train illumination:", train_illum)
print("Val illumination  :", val_illum)
print("Test illumination :", test_illum)

# Confirm no overlap.
assert set(train_illum).isdisjoint(val_illum)
assert set(train_illum).isdisjoint(test_illum)
assert set(val_illum).isdisjoint(test_illum)

assert (
    set(train_illum)
    | set(val_illum)
    | set(test_illum)
) == set(EXPECTED_ILLUM)


# Save illumination split definition.
illum_split_rows = []

for illum in EXPECTED_ILLUM:
    if illum in train_illum:
        split = "train"
    elif illum in val_illum:
        split = "validation"
    else:
        split = "test"

    illum_split_rows.append({
        "illumination": illum,
        "split": split,
    })

illum_split_df = pd.DataFrame(illum_split_rows)

illum_split_df.to_csv(
    SPLIT_DIR / "illumination_split.csv",
    index=False
)


# ============================================================
# 5. MODEL A — FRONTAL-ONLY TRAINING
# ============================================================

train_a = df[
    (df["yaw"] == 0)
    & (df["illumination"].isin(train_illum))
].copy()

train_a["experiment"] = "Model_A"
train_a["split"] = "train"
train_a["training_condition"] = "frontal_only"

# Validation checks.
expected_train_per_person = N_TRAIN_ILLUM

a_counts = train_a.groupby("person").size()

if not (a_counts == expected_train_per_person).all():
    raise ValueError(
        "Model A does not contain exactly "
        f"{expected_train_per_person} training images per identity."
    )


# ============================================================
# 6. MODEL B — POSE-DIVERSE TRAINING
# ============================================================
#
# Fairness strategy:
#
# Model A sees all 12 training illumination IDs once/person.
#
# Model B ALSO sees exactly those same 12 illumination IDs
# once/person, but each illumination is assigned to one of:
#
#   0°, -30°, +30°, -60°, +60°
#
# Illumination-to-pose assignment is shuffled independently
# for each identity so a particular lighting condition is not
# systematically tied to a particular pose across all subjects.
#
# ============================================================

pose_assignment_template = []

for yaw, count in MODEL_B_POSE_COUNTS.items():
    pose_assignment_template.extend([yaw] * count)

if len(pose_assignment_template) != N_TRAIN_ILLUM:
    raise ValueError(
        "MODEL_B_POSE_COUNTS must sum to "
        f"{N_TRAIN_ILLUM}, but currently sums to "
        f"{len(pose_assignment_template)}."
    )

train_b_rows = []

for person_index, person in enumerate(persons):

    # Deterministic but different assignment for each identity.
    person_rng = np.random.default_rng(SEED + person_index + 1)

    person_illum = np.array(train_illum)
    person_illum = person_rng.permutation(person_illum)

    person_pose_assignment = np.array(
        pose_assignment_template,
        dtype=int
    )

    # Shuffle the pose labels too so there is no ordering effect.
    person_pose_assignment = person_rng.permutation(
        person_pose_assignment
    )

    for illum, yaw in zip(
        person_illum,
        person_pose_assignment
    ):

        match = df[
            (df["person"] == person)
            & (df["yaw"] == int(yaw))
            & (df["illumination"] == int(illum))
        ]

        if len(match) != 1:
            raise ValueError(
                "Expected exactly one image for:\n"
                f"person={person}, yaw={yaw}, illumination={illum}\n"
                f"Found {len(match)}"
            )

        train_b_rows.append(
            match.iloc[0].to_dict()
        )

train_b = pd.DataFrame(train_b_rows)

train_b["experiment"] = "Model_B"
train_b["split"] = "train"
train_b["training_condition"] = "pose_diverse"

# Validate Model B total training budget.
b_counts = train_b.groupby("person").size()

if not (b_counts == expected_train_per_person).all():
    raise ValueError(
        "Model B does not contain exactly "
        f"{expected_train_per_person} training images per identity."
    )

# Validate pose allocation for every identity.
expected_pose_counts = MODEL_B_POSE_COUNTS

for person, group in train_b.groupby("person"):

    actual = group["yaw"].value_counts().to_dict()

    for yaw, expected_count in expected_pose_counts.items():

        if actual.get(yaw, 0) != expected_count:
            raise ValueError(
                f"Model B pose allocation error for {person}.\n"
                f"Expected yaw {yaw}: {expected_count}, "
                f"found {actual.get(yaw, 0)}"
            )

# Confirm Model B uses every training illumination exactly once
# per identity.
for person, group in train_b.groupby("person"):

    person_illums = sorted(group["illumination"].tolist())

    if person_illums != train_illum:
        raise ValueError(
            f"Model B illumination mismatch for {person}.\n"
            f"Expected: {train_illum}\n"
            f"Found   : {person_illums}"
        )


# ============================================================
# 7. COMMON VALIDATION SET
# ============================================================

validation = df[
    (df["illumination"].isin(val_illum))
    & (df["yaw"].isin(VAL_YAWS))
].copy()

validation["experiment"] = "common"
validation["split"] = "validation"
validation["training_condition"] = "common_validation"

expected_val_per_person = len(val_illum) * len(VAL_YAWS)

val_counts = validation.groupby("person").size()

if not (val_counts == expected_val_per_person).all():
    raise ValueError(
        "Validation set size is inconsistent across identities."
    )


# ============================================================
# 8. COMMON TEST SET
# ============================================================

test = df[
    (df["illumination"].isin(test_illum))
    & (df["yaw"].isin(TEST_YAWS))
].copy()

test["experiment"] = "common"
test["split"] = "test"
test["training_condition"] = "common_test"

expected_test_per_person = len(test_illum) * len(TEST_YAWS)

test_counts = test.groupby("person").size()

if not (test_counts == expected_test_per_person).all():
    raise ValueError(
        "Test set size is inconsistent across identities."
    )


# ============================================================
# 9. LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE CHECK")
print("=" * 80)

train_a_paths = set(train_a["path"])
train_b_paths = set(train_b["path"])
val_paths = set(validation["path"])
test_paths = set(test["path"])

# A and B are separate experimental training conditions and
# are allowed to share some training samples.
#
# However, neither training set may overlap validation/test.

checks = {
    "Model A vs Validation": train_a_paths & val_paths,
    "Model A vs Test": train_a_paths & test_paths,
    "Model B vs Validation": train_b_paths & val_paths,
    "Model B vs Test": train_b_paths & test_paths,
    "Validation vs Test": val_paths & test_paths,
}

leak_found = False

for name, overlap in checks.items():

    if overlap:
        leak_found = True
        print(
            f"FAIL — {name}: "
            f"{len(overlap)} overlapping images"
        )
    else:
        print(f"PASS — {name}: 0 overlapping images")

if leak_found:
    raise RuntimeError(
        "Data leakage detected. Split files were not saved."
    )


# ============================================================
# 10. ADD HELPER LABELS FOR ANALYSIS
# ============================================================

def add_analysis_columns(dataframe):

    dataframe = dataframe.copy()

    dataframe["abs_yaw"] = dataframe["yaw"].abs()

    dataframe["pose_group"] = np.where(
        dataframe["yaw"] == 0,
        "frontal",
        np.where(
            dataframe["yaw"] < 0,
            "left",
            "right"
        )
    )

    dataframe["yaw_status_for_model_b"] = dataframe["yaw"].map(
        lambda y:
            "seen_training_pose"
            if y in MODEL_B_POSE_COUNTS
            else (
                "interpolation_pose"
                if abs(y) in [15, 45]
                else "extreme_extrapolation_pose"
            )
    )

    return dataframe


train_a = add_analysis_columns(train_a)
train_b = add_analysis_columns(train_b)
validation = add_analysis_columns(validation)
test = add_analysis_columns(test)


# ============================================================
# 11. SORT FOR REPRODUCIBILITY
# ============================================================

sort_columns = [
    "person",
    "yaw",
    "illumination",
    "filename",
]

train_a = train_a.sort_values(sort_columns).reset_index(drop=True)
train_b = train_b.sort_values(sort_columns).reset_index(drop=True)
validation = validation.sort_values(sort_columns).reset_index(drop=True)
test = test.sort_values(sort_columns).reset_index(drop=True)


# ============================================================
# 12. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SPLIT SUMMARY")
print("=" * 80)

print("\nMODEL A — FRONTAL ONLY")
print("-" * 40)
print(f"Total images        : {len(train_a):,}")
print(f"Images / identity   : {len(train_a) // len(persons)}")
print("Yaw distribution:")
print(train_a["yaw"].value_counts().sort_index())

print("\nMODEL B — POSE DIVERSE")
print("-" * 40)
print(f"Total images        : {len(train_b):,}")
print(f"Images / identity   : {len(train_b) // len(persons)}")
print("Yaw distribution:")
print(train_b["yaw"].value_counts().sort_index())

print("\nMODEL B — ABSOLUTE YAW DISTRIBUTION")
print("-" * 40)
print(train_b["abs_yaw"].value_counts().sort_index())

print("\nCOMMON VALIDATION")
print("-" * 40)
print(f"Total images        : {len(validation):,}")
print(f"Images / identity   : {len(validation) // len(persons)}")
print("Yaw distribution:")
print(validation["yaw"].value_counts().sort_index())

print("\nCOMMON TEST")
print("-" * 40)
print(f"Total images        : {len(test):,}")
print(f"Images / identity   : {len(test) // len(persons)}")
print("Yaw distribution:")
print(test["yaw"].value_counts().sort_index())


# ============================================================
# 13. SAVE CSV FILES
# ============================================================

train_a_path = SPLIT_DIR / "train_model_A.csv"
train_b_path = SPLIT_DIR / "train_model_B.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train_a.to_csv(train_a_path, index=False)
train_b.to_csv(train_b_path, index=False)
validation.to_csv(validation_path, index=False)
test.to_csv(test_path, index=False)


# ============================================================
# 14. SAVE SPLIT SUMMARY
# ============================================================

summary_rows = [
    {
        "dataset": "train_model_A",
        "images": len(train_a),
        "identities": train_a["person"].nunique(),
        "images_per_identity": len(train_a) // len(persons),
        "illumination_ids": ",".join(map(str, train_illum)),
        "yaw_conditions": "0",
    },
    {
        "dataset": "train_model_B",
        "images": len(train_b),
        "identities": train_b["person"].nunique(),
        "images_per_identity": len(train_b) // len(persons),
        "illumination_ids": ",".join(map(str, train_illum)),
        "yaw_conditions": "0,-30,+30,-60,+60",
    },
    {
        "dataset": "validation",
        "images": len(validation),
        "identities": validation["person"].nunique(),
        "images_per_identity": len(validation) // len(persons),
        "illumination_ids": ",".join(map(str, val_illum)),
        "yaw_conditions": "0,-30,+30,-60,+60",
    },
    {
        "dataset": "test",
        "images": len(test),
        "identities": test["person"].nunique(),
        "images_per_identity": len(test) // len(persons),
        "illumination_ids": ",".join(map(str, test_illum)),
        "yaw_conditions": (
            "-90,-75,-60,-45,-30,-15,"
            "0,+15,+30,+45,+60,+75,+90"
        ),
    },
]

summary_df = pd.DataFrame(summary_rows)

summary_path = SPLIT_DIR / "split_summary.csv"

summary_df.to_csv(summary_path, index=False)


# ============================================================
# 15. SAVE HUMAN-READABLE SPLIT MANIFEST
# ============================================================

manifest_path = SPLIT_DIR / "split_manifest.txt"

manifest = f"""ML Term Project — Data Split Manifest

Random seed:
{SEED}

Dataset:
{len(df):,} total images
{len(persons)} identities
20 illumination conditions
13 signed yaw conditions

Illumination split:
Train      ({len(train_illum)}): {train_illum}
Validation ({len(val_illum)}): {val_illum}
Test       ({len(test_illum)}): {test_illum}

Model A — Frontal-only:
Training poses: 0°
Images/person: {expected_train_per_person}
Total images: {len(train_a):,}

Model B — Pose-diverse:
Training poses:
  0°   : 4 images/person
  -30° : 2 images/person
  +30° : 2 images/person
  -60° : 2 images/person
  +60° : 2 images/person

Images/person: {expected_train_per_person}
Total images: {len(train_b):,}

Validation:
Poses: {VAL_YAWS}
Held-out illumination IDs: {val_illum}
Images/person: {expected_val_per_person}
Total images: {len(validation):,}

Test:
Poses: {TEST_YAWS}
Held-out illumination IDs: {test_illum}
Images/person: {expected_test_per_person}
Total images: {len(test):,}

Model B interpretation:
Seen training poses:
  0°, ±30°, ±60°

Unseen interpolation poses:
  ±15°, ±45°

Extreme extrapolation poses:
  ±75°, ±90°

Important controls:
- Model A and Model B have equal original-image training budgets.
- Both use the same 12 training illumination conditions.
- Validation and test illuminations are fully held out from training.
- Both models use exactly the same validation and test sets.
- Augmentation has NOT been applied at this stage.
"""

manifest_path.write_text(
    manifest,
    encoding="utf-8"
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("Illumination split :", (SPLIT_DIR / "illumination_split.csv").resolve())
print("Model A train      :", train_a_path.resolve())
print("Model B train      :", train_b_path.resolve())
print("Validation         :", validation_path.resolve())
print("Test               :", test_path.resolve())
print("Summary            :", summary_path.resolve())
print("Manifest           :", manifest_path.resolve())

print("\nStep 2 completed successfully.")
