from pathlib import Path
import re
import pandas as pd

DATASET_ROOT = Path(r"path/to/dataset")

OUTPUT_DIR = Path(r"outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_ILLUM = set(range(1, 21))

print("=" * 80)
print("DATASET PATH CHECK")
print("=" * 80)
print("Dataset root :", DATASET_ROOT)
print("Exists       :", DATASET_ROOT.exists())

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATASET_ROOT}\nPlease check DATASET_ROOT.")

def parse_image_info(image_path):
    person = image_path.parents[1].name
    folder = image_path.parent.name.lower()
    stem = image_path.stem.lower()

    frontal_match = re.fullmatch(r"frontal[_-]?(\d+)", stem)
    if frontal_match:
        illumination = int(frontal_match.group(1))
        return {
            "person": person,
            "direction": "frontal",
            "angle": 0,
            "yaw": 0,
            "illumination": illumination,
            "filename": image_path.name,
            "path": str(image_path)
        }

    side_match = re.fullmatch(r"(\d+)_degree[_-]?(\d+)", stem)
    if side_match:
        angle = int(side_match.group(1))
        illumination = int(side_match.group(2))

        if folder == "left":
            direction = "left"
            yaw = -angle
        elif folder == "right":
            direction = "right"
            yaw = angle
        else:
            direction = folder
            yaw = None

        return {
            "person": person,
            "direction": direction,
            "angle": angle,
            "yaw": yaw,
            "illumination": illumination,
            "filename": image_path.name,
            "path": str(image_path)
        }

    return None

records = []
unparsed_files = []

for image_path in DATASET_ROOT.rglob("*"):
    if not image_path.is_file():
        continue
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    info = parse_image_info(image_path)
    if info is None:
        unparsed_files.append(str(image_path))
    else:
        records.append(info)

df = pd.DataFrame(records)

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Parsed images   : {len(df):,}")
print(f"Unparsed images : {len(unparsed_files):,}")

if len(df) == 0:
    raise RuntimeError("No images were parsed. Check folder structure and filename format.")

print(f"Identities      : {df['person'].nunique()}")
print(f"Angles found    : {sorted(df['angle'].unique())}")
print(f"Yaw values      : {sorted(df['yaw'].dropna().unique())}")
print(f"Illum IDs found : {sorted(df['illumination'].unique())}")

print("\n" + "=" * 80)
print("IMAGE COUNTS BY POSE")
print("=" * 80)

pose_summary = (
    df.groupby(["direction", "angle"])
      .agg(
          images=("filename", "count"),
          identities=("person", "nunique"),
          min_illum=("illumination", "min"),
          max_illum=("illumination", "max")
      )
      .reset_index()
      .sort_values(["angle", "direction"])
)
print(pose_summary.to_string(index=False))

print("\n" + "=" * 80)
print("IMAGES PER PERSON / POSE")
print("=" * 80)

counts = (
    df.groupby(["person", "direction", "angle"])
      .size()
      .reset_index(name="n_images")
)
print(counts["n_images"].value_counts().sort_index())
print("\nMinimum images per person/pose :", counts["n_images"].min())
print("Maximum images per person/pose :", counts["n_images"].max())

print("\n" + "=" * 80)
print("ILLUMINATION COMPLETENESS CHECK")
print("=" * 80)

illumination_problems = []

for (person, direction, angle), group in df.groupby(["person", "direction", "angle"]):
    found_illum = set(group["illumination"].tolist())
    missing = sorted(EXPECTED_ILLUM - found_illum)
    extra = sorted(found_illum - EXPECTED_ILLUM)

    if missing or extra:
        illumination_problems.append({
            "person": person,
            "direction": direction,
            "angle": angle,
            "n_images": len(group),
            "missing": missing,
            "extra": extra
        })

illumination_problem_df = pd.DataFrame(illumination_problems)

if len(illumination_problem_df) == 0:
    print("PASS")
    print("Every person/pose has illumination IDs 01-20.")
else:
    print("WARNING")
    print(f"Found {len(illumination_problem_df)} person/pose combinations with illumination problems.")
    print("\nFirst 30 problems:")
    print(illumination_problem_df.head(30).to_string(index=False))

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

duplicates = df[
    df.duplicated(
        subset=["person", "direction", "angle", "illumination"],
        keep=False
    )
].sort_values(["person", "angle", "direction", "illumination"])

if len(duplicates) == 0:
    print("PASS")
    print("No duplicate person/pose/illumination records.")
else:
    print("WARNING")
    print(f"Duplicate records found: {len(duplicates)}")
    print(
        duplicates[
            ["person", "direction", "angle", "illumination", "filename"]
        ].head(30).to_string(index=False)
    )

print("\n" + "=" * 80)
print("POSE CONDITIONS PER PERSON")
print("=" * 80)

pose_conditions = (
    df[["person", "direction", "angle"]]
      .drop_duplicates()
      .groupby("person")
      .size()
)
print(pose_conditions.value_counts().sort_index())
print("\nMinimum pose conditions/person :", pose_conditions.min())
print("Maximum pose conditions/person :", pose_conditions.max())

print("\n" + "=" * 80)
print("SIGNED YAW SUMMARY")
print("=" * 80)

yaw_summary = (
    df.groupby("yaw")
      .agg(
          images=("filename", "count"),
          identities=("person", "nunique")
      )
      .reset_index()
      .sort_values("yaw")
)
print(yaw_summary.to_string(index=False))

if len(unparsed_files) > 0:
    print("\n" + "=" * 80)
    print("UNPARSED FILES")
    print("=" * 80)

    for path in unparsed_files[:30]:
        print(path)

    if len(unparsed_files) > 30:
        print(f"... and {len(unparsed_files) - 30} more.")

dataset_index_path = OUTPUT_DIR / "dataset_index.csv"
pose_summary_path = OUTPUT_DIR / "pose_summary.csv"

df.to_csv(dataset_index_path, index=False)
pose_summary.to_csv(pose_summary_path, index=False)

if len(illumination_problem_df) > 0:
    illumination_problem_df.to_csv(OUTPUT_DIR / "illumination_problems.csv", index=False)

if len(duplicates) > 0:
    duplicates.to_csv(OUTPUT_DIR / "duplicate_records.csv", index=False)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Dataset index :", dataset_index_path.resolve())
print("Pose summary  :", pose_summary_path.resolve())
print("\nDataset audit completed.")
