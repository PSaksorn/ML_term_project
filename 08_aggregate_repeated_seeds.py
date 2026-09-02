from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 8 — AGGREGATE REPEATED-SEED RESULTS
# ============================================================
#
# Aggregates:
#   Seed 42   (existing Step 5 result)
#   Seed 123  (Step 7)
#   Seed 2026 (Step 7)
#
# Quantitative final report results become:
#   Mean ± SD across seeds
#
# Qualitative error examples can still use seed 42 as the
# representative run.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEEDS = [42, 123, 2026]

SEED42_RESULT_DIR = Path("outputs") / "final_evaluation" / "results"

REPEATED_ROOT = Path("outputs") / "repeated_seeds"

OUTPUT_DIR = Path("outputs") / "repeated_seed_summary"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

def result_dir_for_seed(seed):
    if seed == 42:
        return SEED42_RESULT_DIR

    return (
        REPEATED_ROOT
        / f"seed_{seed}"
        / "results"
    )


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing repeated-seed result:\n{path.resolve()}\n\n"
            "Run Step 7 first for seeds 123 and 2026."
        )


# ============================================================
# 3. LOAD OVERALL METRICS
# ============================================================

overall_rows = []

for seed in SEEDS:

    result_dir = result_dir_for_seed(seed)

    path = result_dir / "overall_test_metrics.csv"
    require(path)

    df = pd.read_csv(path)

    # Seed 42 Step 5 may not contain a seed column.
    if "seed" not in df.columns:
        df.insert(0, "seed", seed)

    # Normalize model naming.
    df["model"] = (
        df["model"]
        .astype(str)
        .str.replace("Model_A", "Model_A", regex=False)
        .str.replace("Model_B", "Model_B", regex=False)
    )

    overall_rows.append(
        df[
            [
                "seed",
                "model",
                "accuracy",
                "macro_f1",
            ]
        ]
    )

overall_all = pd.concat(
    overall_rows,
    ignore_index=True
)

overall_all.to_csv(
    TABLE_DIR / "per_seed_overall_metrics.csv",
    index=False
)


# ============================================================
# 4. OVERALL MEAN ± SD
# ============================================================

overall_agg = (
    overall_all
    .groupby("model")
    .agg(
        n_seeds=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
    )
    .reset_index()
)

overall_agg["accuracy_mean_percent"] = (
    overall_agg["accuracy_mean"] * 100
)

overall_agg["accuracy_sd_percent"] = (
    overall_agg["accuracy_sd"] * 100
)

overall_agg["macro_f1_mean_percent"] = (
    overall_agg["macro_f1_mean"] * 100
)

overall_agg["macro_f1_sd_percent"] = (
    overall_agg["macro_f1_sd"] * 100
)

overall_agg["accuracy_report"] = overall_agg.apply(
    lambda r:
        f"{r['accuracy_mean_percent']:.2f} ± "
        f"{r['accuracy_sd_percent']:.2f}%",
    axis=1
)

overall_agg["macro_f1_report"] = overall_agg.apply(
    lambda r:
        f"{r['macro_f1_mean_percent']:.2f} ± "
        f"{r['macro_f1_sd_percent']:.2f}%",
    axis=1
)

overall_agg.to_csv(
    TABLE_DIR / "aggregate_overall_metrics.csv",
    index=False
)


# ============================================================
# 5. PAIRED OVERALL B-A GAIN
# ============================================================

overall_pivot = overall_all.pivot(
    index="seed",
    columns="model",
    values=["accuracy", "macro_f1"]
)

overall_gain_rows = []

for seed in SEEDS:

    acc_a = overall_pivot.loc[seed, ("accuracy", "Model_A")]
    acc_b = overall_pivot.loc[seed, ("accuracy", "Model_B")]

    f1_a = overall_pivot.loc[seed, ("macro_f1", "Model_A")]
    f1_b = overall_pivot.loc[seed, ("macro_f1", "Model_B")]

    overall_gain_rows.append({
        "seed": seed,
        "accuracy_gain_B_minus_A": acc_b - acc_a,
        "macro_f1_gain_B_minus_A": f1_b - f1_a,
    })

overall_gain_df = pd.DataFrame(overall_gain_rows)

overall_gain_df.to_csv(
    TABLE_DIR / "per_seed_overall_gain.csv",
    index=False
)

overall_gain_summary = pd.DataFrame([{
    "n_seeds": len(SEEDS),
    "accuracy_gain_mean": overall_gain_df[
        "accuracy_gain_B_minus_A"
    ].mean(),
    "accuracy_gain_sd": overall_gain_df[
        "accuracy_gain_B_minus_A"
    ].std(ddof=1),
    "macro_f1_gain_mean": overall_gain_df[
        "macro_f1_gain_B_minus_A"
    ].mean(),
    "macro_f1_gain_sd": overall_gain_df[
        "macro_f1_gain_B_minus_A"
    ].std(ddof=1),
}])

overall_gain_summary.to_csv(
    TABLE_DIR / "aggregate_overall_gain.csv",
    index=False
)


# ============================================================
# 6. LOAD / AGGREGATE ABSOLUTE-YAW METRICS
# ============================================================

abs_rows = []

for seed in SEEDS:

    path = (
        result_dir_for_seed(seed)
        / "metrics_by_absolute_yaw.csv"
    )

    require(path)

    df = pd.read_csv(path)

    if "seed" not in df.columns:
        df.insert(0, "seed", seed)

    abs_rows.append(df)

abs_all = pd.concat(
    abs_rows,
    ignore_index=True
)

abs_all.to_csv(
    TABLE_DIR / "per_seed_metrics_by_absolute_yaw.csv",
    index=False
)

abs_agg = (
    abs_all
    .groupby("abs_yaw")
    .agg(
        n_seeds=("seed", "nunique"),

        model_A_accuracy_mean=(
            "model_A_accuracy",
            "mean"
        ),
        model_A_accuracy_sd=(
            "model_A_accuracy",
            "std"
        ),

        model_B_accuracy_mean=(
            "model_B_accuracy",
            "mean"
        ),
        model_B_accuracy_sd=(
            "model_B_accuracy",
            "std"
        ),

        model_A_macro_f1_mean=(
            "model_A_macro_f1",
            "mean"
        ),
        model_A_macro_f1_sd=(
            "model_A_macro_f1",
            "std"
        ),

        model_B_macro_f1_mean=(
            "model_B_macro_f1",
            "mean"
        ),
        model_B_macro_f1_sd=(
            "model_B_macro_f1",
            "std"
        ),

        accuracy_gain_mean=(
            "accuracy_gain_B_minus_A",
            "mean"
        ),
        accuracy_gain_sd=(
            "accuracy_gain_B_minus_A",
            "std"
        ),
    )
    .reset_index()
)

abs_agg.to_csv(
    TABLE_DIR / "aggregate_metrics_by_absolute_yaw.csv",
    index=False
)

# Report-ready percentage table.
report_yaw = abs_agg.copy()

for col in [
    "model_A_accuracy_mean",
    "model_A_accuracy_sd",
    "model_B_accuracy_mean",
    "model_B_accuracy_sd",
    "model_A_macro_f1_mean",
    "model_A_macro_f1_sd",
    "model_B_macro_f1_mean",
    "model_B_macro_f1_sd",
    "accuracy_gain_mean",
    "accuracy_gain_sd",
]:
    report_yaw[col] = report_yaw[col] * 100

report_yaw["model_A_accuracy_report"] = report_yaw.apply(
    lambda r:
        f"{r['model_A_accuracy_mean']:.2f} ± "
        f"{r['model_A_accuracy_sd']:.2f}%",
    axis=1
)

report_yaw["model_B_accuracy_report"] = report_yaw.apply(
    lambda r:
        f"{r['model_B_accuracy_mean']:.2f} ± "
        f"{r['model_B_accuracy_sd']:.2f}%",
    axis=1
)

report_yaw["accuracy_gain_report"] = report_yaw.apply(
    lambda r:
        f"{r['accuracy_gain_mean']:.2f} ± "
        f"{r['accuracy_gain_sd']:.2f} pp",
    axis=1
)

report_yaw[
    [
        "abs_yaw",
        "model_A_accuracy_report",
        "model_B_accuracy_report",
        "accuracy_gain_report",
    ]
].to_csv(
    TABLE_DIR / "report_table_accuracy_by_yaw_mean_sd.csv",
    index=False
)


# ============================================================
# 7. POSE CATEGORY AGGREGATION
# ============================================================

category_rows = []

for seed in SEEDS:

    path = (
        result_dir_for_seed(seed)
        / "metrics_by_pose_category.csv"
    )

    require(path)

    df = pd.read_csv(path)

    if "seed" not in df.columns:
        df.insert(0, "seed", seed)

    category_rows.append(df)

category_all = pd.concat(
    category_rows,
    ignore_index=True
)

category_all.to_csv(
    TABLE_DIR / "per_seed_metrics_by_pose_category.csv",
    index=False
)

category_agg = (
    category_all
    .groupby("pose_category")
    .agg(
        n_seeds=("seed", "nunique"),
        model_A_accuracy_mean=("model_A_accuracy", "mean"),
        model_A_accuracy_sd=("model_A_accuracy", "std"),
        model_B_accuracy_mean=("model_B_accuracy", "mean"),
        model_B_accuracy_sd=("model_B_accuracy", "std"),
        accuracy_gain_mean=("accuracy_gain_B_minus_A", "mean"),
        accuracy_gain_sd=("accuracy_gain_B_minus_A", "std"),
        model_A_macro_f1_mean=("model_A_macro_f1", "mean"),
        model_A_macro_f1_sd=("model_A_macro_f1", "std"),
        model_B_macro_f1_mean=("model_B_macro_f1", "mean"),
        model_B_macro_f1_sd=("model_B_macro_f1", "std"),
    )
    .reset_index()
)

category_agg.to_csv(
    TABLE_DIR / "aggregate_metrics_by_pose_category.csv",
    index=False
)


# ============================================================
# 8. EMBEDDING SIMILARITY AGGREGATION
# ============================================================

embedding_rows = []

for seed in SEEDS:

    path = (
        result_dir_for_seed(seed)
        / "embedding_similarity_by_absolute_yaw.csv"
    )

    require(path)

    df = pd.read_csv(path)

    if "seed" not in df.columns:
        df.insert(0, "seed", seed)

    embedding_rows.append(df)

embedding_all = pd.concat(
    embedding_rows,
    ignore_index=True
)

embedding_all.to_csv(
    TABLE_DIR / "per_seed_embedding_similarity.csv",
    index=False
)

embedding_agg = (
    embedding_all
    .groupby("abs_yaw")
    .agg(
        n_seeds=("seed", "nunique"),

        model_A_mean_cosine_mean=(
            "model_A_mean_cosine",
            "mean"
        ),
        model_A_mean_cosine_sd=(
            "model_A_mean_cosine",
            "std"
        ),

        model_B_mean_cosine_mean=(
            "model_B_mean_cosine",
            "mean"
        ),
        model_B_mean_cosine_sd=(
            "model_B_mean_cosine",
            "std"
        ),

        cosine_gain_mean=(
            "cosine_gain_B_minus_A",
            "mean"
        ),
        cosine_gain_sd=(
            "cosine_gain_B_minus_A",
            "std"
        ),
    )
    .reset_index()
)

embedding_agg.to_csv(
    TABLE_DIR / "aggregate_embedding_similarity.csv",
    index=False
)


# ============================================================
# 9. FINAL FIGURES — MEAN ± SD
# ============================================================

x = abs_agg["abs_yaw"].to_numpy()

# Accuracy by yaw
plt.figure(figsize=(9, 6))

plt.plot(
    x,
    abs_agg["model_A_accuracy_mean"],
    marker="o",
    label="Model A — Frontal-only"
)

plt.fill_between(
    x,
    abs_agg["model_A_accuracy_mean"]
    - abs_agg["model_A_accuracy_sd"],
    abs_agg["model_A_accuracy_mean"]
    + abs_agg["model_A_accuracy_sd"],
    alpha=0.20
)

plt.plot(
    x,
    abs_agg["model_B_accuracy_mean"],
    marker="o",
    label="Model B — Pose-diverse"
)

plt.fill_between(
    x,
    abs_agg["model_B_accuracy_mean"]
    - abs_agg["model_B_accuracy_sd"],
    abs_agg["model_B_accuracy_mean"]
    + abs_agg["model_B_accuracy_sd"],
    alpha=0.20
)

plt.xlabel("Absolute yaw angle (degrees)")
plt.ylabel("Top-1 accuracy")
plt.title("Cross-Pose Recognition Accuracy — Mean ± SD Across Seeds")
plt.xticks(x)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

accuracy_fig = FIGURE_DIR / "accuracy_by_yaw_mean_sd.png"

plt.savefig(
    accuracy_fig,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# Accuracy gain
plt.figure(figsize=(9, 6))

plt.plot(
    x,
    abs_agg["accuracy_gain_mean"],
    marker="o"
)

plt.fill_between(
    x,
    abs_agg["accuracy_gain_mean"]
    - abs_agg["accuracy_gain_sd"],
    abs_agg["accuracy_gain_mean"]
    + abs_agg["accuracy_gain_sd"],
    alpha=0.20
)

plt.axhline(0, linewidth=1)

plt.xlabel("Absolute yaw angle (degrees)")
plt.ylabel("Accuracy gain (Model B - Model A)")
plt.title("Pose-Diverse Training Gain — Mean ± SD Across Seeds")
plt.xticks(x)
plt.grid(alpha=0.25)
plt.tight_layout()

gain_fig = FIGURE_DIR / "accuracy_gain_by_yaw_mean_sd.png"

plt.savefig(
    gain_fig,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# Embedding similarity
x_emb = embedding_agg["abs_yaw"].to_numpy()

plt.figure(figsize=(9, 6))

plt.plot(
    x_emb,
    embedding_agg["model_A_mean_cosine_mean"],
    marker="o",
    label="Model A — Frontal-only"
)

plt.fill_between(
    x_emb,
    embedding_agg["model_A_mean_cosine_mean"]
    - embedding_agg["model_A_mean_cosine_sd"],
    embedding_agg["model_A_mean_cosine_mean"]
    + embedding_agg["model_A_mean_cosine_sd"],
    alpha=0.20
)

plt.plot(
    x_emb,
    embedding_agg["model_B_mean_cosine_mean"],
    marker="o",
    label="Model B — Pose-diverse"
)

plt.fill_between(
    x_emb,
    embedding_agg["model_B_mean_cosine_mean"]
    - embedding_agg["model_B_mean_cosine_sd"],
    embedding_agg["model_B_mean_cosine_mean"]
    + embedding_agg["model_B_mean_cosine_sd"],
    alpha=0.20
)

plt.xlabel("Absolute yaw angle (degrees)")
plt.ylabel("Mean same-identity cosine similarity to frontal")
plt.title("Pose-Induced Representation Shift — Mean ± SD Across Seeds")
plt.xticks(x_emb)
plt.ylim(-0.1, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

embedding_fig = FIGURE_DIR / "embedding_similarity_mean_sd.png"

plt.savefig(
    embedding_fig,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FINAL SUMMARY JSON / TXT
# ============================================================

model_a_row = overall_agg[
    overall_agg["model"] == "Model_A"
].iloc[0]

model_b_row = overall_agg[
    overall_agg["model"] == "Model_B"
].iloc[0]

summary = {
    "seeds": SEEDS,
    "n_seeds": len(SEEDS),
    "model_A": {
        "accuracy_mean": float(model_a_row["accuracy_mean"]),
        "accuracy_sd": float(model_a_row["accuracy_sd"]),
        "macro_f1_mean": float(model_a_row["macro_f1_mean"]),
        "macro_f1_sd": float(model_a_row["macro_f1_sd"]),
    },
    "model_B": {
        "accuracy_mean": float(model_b_row["accuracy_mean"]),
        "accuracy_sd": float(model_b_row["accuracy_sd"]),
        "macro_f1_mean": float(model_b_row["macro_f1_mean"]),
        "macro_f1_sd": float(model_b_row["macro_f1_sd"]),
    },
    "overall_accuracy_gain_B_minus_A": {
        "mean": float(
            overall_gain_summary.iloc[0]["accuracy_gain_mean"]
        ),
        "sd": float(
            overall_gain_summary.iloc[0]["accuracy_gain_sd"]
        ),
    },
}

with open(
    TABLE_DIR / "final_repeated_seed_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

text = (
    "Repeated-Seed Final Summary\n"
    "===========================\n\n"
    f"Seeds: {SEEDS}\n\n"
    f"Model A Accuracy: "
    f"{model_a_row['accuracy_mean']*100:.2f} ± "
    f"{model_a_row['accuracy_sd']*100:.2f}%\n"
    f"Model A Macro-F1: "
    f"{model_a_row['macro_f1_mean']*100:.2f} ± "
    f"{model_a_row['macro_f1_sd']*100:.2f}%\n\n"
    f"Model B Accuracy: "
    f"{model_b_row['accuracy_mean']*100:.2f} ± "
    f"{model_b_row['accuracy_sd']*100:.2f}%\n"
    f"Model B Macro-F1: "
    f"{model_b_row['macro_f1_mean']*100:.2f} ± "
    f"{model_b_row['macro_f1_sd']*100:.2f}%\n\n"
    f"Overall Accuracy Gain (B-A): "
    f"{overall_gain_summary.iloc[0]['accuracy_gain_mean']*100:.2f} ± "
    f"{overall_gain_summary.iloc[0]['accuracy_gain_sd']*100:.2f} "
    "percentage points\n"
)

(
    TABLE_DIR / "final_repeated_seed_summary.txt"
).write_text(
    text,
    encoding="utf-8"
)


# ============================================================
# 11. PRINT
# ============================================================

print("=" * 80)
print("STEP 8 — REPEATED-SEED AGGREGATION")
print("=" * 80)

print("\nPer-seed overall results:")
print(
    overall_all.to_string(index=False)
)

print("\nFinal aggregate:")
print(
    overall_agg[
        [
            "model",
            "accuracy_report",
            "macro_f1_report",
        ]
    ].to_string(index=False)
)

print(
    "\nOverall Accuracy Gain B-A: "
    f"{overall_gain_summary.iloc[0]['accuracy_gain_mean']*100:.2f} ± "
    f"{overall_gain_summary.iloc[0]['accuracy_gain_sd']*100:.2f} pp"
)

print("\nFigures:")
print(accuracy_fig.resolve())
print(gain_fig.resolve())
print(embedding_fig.resolve())

print("\nStep 8 completed successfully.")
print("Output:", OUTPUT_DIR.resolve())
