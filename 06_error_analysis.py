from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

import matplotlib.pyplot as plt


# ============================================================
# STEP 6 — ERROR ANALYSIS + REPORT-READY FIGURES/TABLES
# ============================================================
#
# This script does NOT train and does NOT re-run inference.
# It consumes outputs from Step 5.
#
# Main goals:
# 1) Quantify where Model A and Model B succeed/fail.
# 2) Break failures down by yaw.
# 3) Inspect left/right asymmetry.
# 4) Inspect confidence on correct vs incorrect predictions.
# 5) Find the most frequent identity-confusion pairs.
# 6) Create representative image grids for:
#       A wrong / B right
#       A right / B wrong
#       Both wrong
# 7) Save report-ready tables and figures.
# ============================================================


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
N_EXAMPLES_PER_CATEGORY = 12

STEP5_DIR = Path("outputs") / "final_evaluation"
RESULT_DIR = STEP5_DIR / "results"
ERROR_INPUT_DIR = STEP5_DIR / "error_analysis"

MODEL_A_PRED = RESULT_DIR / "model_A_test_predictions.csv"
MODEL_B_PRED = RESULT_DIR / "model_B_test_predictions.csv"
ABS_YAW_METRICS = RESULT_DIR / "metrics_by_absolute_yaw.csv"
SIGNED_YAW_METRICS = RESULT_DIR / "metrics_by_signed_yaw.csv"
POSE_CATEGORY_METRICS = RESULT_DIR / "metrics_by_pose_category.csv"

OUTPUT_DIR = Path("outputs") / "final_analysis"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXAMPLE_DIR = OUTPUT_DIR / "examples"

for d in [TABLE_DIR, FIGURE_DIR, EXAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

required = [
    MODEL_A_PRED,
    MODEL_B_PRED,
    ABS_YAW_METRICS,
    SIGNED_YAW_METRICS,
    POSE_CATEGORY_METRICS,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step 5 output:\n{path.resolve()}\n\n"
            "Run 05_final_evaluation.py first."
        )

a = pd.read_csv(MODEL_A_PRED)
b = pd.read_csv(MODEL_B_PRED)

abs_metrics = pd.read_csv(ABS_YAW_METRICS)
signed_metrics = pd.read_csv(SIGNED_YAW_METRICS)
pose_category_metrics = pd.read_csv(POSE_CATEGORY_METRICS)

print("=" * 80)
print("STEP 6 — ERROR ANALYSIS")
print("=" * 80)
print(f"Model A predictions : {len(a):,}")
print(f"Model B predictions : {len(b):,}")


# ============================================================
# 3. MERGE MODEL A / MODEL B PREDICTIONS
# ============================================================

merge_keys = [
    "person",
    "yaw",
    "illumination",
    "path",
]

a_small = a[
    merge_keys
    + [
        "predicted_person",
        "predicted_label",
        "correct",
        "confidence",
        "abs_yaw",
    ]
].rename(columns={
    "predicted_person": "A_predicted_person",
    "predicted_label": "A_predicted_label",
    "correct": "A_correct",
    "confidence": "A_confidence",
    "abs_yaw": "A_abs_yaw",
})

b_small = b[
    merge_keys
    + [
        "predicted_person",
        "predicted_label",
        "correct",
        "confidence",
        "abs_yaw",
    ]
].rename(columns={
    "predicted_person": "B_predicted_person",
    "predicted_label": "B_predicted_label",
    "correct": "B_correct",
    "confidence": "B_confidence",
    "abs_yaw": "B_abs_yaw",
})

merged = pd.merge(
    a_small,
    b_small,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

if len(merged) != len(a) or len(merged) != len(b):
    raise RuntimeError(
        "Merged prediction count does not match original predictions."
    )

merged["abs_yaw"] = merged["yaw"].abs()

merged["outcome"] = np.select(
    [
        (merged["A_correct"] == 1) & (merged["B_correct"] == 1),
        (merged["A_correct"] == 0) & (merged["B_correct"] == 1),
        (merged["A_correct"] == 1) & (merged["B_correct"] == 0),
        (merged["A_correct"] == 0) & (merged["B_correct"] == 0),
    ],
    [
        "Both correct",
        "A wrong / B right",
        "A right / B wrong",
        "Both wrong",
    ],
    default="Unknown",
)

merged.to_csv(
    TABLE_DIR / "all_prediction_comparison.csv",
    index=False
)


# ============================================================
# 4. OVERALL OUTCOME SUMMARY
# ============================================================

outcome_order = [
    "Both correct",
    "A wrong / B right",
    "A right / B wrong",
    "Both wrong",
]

overall_outcomes = (
    merged["outcome"]
    .value_counts()
    .reindex(outcome_order, fill_value=0)
    .rename_axis("outcome")
    .reset_index(name="count")
)

overall_outcomes["percent"] = (
    overall_outcomes["count"]
    / len(merged)
    * 100
)

overall_outcomes.to_csv(
    TABLE_DIR / "overall_outcome_summary.csv",
    index=False
)

print("\n" + "=" * 80)
print("OVERALL A/B OUTCOMES")
print("=" * 80)
print(overall_outcomes.to_string(index=False))


# ============================================================
# 5. OUTCOME SUMMARY BY ABSOLUTE YAW
# ============================================================

yaw_outcomes = (
    merged
    .groupby(["abs_yaw", "outcome"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=outcome_order, fill_value=0)
    .reset_index()
)

yaw_outcomes["total"] = yaw_outcomes[outcome_order].sum(axis=1)

for column in outcome_order:
    yaw_outcomes[f"{column}_percent"] = (
        yaw_outcomes[column]
        / yaw_outcomes["total"]
        * 100
    )

yaw_outcomes.to_csv(
    TABLE_DIR / "outcomes_by_absolute_yaw.csv",
    index=False
)


# ============================================================
# 6. ERROR RATE BY SIGNED YAW
# ============================================================

signed_error_rows = []

for yaw, group in merged.groupby("yaw"):
    signed_error_rows.append({
        "yaw": int(yaw),
        "n_images": len(group),
        "model_A_error_rate": 1.0 - group["A_correct"].mean(),
        "model_B_error_rate": 1.0 - group["B_correct"].mean(),
        "model_A_mean_confidence": group["A_confidence"].mean(),
        "model_B_mean_confidence": group["B_confidence"].mean(),
    })

signed_error_df = (
    pd.DataFrame(signed_error_rows)
    .sort_values("yaw")
)

signed_error_df.to_csv(
    TABLE_DIR / "error_rate_by_signed_yaw.csv",
    index=False
)


# ============================================================
# 7. LEFT / RIGHT ASYMMETRY
# ============================================================

left_right_rows = []

for abs_yaw in sorted(
    x for x in merged["abs_yaw"].unique()
    if x != 0
):
    neg = merged[merged["yaw"] == -abs_yaw]
    pos = merged[merged["yaw"] == abs_yaw]

    if len(neg) == 0 or len(pos) == 0:
        continue

    A_neg = neg["A_correct"].mean()
    A_pos = pos["A_correct"].mean()
    B_neg = neg["B_correct"].mean()
    B_pos = pos["B_correct"].mean()

    left_right_rows.append({
        "abs_yaw": int(abs_yaw),
        "A_negative_yaw_accuracy": A_neg,
        "A_positive_yaw_accuracy": A_pos,
        "A_positive_minus_negative": A_pos - A_neg,
        "B_negative_yaw_accuracy": B_neg,
        "B_positive_yaw_accuracy": B_pos,
        "B_positive_minus_negative": B_pos - B_neg,
    })

left_right_df = pd.DataFrame(left_right_rows)

left_right_df.to_csv(
    TABLE_DIR / "left_right_asymmetry.csv",
    index=False
)


# ============================================================
# 8. CONFIDENCE ANALYSIS
# ============================================================

confidence_rows = []

for model_prefix in ["A", "B"]:
    for abs_yaw, group in merged.groupby("abs_yaw"):
        for correctness in [0, 1]:
            subset = group[group[f"{model_prefix}_correct"] == correctness]

            if len(subset) == 0:
                mean_conf = np.nan
                median_conf = np.nan
            else:
                mean_conf = subset[f"{model_prefix}_confidence"].mean()
                median_conf = subset[f"{model_prefix}_confidence"].median()

            confidence_rows.append({
                "model": f"Model_{model_prefix}",
                "abs_yaw": int(abs_yaw),
                "correct": int(correctness),
                "n_images": len(subset),
                "mean_confidence": mean_conf,
                "median_confidence": median_conf,
            })

confidence_df = pd.DataFrame(confidence_rows)

confidence_df.to_csv(
    TABLE_DIR / "confidence_by_yaw_and_correctness.csv",
    index=False
)


# ============================================================
# 9. HIGH-CONFIDENCE WRONG PREDICTIONS
# ============================================================

high_conf_wrong_A = (
    merged[merged["A_correct"] == 0]
    .sort_values(
        ["A_confidence", "abs_yaw"],
        ascending=[False, False]
    )
    .head(100)
)

high_conf_wrong_B = (
    merged[merged["B_correct"] == 0]
    .sort_values(
        ["B_confidence", "abs_yaw"],
        ascending=[False, False]
    )
    .head(100)
)

high_conf_wrong_A.to_csv(
    TABLE_DIR / "model_A_high_confidence_wrong_top100.csv",
    index=False
)

high_conf_wrong_B.to_csv(
    TABLE_DIR / "model_B_high_confidence_wrong_top100.csv",
    index=False
)


# ============================================================
# 10. MOST FREQUENT CONFUSION PAIRS
# ============================================================

def confusion_pairs(df, model_prefix):
    wrong = df[df[f"{model_prefix}_correct"] == 0].copy()

    if len(wrong) == 0:
        return pd.DataFrame()

    output = (
        wrong
        .groupby(
            [
                "person",
                f"{model_prefix}_predicted_person"
            ]
        )
        .agg(
            count=("path", "size"),
            mean_confidence=(
                f"{model_prefix}_confidence",
                "mean"
            ),
            mean_abs_yaw=("abs_yaw", "mean"),
        )
        .reset_index()
        .sort_values(
            ["count", "mean_confidence"],
            ascending=[False, False]
        )
    )

    return output


confusion_A = confusion_pairs(merged, "A")
confusion_B = confusion_pairs(merged, "B")

confusion_A.head(100).to_csv(
    TABLE_DIR / "model_A_top_confusion_pairs.csv",
    index=False
)

confusion_B.head(100).to_csv(
    TABLE_DIR / "model_B_top_confusion_pairs.csv",
    index=False
)


# ============================================================
# 11. REPRESENTATIVE CASE SAMPLING
# ============================================================

def select_diverse_examples(df, n, ranking_column=None):
    """
    Prefer diversity across abs_yaw and identities.

    If ranking_column is supplied, higher values are preferred
    inside each yaw group.
    """
    if len(df) == 0:
        return df.copy()

    work = df.copy()

    if ranking_column is not None:
        work = work.sort_values(
            ["abs_yaw", ranking_column],
            ascending=[False, False]
        )
    else:
        work = work.sort_values(
            "abs_yaw",
            ascending=False
        )

    selected_indices = []
    used_people = set()

    yaw_values = sorted(
        work["abs_yaw"].unique(),
        reverse=True
    )

    # First pass: at most one unique identity per yaw bin.
    for yaw in yaw_values:
        candidates = work[work["abs_yaw"] == yaw]

        for idx, row in candidates.iterrows():
            if row["person"] not in used_people:
                selected_indices.append(idx)
                used_people.add(row["person"])
                break

        if len(selected_indices) >= n:
            break

    # Second pass: fill remaining slots with unique identities.
    if len(selected_indices) < n:
        for idx, row in work.iterrows():
            if idx in selected_indices:
                continue

            if row["person"] in used_people:
                continue

            selected_indices.append(idx)
            used_people.add(row["person"])

            if len(selected_indices) >= n:
                break

    # Third pass: fill if not enough unique identities.
    if len(selected_indices) < n:
        for idx in work.index:
            if idx not in selected_indices:
                selected_indices.append(idx)

            if len(selected_indices) >= n:
                break

    return work.loc[selected_indices].head(n).reset_index(drop=True)


A_wrong_B_right = merged[
    (merged["A_correct"] == 0)
    & (merged["B_correct"] == 1)
].copy()

A_right_B_wrong = merged[
    (merged["A_correct"] == 1)
    & (merged["B_correct"] == 0)
].copy()

both_wrong = merged[
    (merged["A_correct"] == 0)
    & (merged["B_correct"] == 0)
].copy()

examples_A_wrong_B_right = select_diverse_examples(
    A_wrong_B_right,
    N_EXAMPLES_PER_CATEGORY,
    ranking_column="B_confidence"
)

examples_A_right_B_wrong = select_diverse_examples(
    A_right_B_wrong,
    N_EXAMPLES_PER_CATEGORY,
    ranking_column="A_confidence"
)

examples_both_wrong = select_diverse_examples(
    both_wrong,
    N_EXAMPLES_PER_CATEGORY,
    ranking_column="B_confidence"
)

examples_A_wrong_B_right.to_csv(
    TABLE_DIR / "examples_A_wrong_B_right.csv",
    index=False
)

examples_A_right_B_wrong.to_csv(
    TABLE_DIR / "examples_A_right_B_wrong.csv",
    index=False
)

examples_both_wrong.to_csv(
    TABLE_DIR / "examples_both_wrong.csv",
    index=False
)


# ============================================================
# 12. IMAGE CONTACT SHEETS
# ============================================================

def make_contact_sheet(
    examples,
    title,
    output_path,
    confidence_model="B",
    columns=4,
):
    if len(examples) == 0:
        return None

    thumb_w = 240
    thumb_h = 250
    image_h = 170
    title_h = 55

    rows = math.ceil(len(examples) / columns)

    canvas_w = columns * thumb_w
    canvas_h = title_h + rows * thumb_h

    canvas = Image.new(
        "RGB",
        (canvas_w, canvas_h),
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

        image_path = Path(row["path"])

        try:
            with Image.open(image_path) as img:
                img = img.convert("L")
                img = ImageOps.contain(
                    img,
                    (160, image_h)
                ).convert("RGB")
        except Exception:
            img = Image.new(
                "RGB",
                (160, image_h),
                "white"
            )

        img_x = x0 + (thumb_w - img.width) // 2
        img_y = y0

        canvas.paste(
            img,
            (img_x, img_y)
        )

        text_y = y0 + image_h + 4

        if confidence_model == "B":
            conf = row["B_confidence"]
        else:
            conf = row["A_confidence"]

        lines = [
            f"{row['person']} | yaw {int(row['yaw']):+d}",
            f"A: {row['A_predicted_person']} ({row['A_confidence']:.2f})",
            f"B: {row['B_predicted_person']} ({row['B_confidence']:.2f})",
        ]

        for line_i, line in enumerate(lines):
            draw.text(
                (x0 + 6, text_y + line_i * 16),
                line,
                fill="black"
            )

    canvas.save(output_path)

    return output_path


sheet1 = make_contact_sheet(
    examples_A_wrong_B_right,
    "Representative cases: Model A wrong, Model B right",
    EXAMPLE_DIR / "A_wrong_B_right_examples.png",
    confidence_model="B",
)

sheet2 = make_contact_sheet(
    examples_A_right_B_wrong,
    "Representative cases: Model A right, Model B wrong",
    EXAMPLE_DIR / "A_right_B_wrong_examples.png",
    confidence_model="A",
)

sheet3 = make_contact_sheet(
    examples_both_wrong,
    "Representative cases: Both models wrong",
    EXAMPLE_DIR / "both_wrong_examples.png",
    confidence_model="B",
)


# ============================================================
# 13. REPORT-READY FIGURES
# ============================================================

# Figure 1 — Outcome composition by absolute yaw
plt.figure(figsize=(10, 6))

bottom = np.zeros(len(yaw_outcomes))

for outcome in outcome_order:
    values = yaw_outcomes[f"{outcome}_percent"].to_numpy()

    plt.bar(
        yaw_outcomes["abs_yaw"],
        values,
        bottom=bottom,
        label=outcome,
        width=9
    )

    bottom += values

plt.xlabel("Absolute yaw angle (degrees)")
plt.ylabel("Percentage of test images")
plt.title("Model A/B Outcome Composition by Yaw")
plt.xticks(yaw_outcomes["abs_yaw"])
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

outcome_fig = FIGURE_DIR / "outcome_composition_by_yaw.png"
plt.savefig(outcome_fig, dpi=180, bbox_inches="tight")
plt.close()


# Figure 2 — Signed-yaw accuracy to expose left/right differences
signed_plot = signed_metrics.sort_values("yaw")

plt.figure(figsize=(10, 6))

plt.plot(
    signed_plot["yaw"],
    signed_plot["model_A_accuracy"],
    marker="o",
    label="Model A — Frontal-only"
)

plt.plot(
    signed_plot["yaw"],
    signed_plot["model_B_accuracy"],
    marker="o",
    label="Model B — Pose-diverse"
)

plt.xlabel("Signed yaw angle (degrees)")
plt.ylabel("Top-1 accuracy")
plt.title("Recognition Accuracy by Signed Yaw")
plt.xticks(sorted(signed_plot["yaw"].unique()))
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

signed_fig = FIGURE_DIR / "accuracy_by_signed_yaw.png"
plt.savefig(signed_fig, dpi=180, bbox_inches="tight")
plt.close()


# Figure 3 — Model B confidence: correct vs wrong at each yaw
b_conf = confidence_df[
    confidence_df["model"] == "Model_B"
]

b_correct = (
    b_conf[b_conf["correct"] == 1]
    .sort_values("abs_yaw")
)

b_wrong = (
    b_conf[b_conf["correct"] == 0]
    .sort_values("abs_yaw")
)

plt.figure(figsize=(9, 6))

plt.plot(
    b_correct["abs_yaw"],
    b_correct["mean_confidence"],
    marker="o",
    label="Correct predictions"
)

plt.plot(
    b_wrong["abs_yaw"],
    b_wrong["mean_confidence"],
    marker="o",
    label="Incorrect predictions"
)

plt.xlabel("Absolute yaw angle (degrees)")
plt.ylabel("Mean softmax confidence")
plt.title("Model B Confidence: Correct vs Incorrect Predictions")
plt.xticks(sorted(merged["abs_yaw"].unique()))
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

confidence_fig = FIGURE_DIR / "model_B_confidence_by_yaw.png"
plt.savefig(confidence_fig, dpi=180, bbox_inches="tight")
plt.close()


# ============================================================
# 14. COMPACT FINAL RESULT TABLE
# ============================================================

final_table = abs_metrics.copy()

for column in [
    "model_A_accuracy",
    "model_A_macro_f1",
    "model_B_accuracy",
    "model_B_macro_f1",
    "accuracy_gain_B_minus_A",
    "macro_f1_gain_B_minus_A",
]:
    if column in final_table.columns:
        final_table[column] = final_table[column] * 100

final_table.to_csv(
    TABLE_DIR / "report_table_per_yaw_percent.csv",
    index=False
)


# ============================================================
# 15. SUMMARY JSON + TEXT
# ============================================================

summary = {
    "n_test_images": int(len(merged)),
    "outcomes": {
        row["outcome"]: {
            "count": int(row["count"]),
            "percent": float(row["percent"]),
        }
        for _, row in overall_outcomes.iterrows()
    },
    "n_A_wrong_B_right": int(len(A_wrong_B_right)),
    "n_A_right_B_wrong": int(len(A_right_B_wrong)),
    "n_both_wrong": int(len(both_wrong)),
    "model_B_high_confidence_wrong_ge_0_90": int(
        (
            (merged["B_correct"] == 0)
            & (merged["B_confidence"] >= 0.90)
        ).sum()
    ),
    "largest_model_B_left_right_accuracy_gap_pp": (
        float(
            left_right_df["B_positive_minus_negative"]
            .abs()
            .max()
            * 100
        )
        if len(left_right_df) > 0
        else None
    ),
}

with open(
    TABLE_DIR / "error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)


text_lines = [
    "ML Term Project — Step 6 Error Analysis Summary",
    "",
    f"Test images: {len(merged):,}",
    "",
]

for _, row in overall_outcomes.iterrows():
    text_lines.append(
        f"{row['outcome']}: "
        f"{int(row['count']):,} "
        f"({row['percent']:.2f}%)"
    )

text_lines.extend([
    "",
    f"A wrong / B right: {len(A_wrong_B_right):,}",
    f"A right / B wrong: {len(A_right_B_wrong):,}",
    f"Both wrong: {len(both_wrong):,}",
    "",
    "Interpretation guidance:",
    "- A wrong / B right illustrates cases where pose-diverse training adds robustness.",
    "- A right / B wrong captures trade-offs or unusual failures of Model B.",
    "- Both wrong identifies residual difficult cases, especially useful at extreme yaw.",
    "- High-confidence wrong predictions should be discussed as a limitation: confidence is not correctness.",
    "- Signed-yaw differences should be treated as secondary observations unless consistently directional.",
])

(
    TABLE_DIR
    / "error_analysis_summary.txt"
).write_text(
    "\n".join(text_lines),
    encoding="utf-8"
)


# ============================================================
# 16. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ERROR ANALYSIS SUMMARY")
print("=" * 80)

print(overall_outcomes.to_string(index=False))

print("\nA wrong / B right :", f"{len(A_wrong_B_right):,}")
print("A right / B wrong :", f"{len(A_right_B_wrong):,}")
print("Both wrong        :", f"{len(both_wrong):,}")

high_conf_B_wrong = (
    (merged["B_correct"] == 0)
    & (merged["B_confidence"] >= 0.90)
).sum()

print(
    "Model B wrong with confidence >= 0.90 :",
    f"{high_conf_B_wrong:,}"
)

print("\nReport-ready outputs:")
print("Tables  :", TABLE_DIR.resolve())
print("Figures :", FIGURE_DIR.resolve())
print("Examples:", EXAMPLE_DIR.resolve())

print("\nStep 6 completed successfully.")
