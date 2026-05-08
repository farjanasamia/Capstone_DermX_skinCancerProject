import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Configuration ───────────────────────────────────────────────────────────

DATASET_DIR   = "dataset"
CSV_PATH      = os.path.join(DATASET_DIR, "GroundTruth.csv")
IMAGES_DIR    = os.path.join(DATASET_DIR, "images")
OUTPUT_DIR    = "dataset_split"

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

RANDOM_SEED   = 42

LABEL_COLS    = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
FIGURES_DIR   = "figures"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def decode_label(row: pd.Series) -> str:
    """Return the class name for a one-hot encoded row."""
    return LABEL_COLS[row[LABEL_COLS].values.argmax()]


def save_split(df: pd.DataFrame, split_name: str) -> str:
    """Write a split DataFrame to OUTPUT_DIR/<split_name>/<split_name>.csv."""
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{split_name}.csv")
    df.to_csv(path, index=False)
    return path


def copy_images_for_split(df: pd.DataFrame, split_name: str) -> None:
    """Copy image files listed in *df* into OUTPUT_DIR/<split_name>/images/."""
    dest_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    os.makedirs(dest_dir, exist_ok=True)
    missing = 0
    for img_id in df["image"]:
        found = False
        for ext in (".jpg", ".jpeg", ".png"):
            src = os.path.join(IMAGES_DIR, img_id + ext)
            if os.path.exists(src):
                shutil.copy2(src, dest_dir)
                found = True
                break
        if not found:
            missing += 1
    if missing:
        print(f"  ⚠ {missing} image(s) not found on disk for split '{split_name}'.")


def print_distribution(df: pd.DataFrame, name: str) -> None:
    counts = df["label"].value_counts().sort_index()
    pcts   = (counts / len(df) * 100).round(1)
    print(f"\n  {name} ({len(df)} samples)")
    for cls in LABEL_COLS:
        n = counts.get(cls, 0)
        p = pcts.get(cls,  0.0)
        print(f"    {cls:<6}  {n:>5}  ({p:.1f}%)")


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_class_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    splits = {
        "Train (70%)":      df_train,
        "Validation (15%)": df_val,
        "Test (15%)":       df_test,
    }
    counts = {
        split_name: [
            int((split_df["label"] == cls).sum()) for cls in LABEL_COLS
        ]
        for split_name, split_df in splits.items()
    }

    n_classes = len(LABEL_COLS)
    bar_w     = 0.25
    x         = np.arange(n_classes)

    COLORS = {
        "Train (70%)":      "#4C72B0",
        "Validation (15%)": "#55A868",
        "Test (15%)":       "#C44E52",
    }

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left panel: absolute counts ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")

    for i, (split_name, color) in enumerate(COLORS.items()):
        offsets = x + (i - 1) * bar_w
        bars = ax.bar(
            offsets,
            counts[split_name],
            width=bar_w,
            label=split_name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 12,
                f"{int(h):,}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Class-wise Sample Count per Split", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right panel: stacked % composition per class ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")

    bottom = np.zeros(n_classes)
    total_per_class = np.array([
        sum(counts[s][i] for s in splits) for i in range(n_classes)
    ], dtype=float)

    for split_name, color in COLORS.items():
        vals = np.array(counts[split_name], dtype=float)
        pcts = np.where(total_per_class > 0, vals / total_per_class * 100, 0)
        bars = ax2.bar(
            x, pcts, width=0.55,
            bottom=bottom,
            label=split_name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for j, (bar, pct) in enumerate(zip(bars, pcts)):
            if pct > 5:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[j] + pct / 2,
                    f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white",
                )
        bottom += pcts

    ax2.set_xticks(x)
    ax2.set_xticklabels(LABEL_COLS, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Proportion (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title("Split Proportion per Class (stacked)", fontsize=13, fontweight="bold", pad=12)
    ax2.axhline(70, color="#4C72B0", linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.axhline(85, color="#55A868", linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        fontsize=11,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
    )

    total_train = sum(counts["Train (70%)"])
    total_val   = sum(counts["Validation (15%)"])
    total_test  = sum(counts["Test (15%)"])
    fig.suptitle(
        f"HAM10000 Dataset Split  ·  "
        f"Train {total_train:,}  |  Val {total_val:,}  |  Test {total_test:,}",
        fontsize=14, fontweight="bold", y=1.07,
    )

    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "class_split_distribution.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved → {fig_path}")
    return fig_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load ground-truth CSV
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)

    if "image" not in df.columns:
        raise ValueError("CSV must have an 'image' column.")
    missing_cols = [c for c in LABEL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing label columns: {missing_cols}")

    # 2. Derive a single string label from one-hot encoding (used for stratification)
    df["label"] = df.apply(decode_label, axis=1)

    print(f"Total samples : {len(df)}")
    print(f"Class distribution:")
    for cls, cnt in df["label"].value_counts().sort_index().items():
        print(f"  {cls:<6}  {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    # 3. Stratified train / temp split  (temp = val + test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # 4. Stratified val / test split from temp
    val_of_temp = VAL_RATIO / temp_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_of_temp),
        stratify=df_temp["label"],
        random_state=RANDOM_SEED,
    )

    # 5. Remove the helper label column before saving
    df_train_save = df_train.drop(columns=["label"])
    df_val_save   = df_val.drop(columns=["label"])
    df_test_save  = df_test.drop(columns=["label"])

    # 6. Save CSVs inside each split subfolder
    train_csv = save_split(df_train_save, "train")
    val_csv   = save_split(df_val_save,   "validation")
    test_csv  = save_split(df_test_save,  "test")

    # 7. Copy images into each split subfolder
    print("\nCopying images …")
    copy_images_for_split(df_train, "train")
    copy_images_for_split(df_val,   "validation")
    copy_images_for_split(df_test,  "test")

    # 8. Summary
    print("\n" + "═" * 50)
    print("Split summary")
    print("═" * 50)
    print_distribution(df_train, "Train      (70%)")
    print_distribution(df_val,   "Validation (15%)")
    print_distribution(df_test,  "Test       (15%)")

    total = len(df_train) + len(df_val) + len(df_test)
    print(f"\n  Total accounted for : {total}")
    print("\nOutput structure:")
    print(f"  {train_csv}")
    print(f"  {os.path.join(OUTPUT_DIR, 'train', 'images/')}")
    print(f"  {val_csv}")
    print(f"  {os.path.join(OUTPUT_DIR, 'validation', 'images/')}")
    print(f"  {test_csv}")
    print(f"  {os.path.join(OUTPUT_DIR, 'test', 'images/')}")

    # 9. Plot class-wise distribution across splits
    plot_class_distribution(df_train, df_val, df_test)

    print("\nDone ✓")


if __name__ == "__main__":
    main()