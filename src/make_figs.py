"""生成报告用的静态图，统一放到 report/fig/ 下。

    python -m src.make_figs

做两件事：
  1. 从原始 CSV 画类别分布 + 金额分布（报告 Fig.1，和任何实验无关，只要数据在）
  2. 把主方案 runs/<exp>/ 下的评估图复制到 report/fig/，文件名与 main.tex 一致
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("report/fig")


def plot_class_distribution(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    n_neg = int((df["Class"] == 0).sum())
    n_pos = int((df["Class"] == 1).sum())
    rate = n_pos / len(df) * 100

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))

    bars = ax[0].bar(["Normal", "Fraud"], [n_neg, n_pos],
                     color=["#4C72B0", "#C44E52"], width=0.55)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Count (log scale)")
    ax[0].set_title(f"Class distribution: {n_pos:,} fraud out of {len(df):,} ({rate:.3f}%)")
    for b, v in zip(bars, [n_neg, n_pos]):
        ax[0].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}",
                   ha="center", va="bottom", fontsize=9)
    ax[0].set_ylim(top=n_neg * 8)
    ax[0].set_axisbelow(True)
    ax[0].grid(alpha=.3, axis="y", which="major")
    ax[0].grid(False, axis="y", which="minor")

    # 金额分布：正样本明显偏向小额，说明 Amount 本身有信号。
    # 用 log1p 而不是 log 轴，避免 Amount<1 的样本被压到同一个 bin 上形成假尖峰。
    log_amount = np.log1p(df["Amount"].to_numpy())
    bins = np.linspace(0, np.log1p(3000), 45)
    ax[1].hist(log_amount[df["Class"] == 0], bins=bins, alpha=.6,
               label="Normal", color="#4C72B0", density=True)
    ax[1].hist(log_amount[df["Class"] == 1], bins=bins, alpha=.7,
               label="Fraud", color="#C44E52", density=True)
    ticks = [0, np.log1p(10), np.log1p(100), np.log1p(1000)]
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(["0", "10", "100", "1,000"])
    ax[1].set_xlabel("Transaction amount (log scale)")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Amount distribution by class")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_pipeline(out: Path) -> None:
    """画系统流水线框图（报告 Methodology 的硬性要求）。

    这张图是可编辑的起点：如果时间允许，用 draw.io 重画会更漂亮，
    但内容就是这 9 个方框和它们的顺序，别漏掉三个关键设计点：
      * RobustScaler 只在 train 上 fit
      * 阈值只在 validation 上选
      * test 只用来做最终报告
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("Raw transactions", "284,807 x 31   (Time, V1-V28, Amount, Class)", "#DCE6F1"),
        ("Clean", "de-duplicate 1,081 rows, impute  ->  283,726 x 31, 473 fraud (0.167%)", "#DCE6F1"),
        ("Stratified split 70/15/15", "train 198,608 (331)  |  val 42,559 (71)  |  test 42,559 (71)", "#DCE6F1"),
        ("RobustScaler on Amount, Time", "fit on TRAIN ONLY  ->  no test statistics leak", "#FBE3D6"),
        ("Imbalance handling", "none | class weight | weighted sampler | SMOTE | oversampling", "#DCE6F1"),
        ("Models", "MLP  |  Wide&Deep  |  1D-CNN  |  GBDT (reference)", "#DCE6F1"),
        ("Threshold selection", "on VALIDATION only:  F1-optimal  vs  cost-optimal ($c_{FN}$=20, $c_{FP}$=1)", "#FBE3D6"),
        ("Test-set evaluation", "run once, for reporting only", "#DCE6F1"),
        ("Gradio deployment", "batch CSV scoring  +  single-transaction probe", "#D8EFD8"),
    ]

    fig, ax = plt.subplots(figsize=(4.4, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.6)
    ax.axis("off")

    h, gap = 0.92, 0.62
    y = 14.6 - h
    centers = []
    for title, detail, color in steps:
        box = FancyBboxPatch((0.35, y), 9.3, h,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=0.9, edgecolor="#4A4A4A", facecolor=color)
        ax.add_patch(box)
        ax.text(5.0, y + h * 0.66, title, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color="#1A1A1A")
        ax.text(5.0, y + h * 0.26, detail, ha="center", va="center",
                fontsize=5.6, color="#333333")
        centers.append(y)
        y -= h + gap

    for i in range(len(steps) - 1):
        y0 = centers[i]
        y1 = centers[i + 1] + h
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color="#4A4A4A"))

    ax.text(5.0, 0.45, "orange = leakage-critical step    green = deployed artefact",
            ha="center", va="center", fontsize=5.4, color="#666666", style="italic")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def copy_run_figs(exp: str) -> None:
    src = Path("runs") / exp
    mapping = {
        "pr_curve.png": "pr_curve.png",
        "confusion_matrix.png": "confusion_matrix.png",
        "threshold_curve.png": "threshold_curve.png",
        "cost_curve.png": "cost_curve.png",
    }
    for a, b in mapping.items():
        if (src / a).exists():
            shutil.copy(src / a, FIG / b)
            print(f"[fig] {FIG / b}  <- {src / a}")
        else:
            print(f"[warn] 缺少 {src / a}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--exp", default="mlp_smote", help="主方案的 runs 目录名")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)

    csv = Path(args.csv)
    if csv.exists():
        plot_class_distribution(csv, FIG / "class_dist.png")
    else:
        print(f"[warn] 找不到 {csv}，跳过类别分布图")

    copy_run_figs(args.exp)
    plot_pipeline(FIG / "pipeline.png")

    meta = Path("runs") / args.exp / "metrics_test.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(f"\n主方案 {args.exp}: AUC-PR={m['auc_pr']:.4f} "
              f"AUC-ROC={m['auc_roc']:.4f} "
              f"代价@最优阈值={m['cost']['at_cost_threshold']:.0f}")

    print("\n还需要手工补两张图：")
    print("  report/fig/pipeline.png  —— 系统流水线框图（PPT / draw.io 画）")
    print("  report/fig/demo.png      —— Gradio 界面截图")


if __name__ == "__main__":
    main()
