"""汇总所有实验，生成报告用的对比表。

    python -m src.summarize              # 打印到终端
    python -m src.summarize --md         # 输出 Markdown 表格（可直接粘进报告/群聊）
    python -m src.summarize --tex        # 输出 LaTeX 表格（可直接替换 main.tex 里的 Table II）

数据来源只有一处：runs/<exp>/metrics_test.json
—— 报告里所有数字必须从这里来，不许手抄。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RUNS = Path("runs")

# 实验名的中文说明，报告里当行标题用
LABELS = {
    "mlp_none": "MLP, 不做处理",
    "mlp_weight": "MLP, 类别加权",
    "mlp_sampler": "MLP, 加权采样",
    "mlp_smote": "MLP, SMOTE",
    "mlp_oversample": "MLP, 随机过采样",
    "widedeep_weight": "Wide&Deep, 加权",
    "cnn1d_weight": "1D-CNN, 加权",
    "mlp_lr3e4": "MLP, 加权 + lr 3e-4",
    "gbdt": "GBDT (对照)",
}

ORDER = [
    "mlp_none", "mlp_weight", "mlp_sampler", "mlp_smote", "mlp_oversample",
    "widedeep_weight", "cnn1d_weight", "mlp_lr3e4", "gbdt",
]


def collect() -> list[dict]:
    rows = []
    for f in sorted(RUNS.glob("*/metrics_test.json")):
        name = f.parent.name
        if name == "demo":          # 模拟数据的产物，不能进报告
            continue
        m = json.loads(f.read_text())
        a = m["at_f1_threshold"]
        c = m["cost"]
        p90 = m.get("at_precision90_threshold")
        rows.append({
            "name": name,
            "label": LABELS.get(name, name),
            "auc_pr": m["auc_pr"],
            "auc_roc": m["auc_roc"],
            "precision": a["precision"],
            "recall": a["recall"],
            "f1": a["f1"],
            "threshold": a["threshold"],
            "cost_05": c["at_0.5"],
            "cost_f1": c["at_f1_threshold"],
            "cost_opt": c.get("at_cost_threshold"),
            "thr_opt": m.get("at_cost_threshold", {}).get("threshold"),
            "recall_opt": m.get("at_cost_threshold", {}).get("recall"),
            "precision_opt": m.get("at_cost_threshold", {}).get("precision"),
            "recall_p90": p90["recall"] if p90 else None,
            "n_epochs": _n_epochs(name),
        })

    def key(r):
        i = ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER)
        return (i, r["name"])

    return sorted(rows, key=key)


def _n_epochs(name: str) -> int:
    lc = RUNS / name / "log.csv"
    if not lc.exists():
        return 0
    return max(sum(1 for _ in lc.open()) - 1, 0)


def print_plain(rows: list[dict]) -> None:
    hdr = (f"{'experiment':17s} {'AUC-PR':>7s} {'AUC-ROC':>8s} "
           f"{'P@F1':>6s} {'R@F1':>6s} {'F1':>6s} | "
           f"{'c@0.5':>6s} {'c@F1':>6s} {'c@opt':>6s} | {'R@opt':>6s} {'ep':>3s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        opt = f"{r['cost_opt']:6.0f}" if r["cost_opt"] is not None else "     -"
        ropt = f"{r['recall_opt']:6.3f}" if r["recall_opt"] is not None else "     -"
        print(f"{r['name']:17s} {r['auc_pr']:7.4f} {r['auc_roc']:8.4f} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} | "
              f"{r['cost_05']:6.0f} {r['cost_f1']:6.0f} {opt} | {ropt} {r['n_epochs']:3d}")

    if rows:
        best = max(rows, key=lambda r: r["auc_pr"])
        cheap = min((r for r in rows if r["cost_opt"] is not None),
                    key=lambda r: r["cost_opt"])
        print(f"\n最佳 AUC-PR      : {best['name']} = {best['auc_pr']:.4f}")
        print(f"最低业务代价     : {cheap['name']} = {cheap['cost_opt']:.0f} "
              f"(召回 {cheap['recall_opt']:.3f})")
        print(f"阈值范围         : {min(r['threshold'] for r in rows):.3f} ~ "
              f"{max(r['threshold'] for r in rows):.3f}")
        print("\n提醒：测试集只有 71 笔欺诈，指标波动约 ±3 个点。"
              "报告里要如实说明，并配合 PR 曲线一起看。")


def print_markdown(rows: list[dict]) -> None:
    print("| 配置 | AUC-PR | AUC-ROC | P | R | F1 | 代价@0.5 | 代价@F1阈值 | 代价@最优阈值 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        opt = f"{r['cost_opt']:.0f}" if r["cost_opt"] is not None else "-"
        print(f"| {r['label']} | {r['auc_pr']:.4f} | {r['auc_roc']:.4f} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
              f"{r['cost_05']:.0f} | {r['cost_f1']:.0f} | {opt} |")


def print_tex(rows: list[dict]) -> None:
    tex_label = {
        "mlp_none": "MLP, no handling",
        "mlp_weight": "MLP, class weighting",
        "mlp_sampler": "MLP, weighted sampler",
        "mlp_smote": "MLP, SMOTE",
        "mlp_oversample": "MLP, random oversampling",
        "widedeep_weight": "Wide\\,\\&\\,Deep, weighting",
        "cnn1d_weight": "1D-CNN, weighting",
        "mlp_lr3e4": "MLP, weighting + lr $3\\times10^{-4}$",
        "gbdt": "GBDT (reference)",
    }
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Configuration & AUC-PR & AUC-ROC & P & R & F1 & Cost@opt \\")
    print(r"\midrule")
    best = max(rows, key=lambda r: r["auc_pr"])["name"] if rows else None
    for r in rows:
        label = tex_label.get(r["name"], r["name"].replace("_", r"\_"))
        cost = f"{r['cost_opt']:.0f}" if r["cost_opt"] is not None else "--"
        cells = [f"{r['auc_pr']:.4f}", f"{r['auc_roc']:.4f}", f"{r['precision']:.3f}",
                 f"{r['recall']:.3f}", f"{r['f1']:.3f}", cost]
        if r["name"] == best:
            print(r"\textbf{" + label + "} & " +
                  " & ".join(r"\textbf{" + c + "}" for c in cells) + r" \\")
        else:
            print(label + " & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    ap.add_argument("--tex", action="store_true", help="输出 LaTeX 表格")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        raise SystemExit("runs/ 下还没有任何 metrics_test.json，先跑实验")

    if args.md:
        print_markdown(rows)
    elif args.tex:
        print_tex(rows)
    else:
        print_plain(rows)


if __name__ == "__main__":
    main()
