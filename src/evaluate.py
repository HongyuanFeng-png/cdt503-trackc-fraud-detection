"""评估：在验证集上调阈值，在测试集上出最终指标与图表。

    python -m src.evaluate --ckpt runs/mlp_weight/best.pt --out runs/mlp_weight

为什么要在验证集上定阈值？
  默认 0.5 是给「平衡数据」用的。在 0.17% 欺诈率下，模型输出 0.5 以上极少，
  直接用 0.5 会导致召回率接近 0。阈值本身是一个「业务决策」，必须在验证集上选，
  然后原封不动地用到测试集上。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.data import build_model, load_split, make_loader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(ckpt_path: Path, X: np.ndarray) -> np.ndarray:
    """支持 torch 模型（.pt）与 sklearn 模型（.joblib）。"""
    if ckpt_path.suffix == ".joblib":
        obj = joblib.load(ckpt_path)
        return obj["model"].predict_proba(X)[:, 1]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = get_device()
    model = build_model(ckpt["model"], ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    loader = make_loader(X, np.zeros(len(X), np.int8), batch_size=4096, shuffle=False, device=device)
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def best_f1_threshold(y: np.ndarray, prob: np.ndarray):
    prec, rec, thr = precision_recall_curve(y, prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))          # 最后一个是哨兵点，去掉
    return float(thr[i]), float(f1[i])


def threshold_at_precision(y: np.ndarray, prob: np.ndarray, target: float = 0.90):
    """在「精度不低于 target」的前提下，取召回最高的阈值。"""
    prec, rec, thr = precision_recall_curve(y, prob)
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) == 0:
        return None, None
    i = ok[int(np.argmax(rec[:-1][ok]))]
    return float(thr[i]), float(rec[i])


def best_cost_threshold(y: np.ndarray, prob: np.ndarray,
                        fn_cost: float = 20.0, fp_cost: float = 1.0):
    """在验证集上直接最小化业务代价，返回 (阈值, 验证集代价)。

    为什么需要这个：F1 最优的阈值**不等于**代价最优的阈值。
    F1 把漏检和误报看成等价（precision 和 recall 对称），
    但业务上漏检一笔欺诈的代价是误报的 20 倍。两者会选出不同的工作点，
    在极端不平衡数据上甚至可能差出很多。

    候选阈值取验证集预测概率的所有唯一值，等价于遍历 PR 曲线上的每一个点。
    """
    order = np.argsort(prob, kind="mergesort")
    p_sorted = prob[order]
    y_sorted = y[order].astype(np.int64)

    # 从高阈值往低走：初始全部判负，逐步把样本判正
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    fn, fp = n_pos, 0
    best_thr, best_cost = float(p_sorted[-1]) + 1e-9, fn_cost * fn + fp_cost * fp

    for i in range(len(p_sorted) - 1, -1, -1):
        if y_sorted[i] == 1:
            fn -= 1
        else:
            fp += 1
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_thr = float(p_sorted[i])
    return best_thr, float(best_cost)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fn-cost", type=float, default=20.0,
                    help="漏检一笔欺诈的代价（相对误报=1）")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out or ckpt_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_va, y_va = load_split(args.data, "val")
    X_te, y_te = load_split(args.data, "test")

    p_va = predict(ckpt_path, X_va)
    p_te = predict(ckpt_path, X_te)

    thr_f1, f1_va = best_f1_threshold(y_va, p_va)
    thr_p90, rec_p90_va = threshold_at_precision(y_va, p_va, 0.90)
    thr_cost, cost_va = best_cost_threshold(y_va, p_va, args.fn_cost, 1.0)

    print(f"[threshold] 验证集最优 F1 阈值 = {thr_f1:.4f} (val F1={f1_va:.4f})")
    if thr_p90 is not None:
        print(f"[threshold] 精度>=0.90 的阈值 = {thr_p90:.4f} (val recall={rec_p90_va:.4f})")
    print(f"[threshold] 验证集最小代价阈值 = {thr_cost:.4f} (val cost={cost_va:.0f})")

    # ---------- 测试集指标（用验证集定好的阈值） ----------
    def at(thr):
        pred = (p_te >= thr).astype(int)
        return {
            "threshold": float(thr),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "tp": int(((pred == 1) & (y_te == 1)).sum()),
            "fp": int(((pred == 1) & (y_te == 0)).sum()),
            "fn": int(((pred == 0) & (y_te == 1)).sum()),
            "tn": int(((pred == 0) & (y_te == 0)).sum()),
        }

    m_f1 = at(thr_f1)
    m_p90 = at(thr_p90) if thr_p90 is not None else None
    m_cost = at(thr_cost)
    m_default = at(0.5)

    auc_pr = float(average_precision_score(y_te, p_te))
    auc_roc = float(roc_auc_score(y_te, p_te))

    # 代价分析：漏检代价远高于误报，这是"为什么阈值不能设 0.5"的最直观证据
    def cost(m):
        return m["fn"] * args.fn_cost + m["fp"] * 1.0

    metrics = {
        "n_test": int(len(y_te)),
        "n_pos_test": int(y_te.sum()),
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "at_f1_threshold": m_f1,
        "at_precision90_threshold": m_p90,
        "at_cost_threshold": m_cost,
        "at_default_0.5": m_default,
        "cost": {
            "fn_cost": args.fn_cost,
            "at_0.5": cost(m_default),
            "at_f1_threshold": cost(m_f1),
            "at_precision90": cost(m_p90) if m_p90 else None,
            "at_cost_threshold": cost(m_cost),
        },
    }
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---------- 图 1：PR 曲线（不平衡数据的主图） ----------
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(rec, prec, color="#C44E52", lw=1.8, label=f"ours (AUC-PR = {auc_pr:.4f})")
    ax.axhline(y_te.mean(), ls="--", color="grey", lw=1, label=f"random ({y_te.mean():.4f})")
    ax.scatter([m_f1["recall"]], [m_f1["precision"]], color="black", zorder=5, s=30,
               label=f"F1-optimal (thr={thr_f1:.3f})")
    ax.scatter([m_cost["recall"]], [m_cost["precision"]], color="#55A868", zorder=5, s=45,
               marker="*", label=f"cost-optimal (thr={thr_cost:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "pr_curve.png", dpi=180); plt.close(fig)

    # ---------- 图 2：混淆矩阵 ----------
    cm = confusion_matrix(y_te, (p_te >= thr_f1).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (thr={thr_f1:.3f})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=180); plt.close(fig)

    # ---------- 图 3：分数分布（说明"为什么不平衡下不能用 accuracy"） ----------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bins = np.linspace(0, 1, 60)
    ax.hist(p_te[y_te == 0], bins=bins, alpha=.65, label="Normal", color="#4C72B0", log=True)
    ax.hist(p_te[y_te == 1], bins=bins, alpha=.75, label="Fraud", color="#C44E52", log=True)
    ax.axvline(thr_f1, ls="--", color="black", lw=1, label=f"threshold={thr_f1:.3f}")
    ax.set_xlabel("Predicted fraud probability"); ax.set_ylabel("Count (log)")
    ax.set_title("Score distribution (test)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "score_dist.png", dpi=180); plt.close(fig)

    # ---------- 图 4：阈值-指标曲线（业务决策图） ----------
    thrs = np.linspace(0.01, 0.99, 99)
    ps = [precision_score(y_te, (p_te >= t).astype(int), zero_division=0) for t in thrs]
    rs = [recall_score(y_te, (p_te >= t).astype(int), zero_division=0) for t in thrs]
    fs = [f1_score(y_te, (p_te >= t).astype(int), zero_division=0) for t in thrs]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(thrs, ps, label="Precision")
    ax.plot(thrs, rs, label="Recall")
    ax.plot(thrs, fs, label="F1", lw=2)
    ax.axvline(thr_f1, ls="--", color="black", lw=1)
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold vs. metrics (test)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "threshold_curve.png", dpi=180); plt.close(fig)

    # ---------- 图 5：代价-阈值曲线（在验证集上画，因为阈值是在这里选的） ----------
    grid = np.unique(np.quantile(p_va, np.linspace(0, 1, 300)))
    costs_va = []
    for t in grid:
        pred_va = (p_va >= t).astype(int)
        fn_va = int(((pred_va == 0) & (y_va == 1)).sum())
        fp_va = int(((pred_va == 1) & (y_va == 0)).sum())
        costs_va.append(args.fn_cost * fn_va + fp_va)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(grid, costs_va, color="#C44E52", lw=1.8, label="validation cost")
    for t, style, lab in (
        (0.5, ":", "0.5 (default)"),
        (thr_f1, "--", f"F1-optimal ({thr_f1:.3f})"),
        (thr_cost, "-.", f"cost-optimal ({thr_cost:.3f})"),
    ):
        ax.axvline(t, ls=style, lw=1, color="black", label=lab)
    ax.set_xlabel("Threshold"); ax.set_ylabel(f"Cost ($c_{{FN}}$={args.fn_cost:g}, $c_{{FP}}$=1)")
    ax.set_title("Cost vs. threshold (validation set)")
    ax.set_yscale("log")          # 阈值接近 0 时全部判正，代价会到 10^4 量级，线性轴会压扁有效区间
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(out_dir / "cost_curve.png", dpi=180); plt.close(fig)

    # ---------- 文本报告 ----------
    pred = (p_te >= thr_f1).astype(int)
    rep = classification_report(y_te, pred, target_names=["Normal", "Fraud"], digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(
        f"threshold = {thr_f1:.4f} (F1-optimal, tuned on validation set)\n"
        f"cost-optimal threshold = {thr_cost:.4f} (tuned on validation set)\n\n{rep}\n"
        f"AUC-PR  = {auc_pr:.4f}\nAUC-ROC = {auc_roc:.4f}\n"
        f"Cost (FN={args.fn_cost:g}, FP=1): "
        f"@0.5 = {cost(m_default):.0f}, @F1 = {cost(m_f1):.0f}, "
        f"@cost-optimal = {cost(m_cost):.0f}\n"
    )

    print(rep)
    print(f"AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}")
    print(f"代价(FN={args.fn_cost:g}): 阈值0.5 -> {cost(m_default):.0f} | "
          f"F1阈值 -> {cost(m_f1):.0f} | 代价最优阈值 -> {cost(m_cost):.0f}")
    print(f"产物: {out_dir}/{{pr_curve,confusion_matrix,score_dist,threshold_curve,cost_curve}}.png")


if __name__ == "__main__":
    main()
