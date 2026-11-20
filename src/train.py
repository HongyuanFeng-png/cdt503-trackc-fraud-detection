"""训练脚本：支持 4 种不平衡处理 + 3 种神经网络 + 1 种传统基线

典型用法:
    # 基线：普通 MLP，不做任何不平衡处理（证明"必须处理不平衡"）
    python -m src.train --model mlp --imbalance none --out runs/mlp_none

    # 主方案：MLP + 类别加权
    python -m src.train --model mlp --imbalance weight --out runs/mlp_weight

    # 消融：SMOTE / 过采样 / 加权采样 / 传统 GBDT
    python -m src.train --model mlp --imbalance smote --out runs/mlp_smote
    python -m src.train --model mlp --imbalance sampler --out runs/mlp_sampler
    python -m src.train --model gbdt --imbalance weight --out runs/gbdt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import build_model, count_params, load_split, make_loader, make_weighted_sampler
from src.smote import random_oversample, smote


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_probs(y_true: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "auc_pr": float(average_precision_score(y_true, prob)),
        "auc_roc": float(roc_auc_score(y_true, prob)),
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    probs, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    prob = np.concatenate(probs)
    label = np.concatenate(labels)
    m = evaluate_probs(label, prob)
    m["loss"] = total / max(n, 1)
    return m


def train_gbdt(X_tr, y_tr, X_va, y_va, out_dir: Path, args):
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_va)[:, 1]
    m = evaluate_probs(y_va, prob)
    joblib.dump({"model": clf, "in_dim": X_tr.shape[1]}, out_dir / "best.joblib")
    rows = [{"epoch": 1, "lr": 0, "train_loss": float("nan"), "val_loss": float("nan"),
             "val_auc_pr": m["auc_pr"], "val_auc_roc": m["auc_roc"]}]
    print(f"[gbdt] val AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f}  ({time.time()-t0:.1f}s)")
    return rows, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="mlp", help="mlp | cnn1d | widedeep | gbdt")
    ap.add_argument("--imbalance", default="weight",
                    choices=["none", "weight", "sampler", "smote", "oversample"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = load_split(args.data, "train")
    X_va, y_va = load_split(args.data, "val")
    in_dim = X_tr.shape[1]

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    print(f"[data] train={len(y_tr)} pos={n_pos} ({y_tr.mean()*100:.3f}%) | val={len(y_va)} pos={int(y_va.sum())}")
    print(f"[config] model={args.model} imbalance={args.imbalance} epochs={args.epochs}")

    if args.model == "gbdt":
        rows, best = train_gbdt(X_tr, y_tr, X_va, y_va, out_dir, args)
        with open(out_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
        print(f"[done] -> {out_dir}/best.joblib")
        return

    # ---------- 不平衡处理（只作用于训练集） ----------
    sampler = None
    pos_weight = None
    if args.imbalance == "weight":
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        print(f"[imbalance] 类别加权 pos_weight={pos_weight.item():.1f}")
    elif args.imbalance == "sampler":
        sampler = make_weighted_sampler(y_tr)
        print("[imbalance] WeightedRandomSampler")
    elif args.imbalance == "smote":
        X_tr, y_tr = smote(X_tr, y_tr, k=5, ratio=0.5, seed=args.seed)
    elif args.imbalance == "oversample":
        X_tr, y_tr = random_oversample(X_tr, y_tr, ratio=0.5, seed=args.seed)

    device = get_device()
    print(f"[device] {device}")

    train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True, sampler=sampler, device=device)
    val_loader = make_loader(X_va, y_va, args.batch_size, shuffle=False, device=device)

    model = build_model(args.model, in_dim).to(device)
    print(f"[model] {args.model} trainable params = {count_params(model):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    rows, best_ap, bad = [], -1.0, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_auc_pr": va["auc_pr"],
            "val_auc_roc": va["auc_roc"],
        }
        rows.append(row)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {tr['loss']:.4f} | "
              f"val loss {va['loss']:.4f} | AUC-PR {va['auc_pr']:.4f} | AUC-ROC {va['auc_roc']:.4f}")

        # 用 AUC-PR 选模型：极端不平衡下，AUC-ROC 会虚高，AUC-PR 才反映真实能力
        if va["auc_pr"] > best_ap:
            best_ap, bad = va["auc_pr"], 0
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "imbalance": args.imbalance,
                    "val_auc_pr": best_ap,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
            print(f"  -> 保存最佳模型 (val AUC-PR = {best_ap:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] 连续 {args.patience} 轮未提升")
                break

    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Loss")
    ax[1].plot(ep, [r["val_auc_pr"] for r in rows], label="val AUC-PR", color="#C44E52")
    ax[1].plot(ep, [r["val_auc_roc"] for r in rows], label="val AUC-ROC", linestyle="--", color="#4C72B0")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=160)
    plt.close(fig)

    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟 | 最佳 val AUC-PR = {best_ap:.4f}")
    print(f"产物: {out_dir}/best.pt  {out_dir}/log.csv  {out_dir}/curves.png")


if __name__ == "__main__":
    main()
