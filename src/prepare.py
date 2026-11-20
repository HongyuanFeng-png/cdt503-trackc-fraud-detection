"""数据预处理：清洗 -> 划分 -> 标准化 -> 存成 npz

    python -m src.prepare --csv data/creditcard.csv

输出（data/processed/）：
    train.npz / val.npz / test.npz   每份含 X(float32) 与 y(int8)
    scaler.joblib                    只在训练集上 fit，推理时必须用同一个
    meta.json                        特征名、样本数、不平衡比例

关键设计（报告里要写）：
  1. 标准化只在训练集上 fit —— 否则测试集信息会泄露到训练中。
  2. 划分默认分层，保证验证/测试集里一定有欺诈样本。
  3. 提供 --split time 做「按时间切分」，更接近真实上线场景（未来数据不可见）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

FEATURES_V = [f"V{i}" for i in range(1, 29)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--split", choices=["stratified", "time"], default="stratified")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(
            f"找不到 {csv}\n"
            "请先从 Kaggle 下载 'Credit Card Fraud Detection' 的 creditcard.csv 放到 data/ 下，\n"
            "或先用模拟数据跑通流程: python -m src.make_synth_data"
        )

    df = pd.read_csv(csv)
    print(f"[load] {csv}  shape={df.shape}")

    # ---------- 清洗 ----------
    before = len(df)
    df = df.drop_duplicates()
    if "Class" not in df.columns:
        raise SystemExit("CSV 里没有 Class 列，请确认下载的是正确的数据集")

    df[FEATURES_V] = df[FEATURES_V].fillna(0.0)
    df["Amount"] = df["Amount"].fillna(df["Amount"].median())
    df["Time"] = df["Time"].fillna(df["Time"].median())
    df["Amount"] = df["Amount"].clip(lower=0)
    print(f"[clean] 去重 {before - len(df)} 行 -> {len(df)} 行")

    n_pos = int(df["Class"].sum())
    print(f"[balance] 欺诈 {n_pos} / {len(df)} = {n_pos/len(df)*100:.4f}%  "
          f"(不平衡比 1:{len(df)/max(n_pos,1):.0f})")

    # ---------- 划分 ----------
    if args.split == "time":
        df = df.sort_values("Time").reset_index(drop=True)
        n = len(df)
        n_test = int(n * args.test)
        n_val = int(n * args.val)
        train = df.iloc[: n - n_val - n_test]
        val = df.iloc[n - n_val - n_test : n - n_test]
        test = df.iloc[n - n_test :]
    else:
        train, rest = train_test_split(
            df, test_size=args.val + args.test, stratify=df["Class"], random_state=args.seed
        )
        val, test = train_test_split(
            rest,
            test_size=args.test / (args.val + args.test),
            stratify=rest["Class"],
            random_state=args.seed,
        )

    # ---------- 标准化（只在 train 上 fit） ----------
    scaler = RobustScaler()
    train = train.copy(); val = val.copy(); test = test.copy()
    train[["Amount", "Time"]] = scaler.fit_transform(train[["Amount", "Time"]])
    val[["Amount", "Time"]] = scaler.transform(val[["Amount", "Time"]])
    test[["Amount", "Time"]] = scaler.transform(test[["Amount", "Time"]])

    cols = FEATURES_V + ["Amount", "Time"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        X = part[cols].to_numpy(np.float32)
        y = part["Class"].to_numpy(np.int8)
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)
        stats[name] = {"n": int(len(y)), "pos": int(y.sum()), "rate": float(y.mean())}
        print(f"[split] {name:5s} n={len(y):7d}  pos={int(y.sum()):4d}  ({y.mean()*100:.4f}%)")

    joblib.dump(scaler, out_dir / "scaler.joblib")

    meta = {
        "features": cols,
        "split_mode": args.split,
        "scaler": "RobustScaler on [Amount, Time], fit on train only",
        "stats": stats,
        "imbalance_ratio": (stats["train"]["n"] - stats["train"]["pos"]) / max(stats["train"]["pos"], 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] 特征 {len(cols)} 个 -> {out_dir}/  (meta.json 记录了全部参数)")


if __name__ == "__main__":
    main()
