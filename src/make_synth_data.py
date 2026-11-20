"""生成「模拟」信用卡交易数据，用于在下载真实数据集之前验证整条流水线。

真实数据集（Kaggle Credit Card Fraud Detection）的字段：
    Time, V1..V28, Amount, Class
本脚本按同样的 schema 生成一份数据，正样本比例约 0.4%（真实数据是 0.172%）。

真实数据用法见 README 第 3 节；本脚本只用于「先确认代码是对的」。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/creditcard_synth.csv")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--fraud-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows
    n_pos = max(20, int(n * args.fraud_rate))
    n_neg = n - n_pos

    # 负样本：标准正态；正样本：在 6 个维度上有偏移（模拟真实欺诈的可分性）
    X_neg = rng.normal(0, 1, size=(n_neg, 28))
    X_pos = rng.normal(0, 1, size=(n_pos, 28))
    shift = np.zeros(28)
    shift[[0, 3, 7, 11, 16, 21]] = [-1.35, 1.1, 1.55, -1.05, 1.2, -1.4]
    X_pos += shift + rng.normal(0, 0.9, size=(n_pos, 28))

    X = np.vstack([X_neg, X_pos]).astype(np.float32)
    y = np.concatenate([np.zeros(n_neg, np.int8), np.ones(n_pos, np.int8)])

    # Amount：对数正态，正样本略大
    amount = np.where(
        y == 1,
        rng.lognormal(3.8, 1.0, size=n),
        rng.lognormal(3.2, 1.2, size=n),
    )

    # Time：两天内递增
    time = np.sort(rng.uniform(0, 172800, size=n))

    order = rng.permutation(n)
    X, y, amount, time = X[order], y[order], amount[order], time[order]

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"已生成 {out}  行数={len(df)}  欺诈={int(y.sum())} ({y.mean()*100:.3f}%)")


if __name__ == "__main__":
    main()
