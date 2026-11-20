"""SMOTE：只在训练集上做，用于对抗极端类别不平衡。

原理：对每个少数类样本，在它的 k 个最近邻里随机挑一个，
在两点连线上随机插值，生成"看起来合理"的新样本。

注意：
  * 绝不能在验证/测试集上做 —— 那等于作弊（同一笔交易的近亲同时出现在测试集里）。
  * 284k 行数据上跑 SMOTE 大约需要几十秒，属于可接受范围。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, ratio: float = 0.5, seed: int = 42):
    """把少数类过采样到「多数类数量的 ratio 倍」。

    ratio=0.5 表示采样后少数类占 1/3；ratio=1.0 表示完全 1:1。
    返回 (X_res, y_res)，dtype 与输入保持一致。
    """
    rng = np.random.default_rng(seed)

    X_min = X[y == 1]
    n_min = len(X_min)
    n_maj = int((y == 0).sum())
    if n_min < 2:
        print("[smote] 少数类样本太少，跳过")
        return X, y

    target = int(n_maj * ratio)
    n_new = target - n_min
    if n_new <= 0:
        print("[smote] 当前比例已高于目标，跳过")
        return X, y

    k_eff = min(k + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_min)
    _, idx = nn.kneighbors(X_min)

    base = rng.integers(0, n_min, n_new)
    neigh = idx[base, rng.integers(1, k_eff, n_new)]
    gap = rng.random((n_new, 1)).astype(X.dtype)

    X_new = X_min[base] + gap * (X_min[neigh] - X_min[base])
    y_new = np.ones(n_new, dtype=y.dtype)

    print(f"[smote] 少数类 {n_min} -> {n_min + n_new} (ratio={ratio})")
    return np.vstack([X, X_new]), np.concatenate([y, y_new])


def random_oversample(X: np.ndarray, y: np.ndarray, ratio: float = 0.5, seed: int = 42):
    """最简单的过采样：直接复制少数类样本（作为 SMOTE 的对照）。"""
    rng = np.random.default_rng(seed)
    X_min = X[y == 1]
    n_min = len(X_min)
    n_maj = int((y == 0).sum())
    n_new = int(n_maj * ratio) - n_min
    if n_new <= 0:
        return X, y
    pick = rng.integers(0, n_min, n_new)
    print(f"[oversample] 少数类 {n_min} -> {n_min + n_new} (ratio={ratio})")
    return np.vstack([X, X_min[pick]]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
