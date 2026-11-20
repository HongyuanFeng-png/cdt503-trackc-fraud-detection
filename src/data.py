"""Dataset / DataLoader / 模型工厂。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_split(processed_dir: str | Path, name: str):
    d = np.load(Path(processed_dir) / f"{name}.npz")
    return d["X"], d["y"]


def make_loader(X, y, batch_size: int = 1024, shuffle: bool = True, sampler=None, device=None):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,          # 数据在内存里，多进程反而更慢
        pin_memory=bool(device is not None and device.type == "cuda"),
        drop_last=False,
    )


def make_weighted_sampler(y: np.ndarray, num_samples: int | None = None):
    """按类别频率的倒数加权采样，使每个 batch 里正负样本大致均衡。"""
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    w = 1.0 / counts[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=num_samples or len(y),
        replacement=True,
    )


class MLP(nn.Module):
    """基线模型：3 层全连接 + BatchNorm + Dropout。

    表格数据上，一个调好的 MLP 往往和花哨模型打平 —— 这正是报告里可以讨论的点。
    """

    def __init__(self, in_dim: int, hidden=(128, 64, 32), p: float = 0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """把 30 个特征当成长度为 30 的序列做一维卷积，捕捉特征间的局部组合。

    这是报告里「换个架构」的对照组，参数量小、训练快。
    """

    def __init__(self, in_dim: int, ch: int = 32, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch * 2, 3, padding=1),
            nn.BatchNorm1d(ch * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(ch * 2, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)


class WideDeep(nn.Module):
    """Wide & Deep 思路的简化版：线性部分记住特征、深层部分学交互。

    表格数据上的经典结构，比纯 MLP 稳定，报告里也好讲。
    """

    def __init__(self, in_dim: int, hidden=(128, 64), p: float = 0.3):
        super().__init__()
        self.wide = nn.Linear(in_dim, 1)
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        return (self.wide(x) + self.deep(x)).squeeze(-1)


MODELS = {
    "mlp": MLP,
    "cnn1d": CNN1D,
    "widedeep": WideDeep,
}


def build_model(name: str, in_dim: int) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"未知模型 {name}，可选: {list(MODELS)} 或 gbdt")
    return MODELS[name](in_dim)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
