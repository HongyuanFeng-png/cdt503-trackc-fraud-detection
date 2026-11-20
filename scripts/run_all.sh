#!/usr/bin/env bash
# 一键跑完「报告需要的全部对比实验」。建议睡前跑。
# 用法: bash scripts/run_all.sh
#
# 注意：用 bash 跑，不要用 zsh。zsh 不会对未加引号的变量做分词，
# 会把 "--model mlp --imbalance smote" 当成一个整体参数传给 argparse 然后静默失败。
set -euo pipefail

DATA=${DATA:-data/processed}
EPOCHS=${EPOCHS:-30}

run () {  # $1 = 输出目录, 其余 = 参数
  local out=$1; shift
  if [ -f "$out/metrics_test.json" ]; then
    echo "[skip] $out 已存在"
    return
  fi
  echo "=== $out : $* ==="
  python -m src.train --data "$DATA" --epochs "$EPOCHS" --out "$out" "$@"
  python -m src.evaluate --ckpt "$(ls "$out"/best.* | head -1)" --data "$DATA" --out "$out" >/dev/null
}

# 1) 完全不做不平衡处理 —— 对照组
run runs/mlp_none        --model mlp --imbalance none

# 2) 类别加权（会破坏概率校准，见 README 第 5 节）
run runs/mlp_weight      --model mlp --imbalance weight

# 3) 加权采样（WeightedRandomSampler）
run runs/mlp_sampler     --model mlp --imbalance sampler

# 4) SMOTE 过采样（当前主方案）
run runs/mlp_smote       --model mlp --imbalance smote

# 5) 简单复制过采样（SMOTE 的对照组）
run runs/mlp_oversample  --model mlp --imbalance oversample

# 6) 换架构：Wide & Deep
run runs/widedeep_weight --model widedeep --imbalance weight

# 7) 换架构：1D-CNN
run runs/cnn1d_weight    --model cnn1d --imbalance weight

# 8) 传统方法对照组：梯度提升树
run runs/gbdt            --model gbdt --imbalance weight

# 9) 学习率扫描
run runs/mlp_lr3e4       --model mlp --imbalance weight --lr 3e-4 --epochs 50

echo
echo "================ 汇总（测试集） ================"
python -m src.summarize
