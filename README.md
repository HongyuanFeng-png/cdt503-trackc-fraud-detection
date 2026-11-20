# 信用卡欺诈检测 · Track C 实战手册

> 目标：用**不平衡处理 + 深度模型 + 阈值调优 + Gradio 部署**，做一个稳过且有话可说的原型。
> 每一步都告诉你「做什么」「为什么」「怎么判断做对了」。

---

## 0. 你会得到什么

```
CDT503_TrackC_Fraud/
├── app.py                   # Gradio：批量检测 + 单笔抽查（Level 3）
├── src/
│   ├── make_synth_data.py   # 生成模拟数据（先用它跑通，不用等下载）
│   ├── prepare.py           # 清洗 → 分层划分 → 标准化 → npz
│   ├── smote.py             # SMOTE + 随机过采样
│   ├── data.py              # Dataset / DataLoader / 模型工厂
│   ├── train.py             # 训练（4 种不平衡策略 × 3 种网络 + GBDT 基线）
│   └── evaluate.py          # 阈值调优 + PR 曲线 + 混淆矩阵 + 代价分析
├── scripts/run_all.sh       # 一键跑 8 组对比实验
├── report/main.tex          # IEEEtran 报告骨架
└── data/  runs/             # 数据与实验结果（不进 git）
```

**评分对照**

| 评分要求 | 本方案如何满足 |
|---|---|
| Level 2 · 脏数据实战 | 0.17% 极端不平衡 + SMOTE/加权/采样三路对比 + 数据清洗 |
| Level 2 · 超参数调优 | 阈值调优 + 8 组消融 + 学习率/轮数扫描 |
| Level 2 · 先进架构 | Wide & Deep、1D-CNN（对比 MLP 与 GBDT） |
| Level 3 · 应用部署 | Gradio 批量检测 + 单笔实时打分 |

---

## 1. 环境（已装好）

```bash
cd "/Users/fenghongyuan/Desktop/CDT503  control sys & AI/CDT503_TrackC_Fraud"
source .venv/bin/activate
python -c "import torch, sklearn, gradio; print('ok')"
```

> ⚠️ 这个路径里有空格（`CDT503  control sys & AI`），所以**装包请用 `python -m pip install xxx`**，
> 不要用 `pip install xxx`。已经帮你把 `pip` 改成了兼容空格的包装脚本，但养成 `python -m pip` 的习惯更保险。

没有 `.venv` 就重建：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. 先用「模拟数据」跑通全流程（约 2 分钟）

**为什么要这步**：真实数据要登录 Kaggle 下载，可能卡住。先用模拟数据确认代码没问题，
再去折腾下载，能省掉一半的调试时间。

```bash
# 1) 生成 6 万行模拟交易（0.4% 欺诈）
python -m src.make_synth_data --rows 60000

# 2) 清洗 + 划分 + 标准化
python -m src.prepare --csv data/creditcard_synth.csv

# 3) 训练（约 10 秒）
python -m src.train --model mlp --imbalance weight --epochs 15 --out runs/demo

# 4) 评估
python -m src.evaluate --ckpt runs/demo/best.pt --out runs/demo
```

**做对了的标志**（实测值）：

```
[threshold] 验证集最优 F1 阈值 = 0.9xxx
              precision    recall  f1-score
       Fraud     0.96xx    0.69xx    0.80xx
AUC-PR=0.88xx   AUC-ROC=0.99xx
代价(FN=20): 阈值0.5 -> 504 | 调优阈值 -> 221
```

- AUC-ROC 很高（0.99）但 AUC-PR 只有 0.88 → **这正是不平衡数据的典型现象**
- 阈值不是 0.5，而是 0.9 以上 → 因为类别加权把概率整体抬高了
- 调优阈值后「代价」从 393 降到 281 → 这就是你报告里最值钱的一段分析

> ⚠️ 模拟数据有一个副作用：`--model gbdt` 在这份数据上会拿到接近 1.0 的 AUC-PR。
> 原因是模拟的欺诈信号只是几个维度的简单平移，树模型一击即中。
> **真实数据不会这样**（真实数据上 GBDT 通常在 0.85 左右）。所以模拟数据只用来验证代码，
> 报告里的所有数字必须用真实数据重跑。

---

## 3. 换成真实数据

**数据集**：Kaggle 搜索 `Credit Card Fraud Detection`（ULB Machine Learning Group）
- 284,807 笔交易，492 笔欺诈（**0.172%**）
- 文件 `creditcard.csv`，约 150 MB
- 需要 Kaggle 账号（免费），点 Download 即可

放好后：

```bash
mv ~/Downloads/creditcard.csv data/
python -m src.prepare --csv data/creditcard.csv
```

输出应该类似：

```
[balance] 欺诈 492 / 284807 = 0.1727%  (不平衡比 1:579)
[split] train n= 199364  pos=  344  (0.1727%)
[split] val   n=  42721  pos=   74  (0.1732%)
[split] test  n=  42722  pos=   74  (0.1732%)
```

> **注意**：测试集只有 74 笔欺诈，所以指标的波动会比较大（±3 个点很正常）。
> 报告里要如实说明这一点，不要假装是 0.1% 的精度。

**想讲「时间泄露」这个高级话题**：加 `--split time`，按时间前 70% 训练、后 15% 测试。
真实风控场景里未来数据不可见，时间切分更严谨。两种结果都跑一遍、在报告里对比，很加分。

---

## 4. 训练：参数怎么调

```bash
python -m src.train --model mlp --imbalance weight --epochs 30 --out runs/mlp_weight
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | mlp | `mlp` / `cnn1d` / `widedeep` / `gbdt` |
| `--imbalance` | weight | `none` / `weight` / `sampler` / `smote` / `oversample` |
| `--epochs` | 30 | 早停会自动截断，一般 15–25 轮就收敛 |
| `--batch-size` | 1024 | 内存够就大一点，训练更稳 |
| `--lr` | 1e-3 | AdamW，1e-3 起步；不收敛就降到 3e-4 |
| `--patience` | 8 | 验证集 AUC-PR 连续 8 轮不升就停 |

### 四种不平衡策略，到底有什么区别

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| `none` | 什么都不做 | 对照组 | **排序能力其实不差**，但默认阈值下几乎抓不到（见下方实测） |
| `weight` | 损失函数里给正样本加权 `pos_weight = 579` | 不改数据、训练快、效果稳 | 概率被整体抬高，必须重新定阈值 |
| `sampler` | 每个 batch 里过采样少数类 | 实现简单 | 同一笔交易被反复看到，容易过拟合 |
| `smote` | 在少数类的 k 近邻之间插值造新样本 | 增加多样性，学术上"标准答案" | 慢，且高维空间插值可能造出不真实的交易 |
| `oversample` | 直接复制少数类 | 最简单 | 就是过拟合，作为 SMOTE 的对照 |

> **报告里最有价值的一句**（这一条是实测推翻了教科书说法）：
> 在 30 维、0.17% 不平衡的数据上，`weight` 和 `smote` 的 AUC-PR 差了 **0.098**
> （0.7047 vs 0.8030），不是教科书说的"差不多"。原因不是 SMOTE 的插值更聪明，
> 而是**重采样把每个 batch 里的正例比例从 0.17% 提到了 33%**，
> 让模型每个 epoch 都能在正例上多走几步梯度。加权做不到这一点。
> 所以"工程上应该选加权"这句话，在这份数据上是**错的**。

### 为什么用 AUC-PR 而不是 accuracy / AUC-ROC

这是本项目的**核心论点**，务必背下来：

> 数据里 99.83% 是正常交易。一个"全部预测正常"的模型，accuracy 是 **99.83%**，
> 但它一笔欺诈都抓不到。所以 accuracy 在这里毫无意义。
> AUC-ROC 也不够——因为负样本太多，假阳性率稍微上升一点，ROC 曲线看起来还是很好。
> 只有 **AUC-PR（PR 曲线下面积）** 直接衡量"在抓到的样本里有多少是真的"，才反映真实能力。

---

## 5. 消融实验（拿 Level 2 分的关键）

```bash
bash scripts/run_all.sh
```

跑完会打印这样一张表（以下是真实数据的实测结果）：

```
experiment         AUC-PR  AUC-ROC   P@F1   R@F1     F1 |  c@0.5   c@F1  c@opt
mlp_smote          0.8030   0.9547  0.789  0.789  0.789 |    319    315    289
mlp_oversample     0.7946   0.9655  0.855  0.746  0.797 |    289    369    298
mlp_sampler        0.7887   0.9612  0.839  0.732  0.782 |    309    390    276
mlp_none           0.7196   0.9466  0.812  0.789  0.800 |    312    313    313
widedeep_weight    0.7071   0.9649  0.812  0.789  0.800 |    851    313    324
mlp_weight         0.7047   0.9623  0.836  0.789  0.812 |   1192    311    302
gbdt               0.6923   0.9606  0.821  0.775  0.797 |    327    332    319
mlp_lr3e4          0.6875   0.9612  0.824  0.789  0.806 |   1471    312    320
cnn1d_weight       0.4765   0.9699  0.392  0.718  0.507 |   1046    479    493
```

**报告里要写的四句话**：

1. **重采样提升排序，加权不提升。** SMOTE / 过采样 / 加权采样 的 AUC-PR 是 0.789–0.803，
   而"不处理"是 0.7196、"加权"是 0.7047 —— 差 8 个点。
   机制不是"损失形状"，而是**样本效率**：训练集只有 331 个正例，加权后每个 epoch 还是只看到 331 个；
   重采样后每个 batch 里正例占 1/3。
2. **加权是唯一会破坏校准的策略。** 默认 0.5 阈值下代价 1192，是其他配置（289–327）的 4 倍，
   因为它把正类损失乘了 ~600，概率整体抬高。重采样不改损失，所以分数仍然可校准。
3. **F1 最优 ≠ 代价最优。** 在 9 组里有 3 组，F1 最优阈值的代价**比直接用 0.5 还高**
   （过采样 369 vs 289、加权采样 390 vs 309、GBDT 332 vs 327）。
   原因：F1 把漏检和误报看成等价，而业务上漏检贵 20 倍。
4. **架构可能帮倒忙。** 1D-CNN 的 AUC-PR 只有 0.4765，比普通 MLP 低 33 个点 ——
   把 30 个 PCA 分量当序列做卷积，等于强加了一个"相邻特征相关"的先验，而它们本来就不相关。

> **一个能写进报告的理论点**：代价最优阈值有闭式解
> `τ* = c_FP / (c_FN + c_FP) = 1/21 ≈ 0.0476`。
> 我们在验证集上搜出来的最优阈值是 **0.049**，几乎完全吻合。
> 这一条把"阈值为什么不是 0.5"从经验说法升级成了理论结论。

> 以上数字全部来自真实数据的实测（`runs/*/metrics_test.json`），
> 可用 `python -m src.summarize` 一键复现这张表。

---

## 6. 评估与出图

```bash
python -m src.evaluate --ckpt runs/mlp_weight/best.pt --out runs/mlp_weight
```

| 文件 | 报告里放哪 |
|---|---|
| `pr_curve.png` | **主图**。PR 曲线 + 随机基线 + 工作点 |
| `confusion_matrix.png` | 混淆矩阵（阈值调优后） |
| `score_dist.png` | 分数分布（说明两类是否可分） |
| `threshold_curve.png` | 阈值-Precision/Recall/F1 曲线（业务决策图） |
| `metrics_test.json` | 汇总表数据源 |
| `classification_report.txt` | 附录 |

**关键概念：阈值是业务决策，不是超参数**

- 阈值高 → 少打扰客户，但漏检多（FN 代价高）
- 阈值低 → 抓得多，但误报多（客户被误冻结，体验差）
- 我们用 `--fn-cost 20` 表示"漏检一笔的代价 = 误报的 20 倍"
- 报告里给出「不同阈值下的总代价」曲线，比单纯报 F1 高级得多

---

## 7. Gradio 部署（Level 3）

```bash
python app.py --ckpt runs/mlp_weight/best.pt
# 打开 http://127.0.0.1:7860
```

- **批量检测**：上传 CSV → 每笔输出欺诈概率 + 是否标记，按风险排序，可下载结果
- **单笔抽查**：随机抽一笔真实交易，实时显示概率与真实标签

演示话术：
> "这批 2,060 笔交易里，模型标记了 57 笔可疑，其中 56 笔确实是欺诈，1 笔误报。"
> "注意：这份演示 CSV 是人为把欺诈比例提到 2.9% 的（真实数据只有 0.17%），
> 否则 2,000 笔里只能看到 3 笔欺诈，演示效果出不来。这个提升要说出来，别藏着。"

（这个数字现场跑出来是什么就说什么，别提前编。）

---

## 8. 常见报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `找不到 data/creditcard.csv` | 没下载 | 先跑 `make_synth_data`，或去 Kaggle 下载 |
| `CSV 里没有 Class 列` | 下载错了文件 | 确认是 `creditcard.csv` |
| 训练时 `val AUC-PR` 一直是 0 | 学习率太大 / 没做不平衡处理 | `--imbalance weight --lr 3e-4` |
| 召回率永远是 0 | 用 0.5 当阈值 | 用 `evaluate.py`，它会在验证集上调阈值 |
| 评估时 `predict` 报维度错 | 换数据集后没重新 `prepare` | 重新跑 `prepare.py` |
| `ModuleNotFoundError: src` | 不在项目根目录 | `cd` 到 `CDT503_TrackC_Fraud` |
| Gradio 打不开 | 端口占用 | `--port 7861` |
| SMOTE 很慢 | 20 万行 × 30 维做 kNN | 正常，或用 `--imbalance weight` |

---

## 9. 交作业前检查清单

- [ ] 报告 LaTeX + IEEEtran，**4–6 页**
- [ ] 有**系统流水线框图**（数据 → 预处理 → 模型 → 阈值 → 部署）
- [ ] 有类别分布图（说明 0.17% 有多极端）
- [ ] 有 **PR 曲线**（主图）+ 混淆矩阵 + 阈值曲线
- [ ] 有 **8 组对比表**，并写了三条分析
- [ ] 明确解释了「为什么不用 accuracy」
- [ ] 有 Gradio 截图 + 一次真实批量检测的结果
- [ ] Challenges 写了真实的坑（阈值 0.5 导致召回 0、SMOTE 太慢…）
- [ ] 录了 2 分钟备份视频
- [ ] 三人各练过自己那 1.5–2 分钟
- [ ] 彩排 3 次，总时长 ≤ 7 分钟
