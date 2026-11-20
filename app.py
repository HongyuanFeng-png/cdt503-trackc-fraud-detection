"""Gradio 网页 Demo（Level 3 加分项）

    python app.py --ckpt runs/mlp_smote/best.pt

两个功能：
  1. 批量检测：上传原始格式 CSV（Time, V1..V28, Amount, 可选 Class）
     -> 每笔交易输出欺诈概率、是否标记、并按风险排序
  2. 单笔抽查：从测试集随机抽一笔真实交易，实时看模型的判断

演示技巧：先跑批量检测，说出「共 N 笔，标记 M 笔，其中 K 笔确实是欺诈」，
再切到单笔抽查展示概率分布 —— 比只放一张静态图有说服力得多。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src.data import load_split
from src.evaluate import predict

STATE: dict = {}


def init(ckpt_path: str, data_dir: str = "data/processed"):
    if STATE.get("ckpt") == ckpt_path:
        return STATE
    out_dir = Path(ckpt_path).parent
    metrics_file = out_dir / "metrics_test.json"
    thr = 0.5
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        # 用代价最优阈值：这是业务上真正应该用的工作点，不是 F1 最优
        thr = m.get("at_cost_threshold", m["at_f1_threshold"])["threshold"]

    scaler = joblib.load(Path(data_dir) / "scaler.joblib")
    meta = json.loads((Path(data_dir) / "meta.json").read_text())

    STATE.clear()
    STATE.update(ckpt=ckpt_path, thr=thr, scaler=scaler, meta=meta,
                 X_te=None, y_te=None, data_dir=data_dir)
    return STATE


def _ensure_test():
    if STATE.get("X_te") is None:
        X, y = load_split(STATE["data_dir"], "test")
        STATE["X_te"], STATE["y_te"] = X, y
    return STATE["X_te"], STATE["y_te"]


def batch_detect(file, ckpt_path: str):
    if file is None:
        return None, "请先上传 CSV 文件", None
    init(ckpt_path)
    meta = STATE["meta"]
    cols = meta["features"]

    df = pd.read_csv(file)
    missing = [c for c in ["V1", "Amount"] if c not in df.columns]
    if missing:
        return None, f"CSV 缺少必要列: {missing}", None

    work = df.copy()
    work["Amount"] = pd.to_numeric(work["Amount"], errors="coerce").fillna(0).clip(lower=0)
    if "Time" not in work.columns:
        work["Time"] = 0.0
    work["Time"] = pd.to_numeric(work["Time"], errors="coerce").fillna(0)

    scaled = STATE["scaler"].transform(work[["Amount", "Time"]])
    work["Amount"] = scaled[:, 0]
    work["Time"] = scaled[:, 1]

    X = work[cols].to_numpy(np.float32)
    prob = predict(Path(ckpt_path), X)

    out = df.copy()
    out["fraud_probability"] = np.round(prob, 6)
    out["flagged"] = (prob >= STATE["thr"]).astype(int)
    out = out.sort_values("fraud_probability", ascending=False).reset_index(drop=True)

    n = len(out)
    n_flag = int(out["flagged"].sum())
    msg = f"共检测 **{n}** 笔交易，标记 **{n_flag}** 笔为可疑（阈值 {STATE['thr']:.3f}）"
    if "Class" in out.columns:
        tp = int(((out["flagged"] == 1) & (out["Class"] == 1)).sum())
        pos = int((out["Class"] == 1).sum())
        msg += f"\n\n其中真实欺诈 **{pos}** 笔，命中 **{tp}** 笔"

    save_path = Path("runs") / "flagged_transactions.csv"
    save_path.parent.mkdir(exist_ok=True)
    out.to_csv(save_path, index=False)

    return out.head(200), msg, str(save_path)


def sample_one(ckpt_path: str, fraud_only: bool):
    init(ckpt_path)
    X, y = _ensure_test()
    idx = np.where(y == 1)[0] if fraud_only else np.arange(len(y))
    if len(idx) == 0:
        return "测试集里没有欺诈样本"
    i = int(np.random.choice(idx))
    prob = float(predict(Path(ckpt_path), X[i : i + 1])[0])
    true = int(y[i])
    flag = "⚠️ 标记为可疑" if prob >= STATE["thr"] else "✅ 判定为正常"
    return (f"### {flag}\n\n"
            f"- 欺诈概率：**{prob:.4f}**（阈值 {STATE['thr']:.3f}）\n"
            f"- 真实标签：**{'欺诈' if true else '正常'}**\n"
            f"- 金额：**{X[i][-2]:.2f}**（标准化后）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mlp_smote/best.pt")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    found = sorted(glob.glob("runs/*/best.pt")) + sorted(glob.glob("runs/*/best.joblib"))
    choices = found or [args.ckpt]
    default = args.ckpt if args.ckpt in choices else choices[0]

    with gr.Blocks(title="信用卡欺诈检测 Demo") as demo:
        gr.Markdown("## 信用卡欺诈检测\n"
                    "上传交易 CSV 做批量检测，或随机抽查单笔交易。")

        with gr.Tab("批量检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    f = gr.File(label="上传 CSV（Time, V1..V28, Amount）", file_types=[".csv"])
                    ckpt1 = gr.Dropdown(choices=choices, value=default, label="模型")
                    b1 = gr.Button("开始检测", variant="primary")
                    msg = gr.Markdown()
                    dl = gr.File(label="下载完整结果")
                with gr.Column(scale=2):
                    tbl = gr.Dataframe(label="检测结果（按风险降序，最多显示 200 行）", wrap=True)
            b1.click(batch_detect, inputs=[f, ckpt1], outputs=[tbl, msg, dl])

        with gr.Tab("单笔抽查"):
            with gr.Row():
                with gr.Column():
                    ckpt2 = gr.Dropdown(choices=choices, value=default, label="模型")
                    only_fraud = gr.Checkbox(value=True, label="只抽真实欺诈样本")
                    b2 = gr.Button("随机抽一笔", variant="primary")
                with gr.Column():
                    out = gr.Markdown()
            b2.click(sample_one, inputs=[ckpt2, only_fraud], outputs=out)

        gr.Markdown("> 阈值是在**验证集**上按业务代价选出来的，不是默认 0.5。"
                    "代价假设：漏检一笔欺诈 = 20 次误报。\n\n"
                    "> 演示用的 `data/demo_transactions.csv` 人为把欺诈比例提到了 2.9%"
                    "（真实数据是 0.17%），否则 2,000 笔里只能看到 3 笔欺诈。")

    print(f"\n>>> 浏览器打开: http://127.0.0.1:{args.port}\n")
    demo.launch(server_port=args.port, share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
