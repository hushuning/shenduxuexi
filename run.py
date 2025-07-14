# -*- coding: utf-8 -*-
"""
run.py — 主运行脚本（v6：修复 Bokeh 3.x 兼容性）
================================================
新增内容
---------
* **bokeh_boxes()** 重写：
  * 去掉 `source=` 参数导致的 `AttributeError`，改用 `CDSView(filter=...)`。
  * 兼容 Bokeh ≥ 3.0 语法。
* 其余逻辑与 v5 保持一致。

依赖安装
^^^^^^^^
```bash
pip install ccxt pandas numpy torch matplotlib mplfinance plotly bokeh pyecharts jinja2
```

运行
^^^^
```bash
python run.py
```
会生成 4 种箱体图，并完成模型训练与推理。
"""

import os
from typing import List, Dict

import ccxt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import mplfinance as mpf

# 交互库按需导入
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import BoxAnnotation, ColumnDataSource, CDSView, BooleanFilter
except ImportError:
    figure = None
try:
    from pyecharts.charts import Kline
    from pyecharts import options as opts
except ImportError:
    Kline = None

# ==================================
# 目录准备
# ==================================
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ==================================
# 模型定义
# ==================================
class KLineLSTM(nn.Module):
    """两层 LSTM → FC 输出 3 分类（观望/买入/卖出）"""

    def __init__(self, input_size: int = 4, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==================================
# 数据集封装
# ==================================
class KLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataset(df: pd.DataFrame, seq_len=30):
    X, y = [], []
    for i in range(len(df) - seq_len - 5):
        seq = df[['open','high','low','close']].iloc[i:i+seq_len].values
        fut = df['close'].iloc[i+seq_len+4] / df['close'].iloc[i+seq_len] - 1
        label = 1 if fut > 0.02 else 2 if fut < -0.02 else 0
        X.append(seq); y.append(label)
    return np.array(X), np.array(y)

# ==================================
# 数据下载
# ==================================

def download_gateio_futures(symbol='BTC/USDT:USDT', timeframe='5m', limit=1000):
    exchange = ccxt.gateio({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    print(f"📥 下载 {symbol} {timeframe} …")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv('data/klines_5m.csv', index=False)
    print(f"✅ 保存 data/klines_5m.csv ({len(df)} rows)")
    return df

# ==================================
# 箱体检测
# ==================================

def detect_boxes(df: pd.DataFrame, window=60, tol=0.012):
    boxes, i = [], 0
    while i+window <= len(df):
        sub = df.iloc[i:i+window]
        hi, lo = sub['high'].max(), sub['low'].min()
        if (hi-lo)/lo <= tol:
            j = i+window
            while j < len(df) and df['high'].iloc[j]<=hi and df['low'].iloc[j]>=lo:
                j += 1
            boxes.append({'start': df['timestamp'].iloc[i], 'end': df['timestamp'].iloc[j-1],
                          'hi': hi, 'lo': lo})
            i = j
        else:
            i += 1
    print(f"🔍 检测到 {len(boxes)} 个箱体")
    return boxes


# ==================================
# pyecharts HTML
# ==================================

def echarts_boxes(df, boxes, filename='data/kline_boxes_echarts.html'):
    """生成只包含箱体矩形的 ECharts 交互图"""
    if Kline is None:
        print('⚠️  pyecharts 未安装'); return

    # ---- K 线本体 ----
    kline_data = df[['open', 'close', 'low', 'high']].values.tolist()
    dates = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()

    # ---- 箱体 markArea ----
    markareas = []
    ts_to_idx = {t: i for i, t in enumerate(df['timestamp'])}
    for b in boxes:
        if b['start'] not in ts_to_idx or b['end'] not in ts_to_idx:
            continue  # 安全检查
        markareas.append([
            {
                'xAxis': ts_to_idx[b['start']],
                'yAxis': b['lo']
            },
            {
                'xAxis': ts_to_idx[b['end']],
                'yAxis': b['hi']
            }
        ])

    chart = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis('5m', kline_data,
                    itemstyle_opts=opts.ItemStyleOpts(color="#ef5350", color0="#26a69a"))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],
            title_opts=opts.TitleOpts(title='ECharts Box Detection')
        )
    )

    # 单独设置 markArea
    chart.options['series'][0]['markArea'] = {
        'silent': True,
        'itemStyle': {
            'color': 'rgba(52,152,219,0.15)'
        },
        'data': markareas
    }

    chart.render(filename)
    print(f"📈 pyecharts → {filename}")

# ==================================
# 自动打标签
# ==================================

def label_data(df, forward=5, threshold=0.02):
    df = df.copy()
    df['future_return'] = df['close'].shift(-forward) / df['close'] - 1
    df['label'] = np.select([
        df['future_return'] > threshold,
        df['future_return'] < -threshold
    ], [1, 2], default=0)
    return df

# ==================================
# 训练模型
# ==================================

def train_model(df, epochs=5):
    X, y = create_dataset(df)
    loader = DataLoader(KLineDataset(X, y), batch_size=64, shuffle=True)
    model = KLineLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print('🧠 训练 LSTM…')
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        print(f"  Epoch {epoch}/{epochs}  loss={total/len(loader.dataset):.4f}")
    torch.save(model.state_dict(), 'model/lstm_model.pt')
    print('✅ 保存 model/lstm_model.pt')
    return model

# ==================================
# 推理
# ==================================

def predict_latest(df, model, seq_len=30):
    if len(df) < seq_len:
        print('数据不足，无法推理'); return
    seq = df[['open','high','low','close']].iloc[-seq_len:].values
    inp = torch.tensor(seq[None, ...], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = int(torch.argmax(model(inp), dim=1).item())
    print(f"🔮 最新信号 → {['观望','买入','卖出'][pred]}")

# ==================================
# main
# ==================================
if __name__ == '__main__':
    df = download_gateio_futures(limit=1000)
    boxes = detect_boxes(df)

    # 仅输出 ECharts 交互图
    echarts_boxes(df, boxes)

    df_lbl = label_data(df)
    model = train_model(df_lbl, epochs=5)
    predict_latest(df_lbl, model)

    print('🎉 流程结束')
