# -*- coding: utf-8 -*-
"""
run.py — v9 箱体 + 缠论“笔” + LSTM 训练/推理
================================================
功能一览
--------
1. **数据下载**：Gate.io BTC/USDT 永续 5‑min K 线
2. **箱体检测**：滑动窗口 + 波动率阈值
3. **缠论笔**：最简单分型→笔算法
4. **ECharts 可视化**：箱体 (蓝) + 笔折线 (紫)
5. **LSTM 训练 / 推理**
   * 输入：最近 30 根 OHLC
   * 标签：未来 5 根收盘价 ±2 % 区间（观望/买入/卖出）
   * 训练 3 轮仅作演示

依赖
----
```bash
pip install ccxt pandas numpy torch pyecharts jinja2
```

运行
----
```bash
python run.py
```
* 浏览 `data/kline_boxes_echarts.html` 看图
* 终端显示最新信号
"""

import os
from typing import List, Dict

import ccxt
import numpy as np
import pandas as pd

# ─────────── 机器学习 ───────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─────────── 交互图 ───────────
try:
    from pyecharts.charts import Kline, Line, Grid
    from pyecharts import options as opts
except ImportError:
    Kline = None  # 告知用户安装

# ─────────── 目录 ───────────
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# ============================= 数据 =============================

def download_gateio_futures(symbol='BTC/USDT:USDT', timeframe='5m', limit=1000):
    ex = ccxt.gateio({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    print('📥 下载 K 线 …')
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ============================= 箱体 =============================

def detect_boxes(df: pd.DataFrame, window=60, tol=0.012):
    boxes, i = [], 0
    while i + window <= len(df):
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
    print(f'🔍 箱体: {len(boxes)}')
    return boxes

# ============================= 缠论笔 ============================

def detect_fractals(df: pd.DataFrame):
    highs, lows = [], []
    for i in range(2, len(df)-2):
        if df['high'].iloc[i] > df['high'].iloc[i-2:i+3].drop(df.index[i]).max():
            highs.append((i, df['high'].iloc[i]))
        if df['low'].iloc[i] < df['low'].iloc[i-2:i+3].drop(df.index[i]).min():
            lows.append((i, df['low'].iloc[i]))
    return highs, lows


def merge_fractals(highs, lows):
    pts = sorted(highs + lows, key=lambda x: x[0])
    filt = [pts[0]]
    for idx, price in pts[1:]:
        last_idx, last_price = filt[-1]
        same_trend = (price > last_price and last_price in [p for _, p in highs]) or \
                     (price < last_price and last_price in [p for _, p in lows])
        if same_trend:
            if (price > last_price) == (price in [p for _, p in highs]):
                filt[-1] = (idx, price)
        else:
            filt.append((idx, price))
    res = [filt[0]]
    for idx, price in filt[1:]:
        if (price > res[-1][1]) != (res[-1] in highs):
            res.append((idx, price))
    return res


def detect_bi_strokes(df: pd.DataFrame):
    highs, lows = detect_fractals(df)
    pivots = merge_fractals(highs, lows)
    print(f'✅ 笔端点: {len(pivots)}')
    return pivots  # (idx, price)

# ============================= LSTM 模型 =========================

class KLineLSTM(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden, 2, batch_first=True)
        self.fc = nn.Linear(hidden, 3)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])

class KLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataset(df, seq_len=30, forward=5):
    X, y = [], []
    for i in range(len(df)-seq_len-forward):
        seq = df[['open','high','low','close']].iloc[i:i+seq_len].values
        fut_ret = df['close'].iloc[i+seq_len+forward-1]/df['close'].iloc[i+seq_len-1]-1
        label = 1 if fut_ret>0.02 else 2 if fut_ret<-0.02 else 0
        X.append(seq); y.append(label)
    return np.array(X), np.array(y)


def train_model(df, epochs=3):
    X, y = create_dataset(df)
    model = KLineLSTM()
    loader = DataLoader(KLineDataset(X, y), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        tot=0
        for xb,yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(xb)
        print(f'Epoch {ep}/{epochs} loss {tot/len(loader.dataset):.4f}')
    torch.save(model.state_dict(),'model/lstm_model.pt')
    return model


def predict_latest(df, model, seq_len=30):
    seq = df[['open','high','low','close']].iloc[-seq_len:].values
    pred = model(torch.tensor(seq[None,...], dtype=torch.float32))
    idx = int(torch.argmax(pred, dim=1).item())
    print('🔮 最新信号 →', ['观望','买入','卖出'][idx])

# ============================= ECharts 绘图 ====================

def build_label_points(df: pd.DataFrame, seq_len=30, forward=5, threshold=0.02):
    """返回买入/卖出信号点坐标 (idx, price, label)"""
    points = []
    for i in range(len(df) - seq_len - forward):
        fut_ret = df['close'].iloc[i+seq_len+forward-1]/df['close'].iloc[i+seq_len-1]-1
        label = 1 if fut_ret>threshold else 2 if fut_ret<-threshold else 0
        if label:
            idx = i + seq_len - 1  # 信号时刻 = 窗口末
            price = df['close'].iloc[idx]
            points.append((idx, price, label))
    return points

def detect_non_overlapping_boxes(df: pd.DataFrame, boxes: List[Dict]):
    """查找所有没有重合的最近的两个箱体"""
    non_overlapping_boxes = []
    for i in range(len(boxes) - 1):
        box1 = boxes[i]
        box2 = boxes[i + 1]

        # 检查两个箱体是否有重叠
        if  box2['lo'] > box1['hi'] or box2['hi'] < box1['lo']:
            # 两个箱体没有重叠，标记为红色
            print(f"找到了没有重叠的箱子")
            non_overlapping_boxes.append((box1, box2))
        else:
            print("没有找到没有重叠的箱体")
    return non_overlapping_boxes


def echarts_boxes(df: pd.DataFrame, boxes: List[Dict], pivots, labels_pts=None, filename='data/kline_boxes_echarts.html'):
    """单一 ECharts 图层：箱体 + 缠论笔折线 + 买/卖箭头"""
    if Kline is None:
        print('⚠️  缺少 pyecharts'); return

    # ---- 1. K 线数据 ----
    kline_data = df[['open','close','low','high']].values.tolist()
    dates = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    ts2idx = {t:i for i,t in enumerate(df['timestamp'])}

    # ---- 2. 箱体 markArea ----
    markareas = [[{'xAxis': ts2idx[b['start']], 'yAxis': b['lo']},
                  {'xAxis': ts2idx[b['end']],   'yAxis': b['hi']}]
                 for b in boxes if b['start'] in ts2idx and b['end'] in ts2idx]

    # ---- 3. 找到没有重叠的箱体并标记为红色 ----
    non_overlapping_boxes = detect_non_overlapping_boxes(df, boxes)
    non_overlapping_markareas = []
    for box1, box2 in non_overlapping_boxes:
        # 标记第一个箱体
        non_overlapping_markareas.append([{'xAxis': ts2idx[box1['start']], 'yAxis': box1['lo']},
                                          {'xAxis': ts2idx[box1['end']], 'yAxis': box1['hi']}])
        # 标记第二个箱体
        non_overlapping_markareas.append([{'xAxis': ts2idx[box2['start']], 'yAxis': box2['lo']},
                                          {'xAxis': ts2idx[box2['end']], 'yAxis': box2['hi']}])

    # ---- 4. 笔折线 (追加为第二个 series) ----·
    # line_series = {
    #     'type': 'line',
    #     'name': 'Bi',
    #     'data': [[idx, price] for idx, price in pivots],
    #     'symbol': 'none',
    #     'lineStyle': {'color': '#ab47bc', 'width': 2},
    #     'encode': {'x': 0, 'y': 1},
    #     'tooltip': {'show': False},
    # }

    # ---- 5. 标签 markPoint ----
    mp_data = []
    if labels_pts:
        for idx, price, lab in labels_pts:
            mp_data.append({
                'coord': [idx, price],
                'symbol': 'triangle',
                'symbolRotate': 0 if lab==1 else 180,
                'symbolSize': 10,
                'itemStyle': {'color': "#e6000c" if lab==1 else '#ff1744'}
            })

    # ---- 6. 组合成图表 ----
    chart = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis('5m', kline_data,
                    itemstyle_opts=opts.ItemStyleOpts(color='#ef5350', color0='#26a69a'))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],
            title_opts=opts.TitleOpts(title='箱体 + 缠论笔 + 标签')
        )
    )

    # 添加普通箱体
    chart.options['series'][0]['markArea'] ={ 
        'silent': True,
        'itemStyle': {'color':'rgba(52,152,219,0.12)'},
        'data': non_overlapping_markareas
    
    }
    # chart.options['series'][0]['markArea']['data'].extend([{
    #     'silent': True,
    #     'itemStyle': {'color': 'rgba(255,0,0,0.3)'}  # 红色背景
    # }] + non_overlapping_markareas)  # 合并两者，确保红色箱体与蓝色箱体在同一个数据结构中

    # chart.options['series'][0]['markArea'] = {
    #     'silent': True,
    #     'itemStyle': {'color':'rgba(0,0,0,0.1)'},
    #     'data': non_overlapping_markareas
    # }
    # 添加没有重叠的箱体标记为红色
    # chart.options['series'][0]['markArea']['data'].extend(non_overlapping_markareas)

    # 注入马点（买卖信号）
    if mp_data:
        chart.options['series'][0]['markPoint'] = {'data': mp_data}

    # 追加笔折线 series
    # chart.options['series'].append(line_series)

    chart.render(filename)
    print(f'📈 图表 → {filename}')


# ============================= main =============================
if __name__ == '__main__':
    # 1️⃣ 下载数据
    df = download_gateio_futures(limit=100000)

    # 2️⃣ 箱体 + 缠论笔
    boxes  = detect_boxes(df)
    pivots = detect_bi_strokes(df)

    # 3️⃣ 生成标签点（买 = 绿三角↑，卖 = 红三角↓）
    # label_pts = build_label_points(df)

    # 4️⃣ 绘图：箱体 + 笔 + 标签
    echarts_boxes(df, boxes, pivots,None)

    # 5️⃣ LSTM 训练 + 推理（演示 3 轮即可）
    model = train_model(df, epochs=3)
    predict_latest(df, model)

    print('🎉 流程结束')

