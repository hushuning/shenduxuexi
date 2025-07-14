"""
主运行脚本：run.py
本脚本用于整合整个流程：
1. 下载 Gate.io 永续合约的 5分钟K线（使用 ccxt）
2. 自动打标签（涨跌预测）
3. 训练深度学习模型（LSTM）
4. 用最新K线进行推理（是否买入/卖出/观望）

依赖：
    pip install ccxt pandas torch
"""

import ccxt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# === 模型定义 ===
class KLineLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=3):
        super(KLineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# === 数据集构建 ===
class KLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def create_dataset(df, seq_len=30):
    X, y = [], []
    for i in range(len(df) - seq_len - 5):
        seq = df[['open', 'high', 'low', 'close']].iloc[i:i+seq_len].values
        future_return = df['close'].iloc[i+seq_len+4] / df['close'].iloc[i+seq_len] - 1
        label = 0
        if future_return > 0.02:
            label = 1
        elif future_return < -0.02:
            label = 2
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# === Step 1: 下载永续合约K线数据 ===
def download_gateio_futures(symbol='BTC/USDT:USDT', timeframe='5m', limit=1000):
    exchange = ccxt.gateio({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        }
    })
    print(f"📥 正在下载 {symbol} 的 {timeframe} 永续合约K线...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close']]
    df.to_csv('data/your_5f_data.csv', index=False)
    print(f"✅ 已保存为 data/your_5f_data.csv，共 {len(df)} 条")
    return df

# === Step 2: 自动打标签 ===
def label_data(df, forward=5, threshold=0.02):
    df['future_return'] = df['close'].shift(-forward) / df['close'] - 1
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1
    df.loc[df['future_return'] < -threshold, 'label'] = 2
    return df

# === Step 3: 训练模型 ===
def train_model(df):
    X, y = create_dataset(df)
    dataset = KLineDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = KLineLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🧠 开始训练模型...")
    for epoch in range(10):
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")
    # os.makedirs("model", exist_ok=True)  # 确保目录存在
    torch.save(model.state_dict(), 'model/lstm_model.pt')
    print("✅ 模型已保存为 model/lstm_model.pt")
    return model

# === Step 4: 推理（使用最新30根K线预测） ===
def predict_latest(df, model):
    last_seq = df[['open', 'high', 'low', 'close']].iloc[-30:].values
    input_tensor = torch.tensor([last_seq], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    label_map = ["观望", "买入", "卖出"]
    print("🔮 最新信号预测结果：", label_map[pred])
    return pred

# === 执行完整流程 ===
if __name__ == "__main__":
    df = download_gateio_futures()
    df = label_data(df)
    model = train_model(df)
    predict_latest(df, model)
