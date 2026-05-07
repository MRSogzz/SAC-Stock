"""
經驗回放緩衝區
"""
import numpy as np
import torch
from collections import deque

from configs.base_config import DEVICE
from configs.trading_config import SAC_BUFFER_SIZE
from diagnostics import register


class ReplayBuffer:

    @register(
        module="Agent",
        inputs={"capacity": "int"},
        outputs={"return": "ReplayBuffer"},
        notes="環形緩衝區，預設容量 SAC_BUFFER_SIZE；超過 capacity 時自動覆蓋最舊資料",
    )
    def __init__(self, capacity: int = SAC_BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    @register(
        module="Agent",
        inputs={
            "s":    "np.ndarray (STATE_DIM,)",
            "a":    "np.ndarray (N_STOCKS,)",
            "r":    "float",
            "s_":   "np.ndarray (STATE_DIM,)",
            "done": "bool",
        },
        outputs={},
        notes="推入一筆 transition；超過 capacity 自動丟棄最舊的",
    )
    def push(self, s, a, r, s_, done):
        self.buf.append((s, a, r, s_, done))

    def clear(self):
        """清空 buffer，保留原始容量設定（maxlen 不變）。"""
        self.buf.clear()

    @register(
        module="Agent",
        inputs={"batch": "int"},
        outputs={
            "s":    "torch.Tensor (B, STATE_DIM)",
            "a":    "torch.Tensor (B, N_STOCKS)",
            "r":    "torch.Tensor (B, 1)",
            "s_":   "torch.Tensor (B, STATE_DIM)",
            "done": "torch.Tensor (B, 1)",
        },
        notes="隨機不重複抽樣 batch 筆，回傳 FloatTensor 並搬到 DEVICE",
    )
    def sample(self, batch: int):
        idx  = np.random.choice(len(self.buf), batch, replace=False)
        data = [self.buf[i] for i in idx]
        s, a, r, s_, d = zip(*data)
        t = lambda x: torch.FloatTensor(np.array(x)).to(DEVICE)
        return t(s), t(a), t(r).unsqueeze(1), t(s_), t(d).unsqueeze(1)

    def __len__(self):
        return len(self.buf)


class LogitReplayBuffer:
    """
    LogitDelta 專用 buffer，每筆 transition 額外存入 logit_state（L_{t+1}）。

    push()   → (s, a, r, s_, done, logit_state)
    sample() → (s, a, r, s_, done, logit_state) 全部為 FloatTensor on DEVICE

    設計原則：
      - logit_state 是 SACAgentLogitDelta 的內部記憶，不進入 Critic observation
      - 存入的是 L_{t+1}（push_transition 呼叫時機：env.step() 之後，
        _logit_state 已更新）
      - 其餘行為（容量、環形覆蓋、clear）與 ReplayBuffer 完全相同
    """

    def __init__(self, capacity: int = SAC_BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done, logit_state):
        """
        推入一筆 transition，含 logit_state。

        Args:
            s, a, r, s_, done : 標準 SAC transition
            logit_state       : np.ndarray (N_ACTIONS,)，L_{t+1}
        """
        self.buf.append((s, a, r, s_, done, logit_state))

    def clear(self):
        """清空 buffer，保留原始容量設定（maxlen 不變）。"""
        self.buf.clear()

    def sample(self, batch: int):
        """
        隨機不重複抽樣 batch 筆。

        Returns:
            s, a, r, s_, done : FloatTensor on DEVICE（同 ReplayBuffer）
            logit_t1          : FloatTensor (B, N_ACTIONS) on DEVICE
        """
        idx  = np.random.choice(len(self.buf), batch, replace=False)
        data = [self.buf[i] for i in idx]
        s, a, r, s_, d, logit = zip(*data)
        t = lambda x: torch.FloatTensor(np.array(x)).to(DEVICE)
        return (
            t(s), t(a), t(r).unsqueeze(1), t(s_), t(d).unsqueeze(1),
            t(logit),   # (B, N_ACTIONS)
        )

    def __len__(self):
        return len(self.buf)