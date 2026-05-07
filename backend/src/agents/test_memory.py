"""
tests/src/agents/test_memory.py
==================================
memory.py（ReplayBuffer）的單元測試，對應真實實作：
  - __init__(capacity)  → 環形 deque
  - push(s, a, r, s_, done)
  - sample(batch)       → 5 個 FloatTensor，搬到 DEVICE
  - __len__()

Mock 策略：
  - SAC_BUFFER_SIZE → patch，使用小容量加速測試
  - DEVICE          → patch 為 cpu，不要求 CUDA
"""

import numpy as np
import torch
import pytest
from unittest.mock import patch

STATE_DIM  = 131
N_STOCKS   = 3
CAPACITY   = 50
BATCH_SIZE = 8


@pytest.fixture(autouse=True)
def config_patch():
    with patch("src.agents.memory.SAC_BUFFER_SIZE", CAPACITY), \
         patch("src.agents.memory.DEVICE", torch.device("cpu")):
        yield


@pytest.fixture
def buffer():
    from src.agents.memory import ReplayBuffer
    return ReplayBuffer(capacity=CAPACITY)


def _make_transition(seed: int = 0):
    rng = np.random.default_rng(seed)
    s    = rng.standard_normal(STATE_DIM).astype(np.float32)
    a    = rng.dirichlet(np.ones(N_STOCKS)).astype(np.float32)
    r    = float(rng.standard_normal())
    s_   = rng.standard_normal(STATE_DIM).astype(np.float32)
    done = False
    return s, a, r, s_, done


def _push_n(buffer, n: int):
    for i in range(n):
        buffer.push(*_make_transition(seed=i))


# ═══════════════════════════════════════════════════════════════════════════════
# __init__ / __len__ 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferInit:

    def test_initial_length_zero(self, buffer):
        """T1：初始 len() 應為 0。"""
        assert len(buffer) == 0

    def test_capacity_respected(self, buffer):
        """T2：push 超過 capacity 後，len() 應維持在 capacity。"""
        _push_n(buffer, CAPACITY + 20)
        assert len(buffer) == CAPACITY

    def test_oldest_overwritten(self, buffer):
        """T3：超過 capacity 時，最舊資料應被覆蓋（deque maxlen 行為）。"""
        # 先推入 CAPACITY 筆，再推入 1 筆新的
        for i in range(CAPACITY):
            s = np.full(STATE_DIM, float(i), dtype=np.float32)
            buffer.push(s, np.ones(N_STOCKS) / N_STOCKS, 0.0, s, False)

        # 推入第 CAPACITY+1 筆（值為 999.0）
        new_s = np.full(STATE_DIM, 999.0, dtype=np.float32)
        buffer.push(new_s, np.ones(N_STOCKS) / N_STOCKS, 0.0, new_s, False)

        # 最舊的（值為 0.0）應已被推出
        all_s = np.array([buffer.buf[i][0] for i in range(len(buffer))])
        assert not np.any(np.all(np.isclose(all_s, 0.0), axis=1)), (
            "最舊的資料應已被覆蓋"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# push() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestPush:

    def test_push_increments_length(self, buffer):
        """T4：每次 push 後 len() 應正確增加。"""
        assert len(buffer) == 0
        buffer.push(*_make_transition(0))
        assert len(buffer) == 1
        buffer.push(*_make_transition(1))
        assert len(buffer) == 2

    def test_push_stores_correct_data(self, buffer):
        """T5：push 後取出資料應與原始一致。"""
        s, a, r, s_, done = _make_transition(0)
        buffer.push(s, a, r, s_, done)
        stored_s, stored_a, stored_r, stored_s_, stored_done = buffer.buf[0]
        np.testing.assert_array_almost_equal(stored_s, s)
        np.testing.assert_array_almost_equal(stored_a, a)
        assert stored_r == r
        assert stored_done == done


# ═══════════════════════════════════════════════════════════════════════════════
# sample() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestSample:

    def test_sample_returns_five_tensors(self, buffer):
        """T6：sample() 回傳 5 個元素的 tuple。"""
        _push_n(buffer, BATCH_SIZE * 2)
        result = buffer.sample(BATCH_SIZE)
        assert len(result) == 5

    def test_sample_s_shape(self, buffer):
        """T7：sample() s shape 應為 (B, STATE_DIM)。"""
        _push_n(buffer, BATCH_SIZE * 2)
        s, *_ = buffer.sample(BATCH_SIZE)
        assert s.shape == (BATCH_SIZE, STATE_DIM)

    def test_sample_a_shape(self, buffer):
        """T8：sample() a shape 應為 (B, N_STOCKS)。"""
        _push_n(buffer, BATCH_SIZE * 2)
        _, a, *_ = buffer.sample(BATCH_SIZE)
        assert a.shape == (BATCH_SIZE, N_STOCKS)

    def test_sample_r_shape(self, buffer):
        """T9：sample() r shape 應為 (B, 1)（unsqueeze 後）。"""
        _push_n(buffer, BATCH_SIZE * 2)
        _, _, r, *_ = buffer.sample(BATCH_SIZE)
        assert r.shape == (BATCH_SIZE, 1)

    def test_sample_s_prime_shape(self, buffer):
        """T10：sample() s_ shape 應為 (B, STATE_DIM)。"""
        _push_n(buffer, BATCH_SIZE * 2)
        _, _, _, s_, _ = buffer.sample(BATCH_SIZE)
        assert s_.shape == (BATCH_SIZE, STATE_DIM)

    def test_sample_done_shape(self, buffer):
        """T11：sample() done shape 應為 (B, 1)（unsqueeze 後）。"""
        _push_n(buffer, BATCH_SIZE * 2)
        *_, done = buffer.sample(BATCH_SIZE)
        assert done.shape == (BATCH_SIZE, 1)

    def test_sample_returns_float_tensors(self, buffer):
        """T12：所有回傳值均為 torch.FloatTensor（float32）。"""
        _push_n(buffer, BATCH_SIZE * 2)
        tensors = buffer.sample(BATCH_SIZE)
        for t in tensors:
            assert t.dtype == torch.float32

    def test_sample_no_duplicate_when_possible(self, buffer):
        """T13：buffer 充足時 sample 應不重複（replace=False）。"""
        _push_n(buffer, CAPACITY)
        s, *_ = buffer.sample(BATCH_SIZE)
        # 每筆 state 均不相同（因為各筆 seed 不同）
        unique_rows = torch.unique(s, dim=0)
        assert unique_rows.shape[0] == BATCH_SIZE

    def test_sample_insufficient_raises(self, buffer):
        """T14：buffer 不足 batch_size → raise ValueError（np.random.choice 行為）。"""
        _push_n(buffer, 3)
        with pytest.raises(ValueError):
            buffer.sample(BATCH_SIZE)