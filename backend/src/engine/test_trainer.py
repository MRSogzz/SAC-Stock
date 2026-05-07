"""
tests/src/engine/test_trainer.py
==================================
trainer.py 的單元測試，對應真實實作：
  - save_model / load_model / list_models
  - extract_portfolio_rules
  - train（重度 mock，只測介面契約）
  - validate（重度 mock）
  - predict_next（重度 mock）

Mock 策略：
  - 所有 I/O（API、磁碟、FinMind）完全 mock，不需要真實資料
  - train/validate/predict_next 只測「介面契約」：
      回傳格式、必要 key、不 crash
  - 不測訓練效果（那是 integration test 的工作）
"""

import os
import pickle
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

# ── 測試用常數 ────────────────────────────────────────────────────────────────

N_STOCKS    = 2
N_OBS       = 3
N_FEATURES  = 31
STATE_DIM   = N_OBS * N_FEATURES + N_STOCKS * 2 + 1   # 93 + 4 + 1 = 98
SAC_HIDDEN  = 32
TRADEABLE   = ["2330", "2317"]
OBSERVABLE  = ["2330", "2317", "0050"]
STOCK_POOL  = [{"id": "2330", "name": "台積電"},
               {"id": "2317", "name": "鴻海"},
               {"id": "0050", "name": "元大台灣50"}]
INITIAL_CAP = 1_000_000
PERIOD      = "3y"


# ── 假 payload（模擬存入 pkl 的模型資料）─────────────────────────────────────

def _make_fake_actor():
    from src.models.architectures import PortfolioActor
    with patch("src.models.architectures.N_TRADEABLE",   N_STOCKS), \
         patch("src.models.architectures.N_OBSERVABLE",  N_OBS), \
         patch("src.models.architectures.N_FEATURES",    N_FEATURES), \
         patch("src.models.architectures.SAC_HIDDEN",    SAC_HIDDEN), \
         patch("src.models.architectures.STATE_DIM",     STATE_DIM), \
         patch("src.models.architectures.BENCHMARK_IDX", N_OBS - 1), \
         patch("src.models.architectures.N_STOCK_INPUT", N_FEATURES * 2), \
         patch("src.models.architectures.N_PORTFOLIO",   N_STOCKS * 2 + 1):
        return PortfolioActor(STATE_DIM, N_STOCKS, SAC_HIDDEN)


def _make_fake_scaler():
    s = MagicMock()
    s.transform = lambda x: x
    return s


def _make_payload(actor=None) -> dict:
    if actor is None:
        actor = _make_fake_actor()
    return {
        "actor_state":    actor.state_dict(),
        "critic_state":   {},
        "alpha":          1.0,
        "state_dim":      STATE_DIM,
        "n_stocks":       N_STOCKS,
        "stock_ids":      TRADEABLE,
        "scalers":        {sid: _make_fake_scaler() for sid in OBSERVABLE},
        "rules":          {},
        "summary":        {
            "total_return": 5.0, "bh_return": 3.0,
            "risk_free_return": 1.0, "win_rate": 55.0,
            "n_trades": 10, "episodes": 100,
        },
        "training_curve": [1.0, 1.01, 1.02],
        "saved_at":       "2024-01-01 00:00:00",
        "period":         PERIOD,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def config_patch():
    with patch("src.engine.trainer.TRADEABLE_STOCKS",  TRADEABLE), \
         patch("src.engine.trainer.OBSERVABLE_STOCKS", OBSERVABLE), \
         patch("src.engine.trainer.STOCK_POOL",        STOCK_POOL), \
         patch("src.engine.trainer.N_FEATURES",        N_FEATURES), \
         patch("src.engine.trainer.N_OBSERVABLE",      N_OBS), \
         patch("src.engine.trainer.N_TRADEABLE",       N_STOCKS), \
         patch("src.engine.trainer.STATE_DIM",         STATE_DIM), \
         patch("src.engine.trainer.DEVICE",            torch.device("cpu")):
        yield


# ═══════════════════════════════════════════════════════════════════════════════
# save_model / load_model 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveLoadModel:

    def test_save_creates_pkl_file(self, tmp_path):
        """T1：save_model() 後 .pkl 檔案應存在。"""
        from src.engine.trainer import save_model

        agent = MagicMock()
        agent.actor.state_dict.return_value = {}
        agent.critic.state_dict.return_value = {}
        agent.alpha = 1.0
        env = MagicMock()
        env.state_dim = STATE_DIM
        env.n_tradeable = N_STOCKS
        env.tradeable_ids = TRADEABLE
        env.scalers = {}

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            save_model(PERIOD, agent, env, {}, {"episodes": 10}, [1.0])

        pkl = tmp_path / f"portfolio_{PERIOD}.pkl"
        assert pkl.exists()

    def test_load_returns_none_when_missing(self, tmp_path):
        """T2：load_model() 在檔案不存在時回傳 None。"""
        from src.engine.trainer import load_model
        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            result = load_model("nonexistent_period")
        assert result is None

    def test_save_then_load_roundtrip(self, tmp_path):
        """T3：save 後 load → 取回的 payload 包含正確 key。"""
        from src.engine.trainer import save_model, load_model

        agent = MagicMock()
        agent.actor.state_dict.return_value = {"w": torch.tensor([1.0])}
        agent.critic.state_dict.return_value = {}
        agent.alpha = 0.5
        env = MagicMock()
        env.state_dim    = STATE_DIM
        env.n_tradeable  = N_STOCKS
        env.tradeable_ids = TRADEABLE
        env.scalers      = {}

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            save_model(PERIOD, agent, env, {}, {"episodes": 5}, [1.0, 1.01])
            payload = load_model(PERIOD)

        assert payload is not None
        assert payload["state_dim"] == STATE_DIM
        assert payload["alpha"]     == pytest.approx(0.5)
        assert payload["period"]    == PERIOD

    def test_load_actor_state_consistent(self, tmp_path):
        """T4：load 後 actor_state 應與儲存時一致。"""
        from src.engine.trainer import save_model, load_model

        actor = _make_fake_actor()
        sd_before = {k: v.clone() for k, v in actor.state_dict().items()}

        agent = MagicMock()
        agent.actor.state_dict.return_value = actor.state_dict()
        agent.critic.state_dict.return_value = {}
        agent.alpha = 1.0
        env = MagicMock()
        env.state_dim = STATE_DIM; env.n_tradeable = N_STOCKS
        env.tradeable_ids = TRADEABLE; env.scalers = {}

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            save_model(PERIOD, agent, env, {}, {}, [])
            payload = load_model(PERIOD)

        loaded_actor = _make_fake_actor()
        loaded_actor.load_state_dict(payload["actor_state"])
        for k in sd_before:
            assert torch.allclose(sd_before[k], loaded_actor.state_dict()[k])


# ═══════════════════════════════════════════════════════════════════════════════
# list_models 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestListModels:

    def test_empty_dir_returns_empty_list(self, tmp_path):
        """T5：空目錄 → 回傳空 list。"""
        from src.engine.trainer import list_models
        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            result = list_models()
        assert result == []

    def test_finds_saved_models(self, tmp_path):
        """T6：目錄中有 .pkl 檔案 → 回傳包含該模型的 list。"""
        from src.engine.trainer import list_models

        payload = _make_payload()
        pkl_path = tmp_path / f"portfolio_{PERIOD}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            result = list_models()

        assert len(result) == 1
        assert result[0]["period"] == PERIOD

    def test_sorted_by_saved_at_descending(self, tmp_path):
        """T7：多個模型 → 按 saved_at 降序排列。"""
        from src.engine.trainer import list_models

        for i, ts in enumerate(["2024-01-01 00:00:00", "2024-06-01 00:00:00"]):
            p = _make_payload()
            p["saved_at"] = ts
            p["period"]   = f"period_{i}"
            with open(tmp_path / f"portfolio_period_{i}.pkl", "wb") as f:
                pickle.dump(p, f)

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            result = list_models()

        assert result[0]["saved_at"] > result[1]["saved_at"]

    def test_corrupted_pkl_skipped(self, tmp_path):
        """T8：損壞的 pkl → 跳過，不 crash。"""
        from src.engine.trainer import list_models

        (tmp_path / "portfolio_bad.pkl").write_bytes(b"not valid pickle")
        payload = _make_payload()
        with open(tmp_path / "portfolio_good.pkl", "wb") as f:
            pickle.dump(payload, f)

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            result = list_models()

        assert len(result) == 1  # 損壞的被跳過


# ═══════════════════════════════════════════════════════════════════════════════
# extract_portfolio_rules 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractPortfolioRules:

    def _make_actions(self, n: int = 100) -> list:
        rng = np.random.default_rng(0)
        return rng.dirichlet(np.ones(N_STOCKS), size=n).tolist()

    def _make_feat_df(self, n: int = 100) -> dict:
        feat_names = [f"f{i}" for i in range(N_FEATURES)]
        return {
            sid: pd.DataFrame(
                np.random.default_rng(i).standard_normal((n, N_FEATURES)),
                columns=feat_names,
            )
            for i, sid in enumerate(TRADEABLE)
        }

    def test_returns_dict_with_stock_keys(self):
        """T9：回傳 dict，key 為 stock_ids。"""
        from src.engine.trainer import extract_portfolio_rules
        actions   = self._make_actions()
        feat_dict = self._make_feat_df()
        feat_names = [f"f{i}" for i in range(N_FEATURES)]

        result = extract_portfolio_rules(actions, TRADEABLE, feat_dict, feat_names)
        assert set(result.keys()) == set(TRADEABLE)

    def test_each_result_has_required_fields(self):
        """T10：每個股票的結果應包含 top_features / tree_text / n_buy / n_sell。"""
        from src.engine.trainer import extract_portfolio_rules
        actions    = self._make_actions()
        feat_dict  = self._make_feat_df()
        feat_names = [f"f{i}" for i in range(N_FEATURES)]

        result = extract_portfolio_rules(actions, TRADEABLE, feat_dict, feat_names)
        for sid in TRADEABLE:
            assert "top_features" in result[sid]
            assert "tree_text"    in result[sid]
            assert "n_buy"        in result[sid]
            assert "n_sell"       in result[sid]

    def test_insufficient_samples_returns_empty(self):
        """T11：可交易樣本不足 10 筆 → top_features 為空 list，不 crash。"""
        from src.engine.trainer import extract_portfolio_rules

        # 全部都是 hold（action ≈ 0.5），不觸發 buy 或 sell 閾值
        actions = [[0.5, 0.5]] * 5
        feat_dict  = self._make_feat_df(n=5)
        feat_names = [f"f{i}" for i in range(N_FEATURES)]

        result = extract_portfolio_rules(actions, TRADEABLE, feat_dict, feat_names)
        for sid in TRADEABLE:
            assert result[sid]["top_features"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# validate / predict_next 介面契約測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateInterface:

    def test_validate_raises_when_no_model(self, tmp_path):
        """T12：validate() 在找不到模型時 raise ValueError。"""
        from src.engine.trainer import validate
        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="請先訓練"):
                validate(period="missing_period")

    def test_validate_returns_dict_with_required_keys(self, tmp_path):
        """T13：validate() 成功執行後回傳包含必要 key 的 dict。"""
        from src.engine.trainer import validate

        payload = _make_payload()
        with open(tmp_path / f"portfolio_{PERIOD}.pkl", "wb") as f:
            pickle.dump(payload, f)

        n = 200
        fake_stocks = {
            sid: pd.DataFrame({
                "Open": np.ones(n) * 100, "High": np.ones(n) * 101,
                "Low": np.ones(n) * 99,  "Close": np.ones(n) * 100,
                "Volume": np.ones(n) * 10000,
            }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
            for sid in OBSERVABLE
        }

        mock_bt = {
            "initial_capital": INITIAL_CAP, "final_capital": INITIAL_CAP * 1.05,
            "total_profit": INITIAL_CAP * 0.05, "total_return": 5.0,
            "bh_return": 3.0, "risk_free_return": 1.0, "win_rate": 55.0,
            "n_trades": 8, "portfolio_curve": [INITIAL_CAP] * 50,
            "bh_curve": [INITIAL_CAP] * 50, "dates": ["2023-01-01"] * 50,
            "trade_log": [], "avg_positions": {sid: 10.0 for sid in TRADEABLE},
            "all_actions": [[0.5, 0.5]] * 50,
        }

        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)), \
             patch("src.engine.trainer.load_all_stocks", return_value=fake_stocks), \
             patch("src.engine.trainer.align_features") as mock_align, \
             patch("src.engine.trainer.run_backtest", return_value=mock_bt), \
             patch("src.engine.trainer.extract_portfolio_rules", return_value={}), \
             patch("src.models.architectures.N_TRADEABLE",   N_STOCKS), \
             patch("src.models.architectures.N_OBSERVABLE",  N_OBS), \
             patch("src.models.architectures.N_FEATURES",    N_FEATURES), \
             patch("src.models.architectures.SAC_HIDDEN",    SAC_HIDDEN), \
             patch("src.models.architectures.STATE_DIM",     STATE_DIM), \
             patch("src.models.architectures.BENCHMARK_IDX", N_OBS - 1), \
             patch("src.models.architectures.N_STOCK_INPUT", N_FEATURES * 2), \
             patch("src.models.architectures.N_PORTFOLIO",   N_STOCKS * 2 + 1):

            n_val = 80
            feat_df = {
                sid: pd.DataFrame(
                    np.random.randn(n_val, N_FEATURES),
                    columns=[f"f{i}" for i in range(N_FEATURES)],
                    index=pd.date_range("2023-01-01", periods=n_val, freq="B"),
                )
                for sid in OBSERVABLE
            }
            mock_align.return_value = (
                feat_df,
                {sid: np.ones(n_val) * 100 for sid in OBSERVABLE},
                {sid: np.ones(n_val) * 10000 for sid in OBSERVABLE},
                [f"f{i}" for i in range(N_FEATURES)],
                pd.date_range("2023-01-01", periods=n_val, freq="B")
                  .strftime("%Y-%m-%d").tolist(),
            )

            result = validate(period=PERIOD, val_days=50, initial_capital=INITIAL_CAP)

        required = {"mode", "val_days", "val_start", "val_end", "total_return"}
        missing  = required - set(result.keys())
        assert not missing, f"validate 回傳缺少 key：{missing}"


class TestPredictNextInterface:

    def test_predict_next_raises_when_no_model(self, tmp_path):
        """T14：predict_next() 在找不到模型時 raise ValueError。"""
        from src.engine.trainer import predict_next
        with patch("src.engine.trainer.MODEL_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="請先訓練"):
                predict_next(period="missing_period")

    def test_predict_next_returns_recommendations(self, tmp_path):
        """T15：predict_next() 回傳包含 recommendations 的 dict。"""
        from src.engine.trainer import predict_next

        actor  = _make_fake_actor()
        payload = _make_payload(actor)
        with open(tmp_path / f"portfolio_{PERIOD}.pkl", "wb") as f:
            pickle.dump(payload, f)

        n = 300
        close = np.ones(n) * 100
        fake_stocks = {
            sid: pd.DataFrame({
                "Open": close, "High": close * 1.01,
                "Low": close * 0.99, "Close": close, "Volume": np.ones(n) * 10000,
            }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
            for sid in OBSERVABLE
        }

        n_feat = 300
        feat_df = {
            sid: pd.DataFrame(
                np.random.randn(n_feat, N_FEATURES),
                columns=[f"f{i}" for i in range(N_FEATURES)],
                index=pd.date_range("2020-01-01", periods=n_feat, freq="B"),
            )
            for sid in OBSERVABLE
        }

        with patch("src.engine.trainer.MODEL_DIR",    str(tmp_path)), \
             patch("src.engine.trainer.load_all_stocks", return_value=fake_stocks), \
             patch("src.engine.trainer.align_features",
                   return_value=(feat_df, {}, {}, [], [])), \
             patch("src.models.architectures.N_TRADEABLE",   N_STOCKS), \
             patch("src.models.architectures.N_OBSERVABLE",  N_OBS), \
             patch("src.models.architectures.N_FEATURES",    N_FEATURES), \
             patch("src.models.architectures.SAC_HIDDEN",    SAC_HIDDEN), \
             patch("src.models.architectures.STATE_DIM",     STATE_DIM), \
             patch("src.models.architectures.BENCHMARK_IDX", N_OBS - 1), \
             patch("src.models.architectures.N_STOCK_INPUT", N_FEATURES * 2), \
             patch("src.models.architectures.N_PORTFOLIO",   N_STOCKS * 2 + 1), \
             patch("src.engine.trainer.STATE_DIM", STATE_DIM), \
             patch("src.engine.trainer.N_OBSERVABLE", N_OBS):

            result = predict_next(period=PERIOD)

        assert "recommendations" in result
        assert "cash_pct"        in result
        assert isinstance(result["recommendations"], list)