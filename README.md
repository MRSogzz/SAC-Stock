# 台股 AI 投資組合管理系統

基於 **Soft Actor-Critic（SAC）強化學習**的台股投資組合管理系統，支援 10 支股票同時管理、整張與零股並行交易、每日自動預測，目標是在控制風險的前提下超越無風險利率（年化 1.5%）。

---

## 技術架構

```
專案根目錄
├── visualizer.html                    # 獨立的視覺化頁面
├── backend/                           # Python 後端（核心運算與 API 服務）
│   ├── configs/
│   │   ├── base_config.py             # 路徑、裝置（CPU/GPU）、快取設定
│   │   └── trading_config.py          # 股票池、手續費、SAC 超參數
│   ├── src/
│   │   ├── agents/
│   │   │   ├── base.py                # BaseAgent 抽象類
│   │   │   ├── sac_agent.py           # SAC 完整實現
│   │   │   └── memory.py              # ReplayBuffer / LogitReplayBuffer
│   │   ├── data/
│   │   │   ├── loader.py              # FinMind API 下載 + 24 小時快取
│   │   │   └── processor.py           # 特徵工程（38 個技術特徵）、標準化、對齊
│   │   ├── engine/
│   │   │   ├── backtester.py          # 回測引擎（整張/零股委託、升級邏輯）
│   │   │   ├── trainer_standard.py    # 標準訓練器
│   │   │   └── trainer_walk_forward.py# Walk-Forward 訓練器
│   │   ├── environment/
│   │   │   ├── portfolio.py           # PortfolioEnv（交易環境、整張/零股狀態追蹤）
│   │   │   ├── reshaping_engine.py    # 獎勵重塑引擎
│   │   │   └── reward.py              # CompositeReward（8 項組合式獎勵函數）
│   │   ├── inference/
│   │   │   └── predictor.py           # 使用已訓練模型進行每日預測
│   │   ├── models/
│   │   │   │   └── architectures.py       # PortfolioActorLogitDelta、PortfolioCritic（IQN Twin Q）
│   │   └── utils/
│   │       ├── finance.py             # calc_fee、calc_shares、MDD、Sharpe、Sortino
│   │       └── common.py              # 時間格式、sanitize、safe_float
│   ├── diagnostics/                   # 深度分析工具（因子審計、摩擦歸因、策略解剖等）
│   │   └── output/                    # 診斷圖表與 JSON 報告輸出
│   ├── evaluation/                    # 三層評估框架
│   │   ├── layer1_signal.py           # 信號品質
│   │   ├── layer2_strategy.py         # 策略表現
│   │   └── layer3_robustness.py       # 穩健性測試
│   ├── reports/alpha_validation/      # 已存檔的因子驗證 JSON 報告
│   ├── routers/
│   │   └── alpha_validation.py        # Alpha 因子驗證 API 路由
│   ├── storage/
│   │   ├── cache/                     # 依股票代碼與時間區間快取的市場資料 CSV
│   │   ├── history/                   # 歷史預測紀錄（predictions.csv）與除錯日誌
│   │   └── models/                    # Walk-Forward 訓練的模型 .pkl 檔
│   ├── server.py                      # FastAPI 入口（含 APScheduler）
│   ├── main.py                        # CLI 入口（不需啟動後端）
│   ├── daily_predict.py               # Windows 工作排程器用獨立預測腳本
│   ├── run_daily.bat                  # 批次執行腳本
│   ├── calc_ic.py                     # 計算 IC（資訊係數）
│   └── requirements.txt
└── frontend/                          # React + Vite 前端（儀表板 UI）
    ├── features/
    │   ├── AlphaValidation/           # 因子驗證介面（三層評估、報告清單、判決徽章）
    │   ├── StandardTraining/          # 標準訓練控制介面
    │   ├── WalkForwardTraining/       # Walk-Forward 訓練監控
    │   └── SystemStatus/             # 系統狀態總覽與排程器面板
    ├── components/                    # 共用元件（MetricCard、PortfolioChart 等）
    ├── hooks/                         # 自訂 Hooks（useTraining、useWalkForward 等）
    └── constants/config.js            # 前端組態（後端 API 網址等）
```

---

## 股票池（10 支）

| 代號 | 名稱 | 產業 |
|------|------|------|
| 2330 | 台積電 | 半導體 |
| 2317 | 鴻海 | 電子製造 |
| 2454 | 聯發科 | IC 設計 |
| 2412 | 中華電 | 電信 |
| 2308 | 台達電 | 電源 |
| 2882 | 國泰金 | 金融 |
| 1301 | 台塑 | 石化 |
| 2002 | 中鋼 | 鋼鐵 |
| 2886 | 兆豐金 | 金融 |
| 0050 | 元大台灣50 | ETF |

---

## 安裝與啟動

### 環境需求

- Python 3.10+
- CUDA 12.4（建議，支援 GPU 加速）
- Windows 10/11

### 安裝相依套件

```bash
pip install -r requirements.txt

# GPU 版 PyTorch（RTX xxxx / CUDA 12.4）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# APScheduler
pip install apscheduler pytz
```

### 啟動後端

```bash
cd backend
uvicorn server:app --reload --port 8000
```

啟動時會自動顯示使用的運算裝置：

```
使用 GPU：NVIDIA GeForce RTX xxxx
APScheduler 已啟動，下次執行：2025-xx-xx 15:30:00+08:00
```

### 啟動前端

```bash
cd frontend
npm install
npm run dev
```

---

## CLI 使用方式

不需啟動後端，直接透過命令列訓練或驗證：

```bash
# 訓練
python main.py train --period 6y --episodes 100 --capital 1000000 --val-days 250

# 驗證（載入已存模型，可用不同資金）
python main.py validate --period 6y --val-days 250 --capital 300000

# 預測明日持倉
python main.py predict --period 6y
```

---

## 核心設計

### 強化學習

| 項目 | 設定 |
|------|------|
| 演算法 | Soft Actor-Critic（SAC） |
| 動作空間 | 各股目標倉位（0 ~ 40%，連續） |
| 狀態空間 | 38 特徵 × 10 股 + 9 倉位 + 9 零股比例 + 現金 = 399 維 |
| 批次大小 | 1024 |
| Replay Buffer | 500,000 |
| 更新頻率 | 每 2 步更新一次 |
| Target Entropy | -2.1 |
| α 最小值 | 0.1（防止探索崩潰） |

### Actor 架構（Variant H）

現役 Actor 為 **PortfolioActorLogitDelta**，採用 SharedFeatureExtractor + LogitDelta 設計：

- **SharedFeatureExtractor**：所有股票共用同一套權重，每股 38 維特徵 → 32 維 embedding
- **LogitDelta**：`L_{t+1} = 0.995 × L_t + ΔL`，透過 Leaky Integrator 產生有界連續動作
- **Critic**：`RegimeConditionedIQNCritic`（IQN 分位數 Q-network，支援市場狀態條件化）
- **ReplayBuffer**：`LogitReplayBuffer`，每筆 transition 額外儲存 `logit_state` 與 `regime_label`

- **標準訓練**：固定訓練期間，單次跑完後驗證，適合快速迭代調參。
- **Walk-Forward 訓練**：滾動視窗訓練與驗證，每個視窗結果可跨 Run 比較，用於評估策略穩健性。

### 獎勵函數（8 項）

1. **持倉收益**：非對稱設計，虧損懲罰 ×2
2. **交易成本**：買賣手續費分開計算，含最低手續費近似
3. **半方差懲罰**：Sortino 概念，只懲罰下行波動
4. **最大回撤懲罰**：即時追蹤 peak，回撤越大扣越多
5. **持倉衰減**：趨勢強時衰減慢，趨勢弱時加快，防止過度戀棧
6. **無風險利率門檻**：低於年化 1.5% 額外扣分
7. **零股浪費懲罰**：零股市值佔總資產比例越高越扣，越接近整張二次懲罰越重
8. **升級成本**：零股湊整時評估是否划算，成本計入獎勵

### 台股交易成本

| 項目 | 數值 |
|------|------|
| 手續費 | 0.1425%（買賣各收） |
| 證交稅 | 0.3%（賣出時收） |
| 最低手續費 | 20 元 |
| 整張單位 | 1,000 股 |

### 整張與零股並行

系統依目標倉位金額自動拆分委託：

- **整張委託**：可買幾張就買幾張，各自計費
- **零股委託**：剩餘不足整張的部分用零股補足
- **自動升級**：零股累積至 1000 股時，評估「未來累積懲罰 vs 升級手續費」，划算才執行升級（賣零股 → 買整張）
- **減倉規則**：整張只能整張賣，不足一張的零頭保留為零股

### Alpha 因子驗證

`evaluation/` 模組提供三層結構化評估框架：

| 層級 | 模組 | 職責 |
|------|------|------|
| Layer 1 | `layer1_signal.py` | 信號品質（IC、Rank IC 等） |
| Layer 2 | `layer2_strategy.py` | 策略表現（回測、換手率等） |
| Layer 3 | `layer3_robustness.py` | 穩健性測試（壓力測試、過擬合檢測等） |

驗證結果 JSON 報告存於 `reports/alpha_validation/`，涵蓋 RSI、布林帶、成交量等數十種技術指標。`diagnostics/` 則提供更深度的分析，包含因子存在性審計、Alpha 幾何結構辨識、摩擦歸因（分四階段）與可塑性探測，輸出圖表與 JSON 至 `diagnostics/output/`。

---

## 模型存取

模型自動存於 `storage/models/`，包含：

- Actor / Critic 網路權重
- StandardScaler（各股特徵標準化參數）
- 訓練摘要（報酬率、勝率、回合數）
- 完整訓練曲線
- α 值（繼續訓練用）

**繼續訓練**：直接再次按訓練按鈕，系統自動載入已存模型從上次結束的回合繼續。

**跨資金驗證**：模型學習的是倉位比例，與初始資金無關。同一個模型可用不同資金跑驗證，不需重新訓練。

---

## 每日自動預測

### 方式一：後端常駐（APScheduler）

後端啟動後，APScheduler 自動在每週一至週五 **15:30**（台灣時間）執行預測，結果寫入 `storage/history/predictions.csv`，次日自動回填實際漲跌。

### 方式二：Windows 工作排程器

適合不需要前端的情境，後端不用常駐：

1. 編輯 `run_daily.bat`，填入虛擬環境和後端路徑
2. 在 Windows 工作排程器新增每日 15:30 的任務，指向 `run_daily.bat`
3. 結果存於 `storage/history/`，`run_log.txt` 記錄每次執行狀態

---

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/train` | 開始訓練（背景執行） |
| POST | `/validate` | 驗證已存模型 |
| GET | `/status/{job_id}` | 查詢訓練/驗證進度 |
| GET | `/result/{job_id}` | 取得訓練/驗證結果 |
| GET | `/predict/{period}` | 明日持倉建議 |
| GET | `/models` | 列出已存模型 |
| GET | `/stock-pool` | 股票池清單 |
| GET | `/alpha-validation/{feature}` | 指定特徵的因子驗證結果 |
| GET | `/scheduler/status` | 排程狀態 |
| POST | `/scheduler/run-now` | 立即執行預測 |
| GET | `/scheduler/history` | 歷史預測紀錄 |
| GET | `/health` | 健康檢查 |

---

## 注意事項

- 本系統之預測結果**僅供學術研究與參考**，不構成任何投資建議
- 強化學習模型的回測表現不代表未來實際績效
- 小資金（< 10 萬）因受最低手續費 20 元影響，實際交易成本比例較高，建議以驗證集結果為主要參考
- `torch.compile` 在 Windows 上需要 Triton，目前不支援，系統已自動跳過