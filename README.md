# 台股 AI 投資組合管理系統

基於 **Soft Actor-Critic（SAC）強化學習**的台股投資組合管理系統，支援 10 支股票同時管理、整張與零股並行交易、每日自動預測，目標是在控制風險的前提下超越無風險利率（年化 1.5%）。

---

## 技術架構

```
backend/
├── configs/
│   ├── base_config.py        # 路徑、裝置（CPU/GPU）、快取設定
│   └── trading_config.py     # 股票池、手續費、SAC 超參數
├── src/
│   ├── data/
│   │   ├── loader.py          # FinMind API 下載 + 24 小時快取
│   │   └── processor.py       # 特徵工程（21 個技術特徵）、標準化、對齊
│   ├── environment/
│   │   ├── portfolio.py       # PortfolioEnv（交易環境、整張/零股狀態追蹤）
│   │   └── reward.py          # CompositeReward（8 項組合式獎勵函數）
│   ├── models/
│   │   └── architectures.py   # PortfolioActor、PortfolioCritic（Twin Q）
│   ├── agents/
│   │   ├── base.py            # BaseAgent 抽象類
│   │   ├── sac_agent.py       # SAC 完整實現
│   │   └── memory.py          # ReplayBuffer
│   ├── engine/
│   │   ├── trainer.py         # train、validate、predict_next、模型存取
│   │   └── backtester.py      # 回測引擎（整張/零股委託、升級邏輯）
│   └── utils/
│       ├── finance.py         # calc_fee、calc_shares、MDD、Sharpe、Sortino
│       └── common.py          # 時間格式、sanitize、safe_float
├── server.py                  # FastAPI 入口（含 APScheduler）
├── main.py                    # CLI 入口（不需啟動後端）
├── daily_predict.py           # Windows 工作排程器用獨立預測腳本
├── run_daily.bat              # 批次執行腳本
└── requirements.txt
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

# GPU 版 PyTorch（RTX 3050 / CUDA 12.4）
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
使用 GPU：NVIDIA GeForce RTX 3050
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
python main.py train --period 5y --episodes 100 --capital 1000000 --val-days 250

# 驗證（載入已存模型，可用不同資金）
python main.py validate --period 5y --val-days 250 --capital 300000

# 預測明日持倉
python main.py predict --period 5y
```

---

## 核心設計

### 強化學習

| 項目 | 設定 |
|------|------|
| 演算法 | Soft Actor-Critic（SAC） |
| 動作空間 | 各股目標倉位（0 ~ 40%，連續） |
| 狀態空間 | 21 特徵 × 10 股 + 10 倉位 + 10 零股比例 + 現金 = 231 維 |
| 批次大小 | 1024 |
| Replay Buffer | 300,000 |
| 更新頻率 | 每 4 步更新一次 |
| Target Entropy | -5.0（= -n_stocks × 0.5） |
| α 最小值 | 0.05（防止探索崩潰） |

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

---

## 模型存取

模型自動存於 `storage/models/portfolio_{period}.pkl`，包含：

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