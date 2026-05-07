@echo off
:: 每日自動預測批次檔
:: 由 Windows 工作排程器在每天 15:30 呼叫
::
:: 設定方式：
::   1. 修改下方 VENV 和 BACKEND 路徑
::   2. 開啟「工作排程器」→ 建立基本工作
::   3. 觸發程序：每天 15:30（週一至週五）
::   4. 動作：啟動程式，指向本批次檔完整路徑
::   5. 條件：取消勾選「只在 AC 電源時執行」

:: ── 請修改以下路徑 ──────────────────────────────────────────────────────────
set VENV=D:\work\python\stock\.venv
set BACKEND=D:\work\python\stock-ai\backend
:: ────────────────────────────────────────────────────────────────────────────

echo [%date% %time%] 開始執行每日預測...

call "%VENV%\Scripts\activate.bat"
cd /d "%BACKEND%"
python daily_predict.py

echo [%date% %time%] 執行完畢