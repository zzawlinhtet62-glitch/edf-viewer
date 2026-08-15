# EDF Viewer

# EDF Viewer

以 Streamlit 製作的 EDF 睡眠訊號檢視器，可瀏覽 PSG 波形、
比對技師判讀與 YASA 自動判期，並分析各睡眠階段的 EEG 功率頻譜。

## 功能

- **訊號檢視** — 用 MNE 讀取 EDF，Plotly 互動式波形圖，可縮放平移
- **Epoch 導覽** — ±1 / ±100 跳轉，可直接輸入 epoch 編號，
  或跳到下一個指定睡眠階段（W / N1 / N2 / N3 / REM）
- **睡眠分期** — 自動載入 Hypnogram 標註，顯示當前 epoch 的分期
- **睡眠指標** — 睡眠效率 SE、總睡眠時間 TST、入睡潛伏期 SOL、
  REM 潛伏期、入睡後清醒 WASO、覺醒次數
- **YASA 自動判期** — 用機器學習模型自動分期，與技師判讀並列顯示
- **判期比對** — Confusion matrix、一致率、Cohen's κ
- **各階段時間佔比** — 技師與自動判期的長條圖比較
- **EEG 功率頻譜** — 各睡眠階段的 PSD 曲線與 Delta/Theta/Alpha/Sigma/Beta 頻帶佔比

## 安裝

需要 Python 3.9 以上。

```bash
# 建立環境（建議）
conda create -n edf-viewer python=3.11
conda activate edf-viewer

# 安裝套件
pip install -r requirements.txt
```

主要相依套件：`streamlit`、`mne`、`plotly`、`yasa`、`scipy`、`scikit-learn`

## 資料準備

本專案使用 PhysioNet 的 Sleep-EDF Database。
資料檔案因體積較大未納入版本控制，請自行下載後放入 `data/` 目錄：

```bash
mkdir -p data && cd data
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002E0-PSG.edf
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002EC-Hypnogram.edf
```

## 執行

```bash
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`。

## 操作說明

1. **選擇檔案** — 左側 sidebar 選擇 `data/` 目錄中的檔案，或直接上傳 .edf
2. **選擇頻道** — 勾選要顯示的訊號頻道（EEG / EOG / EMG）
3. **瀏覽波形** — 用波形圖上方的按鈕切換 epoch，或輸入編號直接跳轉
4. **自動判期** — 點 sidebar 的「執行自動判期（YASA）」，需要數分鐘
5. **查看分析** — 波形圖下方的 expander 可展開頻道統計、判期比對、
   各階段時間佔比與功率頻譜

## 注意事項

- YASA 固定使用 30 秒 epoch，其他長度會停用自動判期
- Sleep-EDF 記錄涵蓋上床前後大段清醒時間，
  睡眠指標預設排除記錄前後的長 W 區段以取得合理的 SE

## 課程

AI Agent × Biomedical Signal Analysis — 2026

https://github.com/zzawlinhtet62-glitch/edf-viewer

EDF Viewer — Streamlit 睡眠訊號檢視器
含 epoch 導覽、睡眠指標、YASA 自動判期比對（Cohen κ）
與各階段 EEG 功率頻譜分析。
安裝與操作方式見 README.md。
