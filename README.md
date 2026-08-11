# EDF Viewer

Streamlit-based EDF sleep signal viewer.

完整專案：https://github.com/zzawlinhtet62-glitch/edf-viewer

## 功能
- 逐 epoch 瀏覽 PSG 波形（Plotly 互動圖）
- 依睡眠分期跳轉（W / N1 / N2 / N3 / REM）
- 技師判讀 hypnogram 與睡眠指標（SE / TST / SOL / WASO）
- YASA 自動判期與判期比對（confusion matrix、Cohen κ）
- 各階段 EEG 功率頻譜分析

## 執行
pip install -r requirements.txt
streamlit run app.py
