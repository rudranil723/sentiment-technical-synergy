# Information Synergy in Equity Markets
### A Multi-Ticker Ablation Study of Sentiment and Technical Indicators

[![View Notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/rudranil723/sentiment-technical-synergy/blob/main/newStockMarketproject.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow.svg)]()

---

## 📌 Project Overview

This research investigates whether financial news sentiment carries independent predictive power
when combined with traditional technical indicators for stock direction forecasting.
We employ an 8-phase reproducible pipeline analysing **AAPL, TSLA, and MSFT** across the
2020 trading year — a period containing the COVID-19 market crash and recovery,
providing a strong test of sentiment signal under market stress.

**Research question:** Does adding NLP-derived sentiment features to technical indicators
produce a statistically significant improvement in stock direction prediction?

---

## 🔑 Key Findings

| Finding | Detail |
|---|---|
| **TSLA sent_ewma5 ranks 2nd by SHAP** | Exponential sentiment decay outranks RSI and Bollinger Bands for Tesla |
| **AAPL BiLSTM +0.142 AUC** | Strongest improvement when sentiment added to deep learning model |
| **MSFT Sharpe 1.220 vs B&H 1.014** | Strategy beats buy-and-hold on risk-adjusted return |
| **Sentiment helps LSTM, not tree models** | Model architecture determines whether sentiment adds value |
| **6-pipeline bake-off** | SBERT wins overall; K-VADER best for TSLA (+0.046 AUC) |

---

## 📊 Methodology — 8-Phase Pipeline
```
Phase 1 → Data Collection       yfinance OHLCV + Kaggle/HuggingFace news (ticker-filtered)
Phase 2 → Preprocessing         Market calendar alignment, 18 hard assertions, zero NaN
Phase 3 → Sentiment Bake-off    6 NLP methods compared: VADER, LM, TF-IDF, K-VADER, FinBERT, SBERT
Phase 4 → Feature Engineering   25 features: 11 Technical + 14 Sentiment (incl. z-score, EWMA, divergence)
Phase 5 → Ablation Study        45 runs: 5 models × 3 conditions × 3 stocks, walk-forward CV
Phase 6 → Backtesting           BUY/HOLD/SELL signals, $10,000 capital, 0.1% transaction costs
Phase 7 → Significance Testing  McNemar test + COVID crash regime analysis
Phase 8 → Explainability        SHAP TreeExplainer, feature importance rankings, LaTeX tables
```

---

## 📈 Results Summary

### Ablation Study — Best AUC per Stock and Condition

| Stock | TA Only | Sentiment Only | Hybrid | Best Model |
|---|---|---|---|---|
| AAPL | 0.619 | 0.583 | **0.626** | BiLSTM |
| TSLA | 0.566 | 0.537 | 0.506 | BiLSTM (TA) |
| MSFT | **0.655** | 0.569 | 0.635 | Random Forest |

### Trading Simulation — Strategy vs Buy-and-Hold

| Stock | Strategy Return | B&H Return | Strategy Sharpe | B&H Sharpe |
|---|---|---|---|---|
| AAPL | +6.0% | +77.6% | 0.415 | 1.490 |
| TSLA | +22.1% | +629.1% | 1.061 | 2.717 |
| MSFT | +16.0% | +40.8% | **1.220** | 1.014 |

> Note: 2020 was an extraordinary bull market year. Sharpe ratio is the correct
> benchmark for signal-based strategies — MSFT strategy beats buy-and-hold on
> risk-adjusted return.

---

## 📂 Repository Structure
```
sentiment-technical-synergy/
│
├── newStockMarketproject.ipynb   ← Full 8-phase research pipeline (85 cells)
├── requirements.txt              ← All library versions for reproducibility
├── README.md
│
└── data/
    ├── results/                  ← ablation_results.csv, backtest_summary.csv
    ├── plots/                    ← 19 publication-quality figures (300 DPI)
    ├── latex/                    ← 4 LaTeX tables ready for paper
    ├── models/                   ← Saved XGBoost + Scaler .pkl files
    └── evaluation/               ← McNemar results, regime analysis
```

---

## 🛠️ How to Run
```bash
# Clone the repo
git clone https://github.com/rudranil723/sentiment-technical-synergy.git
cd sentiment-technical-synergy

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook newStockMarketproject.ipynb
```

> **First run:** Execute all cells from Phase 1 onwards. Takes ~45 minutes
> (FinBERT and SBERT model downloads + LSTM training).
>
> **Subsequent runs:** Use the Fast Restart Cell (Cell 0) to reload all
> saved data in ~30 seconds and jump directly to any phase.

---

## 📦 Requirements

Key libraries: `yfinance`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow`,
`transformers` (FinBERT), `sentence-transformers` (SBERT), `shap`, `nltk`, `scipy`

See `requirements.txt` for exact versions.

---

## 📄 Paper

**Title:** Hybrid Stock Prediction: A Comparative Study of Sentiment Pipelines
and Deep Learning Models

**Target venue:** Finance Research Letters (Elsevier, IF 9.8)

**Status:** Manuscript in preparation

---

## ⚠️ Limitations

- Study covers 2020 only (248 trading days) due to free news dataset availability
- McNemar significance limited by small sample size
- Future work: expand to 3–5 years using a paid news API (NewsAPI.org, Bloomberg)

---

## 📜 License

MIT License — free to use with attribution.

---

*If you find this useful, please star the repo ⭐*
