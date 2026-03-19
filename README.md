# Information Synergy in Equity Markets
### A Multi-Ticker Ablation Study of Sentiment and Technical Indicators

[![View Notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/YOUR_NOTEBOOK.ipynb)

## 📌 Project Overview
This research investigates the marginal predictive power of financial news sentiment when combined with traditional technical indicators. We employ an 8-phase pipeline to analyze AAPL, TSLA, and MSFT from 2020-2025.

## 🚀 Key Research Findings
* **Sentiment Value:** Neural sentiment features (SBERT/FinBERT) significantly improve AUC for LSTM-based architectures.
* **Ablation Results:** Hybrid models outperformed technical-only baselines in high-volatility regimes.
* **Explainability:** SHAP analysis identifies "Sentiment Z-Score" as a top-5 predictive feature across all tested tickers.

## 📊 Methodology (8-Phase Pipeline)
1. **Data Collection:** yfinance OHLCV + Kaggle/HuggingFace News.
2. **Preprocessing:** Strict temporal alignment to prevent data leakage.
3. **Sentiment Bake-off:** Comparative analysis of 6 NLP models.
4. **Feature Engineering:** 11 Technical + 14 Sentiment features.
5. **Ablation Study:** 45 models trained across 3 conditions (TA, Sent, Hybrid).
6. **Backtesting:** Simulation with $10,000 capital and 0.1% transaction costs.
7. **Evaluation:** McNemar statistical significance testing.
8. **Explainability:** SHAP & Feature Importance ranking.

## 📂 Repository Structure
* `/data/results`: LaTeX tables and performance metrics.
* `/data/plots`: Publication-quality visualizations.
* `/models`: Pre-trained XGBoost and Scaler objects.

---
*Targeted for publication in **Finance Research Letters**.*
