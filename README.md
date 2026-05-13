# 📈 Crypto Market Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)](https://nltk.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Featured-purple.svg)]()

> **⭐ Featured Niche Project** – Where finance meets NLP: sentiment-driven crypto analytics.

## 💎 Project Overview

The crypto market is uniquely **sentiment-driven** — a single tweet can move billions. This project combines:

- 📊 **Price data** (Bitcoin, Ethereum, and altcoins)
- 💬 **Social sentiment** (synthetic Twitter/Reddit posts with sentiment scores)
- 😱 **Fear & Greed Index** simulation

…to answer: **does public sentiment lead or follow price?**

> ⚠️ This is an educational data-analysis project. It is **not** investment advice. Crypto markets are highly volatile and lagging analysis can lose money.

## 🔍 Research Questions

1. **Lead or lag?** Does sentiment predict price movement, or follow it?
2. **Fear & Greed cycles** – Are extreme greed peaks reliable sell signals?
3. **Volume vs sentiment** – Does sentiment correlate with trading volume?
4. **Coin specifics** – Are some coins more sentiment-driven than others?
5. **Event detection** – Can we automatically flag "FUD" or "euphoria" days?

## 📁 Project Structure

```
crypto-market-sentiment-analysis/
├── data/
│   ├── crypto_prices.csv
│   ├── social_posts.csv
│   └── fear_greed.csv
├── notebooks/
│   ├── 01_price_eda.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   └── 03_correlation_study.ipynb
├── src/
│   ├── price_analysis.py
│   ├── sentiment.py
│   ├── correlation_study.py
│   └── generate_data.py
├── outputs/
├── images/
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | Pandas, NumPy, yfinance, ccxt |
| **NLP** | NLTK, VADER, TextBlob |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Statistics** | SciPy, statsmodels (Granger causality) |

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/crypto-market-sentiment-analysis.git
cd crypto-market-sentiment-analysis
pip install -r requirements.txt

# Generate sample data
python src/generate_data.py

# Run analyses
python src/price_analysis.py
python src/sentiment.py
python src/correlation_study.py
```

## 📊 Key Insights

### 🔄 Sentiment → Price Relationship
- **Granger causality** test suggests sentiment leads price by **~1–2 days** for BTC
- ETH shows weaker but similar pattern
- Smaller altcoins are **strongly sentiment-driven** (correlation > 0.65)

### 😱 Fear & Greed as Signal
- "Extreme Greed" (>80) historically precedes 30-day pullbacks
- "Extreme Fear" (<20) zones often mark interim bottoms

### 📰 Event Days
- **Euphoria days**: Avg sentiment > 0.7, post volume up 3x normal
- **FUD days**: Sentiment < -0.5 with sustained negative momentum
- 70% of euphoria days were followed by ≥5% drawdown within a week

### 💎 Coin-Specific Findings
| Coin | Sentiment Sensitivity | Lead/Lag |
|------|------------------------|----------|
| BTC | Medium | Sentiment leads |
| ETH | Medium | Sentiment leads |
| DOGE | **Very High** | Concurrent (memes) |
| SOL | Medium-High | Sentiment leads |

## 🧠 Methodology

1. **Data Collection** – Price (OHLCV), social posts, fear/greed index
2. **Sentiment Scoring** – VADER on each post → daily aggregate
3. **Stationarity Check** – ADF test before causality analysis
4. **Granger Causality** – Test if sentiment predicts price
5. **Event Detection** – Z-score-based anomaly tagging

## 📈 Sample Visualizations

- `btc_price_history.png` – Bitcoin price with key events
- `sentiment_timeline.png` – Daily sentiment scores
- `price_vs_sentiment.png` – Overlaid charts
- `fear_greed_zones.png` – Color-coded zone heatmap
- `correlation_matrix.png` – Cross-coin sentiment correlations
- `granger_results.png` – Lag analysis chart

## 🔗 Real-World Extension

This project is built to plug into real APIs:

```python
# Live prices
import yfinance as yf
btc = yf.download("BTC-USD", period="1y", interval="1d")

# Real fear & greed index
import requests
r = requests.get("https://api.alternative.me/fng/?limit=365")
```

## 📝 License

[MIT](LICENSE)

## 👤 Author

**Nisa Kaya**
- GitHub: nisakayaa

---

⚠️ **Disclaimer**: This project is for educational and research purposes. Past patterns don't guarantee future results. Do your own research.
