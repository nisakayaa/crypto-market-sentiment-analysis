"""
Sentiment-Price Correlation & Granger Causality Study
======================================================
Tests whether social media sentiment leads cryptocurrency price movements
using Granger causality tests and lagged correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load price and sentiment data."""
    base = Path(__file__).resolve().parents[1] / "data"
    prices = pd.read_csv(base / "crypto_prices.csv", parse_dates=["date"])
    posts = pd.read_csv(base / "social_posts.csv", parse_dates=["date"])
    return prices, posts


def build_daily_panel(prices, posts):
    """Build a daily panel of returns and sentiment per coin."""
    prices = prices.sort_values(["coin", "date"]).copy()
    prices["return"] = prices.groupby("coin")["close"].pct_change()

    daily_sent = (
        posts.groupby(["date", "coin"])["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "sentiment"})
    )

    panel = prices.merge(daily_sent, on=["date", "coin"], how="left")
    panel["sentiment"] = panel.groupby("coin")["sentiment"].ffill().fillna(0)
    return panel.dropna(subset=["return"])


def stationarity_check(series, name):
    """Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    print(f"  ADF on {name}: p-value = {result[1]:.4f} "
          f"({'stationary' if result[1] < 0.05 else 'non-stationary'})")
    return result[1] < 0.05


def granger_test(panel, coin, max_lag=5):
    """Run Granger causality: does sentiment cause returns?"""
    sub = panel[panel["coin"] == coin][["return", "sentiment"]].dropna()
    print(f"\n--- Granger Causality: Sentiment -> {coin} Returns ---")

    stationarity_check(sub["return"], "returns")
    stationarity_check(sub["sentiment"], "sentiment")

    test_data = sub[["return", "sentiment"]]
    results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

    rows = []
    for lag, res in results.items():
        p_val = res[0]["ssr_ftest"][1]
        rows.append({"lag": lag, "p_value": p_val,
                     "significant": p_val < 0.05})
    return pd.DataFrame(rows)


def lagged_correlation(panel, coin, max_lag=7):
    """Pearson correlation between sentiment at t-k and returns at t."""
    sub = panel[panel["coin"] == coin].sort_values("date").copy()
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = sub["return"].corr(sub["sentiment"].shift(-lag))
        else:
            corr = sub["return"].corr(sub["sentiment"].shift(lag))
        correlations[lag] = corr
    return pd.Series(correlations)


def plot_lagged_correlations(panel, output_dir, coins):
    """Plot lagged correlations across coins."""
    fig, axes = plt.subplots(1, len(coins), figsize=(5 * len(coins), 4),
                              sharey=True)
    for ax, coin in zip(axes, coins):
        corrs = lagged_correlation(panel, coin)
        colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in corrs.values]
        ax.bar(corrs.index, corrs.values, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{coin}: Sentiment Lag vs Return")
        ax.set_xlabel("Lag (days)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "lagged_correlations.png", dpi=120,
                bbox_inches="tight")
    plt.close()


def cross_coin_sentiment_corr(posts, output_dir):
    """How correlated is sentiment across different coins?"""
    daily = (posts.groupby(["date", "coin"])["sentiment_score"]
             .mean().unstack())
    corr = daily.corr()

    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.title("Cross-Coin Sentiment Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_coin_sentiment_corr.png", dpi=120,
                bbox_inches="tight")
    plt.close()
    return corr


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    prices, posts = load_data()
    panel = build_daily_panel(prices, posts)

    coins = panel["coin"].unique().tolist()

    print("\n=== Granger Causality Tests ===")
    granger_results = {}
    for coin in coins:
        granger_results[coin] = granger_test(panel, coin)
        print(granger_results[coin].to_string(index=False))

    print("\n=== Lagged Correlation Analysis ===")
    plot_lagged_correlations(panel, output_dir, coins)
    print(f"Saved lagged correlation chart to {output_dir}")

    print("\n=== Cross-Coin Sentiment Correlation ===")
    cross_corr = cross_coin_sentiment_corr(posts, output_dir)
    print(cross_corr.round(2))

    # Save summary
    summary_rows = []
    for coin, df in granger_results.items():
        best = df.loc[df["p_value"].idxmin()]
        summary_rows.append({
            "coin": coin,
            "best_lag": int(best["lag"]),
            "p_value": round(best["p_value"], 4),
            "significant_at_5pct": bool(best["significant"]),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "granger_summary.csv", index=False)
    print(f"\nGranger summary saved to {output_dir / 'granger_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
