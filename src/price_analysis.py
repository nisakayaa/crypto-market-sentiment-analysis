"""
Crypto Price Analysis
=====================
EDA on cryptocurrency OHLCV data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

COIN_COLORS = {
    "BTC": "#F7931A", "ETH": "#627EEA", "SOL": "#9945FF", "DOGE": "#C2A633",
}


def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"✅ Loaded {len(df):,} price rows for {df['coin'].nunique()} coins")
    return df


def daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["coin", "date"]).copy()
    df["return"] = df.groupby("coin")["close"].pct_change()
    df["log_return"] = np.log(df.groupby("coin")["close"].apply(lambda x: x / x.shift())).reset_index(level=0, drop=True)
    return df


def volatility_summary(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = daily_returns(df)
    vol = df.groupby("coin")["return"].agg(
        mean_return="mean",
        volatility="std",
        max_drawup="max",
        max_drawdown="min",
    )
    vol["annualized_vol"] = vol["volatility"] * np.sqrt(365)
    return vol.round(4)


def plot_price_timeline(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(13, 6))
    for coin in df["coin"].unique():
        sub = df[df["coin"] == coin].sort_values("date")
        # Normalize to start at 100
        normalized = sub["close"] / sub["close"].iloc[0] * 100
        ax.plot(sub["date"], normalized, label=coin, color=COIN_COLORS.get(coin), linewidth=1.5)
    ax.set_title("Crypto Prices (Normalized to 100 at Start)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = load(str(root / "data" / "crypto_prices.csv"))

    print("\n📊 VOLATILITY SUMMARY:")
    print(volatility_summary(df))

    images = root / "images"
    images.mkdir(exist_ok=True)
    plot_price_timeline(df, str(images / "btc_price_history.png"))


if __name__ == "__main__":
    main()
