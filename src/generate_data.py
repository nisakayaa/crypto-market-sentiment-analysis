"""
Synthetic Data Generator: Crypto Prices, Social Posts, Fear & Greed Index
==========================================================================
Generates realistic sample data so the project runs out-of-the-box.
Replace with real data from yfinance / Twitter API / Alternative.me in production.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

COINS = {
    "BTC": {"start_price": 42000, "volatility": 0.035, "drift": 0.0008},
    "ETH": {"start_price": 2300,  "volatility": 0.045, "drift": 0.0010},
    "SOL": {"start_price": 95,    "volatility": 0.065, "drift": 0.0012},
    "DOGE": {"start_price": 0.08, "volatility": 0.080, "drift": 0.0005},
}


def generate_prices(start="2024-01-01", days=365):
    """Geometric Brownian Motion OHLCV per coin."""
    dates = pd.date_range(start=start, periods=days, freq="D")
    rows = []
    sentiment_shock = np.random.normal(0, 0.02, size=days)

    for coin, params in COINS.items():
        price = params["start_price"]
        for i, date in enumerate(dates):
            daily_return = np.random.normal(params["drift"],
                                            params["volatility"])
            daily_return += sentiment_shock[i] * 0.3  # market-wide shock

            open_p = price
            close_p = price * (1 + daily_return)
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.01)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(0.5, 2.0) * 1e9 / params["start_price"]

            rows.append({
                "date": date, "coin": coin,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": round(volume, 2),
            })
            price = close_p

    return pd.DataFrame(rows)


def generate_social_posts(prices, posts_per_day=80):
    """Generate sentiment scores per coin per day, correlated with returns."""
    prices = prices.sort_values(["coin", "date"]).copy()
    prices["return"] = prices.groupby("coin")["close"].pct_change().fillna(0)

    rows = []
    for _, row in prices.iterrows():
        # sentiment leads price slightly (we'll mix this in)
        base_sentiment = np.tanh(row["return"] * 8) * 0.4
        n_posts = np.random.poisson(posts_per_day)

        for _ in range(n_posts):
            noise = np.random.normal(0, 0.35)
            score = np.clip(base_sentiment + noise, -1, 1)
            rows.append({
                "date": row["date"],
                "coin": row["coin"],
                "sentiment_score": round(score, 3),
                "source": np.random.choice(["twitter", "reddit", "telegram"],
                                            p=[0.5, 0.35, 0.15]),
            })

    return pd.DataFrame(rows)


def generate_fear_greed(prices):
    """Aggregate fear & greed index (0-100, 0=extreme fear)."""
    btc = prices[prices["coin"] == "BTC"].sort_values("date").copy()
    btc["return_7d"] = btc["close"].pct_change(7).fillna(0)

    # Map return to fear/greed scale with noise
    fg = 50 + btc["return_7d"] * 300 + np.random.normal(0, 8, len(btc))
    fg = np.clip(fg, 0, 100).round(0).astype(int)

    classification = pd.cut(
        fg, bins=[-1, 25, 45, 55, 75, 101],
        labels=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
    )

    return pd.DataFrame({
        "date": btc["date"].values,
        "fear_greed_value": fg.values,
        "classification": classification.values,
    })


def main():
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating crypto prices...")
    prices = generate_prices()
    prices.to_csv(output_dir / "crypto_prices.csv", index=False)
    print(f"  saved {len(prices)} rows -> crypto_prices.csv")

    print("Generating social media posts...")
    posts = generate_social_posts(prices)
    posts.to_csv(output_dir / "social_posts.csv", index=False)
    print(f"  saved {len(posts)} rows -> social_posts.csv")

    print("Generating Fear & Greed index...")
    fg = generate_fear_greed(prices)
    fg.to_csv(output_dir / "fear_greed.csv", index=False)
    print(f"  saved {len(fg)} rows -> fear_greed.csv")

    print("\nAll synthetic data generated in", output_dir)


if __name__ == "__main__":
    main()
