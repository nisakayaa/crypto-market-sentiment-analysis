"""
Sentiment Analysis on Social Posts
====================================
Aggregates daily sentiment per coin and detects sentiment events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"✅ Loaded {len(df):,} social posts")
    return df


def daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment per coin per day."""
    return (
        df.groupby(["date", "coin"])
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            n_posts=("sentiment_score", "count"),
            pct_positive=("sentiment_score", lambda x: (x > 0.2).mean()),
            pct_negative=("sentiment_score", lambda x: (x < -0.2).mean()),
        )
        .reset_index()
    )


def detect_events(daily: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    """Flag days with statistically extreme sentiment or volume."""
    daily = daily.copy()
    daily["sentiment_z"] = daily.groupby("coin")["avg_sentiment"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    daily["volume_z"] = daily.groupby("coin")["n_posts"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    def label(row):
        if row["sentiment_z"] > z_threshold and row["volume_z"] > z_threshold:
            return "🚀 Euphoria"
        if row["sentiment_z"] < -z_threshold and row["volume_z"] > z_threshold:
            return "😱 FUD"
        if row["volume_z"] > z_threshold * 1.5:
            return "📢 High Volume"
        return "—"

    daily["event"] = daily.apply(label, axis=1)
    return daily


def plot_sentiment_timeline(daily: pd.DataFrame, coin: str = "BTC", save_path: str = None):
    sub = daily[daily["coin"] == coin].sort_values("date")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub["date"], sub["avg_sentiment"], color="purple", linewidth=1.2)
    ax.fill_between(sub["date"], 0, sub["avg_sentiment"],
                     where=sub["avg_sentiment"] >= 0, color="green", alpha=0.3, label="Positive")
    ax.fill_between(sub["date"], 0, sub["avg_sentiment"],
                     where=sub["avg_sentiment"] < 0, color="red", alpha=0.3, label="Negative")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"{coin} Social Sentiment Over Time", fontweight="bold")
    ax.set_ylabel("Avg Sentiment (-1 to 1)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = load(str(root / "data" / "social_posts.csv"))

    daily = daily_sentiment(df)
    daily_with_events = detect_events(daily)

    print("\n📊 SENTIMENT SUMMARY:")
    print(daily.groupby("coin")[["avg_sentiment", "n_posts", "pct_positive"]].mean().round(3))

    print("\n🚨 RECENT EVENT DAYS:")
    events_only = daily_with_events[daily_with_events["event"] != "—"].tail(15)
    print(events_only[["date", "coin", "avg_sentiment", "n_posts", "event"]].to_string(index=False))

    # Save
    out = root / "outputs" / "daily_sentiment.csv"
    out.parent.mkdir(exist_ok=True)
    daily_with_events.to_csv(out, index=False)
    print(f"\n✅ Saved → {out}")

    images = root / "images"
    images.mkdir(exist_ok=True)
    plot_sentiment_timeline(daily, "BTC", str(images / "sentiment_timeline.png"))


if __name__ == "__main__":
    main()
