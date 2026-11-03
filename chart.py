
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd

from strategy import chart_indicators


def create_chart(df, analysis, filename="btc_analysis.png"):
    data = chart_indicators(df).tail(100).copy()

    data["date"] = pd.to_datetime(
        data["timestamp"], unit="ms"
    )

    fig, ax = plt.subplots(figsize=(16, 9))

    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    # Candlesticks
    for _, row in data.iterrows():
        x = mdates.date2num(row["date"])

        bullish = row["close"] >= row["open"]
        candle_color = "#00d6a3" if bullish else "#ff5468"

        ax.plot(
            [x, x],
            [row["low"], row["high"]],
            color=candle_color,
            linewidth=1
        )

        body_low = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 0.5)

        rect = Rectangle(
            (x - 0.003, body_low),
            0.006,
            body_height,
            facecolor=candle_color,
            edgecolor=candle_color
        )

        ax.add_patch(rect)

    # EMA lines
    ax.plot(data["date"], data["ema20"], label="EMA 20",
            color="#f4c542", linewidth=1.5)

    ax.plot(data["date"], data["ema50"], label="EMA 50",
            color="#36a2eb", linewidth=1.5)

    ax.plot(data["date"], data["ema200"], label="EMA 200",
            color="#c792ea", linewidth=1.5)

    # Ichimoku cloud
    cloud_a = data["cloud_a"]
    cloud_b = data["cloud_b"]

    ax.fill_between(
        data["date"],
        cloud_a,
        cloud_b,
        where=cloud_a >= cloud_b,
        color="#00d6a3",
        alpha=0.10
    )

    ax.fill_between(
        data["date"],
        cloud_a,
        cloud_b,
        where=cloud_a < cloud_b,
        color="#ff5468",
        alpha=0.10
    )

    # FVG zone
    if analysis["fvg_low"] is not None:
        ax.axhspan(
            analysis["fvg_low"],
            analysis["fvg_high"],
            color="#f4c542",
            alpha=0.18,
            label="FVG zone"
        )

    # Trading levels
    levels = [
        ("Entry", analysis["entry"], "#ffffff"),
        ("Stop Loss", analysis["stop_loss"], "#ff5468"),
        ("TP1", analysis["tp1"], "#00d6a3"),
        ("TP2", analysis["tp2"], "#00d6a3"),
        ("TP3", analysis["tp3"], "#00d6a3")
    ]

    for label, value, color in levels:
        if value is not None:
            ax.axhline(
                value,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"{label}: {value:.2f}"
            )

    signal = analysis["signal"]

    title = (
        f"BTC/USDT FUTURES | {signal}\n"
        f"Price: {analysis['price']:.2f} | "
        f"LONG: {analysis['long_score']} | "
        f"SHORT: {analysis['short_score']}"
    )

    ax.set_title(
        title,
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d %H:%M")
    )

    ax.grid(
        True,
        color="#45515c",
        alpha=0.25,
        linestyle=":"
    )

    for spine in ax.spines.values():
        spine.set_color("#45515c")

    legend = ax.legend(
        loc="upper left",
        facecolor="#182630",
        edgecolor="#45515c",
        fontsize=8
    )

    for text in legend.get_texts():
        text.set_color("white")

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(
        filename,
        dpi=150,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    return filename
