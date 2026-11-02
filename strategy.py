
import pandas as pd
import numpy as np


def analyze_market(df):
    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9).mean()

    # ATR
    previous = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - previous),
        abs(df["low"] - previous)
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    last = df.iloc[-1]

    if pd.isna(last["RSI"]) or pd.isna(last["ATR"]):
        return None

    entry = float(last["close"])
    atr = float(last["ATR"])

    if atr <= 0:
        return None

    long_condition = (
        last["EMA20"] > last["EMA50"]
        and entry > last["EMA20"]
        and 52 <= last["RSI"] <= 70
        and last["MACD"] > last["SIGNAL"]
    )

    short_condition = (
        last["EMA20"] < last["EMA50"]
        and entry < last["EMA20"]
        and 30 <= last["RSI"] <= 48
        and last["MACD"] < last["SIGNAL"]
    )

    if long_condition:
        return {
            "side": "LONG",
            "entry": entry,
            "stop": entry - 1.5 * atr,
            "target": entry + 3 * atr,
            "rsi": round(float(last["RSI"]), 2)
        }

    if short_condition:
        return {
            "side": "SHORT",
            "entry": entry,
            "stop": entry + 1.5 * atr,
            "target": entry - 3 * atr,
            "rsi": round(float(last["RSI"]), 2)
        }

    return None
