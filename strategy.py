
import pandas as pd
import numpy as np


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def prepare_data(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["rsi"] = calculate_rsi(df["close"])

    df["atr"] = calculate_atr(df)

    return df.dropna().reset_index(drop=True)


def get_trend(df):
    last = df.iloc[-1]

    if (
        last["close"] > last["ema20"]
        and last["ema20"] > last["ema50"]
        and last["ema50"] > last["ema200"]
    ):
        return "BULLISH"

    if (
        last["close"] < last["ema20"]
        and last["ema20"] < last["ema50"]
        and last["ema50"] < last["ema200"]
    ):
        return "BEARISH"

    return "SIDEWAYS"


def analyze_market(df_15m, df_1h, df_4h):

    df_15m = prepare_data(df_15m)
    df_1h = prepare_data(df_1h)
    df_4h = prepare_data(df_4h)

    if min(
        len(df_15m),
        len(df_1h),
        len(df_4h)
    ) < 5:
        raise ValueError("Insufficient market data")

    trend_15m = get_trend(df_15m)
    trend_1h = get_trend(df_1h)
    trend_4h = get_trend(df_4h)

    last = df_15m.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])

    long_score = 0
    short_score = 0

    reasons = []

    # تایم فریم اصلی
    if trend_15m == "BULLISH":
        long_score += 3
        reasons.append("15m bullish trend")

    elif trend_15m == "BEARISH":
        short_score += 3
        reasons.append("15m bearish trend")

    # تایم فریم یک ساعته
    if trend_1h == "BULLISH":
        long_score += 2

    elif trend_1h == "BEARISH":
        short_score += 2

    # تایم فریم چهار ساعته
    if trend_4h == "BULLISH":
        long_score += 2

    elif trend_4h == "BEARISH":
        short_score += 2

    # RSI
    if 50 < rsi < 70:
        long_score += 2
        reasons.append("RSI supports LONG")

    elif 30 < rsi < 50:
        short_score += 2
        reasons.append("RSI supports SHORT")

    # EMA confirmation
    if last["ema20"] > last["ema50"]:
        long_score += 1

    elif last["ema20"] < last["ema50"]:
        short_score += 1

    total_score = long_score + short_score

    if total_score:
        confidence = round(
            max(long_score, short_score) / 10 * 100,
            1
        )
    else:
        confidence = 0

    signal = "WAIT"

    if (
        long_score >= 7
        and long_score - short_score >= 3
        and trend_1h == "BULLISH"
    ):
        signal = "LONG"

    elif (
        short_score >= 7
        and short_score - long_score >= 3
        and trend_1h == "BEARISH"
    ):
        signal = "SHORT"

    entry = price
    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None

    if signal == "LONG":
        stop_loss = price - (atr * 1.5)
        tp1 = price + (atr * 1.5)
        tp2 = price + (atr * 2.5)
        tp3 = price + (atr * 4)

    elif signal == "SHORT":
        stop_loss = price + (atr * 1.5)
        tp1 = price - (atr * 1.5)
        tp2 = price - (atr * 2.5)
        tp3 = price - (atr * 4)

    if signal == "WAIT":
        confidence = round(
            max(long_score, short_score) / 10 * 100,
            1
        )

    return {
        "signal": signal,
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "confidence": confidence,
        "long_score": long_score,
        "short_score": short_score,
        "trend_15m": trend_15m,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "rsi": round(rsi, 2),
        "atr": round(atr, 6),
        "reasons": reasons
    }
