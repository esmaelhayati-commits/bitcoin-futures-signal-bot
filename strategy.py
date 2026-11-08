
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

MIN_CANDLES = 200
ATR_PERIOD = 14
RSI_PERIOD = 14


# ==========================================
# INDICATORS
# ==========================================

def calculate_rsi(series, period=RSI_PERIOD):
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


def calculate_atr(df, period=ATR_PERIOD):
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


# ==========================================
# DATA PREPARATION
# ==========================================

def prepare_data(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Need at least {MIN_CANDLES} candles"
        )

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


# ==========================================
# TREND DETECTION
# ==========================================

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


# ==========================================
# MAIN STRATEGY
# ==========================================

def analyze_market(df_15m, df_1h, df_4h):

    data_15m = prepare_data(df_15m)
    data_1h = prepare_data(df_1h)
    data_4h = prepare_data(df_4h)

    trend_15m = get_trend(data_15m)
    trend_1h = get_trend(data_1h)
    trend_4h = get_trend(data_4h)

    last = data_15m.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])

    long_score = 0
    short_score = 0

    reasons = []

    # 15 MINUTES
    if trend_15m == "BULLISH":
        long_score += 3
        reasons.append("15m bullish trend")

    elif trend_15m == "BEARISH":
        short_score += 3
        reasons.append("15m bearish trend")

    # 1 HOUR
    if trend_1h == "BULLISH":
        long_score += 2
        reasons.append("1h bullish trend")

    elif trend_1h == "BEARISH":
        short_score += 2
        reasons.append("1h bearish trend")

    # 4 HOURS
    if trend_4h == "BULLISH":
        long_score += 2
        reasons.append("4h bullish trend")

    elif trend_4h == "BEARISH":
        short_score += 2
        reasons.append("4h bearish trend")

    # RSI
    if 50 < rsi < 70:
        long_score += 2
        reasons.append("RSI supports LONG")

    elif 30 < rsi < 50:
        short_score += 2
        reasons.append("RSI supports SHORT")

    # EMA CONFIRMATION
    if last["ema20"] > last["ema50"]:
        long_score += 1

    elif last["ema20"] < last["ema50"]:
        short_score += 1

    # SIGNAL
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

    confidence = round(
        max(long_score, short_score) / 10 * 100,
        1
    )

    # ENTRY AND TARGETS
    entry = price

    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None

    if signal == "LONG":

        stop_loss = price - atr * 1.5

        tp1 = price + atr * 1.5
        tp2 = price + atr * 2.5
        tp3 = price + atr * 4

    elif signal == "SHORT":

        stop_loss = price + atr * 1.5

        tp1 = price - atr * 1.5
        tp2 = price - atr * 2.5
        tp3 = price - atr * 4

    # UNIFIED OUTPUT
    return {
        "signal": signal,
        "side": signal,

        "entry": entry,
        "entry_price": entry,

        "stop": stop_loss,
        "stop_loss": stop_loss,

        "target": tp1,
        "take_profit": tp1,

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


# ==========================================
# BACKTEST COMPATIBILITY
# ==========================================

def generate_signal(*args, **kwargs):

    frames = []

    for arg in args:
        if isinstance(arg, pd.DataFrame):
            frames.append(arg)

    for value in kwargs.values():
        if isinstance(value, pd.DataFrame):
            frames.append(value)

    if len(frames) >= 3:

        df_15m = frames[0]
        df_1h = frames[1]
        df_4h = frames[2]

    elif len(frames) == 1:

        df_15m = frames[0]
        df_1h = frames[0]
        df_4h = frames[0]

    else:
        raise ValueError(
            f"Expected DataFrames, found {len(frames)}"
        )

    return analyze_market(
        df_15m,
        df_1h,
        df_4h
    )


# ==========================================
# END
# ==========================================
