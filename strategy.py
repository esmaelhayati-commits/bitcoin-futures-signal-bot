import pandas as pd
import numpy as np

MIN_CANDLES = 200
ATR_PERIOD = 14
RSI_PERIOD = 14


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

    return (100 - 100 / (1 + rs)).fillna(50)


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


def prepare_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required = [
        "open", "high", "low", "close", "volume"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df.copy()

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(subset=required)

    if len(data) < MIN_CANDLES:
        raise ValueError(
            f"Need at least {MIN_CANDLES} candles; "
            f"received {len(data)}"
        )

    data["ema20"] = data["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["ema50"] = data["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    data["ema200"] = data["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    data["rsi"] = calculate_rsi(data["close"])
    data["atr"] = calculate_atr(data)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "atr"
        ]
    )

    return data


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

    # 15m trend
    if trend_15m == "BULLISH":
        long_score += 3
        reasons.append("15m bullish trend")

    elif trend_15m == "BEARISH":
        short_score += 3
        reasons.append("15m bearish trend")

    # 1h trend
    if trend_1h == "BULLISH":
        long_score += 2
        reasons.append("1h bullish trend")

    elif trend_1h == "BEARISH":
        short_score += 2
        reasons.append("1h bearish trend")

    # 4h trend
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

    # EMA confirmation
    if last["ema20"] > last["ema50"]:
        long_score += 1

    elif last["ema20"] < last["ema50"]:
        short_score += 1

    # Signal decision
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

    # Entry and targets
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
        "atr": round(atr, 8),

        "reasons": reasons
    }


def generate_signal(
    df_15m,
    df_1h,
    df_4h,
    symbol=None
):
    """
    Compatible with backtest.py:
    generate_signal(history15, history1, history4, symbol)
    """

    if not isinstance(df_15m, pd.DataFrame):
        raise TypeError("15m data must be a DataFrame")

    if not isinstance(df_1h, pd.DataFrame):
        raise TypeError("1h data must be a DataFrame")

    if not isinstance(df_4h, pd.DataFrame):
        raise TypeError("4h data must be a DataFrame")

    return analyze_market(
        df_15m,
        df_1h,
        df_4h
    )
