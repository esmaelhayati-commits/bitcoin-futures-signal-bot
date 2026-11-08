
import pandas as pd
import numpy as np


# ==========================================
# BITCOIN FUTURES SIGNAL BOT - STRATEGY V2
# Exchange: XT
# Entry timeframe: 15m
# Confirmation: 1h + 4h
# Mode: LONG / SHORT
# ==========================================

VERSION = "2.0"

ATR_PERIOD = 14
RSI_PERIOD = 14

STOP_ATR_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0

MIN_15M_CANDLES = 60
MIN_HTF_CANDLES = 205


def validate_dataframe(df, name):
    required = ["open", "high", "low", "close", "volume"]

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    if len(df) < 30:
        raise ValueError(f"Not enough candles in {name}")

    df = df.copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    if len(df) < 30:
        raise ValueError(f"Invalid candle data in {name}")

    if (df["close"] <= 0).any():
        raise ValueError(f"Invalid prices in {name}")

    if (df["volume"] < 0).any():
        raise ValueError(f"Invalid volume in {name}")

    return df.reset_index(drop=True)


def calculate_indicators(df):
    df = validate_dataframe(df, "candles")

    # Moving averages
    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0), "rsi"
    ] = 50

    # ATR
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean()

    # Volume average
    df["volume_sma20"] = df["volume"].rolling(
        20
    ).mean()

    return df


def get_trend(row):
    if (
        row["close"] > row["ema200"]
        and row["ema50"] > row["ema200"]
    ):
        return "BULLISH"

    if (
        row["close"] < row["ema200"]
        and row["ema50"] < row["ema200"]
    ):
        return "BEARISH"

    return "NEUTRAL"


def no_signal(reason, symbol=None):
    return {
        "version": VERSION,
        "symbol": symbol,
        "signal": "NO_SIGNAL",
        "confidence": 0,
        "reason": reason,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": RISK_REWARD_RATIO
    }


def generate_signal(
    df_15m,
    df_1h,
    df_4h,
    symbol=None
):
    """
    Inputs must contain CLOSED candles only.

    df_15m: 15-minute OHLCV data
    df_1h:   1-hour OHLCV data
    df_4h:   4-hour OHLCV data

    Returns a signal dictionary.
    """

    try:
        if len(df_15m) < MIN_15M_CANDLES:
            return no_signal("Insufficient 15m candles", symbol)

        if len(df_1h) < MIN_HTF_CANDLES:
            return no_signal("Insufficient 1h candles", symbol)

        if len(df_4h) < MIN_HTF_CANDLES:
            return no_signal("Insufficient 4h candles", symbol)

        data_15m = calculate_indicators(df_15m)
        data_1h = calculate_indicators(df_1h)
        data_4h = calculate_indicators(df_4h)

        current = data_15m.iloc[-1]
        previous = data_15m.iloc[-2]

        hourly = data_1h.iloc[-1]
        four_hour = data_4h.iloc[-1]

        required_values = [
            current["ema20"],
            current["ema50"],
            current["rsi"],
            current["atr"],
            current["volume_sma20"],
            hourly["ema50"],
            hourly["ema200"],
            four_hour["ema50"],
            four_hour["ema200"]
        ]

        if not np.isfinite(required_values).all():
            return no_signal("Indicators not ready", symbol)

        entry = float(current["close"])
        atr = float(current["atr"])
        rsi = float(current["rsi"])

        if atr <= 0 or entry <= 0:
            return no_signal("Invalid ATR or entry price", symbol)

        if current["volume_sma20"] <= 0:
            return no_signal("Invalid average volume", symbol)

        trend_1h = get_trend(hourly)
        trend_4h = get_trend(four_hour)

        # Momentum confirmation
        bullish_momentum = (
            current["ema20"] > current["ema50"]
            and current["close"] > current["ema20"]
            and current["close"] > previous["close"]
            and 52 <= rsi <= 68
        )

        bearish_momentum = (
            current["ema20"] < current["ema50"]
            and current["close"] < current["ema20"]
            and current["close"] < previous["close"]
            and 32 <= rsi <= 48
        )

        # Volume confirmation
        volume_ok = (
            current["volume"]
            >= current["volume_sma20"] * 0.8
        )

        if not volume_ok:
            return no_signal("Low volume confirmation", symbol)

        # LONG
        if (
            trend_1h == "BULLISH"
            and trend_4h == "BULLISH"
            and bullish_momentum
        ):
            stop_loss = entry - (
                atr * STOP_ATR_MULTIPLIER
            )

            risk = entry - stop_loss

            take_profit = entry + (
                risk * RISK_REWARD_RATIO
            )

            return {
                "version": VERSION,
                "symbol": symbol,
                "signal": "LONG",
                "confidence": 80,
                "reason": "HTF bullish trend + 15m momentum",
                "entry": entry,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "risk_reward": RISK_REWARD_RATIO,
                "rsi": round(rsi, 2),
                "atr": round(atr, 8),
                "trend_1h": trend_1h,
                "trend_4h": trend_4h
            }

        # SHORT
        if (
            trend_1h == "BEARISH"
            and trend_4h == "BEARISH"
            and bearish_momentum
        ):
            stop_loss = entry + (
                atr * STOP_ATR_MULTIPLIER
            )

            risk = stop_loss - entry

            take_profit = entry - (
                risk * RISK_REWARD_RATIO
            )

            return {
                "version": VERSION,
                "symbol": symbol,
                "signal": "SHORT",
                "confidence": 80,
                "reason": "HTF bearish trend + 15m momentum",
                "entry": entry,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "risk_reward": RISK_REWARD_RATIO,
                "rsi": round(rsi, 2),
                "atr": round(atr, 8),
                "trend_1h": trend_1h,
                "trend_4h": trend_4h
            }

        return no_signal(
            "No complete trend and momentum confirmation",
            symbol
        )

    except Exception as exc:
        return no_signal(
            f"Strategy error: {str(exc)[:150]}",
            symbol
        )
