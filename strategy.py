import numpy as np
import pandas as pd

STRATEGY_VERSION = "SMC-V3.0"
MIN_CANDLES = 120

SWING_LOOKBACK = 5
FVG_THRESHOLD_PCT = 0.0015
RR_RATIO = 2.5


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out[
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["high"] >= out["low"])
    ]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """تابع سازگاری برای فایل‌های بک‌تست و آزمایشات گذشته"""
    out = validate_data(df)
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["atr"] = calculate_atr(out)
    return out


def identify_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    highs = df["high"].values
    lows = df["low"].values
    indices = df.index

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((indices[i], highs[i]))

        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((indices[i], lows[i]))

    return swing_highs, swing_lows


def detect_fvg(df: pd.DataFrame):
    fvg_bullish = []
    fvg_bearish = []

    for i in range(2, len(df)):
        c1_high, c1_low = df["high"].iloc[i - 2], df["low"].iloc[i - 2]
        c3_high, c3_low = df["high"].iloc[i], df["low"].iloc[i]
        c2_close = df["close"].iloc[i - 1]

        if c3_low > c1_high:
            gap_pct = (c3_low - c1_high) / c2_close
            if gap_pct >= FVG_THRESHOLD_PCT:
                fvg_bullish.append({
                    "top": float(c3_low),
                    "bottom": float(c1_high),
                    "index": df.index[i - 1]
                })

        if c3_high < c1_low:
            gap_pct = (c1_low - c3_high) / c2_close
            if gap_pct >= FVG_THRESHOLD_PCT:
                fvg_bearish.append({
                    "top": float(c1_low),
                    "bottom": float(c3_high),
                    "index": df.index[i - 1]
                })

    return fvg_bullish, fvg_bearish


def detect_market_structure(df: pd.DataFrame):
    swing_highs, swing_lows = identify_swings(df)
    if not swing_highs or not swing_lows:
        return "RANGE", None, None

    last_swing_high = swing_highs[-1][1]
    last_swing_low = swing_lows[-1][1]
    last_close = df["close"].iloc[-1]

    if last
