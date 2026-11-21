import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("smc_strategy")

STRATEGY_VERSION = "4.1"
MIN_CANDLES = 200               # indicator warm-up (EWM ADX/ATR need ~100+ candles)

SWING_LOOKBACK = 5
STRUCTURE_RECENT_CANDLES = 6    # BOS/CHoCH must have happened within the last N closed 1H candles
MAX_FVG_AGE_CANDLES = 48        # on 15m (= 12 hours)
MIN_FVG_ATR_MULTIPLIER = 0.35
MIN_MOVE_AWAY_ATR = 1.0         # price must leave the FVG by >= 1 ATR before the pullback counts
STOP_BUFFER_ATR = 0.25
MIN_STOP_ATR = 1.0
MAX_STOP_ATR = 2.2
RR_RATIO = 2.0                  # NET risk/reward (after fees + slippage)

FEE_PCT_PER_SIDE = 0.0005       # 0.05% (adjust to your exchange)
SLIPPAGE_PCT_PER_SIDE = 0.0002  # 0.02%
ROUND_TRIP_COST_PCT = 2 * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE)

ATR_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_TREND = 22.0
VOLUME_MA_PERIOD = 20


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def _to_datetime(s: pd.Series) -> pd.Series:
    """Handles datetime, epoch s/ms/us/ns (numeric or numeric strings) and ISO strings. Always UTC."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        med = num.dropna().abs().median()
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def validate_data(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out["timestamp"] = out.index
        else:
            raise ValueError("A 'timestamp' column (or DatetimeIndex) is required")

    out["timestamp"] = _to_datetime(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    out = out[
        (out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)
        & (out["volume"] >= 0) & (out["high"] >= out["low"])
    ].reset_index(drop=True)

    # Check missing candles in the recent active structure window (last 60 candles)
    # Using 2.5x multiplier to tolerate occasional zero-volume skipped candles on XT
    recent_tail = out["timestamp"].iloc[-60:]
    if len(recent_tail) > 1:
        gaps = recent_tail.diff().dropna().dt.total_seconds() / 60.0
        if (gaps > tf_minutes * 2.5).any():
            raise ValueError(f"Significant gap in recent {tf_minutes}m candles")
    return out[["timestamp"] + required]


def drop_unclosed(df: pd.DataFrame, tf_minutes: int, now=None) -> pd.DataFrame:
    """Removes the last candle if it is still forming."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if len(df) and df["timestamp"].iloc[-1] + pd.Timedelta(minutes=tf_minutes) > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def calculate_directional_adx(df: pd.DataFrame, period: int = ADX_PERIOD):
    up_move = df["high"].diff()
    down_move = -df["low"].diff
