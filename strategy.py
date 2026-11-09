import numpy as np
import pandas as pd

MIN_CANDLES = 210
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14
STOP_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 3.0


def _validate_ohlcv(df):
    required = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = df[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100)
    result = result.where(avg_gain != 0, 0)
    result = result.where(
        ~((avg_gain == 0) & (avg_loss == 0)),
        50
    )
    return result.fillna(50)


def calculate_atr(df, period=ATR_PERIOD):
    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_adx(df, period=ADX_PERIOD):
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() / atr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean() / atr.replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean().fillna(0)


def prepare_data(df):
    out = _validate_ohlcv(df)

    out["ema20"] = out["close"].ewm(
        span=20, adjust=False
    ).mean()

    out["ema50"] = out["close"].ewm(
        span=50, adjust=False
    ).mean()

    out["ema200"] = out["close"].ewm(
        span=200, adjust=False
    ).mean()

    out["rsi"] = calculate_rsi(out["close"])
    out["atr"] = calculate_atr(out)
    out["adx"] = calculate_adx(out)

    return out


def get_trend(df):
    data = prepare_data(df)

    if len(data) < MIN_CANDLES:
        return "INSUFFICIENT_DATA"

    last = data.iloc[-1]

    if last["close"] > last["ema20"] > last["ema50"] > last["ema200"]:
        return "BULLISH"

    if last["close"] < last["ema20"] < last["ema50"] < last["ema200"]:
        return "BEARISH"

    return "SIDEWAYS"


def _hold(reason, **extra):
    result = {
        "signal": "HOLD",
        "side": "HOLD",
        "confidence": 0,
        "reason": reason
    }

    result.update(extra)
    return result


def analyze_market(df_15m, df_1h, df_4h, symbol=None):

    try:
        d15 = prepare_data(df_15m)
        d1 = prepare_data(df_1h)
        d4 = prepare_data(df_4h)

    except Exception as exc:
        return _hold(
            f"DATA_ERROR: {exc}",
            symbol=symbol
        )

    if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
        return _hold(
            "INSUFFICIENT_DATA",
            symbol=symbol
        )

    a = d15.iloc[-1]
    previous = d15.iloc[-2]

    h1 = d1.iloc[-1]
    h4 = d4.iloc[-1]

    bullish = (
        h1["close"] > h1["ema20"] > h1["ema50"] > h1["ema200"]
        and
        h4["close"] > h4["ema20"] > h4["ema50"] > h4["ema200"]
    )

    bearish = (
        h1["close"] < h1["ema20"] < h1["ema50"] < h1["ema200"]
        and
        h4["close"] < h4["ema20"] < h4["ema50"] < h4["ema200"]
    )

    if not np.isfinite(a["atr"]) or a["atr"] <= 0:
        return _hold("INVALID_ATR", symbol=symbol)

    # Filter weak trends
    if a["adx"] < 18:
        return _hold(
            "ADX_FILTER",
            symbol=symbol,
            adx=round(float(a["adx"]), 2)
        )

    # Long setup
    long_setup = (
        bullish
        and a["close"] > a["ema200"]
        and a["ema20"] > a["ema50"]
        and a["low"] <= a["ema20"] * 1.003
        and a["close"] > a["ema20"]
        and 50 <= a["rsi"] <= 68
        and a["close"] > previous["close"]
    )

    # Short setup
    short_setup = (
        bearish
        and a["close"] < a["ema200"]
        and a["ema20"] < a["ema50"]
        and a["high"] >= a["ema20"] * 0.997
        and a["close"] < a["ema20"]
        and 32 <= a["rsi"] <= 50
        and a["close"] < previous["close"]
    )

    common = {
        "symbol": symbol,
        "entry": float(a["close"]),
        "entry_price": float(a["close"]),
        "atr": float(a["atr"]),
        "rsi": round(float(a["rsi"]), 2),
        "adx": round(float(a["adx"]), 2),
        "trend_1h": (
            "BULLISH" if bullish
            else ("BEARISH" if bearish else "MIXED")
        ),
        "trend_4h": (
            "BULLISH" if bullish
            else ("BEARISH" if bearish else "MIXED")
        ),
        "stop_atr_multiplier": STOP_ATR_MULTIPLIER,
        "target_atr_multiplier": TARGET_ATR_MULTIPLIER,
        "risk_reward": 2.0
    }

    if long_setup:

        stop = float(
            a["close"] - STOP_ATR_MULTIPLIER * a["atr"]
        )

        target = float(
            a["close"] + TARGET_ATR_MULTIPLIER * a["atr"]
        )

        return {
            **common,
            "signal": "LONG",
            "side": "LONG",
            "stop": stop,
            "stop_loss": stop,
            "target": target,
            "take_profit": target,
            "confidence": 70,
            "reason": "15m pullback in aligned 1h/4h bullish trend"
        }

    if short_setup:

        stop = float(
            a["close"] + STOP_ATR_MULTIPLIER * a["atr"]
        )

        target = float(
            a["close"] - TARGET_ATR_MULTIPLIER * a["atr"]
        )

        return {
            **common,
            "signal": "SHORT",
            "side": "SHORT",
            "stop": stop,
            "stop_loss": stop,
            "target": target,
            "take_profit": target,
            "confidence": 70,
            "reason": "15m pullback in aligned 1h/4h bearish trend"
        }

    return _hold(
        "NO_VALID_PULLBACK",
        **common
    )


def generate_signal(df_15m, df_1h, df_4h, symbol=None):
    return analyze_market(
        df_15m,
        df_1h,
        df_4h,
        symbol
    )
