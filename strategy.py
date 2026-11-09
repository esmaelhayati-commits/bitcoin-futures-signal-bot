
import numpy as np
import pandas as pd

# ==========================================
# BITCOIN FUTURES STRATEGY V2.3
# XT USDT SWAPS
# Entry: 15m
# Confirmation: 1h + 4h
# Backtest compatible
# ==========================================

STRATEGY_VERSION = "2.3"

MIN_CANDLES = 210

ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

STOP_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 3.0

MIN_ADX = 18
MIN_RSI_LONG = 50
MAX_RSI_LONG = 68
MIN_RSI_SHORT = 32
MAX_RSI_SHORT = 50

MAX_ENTRY_DISTANCE_ATR = 0.8
PULLBACK_TOLERANCE_ATR = 0.25
MIN_CANDLE_BODY_RATIO = 0.35


def validate_data(df):
    required = [
        "open", "high", "low", "close", "volume"
    ]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[required].copy()

    for col in required:
        out[col] = pd.to_numeric(
            out[col], errors="coerce"
        )

    out = out.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    out = out[
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["volume"] >= 0)
        & (out["high"] >= out["low"])
    ]

    return out


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

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)

    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss == 0)),
        50
    )

    return rsi.fillna(50)


def calculate_atr(df, period=ATR_PERIOD):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_adx(df, period=ADX_PERIOD):

    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0.0
        ),
        index=df.index
    )

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = (
        100 * plus_smoothed / atr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_smoothed / atr.replace(0, np.nan)
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

    out = validate_data(df)

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
    previous = data.iloc[-6]

    bullish = (
        last["close"] > last["ema200"]
        and last["ema50"] > last["ema200"]
        and last["ema50"] > previous["ema50"]
    )

    bearish = (
        last["close"] < last["ema200"]
        and last["ema50"] < last["ema200"]
        and last["ema50"] < previous["ema50"]
    )

    if bullish:
        return "BULLISH"

    if bearish:
        return "BEARISH"

    return "SIDEWAYS"


def hold(reason, **extra):

    result = {
        "signal": "HOLD",
        "side": "HOLD",
        "confidence": 0,
        "reason": reason,
        "strategy_version": STRATEGY_VERSION
    }

    result.update(extra)

    return result


def analyze_market(
    df_15m,
    df_1h,
    df_4h,
    symbol=None
):

    try:

        d15 = prepare_data(df_15m)
        d1 = prepare_data(df_1h)
        d4 = prepare_data(df_4h)

    except Exception as exc:

        return hold(
            f"DATA_ERROR: {exc}",
            symbol=symbol
        )

    if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:

        return hold(
            "INSUFFICIENT_DATA",
            symbol=symbol
        )

    current = d15.iloc[-1]
    previous = d15.iloc[-2]

    h1 = d1.iloc[-1]
    h4 = d4.iloc[-1]

    price = float(current["close"])
    atr = float(current["atr"])

    if not np.isfinite(atr) or atr <= 0:

        return hold(
            "INVALID_ATR",
            symbol=symbol
        )

    # --------------------------------------
    # 1. Higher timeframe trend confirmation
    # --------------------------------------

    trend_1h = get_trend(df_1h)
    trend_4h = get_trend(df_4h)

    bullish_trend = (
        trend_1h == "BULLISH"
        and trend_4h == "BULLISH"
    )

    bearish_trend = (
        trend_1h == "BEARISH"
        and trend_4h == "BEARISH"
    )

    if not bullish_trend and not bearish_trend:

        return hold(
            "HIGHER_TIMEFRAME_TREND_NOT_ALIGNED",
            symbol=symbol,
            trend_1h=trend_1h,
            trend_4h=trend_4h
        )

    # --------------------------------------
    # 2. Volatility and trend strength
    # --------------------------------------

    adx = float(current["adx"])

    if adx < MIN_ADX:

        return hold(
            "WEAK_TREND_ADX",
            symbol=symbol,
            adx=round(adx, 2)
        )

    # --------------------------------------
    # 3. Candle quality
    # --------------------------------------

    candle_range = float(
        current["high"] - current["low"]
    )

    if candle_range <= 0:

        return hold("INVALID_CANDLE_RANGE")

    body = abs(
        float(current["close"] - current["open"])
    )

    body_ratio = body / candle_range

    if body_ratio < MIN_CANDLE_BODY_RATIO:

        return hold(
            "WEAK_CONFIRMATION_CANDLE",
            body_ratio=round(body_ratio, 3)
        )

    # --------------------------------------
    # 4. Pullback and EMA reclaim
    # --------------------------------------

    ema20 = float(current["ema20"])
    ema50 = float(current["ema50"])

    distance_from_ema = abs(
        price - ema20
    ) / atr

    if distance_from_ema > MAX_ENTRY_DISTANCE_ATR:

        return hold(
            "PRICE_TOO_FAR_FROM_EMA20",
            distance_atr=round(distance_from_ema, 3)
        )

    long_pullback = (
        previous["close"] <= previous["ema20"]
        and current["low"] <= (
            ema20 + PULLBACK_TOLERANCE_ATR * atr
        )
        and price > ema20
        and price > current["open"]
        and current["ema20"] > current["ema50"]
    )

    short_pullback = (
        previous["close"] >= previous["ema20"]
        and current["high"] >= (
            ema20 - PULLBACK_TOLERANCE_ATR * atr
        )
        and price < ema20
        and price < current["open"]
        and current["ema20"] < current["ema50"]
    )

    # --------------------------------------
    # 5. Momentum confirmation
    # --------------------------------------

    rsi = float(current["rsi"])

    long_momentum = (
        MIN_RSI_LONG <= rsi <= MAX_RSI_LONG
        and price > previous["close"]
    )

    short_momentum = (
        MIN_RSI_SHORT <= rsi <= MAX_RSI_SHORT
        and price < previous["close"]
    )

    # --------------------------------------
    # 6. Final signal
    # --------------------------------------

    common = {
        "symbol": symbol,
        "strategy_version": STRATEGY_VERSION,
        "entry": price,
        "entry_price": price,
        "atr": atr,
        "rsi": round(rsi, 2),
        "adx": round(adx, 2),
        "body_ratio": round(body_ratio, 3),
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "ema20": ema20,
        "ema50": ema50,
        "stop_atr_multiplier": STOP_ATR_MULTIPLIER,
        "target_atr_multiplier": TARGET_ATR_MULTIPLIER,
        "risk_reward": 2.0
    }

    if (
        bullish_trend
        and long_pullback
        and long_momentum
    ):

        stop = price - STOP_ATR_MULTIPLIER * atr
        target = price + TARGET_ATR_MULTIPLIER * atr

        return {
            **common,
            "signal": "LONG",
            "side": "LONG",
            "stop": stop,
            "stop_loss": stop,
            "target": target,
            "take_profit": target,
            "confidence": 75,
            "reason": (
                "V2.3 bullish trend + EMA20 reclaim "
                "+ momentum confirmation"
            )
        }

    if (
        bearish_trend
        and short_pullback
        and short_momentum
    ):

        stop = price + STOP_ATR_MULTIPLIER * atr
        target = price - TARGET_ATR_MULTIPLIER * atr

        return {
            **common,
            "signal": "SHORT",
            "side": "SHORT",
            "stop": stop,
            "stop_loss": stop,
            "target": target,
            "take_profit": target,
            "confidence": 75,
            "reason": (
                "V2.3 bearish trend + EMA20 rejection "
                "+ momentum confirmation"
            )
        }

    return hold(
        "NO_VALID_V2.3_ENTRY",
        **common
    )


def generate_signal(
    df_15m,
    df_1h,
    df_4h,
    symbol=None
):

    return analyze_market(
        df_15m,
        df_1h,
        df_4h,
        symbol
    )
