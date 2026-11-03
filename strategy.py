
import pandas as pd
import numpy as np


def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    for period in [20, 50, 200]:
        df[f"ema{period}"] = close.ewm(
            span=period, adjust=False
        ).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    # Ichimoku
    tenkan = (
        high.rolling(9).max() +
        low.rolling(9).min()
    ) / 2

    kijun = (
        high.rolling(26).max() +
        low.rolling(26).min()
    ) / 2

    span_a = (tenkan + kijun) / 2
    span_b = (
        high.rolling(52).max() +
        low.rolling(52).min()
    ) / 2

    df["cloud_a"] = span_a.shift(26)
    df["cloud_b"] = span_b.shift(26)

    # Volume
    df["volume_avg"] = volume.rolling(20).mean()

    # Previous market levels
    df["previous_high"] = high.shift(1).rolling(20).max()
    df["previous_low"] = low.shift(1).rolling(20).min()

    # FVG: three-candle imbalance
    df["bullish_fvg"] = low > high.shift(2)
    df["bearish_fvg"] = high < low.shift(2)

    # Simple Order Block approximation
    df["bullish_ob"] = (
        (close.shift(1) < df["open"].shift(1)) &
        (close > high.shift(1))
    )

    df["bearish_ob"] = (
        (close.shift(1) > df["open"].shift(1)) &
        (close < low.shift(1))
    )

    return df


def get_timeframe_trend(df):
    data = add_indicators(df)
    last = data.iloc[-1]

    if last["ema20"] > last["ema50"]:
        return "Bullish"

    if last["ema20"] < last["ema50"]:
        return "Bearish"

    return "Neutral"


def analyze_market(df_15m, df_1h, df_4h):
    data = add_indicators(df_15m)

    last = data.iloc[-1]
    previous = data.iloc[-2]

    price = float(last["close"])
    atr = float(last["atr"])

    trend_1h = get_timeframe_trend(df_1h)
    trend_4h = get_timeframe_trend(df_4h)

    long_score = 0
    short_score = 0
    reasons = []

    # 15m EMA trend
    if last["ema20"] > last["ema50"]:
        long_score += 2
        reasons.append("15m EMA bullish")

    elif last["ema20"] < last["ema50"]:
        short_score += 2
        reasons.append("15m EMA bearish")

    # Higher timeframe confirmation
    if trend_1h == "Bullish":
        long_score += 2
        reasons.append("1h trend bullish")

    elif trend_1h == "Bearish":
        short_score += 2
        reasons.append("1h trend bearish")

    if trend_4h == "Bullish":
        long_score += 3
        reasons.append("4h trend bullish")

    elif trend_4h == "Bearish":
        short_score += 3
        reasons.append("4h trend bearish")

    # EMA200
    if price > last["ema200"]:
        long_score += 1
    else:
        short_score += 1

    # RSI
    if 50 < last["rsi"] < 70:
        long_score += 1
    elif 30 < last["rsi"] < 50:
        short_score += 1

    # MACD
    if last["macd"] > last["macd_signal"]:
        long_score += 1
    else:
        short_score += 1

    # Ichimoku
    cloud_top = max(last["cloud_a"], last["cloud_b"])
    cloud_bottom = min(last["cloud_a"], last["cloud_b"])

    if price > cloud_top:
        long_score += 1
    elif price < cloud_bottom:
        short_score += 1

    # Volume confirmation
    volume_ratio = (
        last["volume"] / last["volume_avg"]
        if last["volume_avg"] and last["volume_avg"] > 0
        else 0
    )

    if volume_ratio >= 1.2:
        if price > previous["close"]:
            long_score += 1
        elif price < previous["close"]:
            short_score += 1

    # BOS
    bull_bos = price > last["previous_high"]
    bear_bos = price < last["previous_low"]

    if bull_bos:
        long_score += 2
        reasons.append("Bullish BOS")

    if bear_bos:
        short_score += 2
        reasons.append("Bearish BOS")

    # Liquidity sweep
    bull_sweep = (
        last["low"] < last["previous_low"] and
        price > last["previous_low"]
    )

    bear_sweep = (
        last["high"] > last["previous_high"] and
        price < last["previous_high"]
    )

    if bull_sweep:
        long_score += 2
        reasons.append("Sell-side liquidity sweep")

    if bear_sweep:
        short_score += 2
        reasons.append("Buy-side liquidity sweep")

    # FVG
    bullish_fvg = bool(data["bullish_fvg"].tail(10).any())
    bearish_fvg = bool(data["bearish_fvg"].tail(10).any())

    if bullish_fvg:
        long_score += 1
        reasons.append("Bullish FVG detected")

    if bearish_fvg:
        short_score += 1
        reasons.append("Bearish FVG detected")

    # Order Block approximation
    bullish_ob = bool(data["bullish_ob"].tail(10).any())
    bearish_ob = bool(data["bearish_ob"].tail(10).any())

    if bullish_ob:
        long_score += 1
        reasons.append("Bullish OB detected")

    if bearish_ob:
        short_score += 1
        reasons.append("Bearish OB detected")

    # Final decision
    signal = "NO SIGNAL"
    entry = None
    stop_loss = None
    take_profit = None

    if long_score >= 9 and long_score - short_score >= 3:
        signal = "LONG"
        entry = price
        stop_loss = price - (1.5 * atr)
        take_profit = price + (3 * atr)

    elif short_score >= 9 and short_score - long_score >= 3:
        signal = "SHORT"
        entry = price
        stop_loss = price + (1.5 * atr)
        take_profit = price - (3 * atr)

    max_score = 17
    confidence = round(
        min(100, max(long_score, short_score) / max_score * 100),
        1
    )

    return {
        "signal": signal,
        "price": round(price, 2),
        "entry": round(entry, 2) if entry is not None else None,
        "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
        "take_profit": round(take_profit, 2) if take_profit is not None else None,
        "long_score": long_score,
        "short_score": short_score,
        "confidence": confidence,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "volume_ratio": round(float(volume_ratio), 2),
        "reasons": reasons
    }
