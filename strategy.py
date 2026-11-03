
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

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Ichimoku
    tenkan = (
        high.rolling(9).max() + low.rolling(9).min()
    ) / 2

    kijun = (
        high.rolling(26).max() + low.rolling(26).min()
    ) / 2

    df["cloud_a"] = ((tenkan + kijun) / 2).shift(26)

    df["cloud_b"] = (
        (high.rolling(52).max() + low.rolling(52).min()) / 2
    ).shift(26)

    # Volume
    df["volume_avg"] = volume.rolling(20).mean()

    # Previous 20-candle levels
    df["previous_high"] = high.shift(1).rolling(20).max()
    df["previous_low"] = low.shift(1).rolling(20).min()

    # Three-candle FVG
    df["bullish_fvg"] = low > high.shift(2)
    df["bearish_fvg"] = high < low.shift(2)

    return df


def get_trend(df):
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
    price = float(last["close"])
    atr = float(last["atr"])

    trend_15m = get_trend(df_15m)
    trend_1h = get_trend(df_1h)
    trend_4h = get_trend(df_4h)

    long_score = 0
    short_score = 0
    reasons = []

    # Multi-timeframe trend
    for trend, weight, label in [
        (trend_15m, 2, "15m"),
        (trend_1h, 2, "1h"),
        (trend_4h, 3, "4h")
    ]:
        if trend == "Bullish":
            long_score += weight
            reasons.append(f"{label} trend bullish")
        elif trend == "Bearish":
            short_score += weight
            reasons.append(f"{label} trend bearish")

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
    cloud_values = [last["cloud_a"], last["cloud_b"]]

    if all(pd.notna(x) for x in cloud_values):
        cloud_top = max(cloud_values)
        cloud_bottom = min(cloud_values)

        if price > cloud_top:
            long_score += 1
        elif price < cloud_bottom:
            short_score += 1

    # Volume confirmation
    volume_avg = last["volume_avg"]
    volume_ratio = (
        float(last["volume"]) / float(volume_avg)
        if pd.notna(volume_avg) and volume_avg > 0 else 0
    )

    if volume_ratio >= 1.2:
        if price > float(data.iloc[-2]["close"]):
            long_score += 1
            reasons.append("Strong bullish volume")
        elif price < float(data.iloc[-2]["close"]):
            short_score += 1
            reasons.append("Strong bearish volume")

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
        last["low"] < last["previous_low"]
        and price > last["previous_low"]
    )

    bear_sweep = (
        last["high"] > last["previous_high"]
        and price < last["previous_high"]
    )

    if bull_sweep:
        long_score += 2
        reasons.append("Sell-side liquidity sweep")

    if bear_sweep:
        short_score += 2
        reasons.append("Buy-side liquidity sweep")

    # FVG zones from recent candles
    bullish_fvg = False
    bearish_fvg = False
    fvg_low = None
    fvg_high = None

    for i in range(max(2, len(data) - 20), len(data)):
        candle = data.iloc[i]

        if candle["low"] > data.iloc[i-2]["high"]:
            zone_low = float(data.iloc[i-2]["high"])
            zone_high = float(candle["low"])

            # Approximate mitigation check
            later = data.iloc[i+1:]
            if later.empty or later["low"].min() > zone_low:
                bullish_fvg = True
                fvg_low, fvg_high = zone_low, zone_high

        if candle["high"] < data.iloc[i-2]["low"]:
            zone_low = float(candle["high"])
            zone_high = float(data.iloc[i-2]["low"])

            later = data.iloc[i+1:]
            if later.empty or later["high"].max() < zone_high:
                bearish_fvg = True
                fvg_low, fvg_high = zone_low, zone_high

    if bullish_fvg:
        long_score += 1
        reasons.append("Bullish FVG detected")

    if bearish_fvg:
        short_score += 1
        reasons.append("Bearish FVG detected")

    # Simple order block approximation
    bullish_ob = False
    bearish_ob = False

    for i in range(max(1, len(data) - 10), len(data)):
        current = data.iloc[i]
        previous = data.iloc[i-1]

        if (
            previous["close"] < previous["open"]
            and current["close"] > previous["high"]
        ):
            bullish_ob = True

        if (
            previous["close"] > previous["open"]
            and current["close"] < previous["low"]
        ):
            bearish_ob = True

    if bullish_ob:
        long_score += 1
        reasons.append("Bullish Order Block approximation")

    if bearish_ob:
        short_score += 1
        reasons.append("Bearish Order Block approximation")

    # Premium / Discount
    recent_high = float(data["high"].tail(50).max())
    recent_low = float(data["low"].tail(50).min())
    midpoint = (recent_high + recent_low) / 2

    zone = "Premium" if price > midpoint else "Discount"

    # Final signal
    signal = "NO SIGNAL"
    entry = None
    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None

    if atr > 0:
        if long_score >= 10 and long_score - short_score >= 3:
            signal = "LONG"
            entry = price
            stop_loss = price - 1.5 * atr
            tp1 = price + 1.5 * atr
            tp2 = price + 2.5 * atr
            tp3 = price + 3.5 * atr

        elif short_score >= 10 and short_score - long_score >= 3:
            signal = "SHORT"
            entry = price
            stop_loss = price + 1.5 * atr
            tp1 = price - 1.5 * atr
            tp2 = price - 2.5 * atr
            tp3 = price - 3.5 * atr

    max_score = max(long_score, short_score)
    confidence = round(min(100, max_score / 17 * 100), 1)

    return {
        "signal": signal,
        "price": round(price, 2),
        "entry": round(entry, 2) if entry is not None else None,
        "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
        "tp1": round(tp1, 2) if tp1 is not None else None,
        "tp2": round(tp2, 2) if tp2 is not None else None,
        "tp3": round(tp3, 2) if tp3 is not None else None,
        "long_score": long_score,
        "short_score": short_score,
        "confidence": confidence,
        "trend_15m": trend_15m,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "rsi": round(float(last["rsi"]), 2),
        "macd": round(float(last["macd"]), 2),
        "macd_signal": round(float(last["macd_signal"]), 2),
        "atr": round(atr, 2),
        "ema20": round(float(last["ema20"]), 2),
        "ema50": round(float(last["ema50"]), 2),
        "ema200": round(float(last["ema200"]), 2),
        "volume_ratio": round(volume_ratio, 2),
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "fvg_low": fvg_low,
        "fvg_high": fvg_high,
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "bull_bos": bool(bull_bos),
        "bear_bos": bool(bear_bos),
        "bull_sweep": bool(bull_sweep),
        "bear_sweep": bool(bear_sweep),
        "zone": zone,
        "reasons": reasons
    }


def chart_indicators(df):
    return add_indicators(df)
