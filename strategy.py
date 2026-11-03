
import pandas as pd
import numpy as np


def analyze_market(df):
    df = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    if len(df) < 100:
        raise ValueError("At least 100 candles are required")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Ichimoku
    tenkan = (
        high.rolling(9).max() +
        low.rolling(9).min()
    ) / 2

    kijun = (
        high.rolling(26).max() +
        low.rolling(26).min()
    ) / 2

    cloud_a = ((tenkan + kijun) / 2).shift(26)

    cloud_b = (
        (
            high.rolling(52).max() +
            low.rolling(52).min()
        ) / 2
    ).shift(26)

    df["TENKAN"] = tenkan
    df["KIJUN"] = kijun
    df["CLOUD_A"] = cloud_a
    df["CLOUD_B"] = cloud_b

    # Market Structure
    previous_high = high.shift(1).rolling(20).max()
    previous_low = low.shift(1).rolling(20).min()

    bull_bos = close.iloc[-1] > previous_high.iloc[-1]
    bear_bos = close.iloc[-1] < previous_low.iloc[-1]

    # CHoCH - simplified trend change
    previous_trend_bull = (
        close.iloc[-2] > df["EMA50"].iloc[-2]
    )

    bull_choch = bull_bos and not previous_trend_bull
    bear_choch = bear_bos and previous_trend_bull

    # Liquidity Sweep
    bull_sweep = (
        low.iloc[-1] < previous_low.iloc[-1]
        and close.iloc[-1] > previous_low.iloc[-1]
    )

    bear_sweep = (
        high.iloc[-1] > previous_high.iloc[-1]
        and close.iloc[-1] < previous_high.iloc[-1]
    )

    # Fair Value Gap
    bull_fvg = False
    bear_fvg = False

    for i in range(max(2, len(df) - 10), len(df)):
        if low.iloc[i] > high.iloc[i - 2]:
            bull_fvg = True

        if high.iloc[i] < low.iloc[i - 2]:
            bear_fvg = True

    # Volume
    volume_average = volume.rolling(20).mean()

    current_volume = volume.iloc[-1]
    average_volume = volume_average.iloc[-1]

    volume_ratio = (
        current_volume / average_volume
        if average_volume > 0 else 0
    )

    volume_confirmed = volume_ratio >= 1.2

    # Premium / Discount
    range_high = high.rolling(50).max().iloc[-1]
    range_low = low.rolling(50).min().iloc[-1]

    midpoint = (range_high + range_low) / 2
    price = float(close.iloc[-1])

    discount = price < midpoint
    premium = price > midpoint

    # Latest values
    last = df.iloc[-1]

    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("Invalid ATR")

    if not np.isfinite(rsi):
        raise ValueError("Invalid RSI")

    # Ichimoku confirmation
    cloud_values = [last["CLOUD_A"], last["CLOUD_B"]]

    cloud_valid = all(np.isfinite(x) for x in cloud_values)

    if cloud_valid:
        cloud_top = max(cloud_values)
        cloud_bottom = min(cloud_values)

        ichimoku_bull = (
            price > cloud_top
            and last["TENKAN"] > last["KIJUN"]
        )

        ichimoku_bear = (
            price < cloud_bottom
            and last["TENKAN"] < last["KIJUN"]
        )
    else:
        ichimoku_bull = False
        ichimoku_bear = False

    # Scoring
    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    if (
        last["EMA20"] > last["EMA50"]
        and price > last["EMA20"]
        and last["EMA50"] > last["EMA200"]
    ):
        long_score += 2
        long_reasons.append("EMA bullish trend")

    if (
        last["EMA20"] < last["EMA50"]
        and price < last["EMA20"]
        and last["EMA50"] < last["EMA200"]
    ):
        short_score += 2
        short_reasons.append("EMA bearish trend")

    if ichimoku_bull:
        long_score += 2
        long_reasons.append("Ichimoku bullish")

    if ichimoku_bear:
        short_score += 2
        short_reasons.append("Ichimoku bearish")

    if 50 <= rsi <= 68:
        long_score += 1
        long_reasons.append("RSI bullish")

    if 32 <= rsi <= 50:
        short_score += 1
        short_reasons.append("RSI bearish")

    if last["MACD"] > last["MACD_SIGNAL"]:
        long_score += 1
        long_reasons.append("MACD bullish")

    if last["MACD"] < last["MACD_SIGNAL"]:
        short_score += 1
        short_reasons.append("MACD bearish")

    if bull_bos:
        long_score += 2
        long_reasons.append("Bullish BOS")

    if bear_bos:
        short_score += 2
        short_reasons.append("Bearish BOS")

    if bull_choch:
        long_score += 2
        long_reasons.append("Bullish CHoCH")

    if bear_choch:
        short_score += 2
        short_reasons.append("Bearish CHoCH")

    if bull_fvg:
        long_score += 1
        long_reasons.append("Bullish FVG detected")

    if bear_fvg:
        short_score += 1
        short_reasons.append("Bearish FVG detected")

    if bull_sweep:
        long_score += 1
        long_reasons.append("Sell-side liquidity sweep")

    if bear_sweep:
        short_score += 1
        short_reasons.append("Buy-side liquidity sweep")

    if volume_confirmed:
        if last["MACD"] > last["MACD_SIGNAL"]:
            long_score += 1
            long_reasons.append("Volume confirmation")

        if last["MACD"] < last["MACD_SIGNAL"]:
            short_score += 1
            short_reasons.append("Volume confirmation")

    if discount:
        long_score += 1
        long_reasons.append("Discount zone")

    if premium:
        short_score += 1
        short_reasons.append("Premium zone")

    # Final signal
    signal = None

    if long_score >= 7 and long_score >= short_score + 2:
        signal = "LONG"

    elif short_score >= 7 and short_score >= long_score + 2:
        signal = "SHORT"

    if signal == "LONG":
        stop = price - (1.5 * atr)
        target = price + (3 * atr)
        reasons = long_reasons
        confidence = round(long_score / 14 * 100)

    elif signal == "SHORT":
        stop = price + (1.5 * atr)
        target = price - (3 * atr)
        reasons = short_reasons
        confidence = round(short_score / 14 * 100)

    else:
        stop = None
        target = None
        reasons = []
        confidence = round(max(long_score, short_score) / 14 * 100)

    if last["EMA20"] > last["EMA50"]:
        trend = "Bullish"
    elif last["EMA20"] < last["EMA50"]:
        trend = "Bearish"
    else:
        trend = "Neutral"

    return {
        "signal": signal,
        "price": price,
        "stop": stop,
        "target": target,
        "atr": atr,
        "rsi": rsi,
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "ema200": float(last["EMA200"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"]),
        "trend": trend,
        "ichimoku": (
            "Bullish" if ichimoku_bull
            else "Bearish" if ichimoku_bear
            else "Neutral"
        ),
        "bull_bos": bool(bull_bos),
        "bear_bos": bool(bear_bos),
        "bull_choch": bool(bull_choch),
        "bear_choch": bool(bear_choch),
        "bull_fvg": bool(bull_fvg),
        "bear_fvg": bool(bear_fvg),
        "bull_sweep": bool(bull_sweep),
        "bear_sweep": bool(bear_sweep),
        "volume_ratio": float(volume_ratio),
        "zone": "Discount" if discount else "Premium",
        "long_score": long_score,
        "short_score": short_score,
        "confidence": confidence,
        "reasons": reasons
    }
