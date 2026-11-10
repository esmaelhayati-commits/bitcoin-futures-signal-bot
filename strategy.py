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

    if last_close > last_swing_high:
        return "BULLISH_BOS", last_swing_high, last_swing_low
    elif last_close < last_swing_low:
        return "BEARISH_BOS", last_swing_high, last_swing_low

    return "RANGE", last_swing_high, last_swing_low


def hold(reason: str, **extra) -> dict:
    res = {
        "signal": "HOLD",
        "side": "HOLD",
        "confidence": 0,
        "reason": reason,
        "strategy_version": STRATEGY_VERSION,
    }
    res.update(extra)
    return res


def analyze_market(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, symbol: str = None) -> dict:
    try:
        d15 = validate_data(df_15m)
        d1 = validate_data(df_1h)
        d4 = validate_data(df_4h)
    except Exception as exc:
        return hold(f"DATA_ERROR: {exc}", symbol=symbol)

    if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
        return hold("INSUFFICIENT_DATA", symbol=symbol)

    d15["atr"] = calculate_atr(d15)
    atr = float(d15["atr"].iloc[-1])
    current_price = float(d15["close"].iloc[-1])
    current_open = float(d15["open"].iloc[-1])
    current_low = float(d15["low"].iloc[-1])
    current_high = float(d15["high"].iloc[-1])

    if not np.isfinite(atr) or atr <= 0:
        return hold("INVALID_ATR", symbol=symbol)

    structure_1h, _, _ = detect_market_structure(d1)
    ema50_4h = d4["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    d4_bullish = d4["close"].iloc[-1] > ema50_4h

    trend_1h_label = "BULLISH" if structure_1h == "BULLISH_BOS" else ("BEARISH" if structure_1h == "BEARISH_BOS" else "SIDEWAYS")
    trend_4h_label = "BULLISH" if d4_bullish else "BEARISH"

    bullish_fvgs, bearish_fvgs = detect_fvg(d15)

    common = {
        "symbol": symbol,
        "entry": current_price,
        "entry_price": current_price,
        "atr": atr,
        "trend_1h": trend_1h_label,
        "trend_4h": trend_4h_label,
        "strategy_version": STRATEGY_VERSION,
        "risk_reward": RR_RATIO,
    }

    if d4_bullish and structure_1h in ["BULLISH_BOS", "RANGE"]:
        if bullish_fvgs:
            recent_fvg = bullish_fvgs[-1]
            if recent_fvg["bottom"] <= current_low <= recent_fvg["top"] and current_price > current_open:
                stop = min(recent_fvg["bottom"], current_price - (1.3 * atr))
                risk = current_price - stop
                target = current_price + (risk * RR_RATIO)

                return {
                    **common,
                    "signal": "LONG",
                    "side": "LONG",
                    "stop": stop,
                    "stop_loss": stop,
                    "target": target,
                    "take_profit": target,
                    "confidence": 85,
                    "reason": "تایید شکست ساختار 1H + پولبک به FVG نقدینگی در 15M",
                }

    if not d4_bullish and structure_1h in ["BEARISH_BOS", "RANGE"]:
        if bearish_fvgs:
            recent_fvg = bearish_fvgs[-1]
            if recent_fvg["bottom"] <= current_high <= recent_fvg["top"] and current_price < current_open:
                stop = max(recent_fvg["top"], current_price + (1.3 * atr))
                risk = stop - current_price
                target = current_price - (risk * RR_RATIO)

                return {
                    **common,
                    "signal": "SHORT",
                    "side": "SHORT",
                    "stop": stop,
                    "stop_loss": stop,
                    "target": target,
                    "take_profit": target,
                    "confidence": 85,
                    "reason": "شکست ساختار نزولی 1H + تاچ ناحیه عرضه FVG در 15M",
                }

    return hold("NO_SMC_SETUP", **common)


def generate_signal(df_15m, df_1h, df_4h, symbol=None):
    return analyze_market(df_15m, df_1h, df_4h, symbol)
