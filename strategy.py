import numpy as np
import pandas as pd

STRATEGY_VERSION = "SMC-V3.0"
MIN_CANDLES = 120

SWING_LOOKBACK = 5        # تعداد کندل برای تشخیص سویینگ‌های سقف و کف
FVG_THRESHOLD_PCT = 0.0015 # حداقل سایز گپ قیمت (0.15 درصد)
RR_RATIO = 2.5            # نسبت سود به زیان استاندارد اسمارت مانی

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df[required].dropna()
    return out[(out["high"] >= out["low"]) & (out["close"] > 0)]

def identify_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    """شناسایی سقف‌ها و کف‌های سوئینگ (Swing Highs / Swing Lows)"""
    highs = df["high"]
    lows = df["low"]
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        if all(highs.iloc[i] > highs.iloc[i - j] for j in range(1, lookback + 1)) and \
           all(highs.iloc[i] > highs.iloc[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((df.index[i], highs.iloc[i]))
            
        if all(lows.iloc[i] < lows.iloc[i - j] for j in range(1, lookback + 1)) and \
           all(lows.iloc[i] < lows.iloc[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((df.index[i], lows.iloc[i]))
            
    return swing_highs, swing_lows

def detect_fvg(df: pd.DataFrame):
    """تشخیص شکاف نقدینگی و ارزش منصفانه (Fair Value Gap)"""
    fvg_bullish = []
    fvg_bearish = []
    
    for i in range(2, len(df)):
        c1_high, c1_low = df["high"].iloc[i-2], df["low"].iloc[i-2]
        c3_high, c3_low = df["high"].iloc[i], df["low"].iloc[i]
        c2_close = df["close"].iloc[i-1]
        
        # گپ صعودی: کف کندل سوم بالاتر از سقف کندل اول باشد
        if c3_low > c1_high:
            gap_pct = (c3_low - c1_high) / c2_close
            if gap_pct >= FVG_THRESHOLD_PCT:
                fvg_bullish.append({
                    "top": c3_low,
                    "bottom": c1_high,
                    "index": df.index[i-1]
                })
                
        # گپ نزولی: سقف کندل سوم پایین‌تر از کف کندل اول باشد
        if c3_high < c1_low:
            gap_pct = (c1_low - c3_high) / c2_close
            if gap_pct >= FVG_THRESHOLD_PCT:
                fvg_bearish.append({
                    "top": c1_low,
                    "bottom": c3_high,
                    "index": df.index[i-1]
                })
                
    return fvg_bullish, fvg_bearish

def detect_market_structure(df: pd.DataFrame):
    """تعیین شکست ساختار (BOS) و روند اسمارت مانی"""
    swing_highs, swing_lows = identify_swings(df)
    if not swing_highs or not swing_lows:
        return "UNKNOWN", None, None
        
    last_swing_high = swing_highs[-1][1]
    last_swing_low = swing_lows[-1][1]
    last_close = df["close"].iloc[-1]
    
    if last_close > last_swing_high:
        return "BULLISH_BOS", last_swing_high, last_swing_low
    elif last_close < last_swing_low:
        return "BEARISH_BOS", last_swing_high, last_swing_low
        
    return "RANGE", last_swing_high, last_swing_low

def calculate_atr(df: pd.DataFrame, period: int = 14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def analyze_market(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, symbol: str = None) -> dict:
    try:
        d15 = validate_data(df_15m)
        d1 = validate_data(df_1h)
        d4 = validate_data(df_4h)
    except Exception as exc:
        return {"signal": "HOLD", "reason": f"DATA_ERROR: {exc}", "symbol": symbol}

    if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
        return {"signal": "HOLD", "reason": "INSUFFICIENT_DATA", "symbol": symbol}

    d15["atr"] = calculate_atr(d15)
    atr = float(d15["atr"].iloc[-1])
    current_price = float(d15["close"].iloc[-1])
    current_low = float(d15["low"].iloc[-1])
    current_high = float(d15["high"].iloc[-1])

    # 1. جهت کلی تایم‌فریم بزرگتر با ساختار ماژور 1h و 4h
    structure_1h, s_high_1h, s_low_1h = detect_market_structure(d1)
    d4_ema50 = d4["close"].ewm(span=50).mean().iloc[-1]
    d4_bullish = d4["close"].iloc[-1] > d4_ema50

    bullish_fvgs, bearish_fvgs = detect_fvg(d15)

    common_meta = {
        "symbol": symbol,
        "entry_price": current_price,
        "atr": atr,
        "strategy_version": STRATEGY_VERSION,
        "risk_reward": RR_RATIO
    }

    # ================== سناریو لاین اسمارت مانی (LONG) ==================
    # شرط: روند 4h صعودی + شکست ساختار صعودی در 1h + پولبک کندل 15m به داخل آخرین Bullish FVG
    if d4_bullish and structure_1h in ["BULLISH_BOS", "RANGE"]:
        if bullish_fvgs:
            recent_fvg = bullish_fvgs[-1]
            # آیا قیمت به محدوده FVG نفوذ کرده و در حال ریجکت است؟
            if recent_fvg["bottom"] <= current_low <= recent_fvg["top"] and current_price > d15["open"].iloc[-1]:
                # اردر بلاک معمولاً زیر سطح گپ است
                stop_loss = min(recent_fvg["bottom"], current_price - (1.2 * atr))
                risk = current_price - stop_loss
                take_profit = current_price + (risk * RR_RATIO)

                return {
                    **common_meta,
                    "signal": "LONG",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trend_1h": "BULLISH_BOS",
                    "trend_4h": "BULLISH",
                    "confidence": 85,
                    "reason": "SMC Entry: 1H Bullish BOS + 15M Bullish FVG Mitigated + Volume Rejection"
                }

    # ================== سناریو شورت اسمارت مانی (SHORT) ==================
    # شرط: روند 4h نزولی + شکست ساختار نزولی در 1h + پولبک کندل 15m به داخل آخرین Bearish FVG
    if not d4_bullish and structure_1h in ["BEARISH_BOS", "RANGE"]:
        if bearish_fvgs:
            recent_fvg = bearish_fvgs[-1]
            if recent_fvg["bottom"] <= current_high <= recent_fvg["top"] and current_price < d15["open"].iloc[-1]:
                stop_loss = max(recent_fvg["top"], current_price + (1.2 * atr))
                risk = stop_loss - current_price
                take_profit = current_price - (risk * RR_RATIO)

                return {
                    **common_meta,
                    "signal": "SHORT",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trend_1h": "BEARISH_BOS",
                    "trend_4h": "BEARISH",
                    "confidence": 85,
                    "reason": "SMC Entry: 1H Bearish BOS + 15M Bearish FVG Mitigated + Supply Pressure"
                }

    return {
        "signal": "HOLD",
        "reason": "NO_SMC_SETUP (Waiting for BOS confirmation & FVG Mitigations)",
        "symbol": symbol
    }
