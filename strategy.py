import numpy as np
import pandas as pd

STRATEGY_VERSION = "4.0"
MIN_CANDLES = 120

SWING_LOOKBACK = 5
MAX_FVG_AGE_CANDLES = 24
MIN_FVG_ATR_MULTIPLIER = 0.35   # Dynamic FVG threshold relative to ATR
STOP_BUFFER_ATR = 0.25          # Anti-stop-hunt buffer beyond FVG edge
MIN_STOP_ATR = 1.0              # Minimum stop distance
MAX_STOP_ATR = 2.2              # Maximum stop distance cap to avoid unrealistic targets
RR_RATIO = 2.0                  # Realistic 1:2 net risk-to-reward

ATR_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_TREND = 22.0
VOLUME_MA_PERIOD = 20


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    out = out[
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["volume"] >= 0)
        & (out["high"] >= out["low"])
    ]
    return out.reset_index(drop=True)


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_directional_adx(df: pd.DataFrame, period: int = ADX_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def identify_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    """Vectorized swing detection supporting equal highs/lows (>= / <=)."""
    window = 2 * lookback + 1
    rolling_max = df["high"].rolling(window=window, center=True).max()
    rolling_min = df["low"].rolling(window=window, center=True).min()

    swing_high_mask = (df["high"] == rolling_max) & rolling_max.notna()
    swing_low_mask = (df["low"] == rolling_min) & rolling_min.notna()

    swing_highs = [(int(idx), float(val)) for idx, val in zip(df.index[swing_high_mask], df["high"][swing_high_mask])]
    swing_lows = [(int(idx), float(val)) for idx, val in zip(df.index[swing_low_mask], df["low"][swing_low_mask])]
    return swing_highs, swing_lows


def detect_market_structure(df: pd.DataFrame):
    """
    Distinguishes fresh BOS (trend continuation break) vs CHoCH (reversal break)
    and ensures the break happened recently (within last 6 closed candles).
    """
    swing_highs, swing_lows = identify_swings(df)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "RANGE", None, None, -1

    last_sh_idx, last_sh = swing_highs[-1]
    prev_sh_idx, prev_sh = swing_highs[-2]
    last_sl_idx, last_sl = swing_lows[-1]
    prev_sl_idx, prev_sl = swing_lows[-2]

    closes = df["close"].values
    n = len(df)
    recent_window = max(0, n - 6)

    # Check Bullish Break above last swing high
    bull_break_idx = -1
    for i in range(max(last_sh_idx + 1, recent_window), n):
        if closes[i] > last_sh and closes[i - 1] <= last_sh:
            bull_break_idx = i

    # Check Bearish Break below last swing low
    bear_break_idx = -1
    for i in range(max(last_sl_idx + 1, recent_window), n):
        if closes[i] < last_sl and closes[i - 1] >= last_sl:
            bear_break_idx = i

    # Prioritize the most recent structural break
    if bull_break_idx > bear_break_idx and closes[-1] > last_sh:
        structure_type = "BULLISH_BOS" if last_sl > prev_sl else "BULLISH_CHOCH"
        return structure_type, last_sh, last_sl, bull_break_idx

    if bear_break_idx > bull_break_idx and closes[-1] < last_sl:
        structure_type = "BEARISH_BOS" if last_sh < prev_sh else "BEARISH_CHOCH"
        return structure_type, last_sh, last_sl, bear_break_idx

    return "RANGE", last_sh, last_sl, -1


def detect_unmitigated_fvgs(df: pd.DataFrame):
    """
    Detects fresh FVGs scaled by ATR, filters out mitigated (already filled) gaps,
    and ensures price moved away before returning (true pullback).
    """
    bullish_fvgs = []
    bearish_fvgs = []
    n = len(df)
    start_idx = max(2, n - MAX_FVG_AGE_CANDLES)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values

    for i in range(start_idx, n - 2):
        atr_val = atrs[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        min_gap = atr_val * MIN_FVG_ATR_MULTIPLIER

        # Bullish FVG: Candle[i-2].high < Candle[i].low
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            if gap_size >= min_gap:
                top = float(lows[i])
                bottom = float(highs[i - 2])
                mid = (top + bottom) / 2.0

                # Check mitigation by subsequent candles before the current candle (n-1)
                mitigated = False
                moved_away = False
                for k in range(i + 1, n - 1):
                    if highs[k] > top:
                        moved_away = True
                    # Mitigated if a prior candle closed below bottom or penetrated >50% of gap
                    if closes[k] < bottom or lows[k] <= mid:
                        mitigated = True
                        break

                if not mitigated and moved_away:
                    bullish_fvgs.append({"top": top, "bottom": bottom, "pos": i - 1, "gap_atr": gap_size / atr_val})

        # Bearish FVG: Candle[i-2].low > Candle[i].high
        if highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            if gap_size >= min_gap:
                top = float(lows[i - 2])
                bottom = float(highs[i])
                mid = (top + bottom) / 2.0

                mitigated = False
                moved_away = False
                for k in range(i + 1, n - 1):
                    if lows[k] < bottom:
                        moved_away = True
                    if closes[k] > top or highs[k] >= mid:
                        mitigated = True
                        break

                if not mitigated and moved_away:
                    bearish_fvgs.append({"top": top, "bottom": bottom, "pos": i - 1, "gap_atr": gap_size / atr_val})

    return bullish_fvgs, bearish_fvgs


def hold(code: str, reason: str, **extra) -> dict:
    res = {
        "signal": "HOLD",
        "side": "HOLD",
        "code": code,
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

        if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
            return hold("HOLD_INSUFFICIENT_DATA", "تعداد کندل کافی نیست", symbol=symbol)

        # Timestamp synchronization check (max 4h drift allowed between latest closed candles)
        if "timestamp" in d15.columns and "timestamp" in d4.columns:
            time_diff = abs((d15["timestamp"].iloc[-1] - d4["timestamp"].iloc[-1]).total_seconds())
            if time_diff > 4 * 3600 + 900:
                return hold("HOLD_DESYNC_DATA", "عدم هماهنگی زمانی بین تایم‌فریم‌ها", symbol=symbol)

        d15["atr"] = calculate_atr(d15)
        d15["vol_ma"] = d15["volume"].rolling(VOLUME_MA_PERIOD).mean()

        atr = float(d15["atr"].iloc[-1])
        current_price = float(d15["close"].iloc[-1])
        current_open = float(d15["open"].iloc[-1])
        current_low = float(d15["low"].iloc[-1])
        current_high = float(d15["high"].iloc[-1])
        current_vol = float(d15["volume"].iloc[-1])
        vol_ma = float(d15["vol_ma"].iloc[-1]) if np.isfinite(d15["vol_ma"].iloc[-1]) else 0.0

        if not np.isfinite(atr) or atr <= 0:
            return hold("HOLD_INVALID_ATR", "مقدار ATR نامعتبر است", symbol=symbol)

        # 1. Directional ADX on 1H
        adx_series, plus_di_series, minus_di_series = calculate_directional_adx(d1)
        adx_1h = float(adx_series.iloc[-1])
        plus_di = float(plus_di_series.iloc[-1])
        minus_di = float(minus_di_series.iloc[-1])

        if not np.isfinite(adx_1h) or adx_1h < MIN_ADX_TREND:
            return hold("HOLD_LOW_ADX", f"قدرت روند پایین است (ADX={adx_1h:.1f})", symbol=symbol)

        # 2. Fresh 1H Market Structure (BOS or CHoCH)
        structure_1h, _, _, _ = detect_market_structure(d1)
        if structure_1h == "RANGE":
            return hold("HOLD_RANGE_1H", "ساختار تایم‌فریم ۱ ساعته در حالت رنج است", symbol=symbol)

        # 3. 4H Macro Trend & True EMA Slope
        ema20_4h_s = d4["close"].ewm(span=20, adjust=False).mean()
        ema50_4h_s = d4["close"].ewm(span=50, adjust=False).mean()

        ema20_now, ema20_prev = float(ema20_4h_s.iloc[-1]), float(ema20_4h_s.iloc[-4])
        ema50_now = float(ema50_4h_s.iloc[-1])
        d4_close = float(d4["close"].iloc[-1])

        ema20_slope_up = ema20_now > ema20_prev
        ema20_slope_down = ema20_now < ema20_prev

        d4_bullish = (d4_close > ema20_now > ema50_now) and ema20_slope_up
        d4_bearish = (d4_close < ema20_now < ema50_now) and ema20_slope_down

        trend_1h_label = "BULLISH" if "BULLISH" in structure_1h else "BEARISH"
        trend_4h_label = "BULLISH" if d4_bullish else ("BEARISH" if d4_bearish else "TRANSITION")

        bullish_fvgs, bearish_fvgs = detect_unmitigated_fvgs(d15)

        common = {
            "symbol": symbol,
            "entry": current_price,
            "entry_price": current_price,
            "atr": atr,
            "adx_1h": round(adx_1h, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "structure_1h": structure_1h,
            "trend_1h": trend_1h_label,
            "trend_4h": trend_4h_label,
            "candle_time": str(d15["timestamp"].iloc[-1]) if "timestamp" in d15.columns else "",
            "strategy_version": STRATEGY_VERSION,
            "risk_reward": RR_RATIO,
        }

        # ==================== LONG SETUP ====================
        if d4_bullish and structure_1h in ["BULLISH_BOS", "BULLISH_CHOCH"] and plus_di > minus_di:
            for fvg in reversed(bullish_fvgs):
                tapped = (current_low <= fvg["top"]) and (current_price >= fvg["bottom"])
                bullish_rejection = (current_price > current_open) and (current_price > (fvg["top"] + fvg["bottom"]) / 2.0)

                if tapped and bullish_rejection:
                    raw_stop = fvg["bottom"] - (STOP_BUFFER_ATR * atr)
                    risk = current_price - raw_stop

                    # Clamp risk between MIN_STOP_ATR and MAX_STOP_ATR
                    if risk < MIN_STOP_ATR * atr:
                        risk = MIN_STOP_ATR * atr
                        raw_stop = current_price - risk
                    elif risk > MAX_STOP_ATR * atr:
                        continue  # Skip oversized FVG with unrealistic stop/target

                    target = current_price + (risk * RR_RATIO)

                    # Dynamic Confidence Calculation (50 - 98)
                    conf = 60
                    if structure_1h == "BULLISH_BOS":
                        conf += 10
                    if adx_1h >= 28:
                        conf += 10
                    if current_vol > vol_ma and vol_ma > 0:
                        conf += 10
                    if fvg["gap_atr"] >= 0.6:
                        conf += 5

                    return {
                        **common,
                        "signal": "LONG",
                        "side": "LONG",
                        "code": "SIGNAL_LONG",
                        "stop": raw_stop,
                        "stop_loss": raw_stop,
                        "target": target,
                        "take_profit": target,
                        "confidence": min(conf, 98),
                        "reason": f"تایید روند صعودی ({structure_1h}) + پولبک معتبر به FVG دست‌نخورده",
                    }

        # ==================== SHORT SETUP ====================
        if d4_bearish and structure_1h in ["BEARISH_BOS", "BEARISH_CHOCH"] and minus_di > plus_di:
            for fvg in reversed(bearish_fvgs):
                tapped = (current_high >= fvg["bottom"]) and (current_price <= fvg["top"])
                bearish_rejection = (current_price < current_open) and (current_price < (fvg["top"] + fvg["bottom"]) / 2.0)

                if tapped and bearish_rejection:
                    raw_stop = fvg["top"] + (STOP_BUFFER_ATR * atr)
                    risk = raw_stop - current_price

                    if risk < MIN_STOP_ATR * atr:
                        risk = MIN_STOP_ATR * atr
                        raw_stop = current_price + risk
                    elif risk > MAX_STOP_ATR * atr:
                        continue

                    target = current_price - (risk * RR_RATIO)

                    conf = 60
                    if structure_1h == "BEARISH_BOS":
                        conf += 10
                    if adx_1h >= 28:
                        conf += 10
                    if current_vol > vol_ma and vol_ma > 0:
                        conf += 10
                    if fvg["gap_atr"] >= 0.6:
                        conf += 5

                    return {
                        **common,
                        "signal": "SHORT",
                        "side": "SHORT",
                        "code": "SIGNAL_SHORT",
                        "stop": raw_stop,
                        "stop_loss": raw_stop,
                        "target": target,
                        "take_profit": target,
                        "confidence": min(conf, 98),
                        "reason": f"تایید روند نزولی ({structure_1h}) + پولبک معتبر به FVG دست‌نخورده",
                    }

        return hold("HOLD_NO_SETUP", "شرایط کامل ستاپ ورود برقرار نیست", **common)

    except Exception as exc:
        return hold("HOLD_RUNTIME_ERROR", f"خطای داخلی تحلیل: {exc}", symbol=symbol)


def generate_signal(df_15m, df_1h, df_4h, symbol=None):
    return analyze_market(df_15m, df_1h, df_4h, symbol)
