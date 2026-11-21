import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("smc_strategy")

STRATEGY_VERSION = "4.1"
MIN_CANDLES = 200

SWING_LOOKBACK = 5
STRUCTURE_RECENT_CANDLES = 6
MAX_FVG_AGE_CANDLES = 48
MIN_FVG_ATR_MULTIPLIER = 0.35
MIN_MOVE_AWAY_ATR = 1.0
STOP_BUFFER_ATR = 0.25
MIN_STOP_ATR = 1.0
MAX_STOP_ATR = 2.2
RR_RATIO = 2.0

FEE_PCT_PER_SIDE = 0.0005
SLIPPAGE_PCT_PER_SIDE = 0.0002
ROUND_TRIP_COST_PCT = 2 * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE)

ATR_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_TREND = 22.0
VOLUME_MA_PERIOD = 20


def _to_datetime(s: pd.Series) -> pd.Series:
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

    tail = out["timestamp"].iloc[-MIN_CANDLES:]
    if len(tail) > 1:
        gaps = tail.diff().dropna().dt.total_seconds() / 60.0
        if (gaps > tf_minutes * 3.5).any():
            raise ValueError(f"Major gap in {tf_minutes}m candles")
    return out[["timestamp"] + required]


def drop_unclosed(df: pd.DataFrame, tf_minutes: int, now=None) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if len(df) and df["timestamp"].iloc[-1] + pd.Timedelta(minutes=tf_minutes) > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


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
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / denom).fillna(0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def identify_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    window = 2 * lookback + 1
    rmax = df["high"].rolling(window=window, center=True).max()
    rmin = df["low"].rolling(window=window, center=True).min()
    hi_mask = ((df["high"] == rmax) & rmax.notna()).values
    lo_mask = ((df["low"] == rmin) & rmin.notna()).values
    highs, lows = df["high"].values, df["low"].values
    swing_highs = [(int(i), float(highs[i])) for i in np.flatnonzero(hi_mask)]
    swing_lows = [(int(i), float(lows[i])) for i in np.flatnonzero(lo_mask)]
    return swing_highs, swing_lows


def detect_market_structure(df: pd.DataFrame) -> dict:
    swing_highs, swing_lows = identify_swings(df)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": "RANGE", "direction": None}

    last_sh_idx, last_sh = swing_highs[-1]
    _, prev_sh = swing_highs[-2]
    last_sl_idx, last_sl = swing_lows[-1]
    _, prev_sl = swing_lows[-2]

    closes, highs, lows = df["close"].values, df["high"].values, df["low"].values
    n = len(df)
    first = max(1, n - STRUCTURE_RECENT_CANDLES)

    bull_idx = -1
    for i in range(max(last_sh_idx + 1, first), n):
        if closes[i] > last_sh and closes[i - 1] <= last_sh:
            bull_idx = i

    bear_idx = -1
    for i in range(max(last_sl_idx + 1, first), n):
        if closes[i] < last_sl and closes[i - 1] >= last_sl:
            bear_idx = i

    if bull_idx > bear_idx and bull_idx != -1:
        leg_start = last_sh_idx + int(np.argmin(lows[last_sh_idx:bull_idx + 1]))
        if closes[-1] > lows[leg_start]:
            return {
                "type": "BULLISH_BOS" if last_sl > prev_sl else "BULLISH_CHOCH",
                "direction": "LONG", "break_idx": bull_idx, "level": last_sh,
                "leg_start_idx": leg_start, "invalidation": float(lows[leg_start]),
            }

    if bear_idx > bull_idx and bear_idx != -1:
        leg_start = last_sl_idx + int(np.argmax(highs[last_sl_idx:bear_idx + 1]))
        if closes[-1] < highs[leg_start]:
            return {
                "type": "BEARISH_BOS" if last_sh < prev_sh else "BEARISH_CHOCH",
                "direction": "SHORT", "break_idx": bear_idx, "level": last_sl,
                "leg_start_idx": leg_start, "invalidation": float(highs[leg_start]),
            }

    return {"type": "RANGE", "direction": None}


def detect_unmitigated_fvgs(df: pd.DataFrame, side: str) -> list:
    n = len(df)
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    atrs = df["atr"].values
    ts = df["timestamp"]
    out = []

    for i in range(max(2, n - MAX_FVG_AGE_CANDLES), n - 2):
        atr_i = atrs[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        if side == "LONG":
            if lows[i] <= highs[i - 2]:
                continue
            top, bottom = float(lows[i]), float(highs[i - 2])
        else:
            if highs[i] >= lows[i - 2]:
                continue
            top, bottom = float(lows[i - 2]), float(highs[i])

        gap = top - bottom
        if gap < MIN_FVG_ATR_MULTIPLIER * atr_i:
            continue

        mid = (top + bottom) / 2.0
        mitigated, away = False, 0.0
        for k in range(i + 1, n - 1):
            if side == "LONG":
                if closes[k] < bottom or lows[k] <= mid:
                    mitigated = True
                    break
                away = max(away, highs[k] - top)
            else:
                if closes[k] > top or highs[k] >= mid:
                    mitigated = True
                    break
                away = max(away, bottom - lows[k])

        if mitigated or away < MIN_MOVE_AWAY_ATR * atr_i:
            continue

        mid_ts = ts.iloc[i - 1]
        out.append({
            "top": top, "bottom": bottom, "mid": mid, "gap_atr": gap / atr_i,
            "mid_ts": mid_ts, "zone_id": f"{side}:{mid_ts.isoformat()}:{bottom:.4f}:{top:.4f}",
        })
    return out


def hold(code: str, reason: str, **extra) -> dict:
    res = {
        "signal": "HOLD", "side": "HOLD", "code": code, "confidence": 0,
        "reason": reason, "strategy_version": STRATEGY_VERSION,
    }
    res.update(extra)
    return res


def _evaluate_side(side: str, c: dict):
    is_long = side == "LONG"
    s = c["struct"]

    if not (s["direction"] == side and c["d4_ok"][side] and c["di_ok"][side]):
        return None, 0, "HOLD_NO_ALIGNMENT", "هم‌جهتی ۴H / ساختار ۱H / DI برقرار نیست"

    fvgs = detect_unmitigated_fvgs(c["d15"], side)
    if not fvgs:
        return None, 2, "HOLD_NO_FVG", "FVG دست‌نخورده با فاصله‌گرفتن کافی وجود ندارد"

    linked = [f for f in fvgs if c["leg_start_ts"] <= f["mid_ts"] <= c["leg_end_ts"]]
    if not linked:
        return None, 3, "HOLD_FVG_NOT_LINKED", "FVG داخل لگ شکست ساختار ۱H نیست"

    price, op, low, high = c["price"], c["open"], c["low"], c["high"]
    atr = c["atr"]
    stage, code, msg = 3, "HOLD_FVG_NOT_LINKED", ""

    for fvg in reversed(linked):
        mid = fvg["mid"]
        if is_long:
            tapped = low <= fvg["top"] and price >= fvg["bottom"]
            rejection = price > op and price > mid
        else:
            tapped = high >= fvg["bottom"] and price <= fvg["top"]
            rejection = price < op and price < mid

        if not tapped:
            if stage < 4:
                stage, code, msg = 4, "HOLD_NO_TAP", "قیمت هنوز به FVG نرسیده"
            continue
        if not rejection:
            if stage < 5:
                stage, code, msg = 5, "HOLD_NO_REJECTION", "تپ شده ولی کندل برگشتی بسته نشده"
            continue

        if is_long:
            stop = fvg["bottom"] - STOP_BUFFER_ATR * atr
            risk = price - stop
        else:
            stop = fvg["top"] + STOP_BUFFER_ATR * atr
            risk = stop - price

        if risk > MAX_STOP_ATR * atr:
            if stage < 6:
                stage, code, msg = 6, "HOLD_RISK_TOO_WIDE", "استاپ از سقف مجاز ATR بزرگ‌تر است"
            continue
        if risk < MIN_STOP_ATR * atr:
            risk = MIN_STOP_ATR * atr
            stop = price - risk if is_long else price + risk

        cost = price * ROUND_TRIP_COST_PCT
        reward = RR_RATIO * (risk + cost) + cost
        target = price + reward if is_long else price - reward

        conf = 55
        if "BOS" in s["type"]:
            conf += 10
        if c["adx_1h"] >= 28:
            conf += 10
        if c["vol_ma"] > 0 and c["vol"] > c["vol_ma"]:
            conf += 10
        if fvg["gap_atr"] >= 0.6:
            conf += 5
        if abs(c["plus_di"] - c["minus_di"]) >= 10:
            conf += 5

        word = "صعودی" if is_long else "نزولی"
        return {
            **c["common"],
            "signal": side, "side": side, "code": f"SIGNAL_{side}",
            "stop": stop, "stop_loss": stop, "target": target, "take_profit": target,
            "risk_dist": risk, "cost_pct": ROUND_TRIP_COST_PCT,
            "zone_id": fvg["zone_id"],
            "confidence": min(conf, 95),
            "reason": f"روند {word} ({s['type']}) + پولبک به FVG دست‌نخورده داخل لگ شکست",
        }, 7, f"SIGNAL_{side}", ""

    return None, stage, code, msg


def analyze_market(df_15m, df_1h, df_4h, symbol: str = None, now=None, only_closed: bool = True) -> dict:
    try:
        d15 = validate_data(df_15m, 15)
        d1 = validate_data(df_1h, 60)
        d4 = validate_data(df_4h, 240)

        if only_closed:
            d15 = drop_unclosed(d15, 15, now)
            d1 = drop_unclosed(d1, 60, now)
            d4 = drop_unclosed(d4, 240, now)

        if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
            return hold("HOLD_INSUFFICIENT_DATA", "تعداد کندل کافی نیست", symbol=symbol)

        t15_end = d15["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
        for df_x, tf in ((d1, 60), (d4, 240)):
            end_x = df_x["timestamp"].iloc[-1] + pd.Timedelta(minutes=tf)
            lag_min = (t15_end - end_x).total_seconds() / 60.0
            if lag_min < -1 or lag_min > tf:
                return hold("HOLD_DESYNC_DATA", f"عدم هماهنگی زمانی تایم‌فریم {tf}m", symbol=symbol)

        d15["atr"] = calculate_atr(d15)
        d15["vol_ma"] = d15["volume"].rolling(VOLUME_MA_PERIOD).mean().shift(1)

        atr = float(d15["atr"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return hold("HOLD_INVALID_ATR", "مقدار ATR نامعتبر است", symbol=symbol)

        vm = d15["vol_ma"].iloc[-1]
        price = float(d15["close"].iloc[-1])

        adx_s, pdi_s, mdi_s = calculate_directional_adx(d1)
        adx_1h, plus_di, minus_di = float(adx_s.iloc[-1]), float(pdi_s.iloc[-1]), float(mdi_s.iloc[-1])
        if not np.isfinite(adx_1h) or adx_1h < MIN_ADX_TREND:
            return hold("HOLD_LOW_ADX", f"قدرت روند پایین است (ADX={adx_1h:.1f})", symbol=symbol)

        struct = detect_market_structure(d1)
        if struct["type"] == "RANGE":
            return hold("HOLD_RANGE_1H", "ساختار ۱ ساعته رنج یا نامعتبر است", symbol=symbol)

        ema20_s = d4["close"].ewm(span=20, adjust=False).mean()
        ema50_s = d4["close"].ewm(span=50, adjust=False).mean()
        ema20, ema20_prev = float(ema20_s.iloc[-1]), float(ema20_s.iloc[-4])
        ema50 = float(ema50_s.iloc[-1])
        d4_close = float(d4["close"].iloc[-1])
        d4_bull = d4_close > ema20 > ema50 and ema20 > ema20_prev
        d4_bear = d4_close < ema20 < ema50 and ema20 < ema20_prev

        candle_time = d15["timestamp"].iloc[-1]
        common = {
            "symbol": symbol, "entry": price, "entry_price": price, "atr": atr,
            "adx_1h": round(adx_1h, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2),
            "structure_1h": struct["type"],
            "trend_1h": "BULLISH" if struct["direction"] == "LONG" else "BEARISH",
            "trend_4h": "BULLISH" if d4_bull else ("BEARISH" if d4_bear else "TRANSITION"),
            "candle_time": candle_time.isoformat(),
            "strategy_version": STRATEGY_VERSION, "risk_reward": RR_RATIO,
        }

        ctx = {
            "d15": d15, "struct": struct, "common": common, "atr": atr, "price": price,
            "open": float(d15["open"].iloc[-1]), "low": float(d15["low"].iloc[-1]),
            "high": float(d15["high"].iloc[-1]), "vol": float(d15["volume"].iloc[-1]),
            "vol_ma": float(vm) if np.isfinite(vm) else 0.0,
            "adx_1h": adx_1h, "plus_di": plus_di, "minus_di": minus_di,
            "d4_ok": {"LONG": d4_bull, "SHORT": d4_bear},
            "di_ok": {"LONG": plus_di > minus_di, "SHORT": minus_di > plus_di},
            "leg_start_ts": d1["timestamp"].iloc[struct["leg_start_idx"]],
            "leg_end_ts": d1["timestamp"].iloc[struct["break_idx"]] + pd.Timedelta(minutes=60),
        }

        best = (-1, "HOLD_NO_SETUP", "شرایط ستاپ برقرار نیست")
        for side in ("LONG", "SHORT"):
            res, stage, code, msg = _evaluate_side(side, ctx)
            if res:
                return res
            if stage > best[0]:
                best = (stage, code, msg)

        return hold(best[1], best[2], **common)

    except Exception as exc:
        logger.exception("analyze_market failed for %s", symbol)
        return hold("HOLD_RUNTIME_ERROR", f"خطای داخلی تحلیل: {exc}", symbol=symbol)


def generate_signal(df_15m, df_1h, df_4h, symbol=None, now=None, only_closed=True):
    return analyze_market(df_15m, df_1h, df_4h, symbol, now=now, only_closed=only_closed)


class SignalGuard:
    def __init__(self, cooldown_minutes: int = 60, path: str = None):
        self.cooldown = pd.Timedelta(minutes=cooldown_minutes)
        self.path = path
        self.state = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                logger.exception("could not load guard state")

    def allow(self, result: dict) -> bool:
        if result.get("signal") not in ("LONG", "SHORT"):
            return False
        key = f"{result.get('symbol')}:{result['side']}"
        t = pd.Timestamp(result["candle_time"])
        last = self.state.get(key)
        if last:
            if last["zone"] == result.get("zone_id"):
                return False
            if t - pd.Timestamp(last["time"]) < self.cooldown:
                return False
        self.state[key] = {"time": t.isoformat(), "zone": result.get("zone_id")}
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.state, f)
            except Exception:
                logger.exception("could not save guard state")
        return True
