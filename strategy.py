import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("smc_strategy")

STRATEGY_VERSION = "5.0"
MIN_CANDLES = 200

ATR_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX_TREND = 25.0
MIN_STOP_ATR = 0.6
MAX_STOP_ATR = 2.2
STOP_BUFFER_ATR = 0.3
RR_RATIO = 2.0

FEE_PCT_PER_SIDE = 0.0005
SLIPPAGE_PCT_PER_SIDE = 0.0002
ROUND_TRIP_COST_PCT = 2 * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE)


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

    if len(out) < 2:
        return out[["timestamp"] + required]

    tail = out["timestamp"].iloc[-MIN_CANDLES:]
    if len(tail) > 1:
        gaps = tail.diff().dropna().dt.total_seconds() / 60.0
        if (gaps > tf_minutes * 3.5).any():
            raise ValueError(f"Major gap in {tf_minutes}m candles")

    freq = f"{tf_minutes}min"
    on_grid = out["timestamp"].dt.floor(freq).eq(out["timestamp"]).all()
    if on_grid and len(out) > 1:
        full = pd.date_range(out["timestamp"].iloc[0], out["timestamp"].iloc[-1], freq=freq)
        if len(full) != len(out):
            out = out.set_index("timestamp").reindex(full)
            out["close"] = out["close"].ffill()
            for c in ("open", "high", "low"):
                out[c] = out[c].fillna(out["close"])
            out["volume"] = out["volume"].fillna(0.0)
            out = out.rename_axis("timestamp").reset_index()

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


def hold(code: str, reason: str, **extra) -> dict:
    res = {
        "signal": "HOLD", "side": "HOLD", "code": code, "confidence": 0,
        "reason": reason, "strategy_version": STRATEGY_VERSION,
    }
    res.update(extra)
    return res


def analyze_market(df_15m, df_1h, df_4h, symbol: str = None, now=None, only_closed: bool = True) -> dict:
    """
    Strategy v5.0: 4H Trend Alignment + 1H Clean EMA20 Pullback & Rejection.
    Evaluates on newly closed 1H candles (or any closed 1H candle via SignalGuard).
    """
    try:
        d1 = validate_data(df_1h, 60)
        d4 = validate_data(df_4h, 240)

        if only_closed:
            d1 = drop_unclosed(d1, 60, now)
            d4 = drop_unclosed(d4, 240, now)

        if min(len(d1), len(d4)) < MIN_CANDLES:
            return hold("HOLD_INSUFFICIENT_DATA", "تعداد کندل کافی نیست", symbol=symbol)

        t1_end = d1["timestamp"].iloc[-1] + pd.Timedelta(minutes=60)
        t4_end = d4["timestamp"].iloc[-1] + pd.Timedelta(minutes=240)
        lag_min = (t1_end - t4_end).total_seconds() / 60.0
        if lag_min < -1 or lag_min > 240:
            return hold("HOLD_DESYNC_DATA", "عدم هماهنگی زمانی تایم‌فریم ۱H و ۴H", symbol=symbol)

        # 4H Trend Filter
        ema20_4h_s = d4["close"].ewm(span=20, adjust=False).mean()
        ema50_4h_s = d4["close"].ewm(span=50, adjust=False).mean()
        ema20_4h = float(ema20_4h_s.iloc[-1])
        ema20_4h_prev = float(ema20_4h_s.iloc[-3])
        ema50_4h = float(ema50_4h_s.iloc[-1])
        close_4h = float(d4["close"].iloc[-1])

        # Stricter slope check for LONG to prevent weak bounce traps
        bull_4h = (close_4h > ema20_4h > ema50_4h) and (ema20_4h > ema20_4h_prev)
        bear_4h = (close_4h < ema20_4h < ema50_4h)

        # 1H Indicators
        ema20_1h = float(d1["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50_1h = float(d1["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        atr_1h = float(calculate_atr(d1).iloc[-1])
        adx_s, pdi_s, mdi_s = calculate_directional_adx(d1)
        adx_1h = float(adx_s.iloc[-1])
        plus_di = float(pdi_s.iloc[-1])
        minus_di = float(mdi_s.iloc[-1])

        if not np.isfinite(atr_1h) or atr_1h <= 0:
            return hold("HOLD_INVALID_ATR", "مقدار ATR نامعتبر است", symbol=symbol)

        if not np.isfinite(adx_1h) or adx_1h < MIN_ADX_TREND:
            return hold("HOLD_LOW_ADX", f"قدرت روند ۱H پایین است (ADX={adx_1h:.1f})", symbol=symbol)

        bar = d1.iloc[-1]
        op = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        price = float(bar["close"])
        candle_time = d1["timestamp"].iloc[-1]

        common = {
            "symbol": symbol,
            "entry": price,
            "entry_price": price,
            "atr": atr_1h,
            "adx_1h": round(adx_1h, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "structure_1h": "1H_EMA20_PULLBACK",
            "trend_4h": "BULLISH" if bull_4h else ("BEARISH" if bear_4h else "NEUTRAL"),
            "candle_time": candle_time.isoformat(),
            "strategy_version": STRATEGY_VERSION,
            "risk_reward": RR_RATIO,
        }

        signal_side = None
        stop_loss = 0.0

        # LONG Setup: 4H Bullish + 1H EMA20 > EMA50 + Pullback wick into EMA20 + Bullish Close above EMA20
        if bull_4h and (ema20_1h > ema50_1h) and (plus_di > minus_di):
            if low <= ema20_1h * 1.001 and price > ema20_1h and price > op:
                sl_cand = min(low, ema50_1h) - STOP_BUFFER_ATR * atr_1h
                risk_cand = price - sl_cand
                if MIN_STOP_ATR * atr_1h <= risk_cand <= MAX_STOP_ATR * atr_1h:
                    signal_side = "LONG"
                    stop_loss = sl_cand

        # SHORT Setup: 4H Bearish + 1H EMA20 < EMA50 + Pullback wick into EMA20 + Bearish Close below EMA20
        elif bear_4h and (ema20_1h < ema50_1h) and (minus_di > plus_di):
            if high >= ema20_1h * 0.999 and price < ema20_1h and price < op:
                sl_cand = max(high, ema50_1h) + STOP_BUFFER_ATR * atr_1h
                risk_cand = sl_cand - price
                if MIN_STOP_ATR * atr_1h <= risk_cand <= MAX_STOP_ATR * atr_1h:
                    signal_side = "SHORT"
                    stop_loss = sl_cand

        if not signal_side:
            return hold("HOLD_NO_PULLBACK", "شرایط پولبک تمیز به EMA20 در ۱H برقرار نیست", **common)

        risk_dist = abs(price - stop_loss)
        cost = price * ROUND_TRIP_COST_PCT
        reward = RR_RATIO * (risk_dist + cost) + cost
        target = price + reward if signal_side == "LONG" else price - reward

        conf = 65
        if adx_1h >= 30:
            conf += 10
        if abs(plus_di - minus_di) >= 10:
            conf += 10

        word = "صعودی" if signal_side == "LONG" else "نزولی"
        return {
            **common,
            "signal": signal_side,
            "side": signal_side,
            "code": f"SIGNAL_{signal_side}",
            "stop": stop_loss,
            "stop_loss": stop_loss,
            "target": target,
            "take_profit": target,
            "risk_dist": risk_dist,
            "cost_pct": ROUND_TRIP_COST_PCT,
            "zone_id": f"{signal_side}:1H_EMA20:{candle_time.isoformat()}",
            "confidence": min(conf, 95),
            "reason": f"روند {word} ۴H + پولبک تمیز به EMA20 یک‌ساعته (ADX={adx_1h:.1f})",
        }

    except Exception as exc:
        logger.exception("analyze_market failed for %s", symbol)
        return hold("HOLD_RUNTIME_ERROR", f"خطای داخلی تحلیل: {exc}", symbol=symbol)


def generate_signal(df_15m, df_1h, df_4h, symbol=None, now=None, only_closed=True):
    return analyze_market(df_15m, df_1h, df_4h, symbol, now=now, only_closed=only_closed)


class SignalGuard:
    """Prevents duplicate entries on the same 1H candle or within cooldown."""

    def __init__(self, cooldown_minutes: int = 120, path: str = None):
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
