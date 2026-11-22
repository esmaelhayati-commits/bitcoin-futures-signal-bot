import time
import logging
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import numpy as np

from strategy import (
    validate_data,
    calculate_atr,
    calculate_directional_adx,
    SignalGuard,
    MIN_CANDLES,
    ROUND_TRIP_COST_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DAYS_BACK = 90
INITIAL_BALANCE = 100.0
LEVERAGE = 20
RISK_PER_TRADE = 0.015
MAX_OPEN_POSITIONS = 2
MAX_SAME_SIDE_POSITIONS = 2
MIN_SL_PCT = 0.004
COOLDOWN_MINUTES = 120
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT"
]

exchange = ccxt.xt({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def fetch_historical_ohlcv(symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    all_rows = []
    current_since = since_ms
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while current_since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=500)
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + tf_ms
            time.sleep(0.2)
        except Exception:
            time.sleep(1.5)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def prepare_indicators(data_store: dict):
    """Vectorized precomputation of indicators across all symbols for ultra-fast simulation."""
    for sym, d in data_store.items():
        d15 = validate_data(d["15m"], 15)
        d1 = validate_data(d["1h"], 60)
        d4 = validate_data(d["4h"], 240)

        # 4H Trend Indicators (shifted by 1 so only closed 4H candles are read)
        d4["ema20"] = d4["close"].ewm(span=20, adjust=False).mean()
        d4["ema50"] = d4["close"].ewm(span=50, adjust=False).mean()
        d4["bull_4h"] = (d4["close"] > d4["ema20"]) & (d4["ema20"] > d4["ema50"])
        d4["bear_4h"] = (d4["close"] < d4["ema20"]) & (d4["ema20"] < d4["ema50"])
        d4["close_time"] = d4["timestamp"] + pd.Timedelta(minutes=240)

        # 1H Trend, Breakout & Pullback Indicators
        d1["ema20"] = d1["close"].ewm(span=20, adjust=False).mean()
        d1["ema50"] = d1["close"].ewm(span=50, adjust=False).mean()
        d1["atr"] = calculate_atr(d1)
        adx1, pdi1, mdi1 = calculate_directional_adx(d1)
        d1["adx"] = adx1
        d1["pdi"] = pdi1
        d1["mdi"] = mdi1
        d1["vol_ma"] = d1["volume"].rolling(20).mean().shift(1)
        d1["donchian_hi"] = d1["high"].rolling(20).max().shift(1)
        d1["donchian_lo"] = d1["low"].rolling(20).min().shift(1)
        d1["close_time"] = d1["timestamp"] + pd.Timedelta(minutes=60)

        # 15m Liquidity Sweep Indicators (24-bar = 6-hour rolling swing high/low)
        d15["atr"] = calculate_atr(d15)
        d15["swing_lo_24"] = d15["low"].rolling(24).min().shift(1)
        d15["swing_hi_24"] = d15["high"].rolling(24).max().shift(1)
        d15["vol_ma"] = d15["volume"].rolling(20).mean().shift(1)
        d15["close_time"] = d15["timestamp"] + pd.Timedelta(minutes=15)

        d["15m_p"] = d15
        d["1h_p"] = d1
        d["4h_p"] = d4
        d["ts_15m"] = d15["timestamp"].values
        d["ct_1h"] = d1["close_time"].values
        d["ct_4h"] = d4["close_time"].values


def run_strategy_sim(mode: str, rr_ratio: float, be_r: float, data_store: dict, all_timestamps: list) -> dict:
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    open_positions = {}
    closed_trades = []
    guard = SignalGuard(cooldown_minutes=COOLDOWN_MINUTES, path=None)

    for candle_open_ts in all_timestamps:
        sim_now = candle_open_ts + pd.Timedelta(minutes=15)
        sim_now_np = np.datetime64(sim_now.tz_convert(None))
        candle_open_np = np.datetime64(candle_open_ts.tz_convert(None))
        is_hour_close = (sim_now.minute == 0)

        # 1. Manage Open Positions on every 15m bar
        for sym in list(open_positions.keys()):
            d = data_store[sym]
            idx15 = np.searchsorted(d["ts_15m"], candle_open_np)
            if idx15 >= len(d["15m_p"]) or d["ts_15m"][idx15] != candle_open_np:
                continue

            bar = d["15m_p"].iloc[idx15]
            high, low = float(bar["high"]), float(bar["low"])
            pos = open_positions[sym]
            side, entry, sl, tp, qty = pos["side"], pos["entry"], pos["stop_loss"], pos["take_profit"], pos["quantity"]

            if side == "LONG":
                if low <= sl:
                    exit_p = sl * (1 - SLIPPAGE_RATE)
                    net = (exit_p - entry) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]
                    continue

                if be_r and not pos["is_be"] and high >= entry + pos["risk_dist"] * be_r:
                    pos["stop_loss"] = entry * (1 + (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_be"] = True

                if high >= tp:
                    exit_p = tp * (1 - SLIPPAGE_RATE)
                    net = (exit_p - entry) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]
            else:
                if high >= sl:
                    exit_p = sl * (1 + SLIPPAGE_RATE)
                    net = (entry - exit_p) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]
                    continue

                if be_r and not pos["is_be"] and low <= entry - pos["risk_dist"] * be_r:
                    pos["stop_loss"] = entry * (1 - (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_be"] = True

                if low <= tp:
                    exit_p = tp * (1 + SLIPPAGE_RATE)
                    net = (entry - exit_p) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]

        if balance > peak_balance:
            peak_balance = balance
        dd = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

        # 2. Scan Signals
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        # 1H strategies only evaluate on closed 1H candles
        if mode in ("1H_EMA_PULLBACK", "1H_BREAKOUT") and not is_hour_close:
            continue

        for sym, d in data_store.items():
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            if sym in open_positions:
                continue

            idx15 = np.searchsorted(d["ts_15m"], candle_open_np)
            if idx15 < MIN_CANDLES or idx15 >= len(d["15m_p"]) or d["ts_15m"][idx15] != candle_open_np:
                continue

            idx1 = np.searchsorted(d["ct_1h"], sim_now_np, side="right") - 1
            idx4 = np.searchsorted(d["ct_4h"], sim_now_np, side="right") - 1
            if idx1 < MIN_CANDLES or idx4 < 60:
                continue

            b15 = d["15m_p"].iloc[idx15]
            b1 = d["1h_p"].iloc[idx1]
            b4 = d["4h_p"].iloc[idx4]

            signal_side = None
            stop_loss = 0.0
            entry_raw = float(b15["close"])

            # --- STRATEGY 1 & 4: Liquidity Sweep + Reclaim (15m) ---
            if mode in ("SWEEP_15M", "SWEEP_STRICT_1H"):
                atr15 = float(b15["atr"])
                if not np.isfinite(atr15) or atr15 <= 0:
                    continue

                trend_ok_long = bool(b4["bull_4h"]) and float(b1["adx"]) >= 22.0 and float(b1["pdi"]) > float(b1["mdi"])
                trend_ok_short = bool(b4["bear_4h"]) and float(b1["adx"]) >= 22.0 and float(b1["mdi"]) > float(b1["pdi"])

                if mode == "SWEEP_STRICT_1H":
                    trend_ok_long = trend_ok_long and (float(b1["close"]) > float(b1["ema20"]) > float(b1["ema50"]))
                    trend_ok_short = trend_ok_short and (float(b1["close"]) < float(b1["ema20"]) < float(b1["ema50"]))

                # Bullish Sweep: low pierces 24-bar swing low, but closes back above it with bullish body
                if trend_ok_long and float(b15["low"]) < float(b15["swing_lo_24"]) and float(b15["close"]) > float(b15["swing_lo_24"]):
                    if float(b15["close"]) > float(b15["open"]):
                        sl_cand = float(b15["low"]) - 0.3 * atr15
                        if 0.5 * atr15 <= (entry_raw - sl_cand) <= 2.5 * atr15:
                            signal_side = "LONG"
                            stop_loss = sl_cand

                # Bearish Sweep: high pierces 24-bar swing high, but closes back below it with bearish body
                elif trend_ok_short and float(b15["high"]) > float(b15["swing_hi_24"]) and float(b15["close"]) < float(b15["swing_hi_24"]):
                    if float(b15["close"]) < float(b15["open"]):
                        sl_cand = float(b15["high"]) + 0.3 * atr15
                        if 0.5 * atr15 <= (sl_cand - entry_raw) <= 2.5 * atr15:
                            signal_side = "SHORT"
                            stop_loss = sl_cand

            # --- STRATEGY 2: 1H Clean EMA20 Pullback ---
            elif mode == "1H_EMA_PULLBACK":
                entry_raw = float(b1["close"])
                atr1 = float(b1["atr"])
                if not np.isfinite(atr1) or atr1 <= 0 or float(b1["adx"]) < 25.0:
                    continue

                if bool(b4["bull_4h"]) and float(b1["ema20"]) > float(b1["ema50"]) and float(b1["pdi"]) > float(b1["mdi"]):
                    # Low dips into/below 1H EMA20 and closes strongly above EMA20
                    if float(b1["low"]) <= float(b1["ema20"]) * 1.001 and entry_raw > float(b1["ema20"]) and entry_raw > float(b1["open"]):
                        sl_cand = min(float(b1["low"]), float(b1["ema50"])) - 0.3 * atr1
                        if 0.6 * atr1 <= (entry_raw - sl_cand) <= 2.2 * atr1:
                            signal_side = "LONG"
                            stop_loss = sl_cand

                elif bool(b4["bear_4h"]) and float(b1["ema20"]) < float(b1["ema50"]) and float(b1["mdi"]) > float(b1["pdi"]):
                    if float(b1["high"]) >= float(b1["ema20"]) * 0.999 and entry_raw < float(b1["ema20"]) and entry_raw < float(b1["open"]):
                        sl_cand = max(float(b1["high"]), float(b1["ema50"])) + 0.3 * atr1
                        if 0.6 * atr1 <= (sl_cand - entry_raw) <= 2.2 * atr1:
                            signal_side = "SHORT"
                            stop_loss = sl_cand

            # --- STRATEGY 3: 1H Donchian 20-Bar Momentum Breakout ---
            elif mode == "1H_BREAKOUT":
                entry_raw = float(b1["close"])
                atr1 = float(b1["atr"])
                if not np.isfinite(atr1) or atr1 <= 0 or float(b1["adx"]) < 24.0:
                    continue
                vol_ok = float(b1["volume"]) > float(b1["vol_ma"]) * 1.2 if np.isfinite(b1["vol_ma"]) else False
                if not vol_ok:
                    continue

                if bool(b4["bull_4h"]) and entry_raw > float(b1["donchian_hi"]) and float(b1["pdi"]) > float(b1["mdi"]):
                    sl_cand = entry_raw - 1.5 * atr1
                    signal_side = "LONG"
                    stop_loss = sl_cand
                elif bool(b4["bear_4h"]) and entry_raw < float(b1["donchian_lo"]) and float(b1["mdi"]) > float(b1["pdi"]):
                    sl_cand = entry_raw + 1.5 * atr1
                    signal_side = "SHORT"
                    stop_loss = sl_cand

            if not signal_side:
                continue

            if sum(1 for p in open_positions.values() if p["side"] == signal_side) >= MAX_SAME_SIDE_POSITIONS:
                continue

            sig_stub = {
                "signal": signal_side, "side": signal_side, "symbol": sym,
                "candle_time": candle_open_ts.isoformat(), "zone_id": f"{mode}:{sym}:{candle_open_ts.isoformat()}"
            }
            if not guard.allow(sig_stub):
                continue

            entry = entry_raw * (1 + SLIPPAGE_RATE) if signal_side == "LONG" else entry_raw * (1 - SLIPPAGE_RATE)
            min_sl = entry * MIN_SL_PCT
            price_risk = max(abs(entry - stop_loss), min_sl)
            stop_loss = entry - price_risk if signal_side == "LONG" else entry + price_risk

            cost = entry * ROUND_TRIP_COST_PCT
            reward = rr_ratio * (price_risk + cost) + cost
            target = entry + reward if signal_side == "LONG" else entry - reward

            risk_amount = balance * RISK_PER_TRADE
            qty = risk_amount / (price_risk + cost)
            notional = qty * entry
            margin = notional / LEVERAGE
            entry_fee = notional * FEE_RATE
            used_margin = sum(p["margin"] for p in open_positions.values())

            if margin + entry_fee > (balance - used_margin) * 0.9:
                continue

            balance -= entry_fee
            open_positions[sym] = {
                "side": signal_side, "entry": entry, "quantity": qty, "stop_loss": stop_loss,
                "take_profit": target, "risk_dist": price_risk, "margin": margin, "is_be": False,
            }

    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] < 0]
    longs = [t for t in closed_trades if t["side"] == "LONG"]
    shorts = [t for t in closed_trades if t["side"] == "SHORT"]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = (gross_w / gross_l) if gross_l > 0 else 0.0
    wr = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    return {
        "final": balance,
        "net_pct": ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100,
        "dd": max_dd,
        "pf": pf,
        "trades": len(closed_trades),
        "wr": wr,
        "long_pnl": sum(t["pnl"] for t in longs),
        "short_pnl": sum(t["pnl"] for t in shorts),
    }


def main():
    now_utc = datetime.now(timezone.utc)
    warmup_start = now_utc - timedelta(days=DAYS_BACK + 35)
    test_start = pd.Timestamp(now_utc - timedelta(days=DAYS_BACK))
    since_ms = int(warmup_start.timestamp() * 1000)

    logging.info("Fetching 90-day (+warmup) historical data from XT...")
    data_store = {}
    for sym in SYMBOLS:
        logging.info("Fetching %s...", sym)
        df15 = fetch_historical_ohlcv(sym, "15m", since_ms)
        df1 = fetch_historical_ohlcv(sym, "1h", since_ms)
        df4 = fetch_historical_ohlcv(sym, "4h", since_ms)
        if min(len(df15), len(df1), len(df4)) > MIN_CANDLES:
            data_store[sym] = {"15m": df15, "1h": df1, "4h": df4}

    logging.info("Precomputing indicators...")
    prepare_indicators(data_store)

    all_timestamps = sorted(set(
        ts for sym_data in data_store.values()
        for ts in sym_data["15m_p"]["timestamp"]
        if ts >= test_start
    ))

    tests = [
        ("1. Liquidity Sweep 15m (RR 2.0)", "SWEEP_15M", 2.0, 1.2),
        ("2. 1H EMA20 Pullback (RR 2.0)", "1H_EMA_PULLBACK", 2.0, 1.2),
        ("3. 1H Donchian Breakout (RR 2.2)", "1H_BREAKOUT", 2.2, 1.2),
        ("4. Sweep 15m + Strict 1H (RR 2.2)", "SWEEP_STRICT_1H", 2.2, 1.2),
    ]

    print("\n" + "=" * 105)
    print(f"{'STRATEGY MODEL':<36} | {'FINAL $':>8} | {'NET %':>8} | {'MAX DD':>7} | {'PF':>5} | {'TRADES':>6} | {'WIN %':>6} | {'LONG $':>7} | {'SHORT $':>7}")
    print("=" * 105)
    for label, mode, rr, be in tests:
        res = run_strategy_sim(mode, rr, be, data_store, all_timestamps)
        print(
            f"{label:<36} | "
            f"${res['final']:>7.2f} | "
            f"{res['net_pct']:>+7.2f}% | "
            f"{res['dd']:>6.2f}% | "
            f"{res['pf']:>5.2f} | "
            f"{res['trades']:>6d} | "
            f"{res['wr']:>5.1f}% | "
            f"{res['long_pnl']:>+7.2f} | "
            f"{res['short_pnl']:>+7.2f}"
        )
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
