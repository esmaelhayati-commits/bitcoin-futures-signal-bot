import time
import logging
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import numpy as np

from strategy import (
    validate_data,
    drop_unclosed,
    calculate_atr,
    calculate_directional_adx,
    detect_market_structure,
    detect_unmitigated_fvgs,
    SignalGuard,
    MIN_CANDLES,
    STOP_BUFFER_ATR,
    MIN_STOP_ATR,
    MAX_STOP_ATR,
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
COOLDOWN_MINUTES = 60
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002

ALL_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT"
]
MAJORS_ONLY = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT", "LINK/USDT:USDT"]

EXPERIMENTS = [
    {
        "name": "0. Baseline (v4.2)",
        "symbols": ALL_SYMBOLS,
        "use_partial_tp1": True,
        "be_trigger_r": None,
        "rr_ratio": 2.0,
        "bos_only": False,
        "max_adx": 999.0,
        "require_rejection": True,
    },
    {
        "name": "A. Full Exit (No TP1, BE @ 1.2R)",
        "symbols": ALL_SYMBOLS,
        "use_partial_tp1": False,
        "be_trigger_r": 1.2,
        "rr_ratio": 2.0,
        "bos_only": False,
        "max_adx": 999.0,
        "require_rejection": True,
    },
    {
        "name": "B. BOS Only + ADX<=42 + Majors",
        "symbols": MAJORS_ONLY,
        "use_partial_tp1": True,
        "be_trigger_r": None,
        "rr_ratio": 2.0,
        "bos_only": True,
        "max_adx": 42.0,
        "require_rejection": True,
    },
    {
        "name": "C. Early FVG Tap (No Rejection Wait)",
        "symbols": ALL_SYMBOLS,
        "use_partial_tp1": True,
        "be_trigger_r": None,
        "rr_ratio": 2.0,
        "bos_only": False,
        "max_adx": 999.0,
        "require_rejection": False,
    },
    {
        "name": "ABC. Combined Optimal",
        "symbols": MAJORS_ONLY,
        "use_partial_tp1": False,
        "be_trigger_r": 1.2,
        "rr_ratio": 2.2,
        "bos_only": True,
        "max_adx": 42.0,
        "require_rejection": False,
    },
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


def precompute_signals(data_store: dict, all_timestamps: list, test_start: pd.Timestamp):
    raw_candidates = {}
    for candle_open_ts in all_timestamps:
        if candle_open_ts < test_start:
            continue
        sim_now = candle_open_ts + pd.Timedelta(minutes=15)
        sim_now_np = np.datetime64(sim_now.tz_convert(None))
        candle_open_np = np.datetime64(candle_open_ts.tz_convert(None))
        last_closed_1h_ts = sim_now.floor("1h") - pd.Timedelta(hours=1)

        step_signals = []
        for sym, sym_data in data_store.items():
            if last_closed_1h_ts not in sym_data["active_1h"]:
                continue

            end15 = np.searchsorted(sym_data["ts_15m"], candle_open_np, side="right")
            end1 = np.searchsorted(sym_data["ts_1h"], sim_now_np, side="left")
            end4 = np.searchsorted(sym_data["ts_4h"], sim_now_np, side="left")
            if min(end15, end1, end4) < MIN_CANDLES:
                continue

            d15 = validate_data(sym_data["15m"].iloc[max(0, end15 - 220):end15], 15)
            d1 = validate_data(sym_data["1h"].iloc[max(0, end1 - 220):end1], 60)
            d4 = validate_data(sym_data["4h"].iloc[max(0, end4 - 220):end4], 240)

            d15 = drop_unclosed(d15, 15, sim_now)
            d1 = drop_unclosed(d1, 60, sim_now)
            d4 = drop_unclosed(d4, 240, sim_now)
            if min(len(d15), len(d1), len(d4)) < MIN_CANDLES:
                continue

            d15["atr"] = calculate_atr(d15)
            atr = float(d15["atr"].iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                continue

            adx_s, pdi_s, mdi_s = calculate_directional_adx(d1)
            adx_1h, plus_di, minus_di = float(adx_s.iloc[-1]), float(pdi_s.iloc[-1]), float(mdi_s.iloc[-1])
            if adx_1h < 22.0:
                continue

            struct = detect_market_structure(d1)
            if struct["type"] == "RANGE":
                continue

            ema20_s = d4["close"].ewm(span=20, adjust=False).mean()
            ema50_s = d4["close"].ewm(span=50, adjust=False).mean()
            ema20, ema20_prev = float(ema20_s.iloc[-1]), float(ema20_s.iloc[-4])
            ema50 = float(ema50_s.iloc[-1])
            d4_close = float(d4["close"].iloc[-1])
            d4_bull = d4_close > ema20 > ema50 and ema20 > ema20_prev
            d4_bear = d4_close < ema20 < ema50 and ema20 < ema20_prev

            side = struct["direction"]
            if side == "LONG" and not (d4_bull and plus_di > minus_di):
                continue
            if side == "SHORT" and not (d4_bear and minus_di > plus_di):
                continue

            fvgs = detect_unmitigated_fvgs(d15, side)
            leg_start_ts = d1["timestamp"].iloc[struct["leg_start_idx"]]
            leg_end_ts = d1["timestamp"].iloc[struct["break_idx"]] + pd.Timedelta(minutes=60)
            linked = [f for f in fvgs if leg_start_ts <= f["mid_ts"] <= leg_end_ts]
            if not linked:
                continue

            price = float(d15["close"].iloc[-1])
            op = float(d15["open"].iloc[-1])
            low = float(d15["low"].iloc[-1])
            high = float(d15["high"].iloc[-1])

            for fvg in reversed(linked):
                if side == "LONG":
                    tapped = low <= fvg["top"] and price >= fvg["bottom"]
                    rejection = price > op and price > fvg["mid"]
                    stop = fvg["bottom"] - STOP_BUFFER_ATR * atr
                    risk = price - stop
                else:
                    tapped = high >= fvg["bottom"] and price <= fvg["top"]
                    rejection = price < op and price < fvg["mid"]
                    stop = fvg["top"] + STOP_BUFFER_ATR * atr
                    risk = stop - price

                if not tapped or risk > MAX_STOP_ATR * atr:
                    continue
                if risk < MIN_STOP_ATR * atr:
                    risk = MIN_STOP_ATR * atr
                    stop = price - risk if side == "LONG" else price + risk

                step_signals.append({
                    "symbol": sym,
                    "side": side,
                    "structure_type": struct["type"],
                    "adx_1h": adx_1h,
                    "rejection": rejection,
                    "entry_price": price,
                    "stop_loss": stop,
                    "risk_dist": risk,
                    "zone_id": fvg["zone_id"],
                    "candle_time": d15["timestamp"].iloc[-1].isoformat(),
                })
                break

        raw_candidates[candle_open_ts] = step_signals
    return raw_candidates


def simulate_experiment(exp: dict, data_store: dict, all_timestamps: list, raw_candidates: dict) -> dict:
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    open_positions = {}
    closed_trades = []
    guard = SignalGuard(cooldown_minutes=COOLDOWN_MINUTES, path=None)
    allowed_syms = set(exp["symbols"])

    for candle_open_ts in all_timestamps:
        candle_open_np = np.datetime64(candle_open_ts.tz_convert(None))

        for sym in list(open_positions.keys()):
            sym_data = data_store[sym]
            idx15 = np.searchsorted(sym_data["ts_15m"], candle_open_np)
            if idx15 >= len(sym_data["15m"]) or sym_data["ts_15m"][idx15] != candle_open_np:
                continue

            bar = sym_data["15m"].iloc[idx15]
            high, low = float(bar["high"]), float(bar["low"])
            pos = open_positions[sym]
            side, entry, sl, tp1, tp2, qty = (
                pos["side"], pos["entry"], pos["stop_loss"], pos["tp1"], pos["take_profit"], pos["quantity"]
            )

            if side == "LONG":
                if low <= sl:
                    exit_p = sl * (1 - SLIPPAGE_RATE)
                    net = (exit_p - entry) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]
                    continue

                if exp["use_partial_tp1"] and not pos["is_tp1_hit"] and high >= tp1:
                    c_qty = qty / 2.0
                    exit_p = tp1 * (1 - SLIPPAGE_RATE)
                    net = (exit_p - entry) * c_qty - (exit_p * c_qty * FEE_RATE)
                    balance += net
                    pos["quantity"] = c_qty
                    pos["stop_loss"] = entry * (1 + (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_tp1_hit"] = True
                    closed_trades.append({"side": side, "pnl": net})
                    qty = c_qty
                elif not exp["use_partial_tp1"] and exp["be_trigger_r"] and not pos["is_be"]:
                    if high >= entry + pos["risk_dist"] * exp["be_trigger_r"]:
                        pos["stop_loss"] = entry * (1 + (FEE_RATE + SLIPPAGE_RATE) * 2)
                        pos["is_be"] = True

                if high >= tp2:
                    exit_p = tp2 * (1 - SLIPPAGE_RATE)
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

                if exp["use_partial_tp1"] and not pos["is_tp1_hit"] and low <= tp1:
                    c_qty = qty / 2.0
                    exit_p = tp1 * (1 + SLIPPAGE_RATE)
                    net = (entry - exit_p) * c_qty - (exit_p * c_qty * FEE_RATE)
                    balance += net
                    pos["quantity"] = c_qty
                    pos["stop_loss"] = entry * (1 - (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_tp1_hit"] = True
                    closed_trades.append({"side": side, "pnl": net})
                    qty = c_qty
                elif not exp["use_partial_tp1"] and exp["be_trigger_r"] and not pos["is_be"]:
                    if low <= entry - pos["risk_dist"] * exp["be_trigger_r"]:
                        pos["stop_loss"] = entry * (1 - (FEE_RATE + SLIPPAGE_RATE) * 2)
                        pos["is_be"] = True

                if low <= tp2:
                    exit_p = tp2 * (1 + SLIPPAGE_RATE)
                    net = (entry - exit_p) * qty - (exit_p * qty * FEE_RATE)
                    balance += net
                    closed_trades.append({"side": side, "pnl": net})
                    del open_positions[sym]

        if balance > peak_balance:
            peak_balance = balance
        dd = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        for cand in raw_candidates.get(candle_open_ts, []):
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            sym = cand["symbol"]
            if sym not in allowed_syms or sym in open_positions:
                continue
            if exp["bos_only"] and "BOS" not in cand["structure_type"]:
                continue
            if cand["adx_1h"] > exp["max_adx"]:
                continue
            if exp["require_rejection"] and not cand["rejection"]:
                continue

            sig_stub = {"signal": cand["side"], "side": cand["side"], "symbol": sym,
                        "candle_time": cand["candle_time"], "zone_id": cand["zone_id"]}
            if not guard.allow(sig_stub):
                continue

            side = cand["side"]
            if sum(1 for p in open_positions.values() if p["side"] == side) >= MAX_SAME_SIDE_POSITIONS:
                continue

            raw_entry = cand["entry_price"]
            entry = raw_entry * (1 + SLIPPAGE_RATE) if side == "LONG" else raw_entry * (1 - SLIPPAGE_RATE)
            stop = cand["stop_loss"]
            min_sl = entry * MIN_SL_PCT
            price_risk = max(abs(entry - stop), min_sl)
            stop = entry - price_risk if side == "LONG" else entry + price_risk

            cost = entry * ROUND_TRIP_COST_PCT
            reward = exp["rr_ratio"] * (price_risk + cost) + cost
            target = entry + reward if side == "LONG" else entry - reward

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
                "side": side, "entry": entry, "quantity": qty, "stop_loss": stop,
                "take_profit": target, "tp1": entry + (target - entry) * 0.5,
                "risk_dist": price_risk, "margin": margin, "is_tp1_hit": False, "is_be": False,
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
        "name": exp["name"],
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

    logging.info("Fetching 90-day (+warmup) historical data once for A/B experiments...")
    data_store = {}
    for sym in ALL_SYMBOLS:
        logging.info("Fetching %s...", sym)
        df15 = fetch_historical_ohlcv(sym, "15m", since_ms)
        df1 = fetch_historical_ohlcv(sym, "1h", since_ms)
        df4 = fetch_historical_ohlcv(sym, "4h", since_ms)
        if min(len(df15), len(df1), len(df4)) > MIN_CANDLES:
            d1_clean = validate_data(df1, 60)
            adx_s, _, _ = calculate_directional_adx(d1_clean)
            active_1h = set(d1_clean.loc[adx_s >= 22.0, "timestamp"])
            data_store[sym] = {
                "15m": df15, "1h": df1, "4h": df4, "active_1h": active_1h,
                "ts_15m": df15["timestamp"].values, "ts_1h": df1["timestamp"].values, "ts_4h": df4["timestamp"].values,
            }

    all_timestamps = sorted(set(
        ts for sym_data in data_store.values()
        for ts in sym_data["15m"]["timestamp"]
        if ts >= test_start
    ))

    logging.info("Precomputing candidate setups across %d steps...", len(all_timestamps))
    raw_candidates = precompute_signals(data_store, all_timestamps, test_start)

    print("\n" + "=" * 105)
    print(f"{'EXPERIMENT':<36} | {'FINAL $':>8} | {'NET %':>8} | {'MAX DD':>7} | {'PF':>5} | {'TRADES':>6} | {'WIN %':>6} | {'LONG $':>7} | {'SHORT $':>7}")
    print("=" * 105)
    for exp in EXPERIMENTS:
        res = simulate_experiment(exp, data_store, all_timestamps, raw_candidates)
        print(
            f"{res['name']:<36} | "
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
