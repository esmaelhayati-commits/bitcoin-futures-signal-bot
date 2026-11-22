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
LEVERAGE = 25
RISK_PER_TRADE = 0.025          # 2.5% risk per trade
BREAKEVEN_TRIGGER_R = 1.2
MAX_OPEN_POSITIONS = 3
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


def run_v5_backtest():
    now_utc = datetime.now(timezone.utc)
    warmup_start = now_utc - timedelta(days=DAYS_BACK + 35)
    test_start = pd.Timestamp(now_utc - timedelta(days=DAYS_BACK))
    since_ms = int(warmup_start.timestamp() * 1000)

    logging.info("Fetching 90-day (+warmup) historical data for Strategy v5.0...")
    data_store = {}
    for sym in SYMBOLS:
        logging.info("Fetching %s...", sym)
        df1 = fetch_historical_ohlcv(sym, "1h", since_ms)
        df4 = fetch_historical_ohlcv(sym, "4h", since_ms)
        if min(len(df1), len(df4)) > MIN_CANDLES:
            d1 = validate_data(df1, 60)
            d4 = validate_data(df4, 240)

            d4["ema20"] = d4["close"].ewm(span=20, adjust=False).mean()
            d4["ema50"] = d4["close"].ewm(span=50, adjust=False).mean()
            d4["bull_4h"] = (d4["close"] > d4["ema20"]) & (d4["ema20"] > d4["ema50"]) & (d4["ema20"] > d4["ema20"].shift(2))
            d4["bear_4h"] = (d4["close"] < d4["ema20"]) & (d4["ema20"] < d4["ema50"])
            d4["close_time"] = d4["timestamp"] + pd.Timedelta(minutes=240)

            d1["ema20"] = d1["close"].ewm(span=20, adjust=False).mean()
            d1["ema50"] = d1["close"].ewm(span=50, adjust=False).mean()
            d1["atr"] = calculate_atr(d1)
            adx1, pdi1, mdi1 = calculate_directional_adx(d1)
            d1["adx"] = adx1
            d1["pdi"] = pdi1
            d1["mdi"] = mdi1
            d1["close_time"] = d1["timestamp"] + pd.Timedelta(minutes=60)

            data_store[sym] = {
                "1h": d1, "4h": d4,
                "ts_1h": d1["timestamp"].values,
                "ct_4h": d4["close_time"].values,
            }

    all_hours = sorted(set(
        ts for d in data_store.values()
        for ts in d["1h"]["timestamp"]
        if ts >= test_start
    ))

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    open_positions = {}
    closed_trades = []
    guard = SignalGuard(cooldown_minutes=COOLDOWN_MINUTES, path=None)

    for hour_open_ts in all_hours:
        sim_now = hour_open_ts + pd.Timedelta(minutes=60)
        sim_now_np = np.datetime64(sim_now.tz_convert(None))
        hour_open_np = np.datetime64(hour_open_ts.tz_convert(None))

        # 1. Manage Open Positions on each 1H bar
        for sym in list(open_positions.keys()):
            d = data_store[sym]
            idx1 = np.searchsorted(d["ts_1h"], hour_open_np)
            if idx1 >= len(d["1h"]) or d["ts_1h"][idx1] != hour_open_np:
                continue

            bar = d["1h"].iloc[idx1]
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

                if not pos["is_be"] and high >= entry + pos["risk_dist"] * BREAKEVEN_TRIGGER_R:
                    pos["stop_loss"] = entry * (1 + (FEE_RATE + SLIPPAGE_RATE) * 2.5)
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

                if not pos["is_be"] and low <= entry - pos["risk_dist"] * BREAKEVEN_TRIGGER_R:
                    pos["stop_loss"] = entry * (1 - (FEE_RATE + SLIPPAGE_RATE) * 2.5)
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

        # 2. Scan for New 1H EMA20 Pullback Signals
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        for sym, d in data_store.items():
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            if sym in open_positions:
                continue

            idx1 = np.searchsorted(d["ts_1h"], hour_open_np)
            idx4 = np.searchsorted(d["ct_4h"], sim_now_np, side="right") - 1
            if idx1 < MIN_CANDLES or idx1 >= len(d["1h"]) or d["ts_1h"][idx1] != hour_open_np or idx4 < 60:
                continue

            b1 = d["1h"].iloc[idx1]
            b4 = d["4h"].iloc[idx4]
            atr1 = float(b1["atr"])
            if not np.isfinite(atr1) or atr1 <= 0 or float(b1["adx"]) < 25.0:
                continue

            entry_raw = float(b1["close"])
            signal_side = None
            stop_loss = 0.0

            if bool(b4["bull_4h"]) and float(b1["ema20"]) > float(b1["ema50"]) and float(b1["pdi"]) > float(b1["mdi"]):
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

            if not signal_side:
                continue
            if sum(1 for p in open_positions.values() if p["side"] == signal_side) >= MAX_SAME_SIDE_POSITIONS:
                continue

            sig_stub = {
                "signal": signal_side, "side": signal_side, "symbol": sym,
                "candle_time": hour_open_ts.isoformat(), "zone_id": f"{signal_side}:1H_EMA20:{hour_open_ts.isoformat()}"
            }
            if not guard.allow(sig_stub):
                continue

            entry = entry_raw * (1 + SLIPPAGE_RATE) if signal_side == "LONG" else entry_raw * (1 - SLIPPAGE_RATE)
            min_sl = entry * MIN_SL_PCT
            price_risk = max(abs(entry - stop_loss), min_sl)
            stop_loss = entry - price_risk if signal_side == "LONG" else entry + price_risk

            cost = entry * ROUND_TRIP_COST_PCT
            reward = 2.0 * (price_risk + cost) + cost
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

    print("\n" + "=" * 55)
    print(f"📊 90-DAY BACKTEST REPORT (Strategy v5.0 | 25x | 2.5% Risk)")
    print("=" * 55)
    print(f"Initial Balance  : ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance    : ${balance:.2f}")
    print(f"Net Profit/Loss  : ${balance - INITIAL_BALANCE:+.2f} ({((balance - INITIAL_BALANCE)/INITIAL_BALANCE)*100:+.2f}%)")
    print(f"Max Drawdown     : {max_dd:.2f}%")
    print(f"Profit Factor    : {pf:.2f}")
    print("-" * 55)
    print(f"Total Trades     : {len(closed_trades)}")
    print(f"Win Rate         : {wr:.1f}% (Wins: {len(wins)} | Losses: {len(losses)})")
    print(f"LONG Trades      : {len(longs)} | Net PnL: ${sum(t['pnl'] for t in longs):+.2f}")
    print(f"SHORT Trades     : {len(shorts)} | Net PnL: ${sum(t['pnl'] for t in shorts):+.2f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_v5_backtest()
