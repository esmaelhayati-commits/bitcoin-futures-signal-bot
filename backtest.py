import time
import logging
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import numpy as np

from strategy import (
    analyze_market,
    SignalGuard,
    MIN_CANDLES,
    calculate_directional_adx,
    validate_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ==================== BACKTEST CONFIGURATION ====================
DAYS_BACK = 90                  # 3 months historical data
INITIAL_BALANCE = 100.0
LEVERAGE = 20
RISK_PER_TRADE = 0.015          # 1.5% max risk per trade
MAX_OPEN_POSITIONS = 2
MAX_SAME_SIDE_POSITIONS = 2
MIN_SL_PCT = 0.004              # 0.4% minimum SL distance
COOLDOWN_MINUTES = 60
FEE_RATE = 0.0005               # 0.05% taker fee per side
SLIPPAGE_RATE = 0.0002          # 0.02% slippage per side

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "LTC/USDT:USDT"
]

exchange = ccxt.xt({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})


def fetch_historical_ohlcv(symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Fetches paginated historical candles from XT exchange."""
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
        except Exception as e:
            logging.warning("Fetch retry for %s (%s): %s", symbol, timeframe, e)
            time.sleep(1.5)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def run_backtest():
    now_utc = datetime.now(timezone.utc)
    warmup_start = now_utc - timedelta(days=DAYS_BACK + 35)
    test_start = pd.Timestamp(now_utc - timedelta(days=DAYS_BACK))
    since_ms = int(warmup_start.timestamp() * 1000)

    logging.info("Downloading 3-month (+warmup) historical data for %d symbols from XT...", len(SYMBOLS))
    data_store = {}
    for sym in SYMBOLS:
        logging.info("Fetching %s...", sym)
        df15 = fetch_historical_ohlcv(sym, "15m", since_ms)
        df1 = fetch_historical_ohlcv(sym, "1h", since_ms)
        df4 = fetch_historical_ohlcv(sym, "4h", since_ms)

        if min(len(df15), len(df1), len(df4)) > MIN_CANDLES:
            d1_clean = validate_data(df1, 60)
            adx_s, _, _ = calculate_directional_adx(d1_clean)
            d1_clean["fast_adx"] = adx_s
            active_1h_hours = set(d1_clean.loc[d1_clean["fast_adx"] >= 20.0, "timestamp"])

            data_store[sym] = {
                "15m": df15,
                "1h": df1,
                "4h": df4,
                "active_1h": active_1h_hours,
                "ts_15m": df15["timestamp"].values,
                "ts_1h": df1["timestamp"].values,
                "ts_4h": df4["timestamp"].values,
            }
        else:
            logging.warning("Insufficient historical data for %s, skipping.", sym)

    if not data_store:
        logging.error("No data fetched. Aborting backtest.")
        return

    all_timestamps = sorted(set(
        ts for sym_data in data_store.values()
        for ts in sym_data["15m"]["timestamp"]
        if ts >= test_start
    ))

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown_pct = 0.0
    open_positions = {}
    closed_trades = []
    guard = SignalGuard(cooldown_minutes=COOLDOWN_MINUTES, path=None)

    total_steps = len(all_timestamps)
    logging.info("Simulating %d 15m steps across %d symbols...", total_steps, len(data_store))

    for step_idx, candle_open_ts in enumerate(all_timestamps, 1):
        if step_idx % 2000 == 0 or step_idx == total_steps:
            logging.info("Progress: %d / %d steps (%.1f%%) | Balance: $%.2f",
                         step_idx, total_steps, (step_idx / total_steps) * 100, balance)

        sim_now = candle_open_ts + pd.Timedelta(minutes=15)
        sim_now_np = np.datetime64(sim_now.tz_convert(None))
        candle_open_np = np.datetime64(candle_open_ts.tz_convert(None))

        # 1. Manage existing open positions
        for sym in list(open_positions.keys()):
            sym_data = data_store[sym]
            idx15 = np.searchsorted(sym_data["ts_15m"], candle_open_np)
            if idx15 >= len(sym_data["15m"]) or sym_data["ts_15m"][idx15] != candle_open_np:
                continue

            bar = sym_data["15m"].iloc[idx15]
            high = float(bar["high"])
            low = float(bar["low"])
            pos = open_positions[sym]
            side = pos["side"]
            entry = pos["entry"]
            sl = pos["stop_loss"]
            tp1 = pos["tp1"]
            tp2 = pos["take_profit"]
            qty = pos["quantity"]

            if side == "LONG":
                if low <= sl:
                    exit_price = sl * (1 - SLIPPAGE_RATE)
                    gross = (exit_price - entry) * qty
                    fee = (exit_price * qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    reason = "BREAKEVEN" if pos["is_breakeven"] and net >= 0 else "STOP_LOSS"
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": reason, "time": str(sim_now)})
                    del open_positions[sym]
                    continue

                if not pos["is_tp1_hit"] and high >= tp1:
                    close_qty = qty / 2.0
                    exit_price = tp1 * (1 - SLIPPAGE_RATE)
                    gross = (exit_price - entry) * close_qty
                    fee = (exit_price * close_qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    pos["quantity"] = close_qty
                    pos["stop_loss"] = entry * (1 + (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_tp1_hit"] = True
                    pos["is_breakeven"] = True
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": "PARTIAL_TP1", "time": str(sim_now)})
                    qty = close_qty

                if high >= tp2:
                    exit_price = tp2 * (1 - SLIPPAGE_RATE)
                    gross = (exit_price - entry) * qty
                    fee = (exit_price * qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": "TAKE_PROFIT_2", "time": str(sim_now)})
                    del open_positions[sym]

            else:  # SHORT
                if high >= sl:
                    exit_price = sl * (1 + SLIPPAGE_RATE)
                    gross = (entry - exit_price) * qty
                    fee = (exit_price * qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    reason = "BREAKEVEN" if pos["is_breakeven"] and net >= 0 else "STOP_LOSS"
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": reason, "time": str(sim_now)})
                    del open_positions[sym]
                    continue

                if not pos["is_tp1_hit"] and low <= tp1:
                    close_qty = qty / 2.0
                    exit_price = tp1 * (1 + SLIPPAGE_RATE)
                    gross = (entry - exit_price) * close_qty
                    fee = (exit_price * close_qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    pos["quantity"] = close_qty
                    pos["stop_loss"] = entry * (1 - (FEE_RATE + SLIPPAGE_RATE) * 2)
                    pos["is_tp1_hit"] = True
                    pos["is_breakeven"] = True
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": "PARTIAL_TP1", "time": str(sim_now)})
                    qty = close_qty

                if low <= tp2:
                    exit_price = tp2 * (1 + SLIPPAGE_RATE)
                    gross = (entry - exit_price) * qty
                    fee = (exit_price * qty) * FEE_RATE
                    net = gross - fee
                    balance += net
                    closed_trades.append({"symbol": sym, "side": side, "pnl": net, "reason": "TAKE_PROFIT_2", "time": str(sim_now)})
                    del open_positions[sym]

        if balance > peak_balance:
            peak_balance = balance
        dd = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > 0 else 0
        max_drawdown_pct = max(max_drawdown_pct, dd)

        # 2. Scan for new entries
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        last_closed_1h_ts = sim_now.floor("1h") - pd.Timedelta(hours=1)

        for sym, sym_data in data_store.items():
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            if sym in open_positions:
                continue

            if last_closed_1h_ts not in sym_data["active_1h"]:
                continue

            end15 = np.searchsorted(sym_data["ts_15m"], candle_open_np, side="right")
            end1 = np.searchsorted(sym_data["ts_1h"], sim_now_np, side="left")
            end4 = np.searchsorted(sym_data["ts_4h"], sim_now_np, side="left")

            if min(end15, end1, end4) < MIN_CANDLES:
                continue

            d15_slice = sym_data["15m"].iloc[max(0, end15 - 220):end15]
            d1_slice = sym_data["1h"].iloc[max(0, end1 - 220):end1]
            d4_slice = sym_data["4h"].iloc[max(0, end4 - 220):end4]

            sig = analyze_market(d15_slice, d1_slice, d4_slice, symbol=sym, now=sim_now, only_closed=True)
            if sig.get("signal") in ("LONG", "SHORT") and guard.allow(sig):
                side = sig["signal"]
                same_side = sum(1 for p in open_positions.values() if p["side"] == side)
                if same_side >= MAX_SAME_SIDE_POSITIONS:
                    continue

                raw_entry = float(sig["entry_price"])
                entry = raw_entry * (1 + SLIPPAGE_RATE) if side == "LONG" else raw_entry * (1 - SLIPPAGE_RATE)
                stop = float(sig["stop_loss"])
                target = float(sig["take_profit"])

                min_sl_dist = entry * MIN_SL_PCT
                raw_risk = abs(entry - stop)
                if raw_risk == 0:
                    continue
                if raw_risk < min_sl_dist:
                    stop = entry - min_sl_dist if side == "LONG" else entry + min_sl_dist
                    price_risk = min_sl_dist
                else:
                    price_risk = raw_risk

                risk_amount = balance * RISK_PER_TRADE
                eff_risk = price_risk + (entry * (FEE_RATE + SLIPPAGE_RATE) * 2)
                qty = risk_amount / eff_risk

                notional = qty * entry
                margin = notional / LEVERAGE
                entry_fee = notional * FEE_RATE
                used_margin = sum(p["margin"] for p in open_positions.values())

                if margin + entry_fee > (balance - used_margin) * 0.9:
                    continue

                balance -= entry_fee
                tp1 = entry + (target - entry) * 0.5

                open_positions[sym] = {
                    "side": side,
                    "entry": entry,
                    "quantity": qty,
                    "stop_loss": stop,
                    "take_profit": target,
                    "tp1": tp1,
                    "margin": margin,
                    "is_tp1_hit": False,
                    "is_breakeven": False
                }

    # ==================== PRINT SUMMARY REPORT ====================
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] < 0]
    long_trades = [t for t in closed_trades if t["side"] == "LONG"]
    short_trades = [t for t in closed_trades if t["side"] == "SHORT"]

    net_profit = balance - INITIAL_BALANCE
    net_pct = (net_profit / INITIAL_BALANCE) * 100
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    print("\n" + "=" * 55)
    print(f"📊 90-DAY BACKTEST REPORT (Strategy v4.2)")
    print("=" * 55)
    print(f"Initial Balance  : ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance    : ${balance:.2f}")
    print(f"Net Profit/Loss  : ${net_profit:+.2f} ({net_pct:+.2f}%)")
    print(f"Max Drawdown     : {max_drawdown_pct:.2f}%")
    print(f"Profit Factor    : {profit_factor:.2f}")
    print("-" * 55)
    print(f"Total Executions : {len(closed_trades)} (Includes TP1 partials)")
    print(f"Win Rate         : {win_rate:.1f}% (Wins: {len(wins)} | Losses: {len(losses)})")
    print(f"LONG Executions  : {len(long_trades)} | Net PnL: ${sum(t['pnl'] for t in long_trades):+.2f}")
    print(f"SHORT Executions : {len(short_trades)} | Net PnL: ${sum(t['pnl'] for t in short_trades):+.2f}")
    print(f"Open at End      : {len(open_positions)}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_backtest()
