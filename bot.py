import os
import time
import json
import logging
from datetime import datetime, timezone
import requests
import ccxt
import pandas as pd
from dotenv import load_dotenv
from strategy import analyze_market

load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("XT_API_KEY")
API_SECRET = os.getenv("XT_API_SECRET")

LIVE_TRADING = False
LEVERAGE = 10
RISK_PER_TRADE = 0.015
MAX_OPEN_POSITIONS = 3
INITIAL_BALANCE = 100.0
STATE_FILE = "live_trader_state.json"

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

exchange_params = {
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
}
if LIVE_TRADING and API_KEY and API_SECRET:
    exchange_params.update({"apiKey": API_KEY, "secret": API_SECRET})

exchange = ccxt.xt(exchange_params)

# ==================== STATE MANAGEMENT ====================
def load_state():
    now_utc = datetime.now(timezone.utc)
    default_state = {
        "balance": INITIAL_BALANCE,
        "positions": {},
        "trade_history": [],
        "last_daily_report": str(now_utc.date()),
        "last_weekly_report": f"{now_utc.year}-W{now_utc.isocalendar()[1]}",
        "last_monthly_report": f"{now_utc.year}-{now_utc.month:02d}"
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_state.update(saved)
                return default_state
        except Exception as e:
            logging.error("Failed to load state: %s", e)
    return default_state

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error("Failed to save state: %s", e)

# ==================== TELEGRAM NOTIFIER ====================
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        logging.error("Telegram send error: %s", e)

# ==================== MARKET DATA ====================
def fetch_candles(symbol: str, timeframe: str):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.iloc[:-1].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)

# ==================== POSITION & EXIT MANAGEMENT ====================
def manage_positions(state):
    positions = state.get("positions", {})
    remaining_positions = {}
    now_str = str(datetime.now(timezone.utc))

    for symbol, pos in positions.items():
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker["last"])
            side = pos["side"]
            entry = pos["entry"]
            stop_loss = pos["stop_loss"]
            take_profit = pos["take_profit"]
            tp1 = pos["tp1"]
            is_tp1_hit = pos.get("is_tp1_hit", False)
            qty = pos["quantity"]

            if side == "LONG":
                pnl_pct = ((current_price - entry) / entry) * 100
                hit_sl = current_price <= stop_loss
                hit_tp2 = current_price >= take_profit
                hit_tp1 = not is_tp1_hit and current_price >= tp1
            else:
                pnl_pct = ((entry - current_price) / entry) * 100
                hit_sl = current_price >= stop_loss
                hit_tp2 = current_price <= take_profit
                hit_tp1 = not is_tp1_hit and current_price <= tp1

            # 1. Stop Loss exit
            if hit_sl:
                net_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                state["balance"] += net_pnl
                reason = "BREAKEVEN" if pos.get("is_breakeven") and net_pnl >= 0 else "STOP_LOSS"

                lines = [
                    f"🛑 <b>Position Closed: {reason}</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Entry: <code>{entry:.4f}</code> | Exit: <code>{current_price:.4f}</code>",
                    f"PnL: <b>{net_pnl:+.4f} USDT</b> ({pnl_pct:+.2f}%)",
                    f"Balance: <code>{state['balance']:.2f} USDT</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": net_pnl, "pct": pnl_pct,
                    "reason": reason, "time": now_str
                })
                continue

            # 2. Final TP2 hit
            if hit_tp2:
                net_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                state["balance"] += net_pnl

                lines = [
                    "🎉 <b>Target 2 Hit (Full Close)</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Entry: <code>{entry:.4f}</code> | Exit: <code>{current_price:.4f}</code>",
                    f"Net PnL: <b>{net_pnl:+.4f} USDT</b> ({pnl_pct:+.2f}%)",
                    f"Balance: <code>{state['balance']:.2f} USDT</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": net_pnl, "pct": pnl_pct,
                    "reason": "TAKE_PROFIT_2", "time": now_str
                })
                continue

            # 3. Partial TP1 hit
            if hit_tp1:
                close_qty = qty / 2
                realized_pnl = (tp1 - entry) * close_qty if side == "LONG" else (entry - tp1) * close_qty
                state["balance"] += realized_pnl

                pos["quantity"] = close_qty
                pos["stop_loss"] = entry
                pos["is_breakeven"] = True
                pos["is_tp1_hit"] = True

                lines = [
                    "💰 <b>Partial TP1 Hit (50% Closed)</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Realized PnL: <b>{realized_pnl:+.4f} USDT</b>",
                    f"Stop Loss moved to Entry: <code>{entry:.4f}</code> (Risk-Free)",
                    f"Target 2: <code>{take_profit:.4f}</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": realized_pnl, "pct": pnl_pct,
                    "reason": "PARTIAL_TP1", "time": now_str
                })

            remaining_positions[symbol] = pos

        except Exception as e:
            logging.error("Error managing %s: %s", symbol, e)
            remaining_positions[symbol] = pos

    state["positions"] = remaining_positions

# ==================== OPENING POSITIONS ====================
def open_position(symbol, signal_data, state):
    side = signal_data["signal"]
    entry = signal_data["entry_price"]
    stop = signal_data["stop_loss"]
    target = signal_data["take_profit"]

    tp1 = entry + (target - entry) * 0.5
    price_risk = abs(entry - stop)
    if price_risk == 0:
        return

    balance = state.get("balance", INITIAL_BALANCE)
    if LIVE_TRADING:
        try:
            bal_data = exchange.fetch_balance()
            balance = float(bal_data["USDT"]["free"])
            state["balance"] = balance
        except Exception:
            pass

    risk_amount = balance * RISK_PER_TRADE
    quantity = risk_amount / price_risk

    state["positions"][symbol] = {
        "side": side,
        "entry": entry,
        "quantity": quantity,
        "stop_loss": stop,
        "take_profit": target,
        "tp1": tp1,
        "is_tp1_hit": False,
        "is_breakeven": False,
        "opened_at": str(datetime.now(timezone.utc))
    }

    lines = [
        f"⚡ <b>New Position Opened ({side})</b>",
        "------------------------------------",
        f"Symbol: <code>{symbol}</code>",
        f"Entry: <code>{entry:.4f}</code>",
        f"Stop Loss: <code>{stop:.4f}</code>",
        f"TP1 (50% Close): <code>{tp1:.4f}</code>",
        f"TP2 (Final): <code>{target:.4f}</code>",
        f"Quantity: <code>{quantity:.4f}</code>"
    ]
    send_telegram("\n".join(lines))

# ==================== REPORTING (DAILY / WEEKLY / MONTHLY) ====================
def generate_report_text(title: str, timeframe_label: str, trades: list, current_open_count: int, balance: float) -> str:
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    breakevens = [t for t in trades if t["pnl"] == 0]

    sum_profit = sum(t["pnl"] for t in wins)
    sum_loss = abs(sum(t["pnl"] for t in losses))
    net_pnl = sum_profit - sum_loss
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    lines = [
        f"📊 <b>{title}</b>",
        "------------------------------------",
        f"Period: <b>{timeframe_label}</b>",
        f"Current Balance: <b>{balance:.2f} USDT</b>",
        f"Net PnL: <b>{net_pnl:+.4f} USDT</b>",
        f"Total Profit: <code>+{sum_profit:.4f} USDT</code>",
        f"Total Loss: <code>-{sum_loss:.4f} USDT</code>",
        "",
        f"Total Closed Trades: <b>{total_trades}</b>",
        f"Wins: <code>{len(wins)}</code> | Losses: <code>{len(losses)}</code> | Breakeven: <code>{len(breakevens)}</code>",
        f"Win Rate: <b>{win_rate:.1f}%</b>",
        "",
        f"Active Open Positions: <b>{current_open_count}</b>",
        "------------------------------------"
    ]
    return "\n".join(lines)

def check_and_send_reports(state):
    now_utc = datetime.now(timezone.utc)
    today_str = str(now_utc.date())
    current_week = f"{now_utc.year}-W{now_utc.isocalendar()[1]}"
    current_month = f"{now_utc.year}-{now_utc.month:02d}"

    trades = state.get("trade_history", [])
    open_count = len(state.get("positions", {}))
    balance = state.get("balance", INITIAL_BALANCE)

    # 1. Daily Report
    if state.get("last_daily_report") != today_str:
        daily_trades = [t for t in trades if t.get("time", "").startswith(state.get("last_daily_report", ""))]
        rep = generate_report_text("Daily Performance Report (24H)", state.get("last_daily_report"), daily_trades, open_count, balance)
        send_telegram(rep)
        state["last_daily_report"] = today_str

    # 2. Weekly Report
    if state.get("last_weekly_report") != current_week:
        rep = generate_report_text("Weekly Performance Report", state.get("last_weekly_report"), trades, open_count, balance)
        send_telegram(rep)
        state["last_weekly_report"] = current_week

    # 3. Monthly Report
    if state.get("last_monthly_report") != current_month:
        rep = generate_report_text("Monthly Performance Report", state.get("last_monthly_report"), trades, open_count, balance)
        send_telegram(rep)
        state["last_monthly_report"] = current_month

# ==================== MAIN SCANNER ====================
def main():
    state = load_state()
    exchange.load_markets()

    manage_positions(state)

    current_open_count = len(state.get("positions", {}))

    for symbol in SYMBOLS:
        if current_open_count >= MAX_OPEN_POSITIONS:
            break
        if symbol in state.get("positions", {}):
            continue

        try:
            df15 = fetch_candles(symbol, "15m")
            df1 = fetch_candles(symbol, "1h")
            df4 = fetch_candles(symbol, "4h")

            sig = analyze_market(df15, df1, df4, symbol=symbol)
            if sig.get("signal") in ["LONG", "SHORT"]:
                open_position(symbol, sig, state)
                current_open_count += 1

            time.sleep(0.3)
        except Exception as e:
            logging.error("Scan error on %s: %s", symbol, e)

    check_and_send_reports(state)
    save_state(state)

if __name__ == "__main__":
    main()
