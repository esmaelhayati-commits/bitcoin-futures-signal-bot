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

LIVE_TRADING = False            # Switch to True only after XT API setup and backtest verification
LEVERAGE = 20
RISK_PER_TRADE = 0.015          # 1.5% max risk per trade
MAX_OPEN_POSITIONS = 2
MAX_SAME_SIDE_POSITIONS = 2
MIN_SL_PCT = 0.004              # Minimum 0.4% SL distance
COOLDOWN_MINUTES = 60           # Prevent re-entering the same symbol within 60 minutes
FEE_RATE = 0.0005               # 0.05% taker fee per side
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
        "symbol_cooldowns": {},
        "last_signal_candles": {},
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


# ==================== MARKET DATA (CLOSED CANDLES ONLY) ====================
def fetch_candles(symbol: str, timeframe: str, retries: int = 3):
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # Strictly drop the unclosed forming candle (iloc[-1]) to prevent repainting
            df = df.iloc[:-1].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df.reset_index(drop=True)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1.5)


# ==================== LIVE EXCHANGE EXECUTION HELPERS ====================
def execute_live_entry(symbol: str, side: str, raw_qty: float, stop_loss: float, take_profit: float):
    """Places real market entry + server-side SL/TP orders on XT exchange when LIVE_TRADING=True."""
    exchange.set_leverage(LEVERAGE, symbol)
    qty = float(exchange.amount_to_precision(symbol, raw_qty))
    sl_price = float(exchange.price_to_precision(symbol, stop_loss))
    tp_price = float(exchange.price_to_precision(symbol, take_profit))

    order_side = "buy" if side == "LONG" else "sell"
    close_side = "sell" if side == "LONG" else "buy"

    entry_order = exchange.create_market_order(symbol, order_side, qty)
    filled_price = float(entry_order.get("average") or entry_order.get("price") or 0)

    # Attach server-side Stop Loss and Take Profit orders
    exchange.create_order(
        symbol, "market", close_side, qty, None,
        params={"stopPrice": sl_price, "reduceOnly": True}
    )
    exchange.create_order(
        symbol, "market", close_side, qty, None,
        params={"stopPrice": tp_price, "reduceOnly": True}
    )
    return qty, filled_price


def close_live_position(symbol: str, side: str, qty: float):
    close_side = "sell" if side == "LONG" else "buy"
    clean_qty = float(exchange.amount_to_precision(symbol, qty))
    if clean_qty > 0:
        exchange.create_market_order(symbol, close_side, clean_qty, params={"reduceOnly": True})


# ==================== POSITION & EXIT MANAGEMENT ====================
def manage_positions(state):
    positions = state.get("positions", {})
    remaining_positions = {}
    now_dt = datetime.now(timezone.utc)
    now_str = str(now_dt)

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

            # 1. Stop Loss or Breakeven Exit
            if hit_sl:
                if LIVE_TRADING:
                    try:
                        exchange.cancel_all_orders(symbol)
                        close_live_position(symbol, side, qty)
                    except Exception as ex:
                        logging.error("Live SL close error on %s: %s", symbol, ex)

                gross_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                exit_fee = (current_price * qty) * FEE_RATE
                net_pnl = gross_pnl - exit_fee
                state["balance"] += net_pnl
                state.setdefault("symbol_cooldowns", {})[symbol] = now_dt.isoformat()

                reason = "BREAKEVEN (Risk-Free)" if pos.get("is_breakeven") and gross_pnl >= 0 else "STOP_LOSS"

                lines = [
                    f"🛑 <b>Position Closed: {reason}</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Entry: <code>{entry:.4f}</code> | Exit: <code>{current_price:.4f}</code>",
                    f"Net PnL (after fee): <b>{net_pnl:+.4f} USDT</b> ({pnl_pct:+.2f}%)",
                    f"Fee Paid: <code>-{exit_fee:.4f} USDT</code>",
                    f"Balance: <code>{state['balance']:.2f} USDT</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": net_pnl, "pct": pnl_pct,
                    "fee": exit_fee, "reason": reason, "time": now_str
                })
                continue

            # 2. Final TP2 Hit (Full Close)
            if hit_tp2:
                if LIVE_TRADING:
                    try:
                        exchange.cancel_all_orders(symbol)
                        close_live_position(symbol, side, qty)
                    except Exception as ex:
                        logging.error("Live TP2 close error on %s: %s", symbol, ex)

                gross_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                exit_fee = (current_price * qty) * FEE_RATE
                net_pnl = gross_pnl - exit_fee
                state["balance"] += net_pnl
                state.setdefault("symbol_cooldowns", {})[symbol] = now_dt.isoformat()

                lines = [
                    "🎉 <b>Target 2 Hit (Full Close)</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Entry: <code>{entry:.4f}</code> | Exit: <code>{current_price:.4f}</code>",
                    f"Net PnL (after fee): <b>{net_pnl:+.4f} USDT</b> ({pnl_pct:+.2f}%)",
                    f"Fee Paid: <code>-{exit_fee:.4f} USDT</code>",
                    f"Balance: <code>{state['balance']:.2f} USDT</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": net_pnl, "pct": pnl_pct,
                    "fee": exit_fee, "reason": "TAKE_PROFIT_2", "time": now_str
                })
                continue

            # 3. Partial TP1 Hit (50% Close + Breakeven)
            if hit_tp1:
                close_qty = qty / 2.0
                if LIVE_TRADING:
                    try:
                        close_live_position(symbol, side, close_qty)
                    except Exception as ex:
                        logging.error("Live TP1 partial close error on %s: %s", symbol, ex)

                gross_pnl = (tp1 - entry) * close_qty if side == "LONG" else (entry - tp1) * close_qty
                exit_fee = (tp1 * close_qty) * FEE_RATE
                realized_pnl = gross_pnl - exit_fee
                state["balance"] += realized_pnl

                fee_buffer = entry * (FEE_RATE * 2)
                breakeven_sl = entry + fee_buffer if side == "LONG" else entry - fee_buffer

                pos["quantity"] = close_qty
                pos["stop_loss"] = breakeven_sl
                pos["is_breakeven"] = True
                pos["is_tp1_hit"] = True

                lines = [
                    "💰 <b>Partial TP1 Hit (50% Closed)</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code>",
                    f"Side: <b>{side}</b>",
                    f"Realized Net PnL: <b>{realized_pnl:+.4f} USDT</b>",
                    f"Stop Loss moved to Breakeven (+fee buffer): <code>{breakeven_sl:.4f}</code>",
                    f"Target 2: <code>{take_profit:.4f}</code>"
                ]
                send_telegram("\n".join(lines))

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": realized_pnl, "pct": pnl_pct,
                    "fee": exit_fee, "reason": "PARTIAL_TP1", "time": now_str
                })

            remaining_positions[symbol] = pos

        except Exception as e:
            logging.error("Error managing %s: %s", symbol, e)
            remaining_positions[symbol] = pos

    state["positions"] = remaining_positions


# ==================== OPENING POSITIONS ====================
def is_in_cooldown(symbol: str, state: dict) -> bool:
    last_closed_str = state.get("symbol_cooldowns", {}).get(symbol)
    if not last_closed_str:
        return False
    try:
        last_closed = datetime.fromisoformat(last_closed_str)
        elapsed_minutes = (datetime.now(timezone.utc) - last_closed).total_seconds() / 60.0
        return elapsed_minutes < COOLDOWN_MINUTES
    except Exception:
        return False


def open_position(symbol, signal_data, state):
    side = signal_data["signal"]
    entry = float(signal_data["entry_price"])
    stop = float(signal_data["stop_loss"])
    target = float(signal_data["take_profit"])
    candle_time = signal_data.get("candle_time", "")
    confidence = signal_data.get("confidence", 0)

    # Prevent duplicate signal on the exact same 15m candle
    if candle_time and state.get("last_signal_candles", {}).get(symbol) == candle_time:
        return False

    # Prevent re-entry during cooldown
    if is_in_cooldown(symbol, state):
        logging.info("Skipping %s: in %d-min cooldown.", symbol, COOLDOWN_MINUTES)
        return False

    # Correlation protection
    same_side_count = sum(1 for p in state.get("positions", {}).values() if p.get("side") == side)
    if same_side_count >= MAX_SAME_SIDE_POSITIONS:
        logging.info("Skipping %s (%s): max same-side positions reached.", symbol, side)
        return False

    # Minimum Stop Loss distance (0.4%)
    min_sl_dist = entry * MIN_SL_PCT
    raw_risk = abs(entry - stop)
    if raw_risk == 0:
        return False

    if raw_risk < min_sl_dist:
        stop = entry - min_sl_dist if side == "LONG" else entry + min_sl_dist
        price_risk = min_sl_dist
    else:
        price_risk = raw_risk

    tp1 = entry + (target - entry) * 0.5

    balance = state.get("balance", INITIAL_BALANCE)
    if LIVE_TRADING:
        try:
            bal_data = exchange.fetch_balance()
            balance = float(bal_data["USDT"]["free"])
            state["balance"] = balance
        except Exception as e:
            logging.error("Live balance fetch error: %s", e)
            return False

    risk_amount = balance * RISK_PER_TRADE
    effective_risk_per_unit = price_risk + (entry * FEE_RATE * 2)
    quantity = risk_amount / effective_risk_per_unit

    if LIVE_TRADING:
        try:
            quantity, filled_entry = execute_live_entry(symbol, side, quantity, stop, target)
            if filled_entry > 0:
                entry = filled_entry
                tp1 = entry + (target - entry) * 0.5
        except Exception as e:
            logging.error("Live order execution failed on %s: %s", symbol, e)
            send_telegram(f"⚠️ <b>Live Order Failed</b> on <code>{symbol}</code>: {e}")
            return False

    notional_value = quantity * entry
    margin_cost = notional_value / LEVERAGE
    entry_fee = notional_value * FEE_RATE

    used_margin = sum(p.get("margin_cost", 0.0) for p in state.get("positions", {}).values())
    free_balance = balance - used_margin
    if margin_cost + entry_fee > free_balance * 0.9:
        logging.info("Skipping %s: insufficient free margin.", symbol)
        return False

    state["balance"] -= entry_fee
    remaining_free = state["balance"] - (used_margin + margin_cost)

    if candle_time:
        state.setdefault("last_signal_candles", {})[symbol] = candle_time

    state["positions"][symbol] = {
        "side": side,
        "entry": entry,
        "quantity": quantity,
        "stop_loss": stop,
        "take_profit": target,
        "tp1": tp1,
        "margin_cost": margin_cost,
        "entry_fee": entry_fee,
        "is_tp1_hit": False,
        "is_breakeven": False,
        "opened_at": str(datetime.now(timezone.utc))
    }

    mode_tag = "LIVE" if LIVE_TRADING else "PAPER"
    lines = [
        f"⚡ <b>New Position Opened ({side} | {mode_tag})</b>",
        "------------------------------------",
        f"Symbol: <code>{symbol}</code>",
        f"Confidence: <b>{confidence}%</b> | Leverage: <b>{LEVERAGE}x</b>",
        f"Entry: <code>{entry:.4f}</code>",
        f"Stop Loss: <code>{stop:.4f}</code> (Max Risk: ${risk_amount:.2f})",
        f"TP1 (50% Close): <code>{tp1:.4f}</code>",
        f"TP2 (Final): <code>{target:.4f}</code>",
        f"Quantity: <code>{quantity:.4f}</code>",
        "------------------------------------",
        f"💵 <b>سرمایه درگیر (مارجین):</b> <code>{margin_cost:.2f} USDT</code>",
        f"🧾 <b>کارمزد ورود:</b> <code>-{entry_fee:.4f} USDT</code>",
        f"💰 <b>مانده حساب آزاد:</b> <code>{remaining_free:.2f} USDT</code>"
    ]
    send_telegram("\n".join(lines))
    return True


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

    if state.get("last_daily_report") != today_str:
        daily_trades = [t for t in trades if t.get("time", "").startswith(state.get("last_daily_report", ""))]
        rep = generate_report_text("Daily Performance Report (24H)", state.get("last_daily_report"), daily_trades, open_count, balance)
        send_telegram(rep)
        state["last_daily_report"] = today_str

    if state.get("last_weekly_report") != current_week:
        rep = generate_report_text("Weekly Performance Report", state.get("last_weekly_report"), trades, open_count, balance)
        send_telegram(rep)
        state["last_weekly_report"] = current_week

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
                if open_position(symbol, sig, state):
                    current_open_count += 1

            time.sleep(0.3)
        except Exception as e:
            logging.error("Scan error on %s: %s", symbol, e)

    check_and_send_reports(state)
    save_state(state)


if __name__ == "__main__":
    main()
