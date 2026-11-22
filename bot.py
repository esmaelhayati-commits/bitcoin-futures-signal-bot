import os
import time
import json
import logging
from datetime import datetime, timezone
import requests
import ccxt
import pandas as pd
from dotenv import load_dotenv
from strategy import analyze_market, SignalGuard

load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("XT_API_KEY")
API_SECRET = os.getenv("XT_API_SECRET")

LIVE_TRADING = False            # Switch to True after final v5.0 backtest check
LEVERAGE = 25                   # High capital efficiency with safe liquidation distance
RISK_PER_TRADE = 0.025          # 2.5% risk per trade for higher net returns
BREAKEVEN_TRIGGER_R = 1.2       # Move SL to breakeven (+fee buffer) once price reaches +1.2R
MAX_OPEN_POSITIONS = 3
MAX_SAME_SIDE_POSITIONS = 2
MIN_SL_PCT = 0.004              # Minimum 0.4% SL distance
COOLDOWN_MINUTES = 120          # Prevent re-entering same symbol/side within 2 hours
FEE_RATE = 0.0005               # 0.05% taker fee per side
INITIAL_BALANCE = 100.0

STATE_FILE = "live_trader_state.json"
GUARD_STATE_FILE = "signal_guard_state.json"

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
signal_guard = SignalGuard(cooldown_minutes=COOLDOWN_MINUTES, path=GUARD_STATE_FILE)


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
def fetch_candles(symbol: str, timeframe: str, retries: int = 3):
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df.reset_index(drop=True)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1.5)


# ==================== LIVE EXCHANGE EXECUTION HELPERS ====================
def place_protection_orders(symbol: str, side: str, qty: float, stop_loss: float, take_profit: float):
    close_side = "sell" if side == "LONG" else "buy"
    clean_qty = float(exchange.amount_to_precision(symbol, qty))
    sl_price = float(exchange.price_to_precision(symbol, stop_loss))
    tp_price = float(exchange.price_to_precision(symbol, take_profit))

    exchange.create_order(
        symbol, "market", close_side, clean_qty, None,
        params={"stopPrice": sl_price, "reduceOnly": True}
    )
    exchange.create_order(
        symbol, "market", close_side, clean_qty, None,
        params={"stopPrice": tp_price, "reduceOnly": True}
    )


def execute_live_entry(symbol: str, side: str, raw_qty: float, stop_loss: float, take_profit: float):
    exchange.set_leverage(LEVERAGE, symbol)
    qty = float(exchange.amount_to_precision(symbol, raw_qty))
    order_side = "buy" if side == "LONG" else "sell"

    entry_order = exchange.create_market_order(symbol, order_side, qty)
    filled_price = float(entry_order.get("average") or entry_order.get("price") or 0)
    place_protection_orders(symbol, side, qty, stop_loss, take_profit)
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
            risk_dist = pos.get("risk_dist", abs(entry - stop_loss))
            is_be = pos.get("is_breakeven", False)
            qty = pos["quantity"]

            if side == "LONG":
                pnl_pct = ((current_price - entry) / entry) * 100
                hit_sl = current_price <= stop_loss
                hit_tp = current_price >= take_profit
                hit_be_trigger = (not is_be) and (current_price >= entry + risk_dist * BREAKEVEN_TRIGGER_R)
            else:
                pnl_pct = ((entry - current_price) / entry) * 100
                hit_sl = current_price >= stop_loss
                hit_tp = current_price <= take_profit
                hit_be_trigger = (not is_be) and (current_price <= entry - risk_dist * BREAKEVEN_TRIGGER_R)

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

                reason = "BREAKEVEN (Risk-Free)" if is_be and gross_pnl >= 0 else "STOP_LOSS"
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

            # 2. Full Take Profit Hit (100% Volume)
            if hit_tp:
                if LIVE_TRADING:
                    try:
                        exchange.cancel_all_orders(symbol)
                        close_live_position(symbol, side, qty)
                    except Exception as ex:
                        logging.error("Live TP close error on %s: %s", symbol, ex)

                gross_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                exit_fee = (current_price * qty) * FEE_RATE
                net_pnl = gross_pnl - exit_fee
                state["balance"] += net_pnl

                lines = [
                    "🎉 <b>Target Hit (Full 100% Profit)</b>",
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
                    "fee": exit_fee, "reason": "TAKE_PROFIT", "time": now_str
                })
                continue

            # 3. Breakeven Lock at +1.2R (Keep 100% volume running for full TP)
            if hit_be_trigger:
                fee_buffer = entry * (FEE_RATE * 2.5)
                breakeven_sl = entry + fee_buffer if side == "LONG" else entry - fee_buffer
                pos["stop_loss"] = breakeven_sl
                pos["is_breakeven"] = True

                if LIVE_TRADING:
                    try:
                        exchange.cancel_all_orders(symbol)
                        place_protection_orders(symbol, side, qty, breakeven_sl, take_profit)
                    except Exception as ex:
                        logging.error("Live BE order update error on %s: %s", symbol, ex)

                lines = [
                    "🛡️ <b>Risk-Free Mode Activated (+1.2R Reached)</b>",
                    "------------------------------------",
                    f"Symbol: <code>{symbol}</code> | Side: <b>{side}</b>",
                    f"Stop Loss moved to Breakeven: <code>{breakeven_sl:.4f}</code>",
                    f"100% Position running toward Target: <code>{take_profit:.4f}</code>"
                ]
                send_telegram("\n".join(lines))

            remaining_positions[symbol] = pos

        except Exception as e:
            logging.error("Error managing %s: %s", symbol, e)
            remaining_positions[symbol] = pos

    state["positions"] = remaining_positions


# ==================== OPENING POSITIONS ====================
def open_position(symbol, signal_data, state):
    side = signal_data["signal"]
    entry = float(signal_data["entry_price"])
    stop = float(signal_data["stop_loss"])
    target = float(signal_data["take_profit"])
    confidence = signal_data.get("confidence", 0)
    structure_1h = signal_data.get("structure_1h", "")
    adx_1h = signal_data.get("adx_1h", 0)

    same_side_count = sum(1 for p in state.get("positions", {}).values() if p.get("side") == side)
    if same_side_count >= MAX_SAME_SIDE_POSITIONS:
        logging.info("Skipping %s (%s): max same-side positions reached.", symbol, side)
        return False

    min_sl_dist = entry * MIN_SL_PCT
    raw_risk = abs(entry - stop)
    if raw_risk == 0:
        return False

    if raw_risk < min_sl_dist:
        stop = entry - min_sl_dist if side == "LONG" else entry + min_sl_dist
        price_risk = min_sl_dist
    else:
        price_risk = raw_risk

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

    state["positions"][symbol] = {
        "side": side,
        "entry": entry,
        "quantity": quantity,
        "stop_loss": stop,
        "take_profit": target,
        "risk_dist": price_risk,
        "margin_cost": margin_cost,
        "entry_fee": entry_fee,
        "zone_id": signal_data.get("zone_id", ""),
        "is_breakeven": False,
        "opened_at": str(datetime.now(timezone.utc))
    }

    mode_tag = "LIVE" if LIVE_TRADING else "PAPER"
    lines = [
        f"⚡ <b>New Position Opened ({side} | {mode_tag})</b>",
        "------------------------------------",
        f"Symbol: <code>{symbol}</code>",
        f"Setup: <b>{structure_1h}</b> | ADX: <b>{adx_1h}</b> | Score: <b>{confidence}%</b>",
        f"Leverage: <b>{LEVERAGE}x</b> | Risk: <b>{RISK_PER_TRADE*100:.1f}%</b>",
        f"Entry: <code>{entry:.4f}</code>",
        f"Stop Loss: <code>{stop:.4f}</code> (Max Risk: ${risk_amount:.2f})",
        f"Target (Full 2.0R): <code>{target:.4f}</code>",
        f"Quantity: <code>{quantity:.4f}</code>",
        "------------------------------------",
        f"💵 <b>Margin Used:</b> <code>{margin_cost:.2f} USDT</code>",
        f"🧾 <b>Entry Fee:</b> <code>-{entry_fee:.4f} USDT</code>",
        f"💰 <b>Free Balance:</b> <code>{remaining_free:.2f} USDT</code>"
    ]
    send_telegram("\n".join(lines))
    return True


# ==================== REPORTING ====================
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
            df1 = fetch_candles(symbol, "1h")
            df4 = fetch_candles(symbol, "4h")

            sig = analyze_market(None, df1, df4, symbol=symbol, only_closed=True)
            logging.info("%s -> %s (%s)", symbol, sig.get("code"), sig.get("reason"))

            if sig.get("signal") in ["LONG", "SHORT"]:
                if signal_guard.allow(sig):
                    if open_position(symbol, sig, state):
                        current_open_count += 1
                else:
                    logging.info("Signal blocked by SignalGuard on %s", symbol)

            time.sleep(0.3)
        except Exception as e:
            logging.error("Scan error on %s: %s", symbol, e)

    check_and_send_reports(state)
    save_state(state)


if __name__ == "__main__":
    main()
