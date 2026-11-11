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

            # 1. برخورد به حد ضرر (خروج کامل)
            if hit_sl:
                net_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                state["balance"] += net_pnl
                reason_label = "ریسک‌فری (بدون ضرر)" if pos.get("is_breakeven") and net_pnl >= 0 else "حد ضرر اولیه (SL)"

                msg = (
                    f"🛑 <b>معامله بسته شد: {reason_label}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🪙 ارز: <code>{symbol}</code>\n"
                    f"🎯 جهت: <b>{side}</b>\n"
                    f"🔹 ورود: <code>{entry:.4f}</code> | خروج: <code>{current_price:.4f}</code>\n"
                    f"📉 سود/زیان: <b>{net_pnl:+.4f} USDT</b> ({pnl_pct:+.2f}%)\n"
                    f"💰 بالانس حساب: <code>{state['balance']:.2f} USDT</code>"
                )
                send_telegram(msg)

                state["trade_history"].append({
                    "symbol": symbol, "side": side, "pnl": net_pnl, "pct": pnl_pct,
                    "reason": "STOP_LOSS", "time": now_str
                })
                continue

            # 2. برخورد به تارگت نهایی TP2 (خروج کامل با سود)
            if hit_tp2:
                net_pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                state["balance"] += net_pnl

                msg = (
                    f"🎉 <b>تارگت نهایی (TP2) تاچ شد! (فروش کامل)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🪙 ارز: <code>{symbol}</code>\n"
                    f"🎯 جهت: <b>{side}</b>\n"
                    f"🔹 ورود: <code>{entry:.4f}</code> | خروج
