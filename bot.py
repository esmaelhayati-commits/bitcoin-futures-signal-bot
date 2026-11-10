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

# ---------------- تنظیمات و متغیرها ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("XT_API_KEY")
API_SECRET = os.getenv("XT_API_SECRET")

# فعال‌سازی معامله واقعی (در صورت False بودن، روی کاغذ معامله می‌کند)
LIVE_TRADING = False  
LEVERAGE = 10
RISK_PER_TRADE = 0.015  # ریسک ۱.۵ درصد در هر پوزیشن
MAX_OPEN_POSITIONS = 3
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

# ---------------- اتصال به صرافی XT ----------------
exchange_params = {
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
}
if LIVE_TRADING and API_KEY and API_SECRET:
    exchange_params.update({"apiKey": API_KEY, "secret": API_SECRET})

exchange = ccxt.xt(exchange_params)

# ---------------- مدیریت وضعیت پوزیشن‌ها ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"positions": {}, "daily_history": [], "last_daily_report": str(datetime.now(timezone.utc).date())}

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logging.error("Failed to save state: %s", e)

# ---------------- پیام‌رسان تلگرام ----------------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        logging.error("Telegram send error: %s", e)

# ---------------- دریافت کندل‌ها ----------------
def fetch_candles(symbol: str, timeframe: str):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.iloc[:-1].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)

# ---------------- مدیریت پوزیشن و سیو سود ----------------
def manage_open_positions(state):
    updated_positions = {}
    
    for symbol, pos in state.get("positions", {}).items():
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker["last"])
            side = pos["side"]
            entry = pos["entry"]
            stop_loss = pos["stop_loss"]
            take_profit = pos["take_profit"]
            half_tp = pos["half_tp"]
            is_breakeven = pos.get("is_breakeven", False)
            qty = pos["quantity"]

            # سناریوی خروج حد ضرر
            hit_sl = (side == "LONG" and current_price <= stop_loss) or (side == "SHORT" and current_price >= stop_loss)
            # سناریوی خروج کامل حد سود
            hit_tp = (side == "LONG" and current_price >= take_profit) or (side == "SHORT" and current_price <= take_profit)
            # سناریوی سیو سود ۵۰٪ اولیه
            hit_half = not is_breakeven and ((side == "LONG" and current_price >= half_tp) or (side == "SHORT" and current_price <= half_tp))

            if hit_sl:
                pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                msg = f"🛑 <b>حد ضرر فعال شد!</b>\nارز: {symbol}\nجهت: {side}\nقیمت خروج: {current_price}\nسود/ضرر: {pnl:+.4f} USDT"
                send_telegram(msg)
                state["daily_history"].append({"symbol": symbol, "pnl": pnl, "time": str(datetime.now(timezone.utc))})
                continue

            if hit_tp:
                pnl = (current_price - entry) * qty if side == "LONG" else (entry - current_price) * qty
                msg = f"🎯 <b>تارگت کامل تاچ شد!</b>\nارز: {symbol}\nجهت: {side}\nقیمت خروج: {current_price}\nسود: {pnl:+.4f} USDT"
                send_telegram(msg)
                state["daily_history"].append({"symbol": symbol, "pnl": pnl, "time": str(datetime.now(timezone.utc))})
                continue

            if hit_half:
                realized_pnl = ((half_tp - entry) * (qty / 2)) if side == "LONG" else ((entry - half_tp) * (qty / 2))
                pos["quantity"] = qty / 2
                pos["stop_loss"] = entry  # ریسک‌فری کردن معامله
                pos["is_breakeven"] = True
                msg = f"💰 <b>سیو سود ۵۰٪ و ریسک‌فری!</b>\nارز: {symbol}\nجهت: {side}\nقیمت: {current_price}\nسود نقدشده: {realized_pnl:+.4f} USDT\nاستاپ به نقطه ورود منتقل شد."
                send_telegram(msg)
                state["daily_history"].append({"symbol": symbol, "pnl": realized_pnl, "time": str(datetime.now(timezone.utc))})

            updated_positions[symbol] = pos

        except Exception as e:
            logging.error("Error managing %s: %s", symbol, e)
            updated_positions[symbol] = pos

    state["positions"] = updated_positions

# ---------------- باز کردن معامله جدید ----------------
def execute_signal(symbol, signal_data, state):
    side = signal_data["signal"]
    entry = signal_data["entry_price"]
    stop = signal_data["stop_loss"]
    target = signal_data["take_profit"]

    # محاسبه حد سود میانی برای سیو سود
    half_tp = entry + (target - entry) * 0.5

    # بالانس فرضی یا واقعی
    balance = 100.0
    if LIVE_TRADING:
        try:
            bal_data = exchange.fetch_balance()
            balance = float(bal_data["USDT"]["free"])
        except Exception:
            pass

    risk_amount = balance * RISK_PER_TRADE
    price_risk = abs(entry - stop)
    if price_risk == 0:
        return

    quantity = (risk_amount / price_risk)

    state["positions"][symbol] = {
        "side": side,
        "entry": entry,
        "quantity": quantity,
        "stop_loss": stop,
        "take_profit": target,
        "half_tp": half_tp,
        "is_breakeven": False,
        "opened_at": str(datetime.now(timezone.utc))
    }

    report = (
        f"⚡ <b>سفارش جدید اجرا شد ({side})</b>\n\n"
        f"🪙 ارز: <code>{symbol}</code>\n"
        f"🔹 قیمت ورود: <code>{entry:.4f}</code>\n"
        f"🛑 حد ضرر: <code>{stop:.4f}</code>\n"
        f"💰 سیو سود (TP1): <code>{half_tp:.4f}</code>\n"
        f"🎯 تارگت نهایی: <code>{target:.4f}</code>\n"
        f"⚖️ حجم پوزیشن: <code>{quantity:.4f}</code>"
    )
    send_telegram(report)

# ---------------- گزارش سود و زیان روزانه ----------------
def send_daily_report_if_needed(state):
    today = str(datetime.now(timezone.utc).date())
    if state.get("last_daily_report") != today:
        history = state.get("daily_history", [])
        total_pnl = sum(item["pnl"] for item in history)
        wins = len([item for item in history if item["pnl"] > 0])
        losses = len([item for item in history if item["pnl"] <= 0])
        total_trades = wins + losses
        winrate = (wins / total_trades * 100) if total_trades > 0 else 0

        msg = (
            f"📅 <b>گزارش عملکرد روزانه ({state.get('last_daily_report')})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"💵 کل سود/زیان محقق‌شده: <b>{total_pnl:+.4f} USDT</b>\n"
            f"📊 تعداد معاملات بسته: {total_trades}\n"
            f"✅ معاملات برنده: {wins}\n"
            f"❌ معاملات بازنده: {losses}\n"
            f"🎯 وین‌ریت روز: <b>{winrate:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━"
        )
        send_telegram(msg)
        state["daily_history"] = []
        state["last_daily_report"] = today

# ---------------- اسکنر اصلی ----------------
def main():
    state = load_state()
    exchange.load_markets()

    # ۱. مدیریت معاملات فعال جاری (خروج، سیو سود و ریسک‌فری)
    manage_open_positions(state)

    # ۲. اسکن جفت‌ارزها برای موقعیت‌های جدید
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
                execute_signal(symbol, sig, state)
                current_open_count += 1

            time.sleep(0.4)
        except Exception as e:
            logging.error("Scan error on %s: %s", symbol, e)

    # ۳. بررسی زمان ارسال گزارش روزانه
    send_daily_report_if_needed(state)
    save_state(state)

if __name__ == "__main__":
    main()
