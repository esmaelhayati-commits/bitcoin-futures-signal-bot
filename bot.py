import os
import time
import json
import logging
import requests
import ccxt
import pandas as pd
from dotenv import load_dotenv
from strategy import analyze_market

load_dotenv()

# ---------------- CONFIG ----------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CANDLE_LIMIT = 250
STATE_FILE = "state.json"

# لیست ۱۰ ارز فیوچرز پرحجم
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
    "NEAR/USDT:USDT",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- EXCHANGE ----------------

exchange = ccxt.xt({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",
    },
})

# ---------------- STATE STORAGE ----------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logging.error("Failed to write state: %s", e)

# ---------------- TELEGRAM NOTIFICATION ----------------

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials are not set.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, data=payload, timeout=20)
        data = resp.json()
        if not data.get("ok"):
            logging.error("Telegram error: %s", data)
            return False
        return True
    except Exception as exc:
        logging.exception("Connection error with Telegram: %s", exc)
        return False

# ---------------- DATA FETCHING ----------------

def fetch_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
    if not candles or len(candles) < 100:
        raise ValueError(f"Insufficient candles for {symbol} on {timeframe}")

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    # حذف کندل زنده باز برای جلوگیری از خطا در محاسبات پایانی
    df = df.iloc[:-1].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)

# ---------------- FORMATTING ----------------

def format_price(value) -> str:
    try:
        val = float(value)
        if val >= 100:
            return f"{val:,.2f}"
        elif val >= 1:
            return f"{val:,.4f}"
        else:
            return f"{val:,.6f}"
    except (TypeError, ValueError):
        return "N/A"

def trend_text(value) -> str:
    trends = {
        "BULLISH": "🟢 صعودی",
        "BEARISH": "🔴 نزولی",
        "SIDEWAYS": "🟡 خنثی / رِنج",
    }
    return trends.get(str(value).upper(), str(value))

# ---------------- REPORT BUILDER ----------------

def create_signal_report(symbol: str, result: dict) -> str:
    signal = str(result.get("signal", "HOLD")).upper()
    emoji = "🟢" if signal == "LONG" else "🔴"

    entry = result.get("entry_price")
    stop = result.get("stop_loss")
    target = result.get("take_profit")
    rr = result.get("risk_reward", 2.5)

    trend_1h = trend_text(result.get("trend_1h"))
    trend_4h = trend_text(result.get("trend_4h"))
    reason = result.get("reason", "تایید ساختار اسمارت مانی")

    return f"""
{emoji} <b>سیگنال معتبر اسمارت مانی (SMC)</b>

━━━━━━━━━━━━━━
🪙 <b>ارز:</b> <code>{symbol}</code>
🎯 <b>جهت:</b> <b>{signal}</b>
⚙️ <b>نسخه:</b> <code>{result.get('strategy_version', '3.0')}</code>

━━━━━━━━━━━━━━
📊 <b>تحلیل ساختار تایم‌فریم‌ها</b>
• روند 1 ساعته (BOS): {trend_1h}
• روند 4 ساعته (کلی): {trend_4h}

━━━━━━━━━━━━━━
📌 <b>مشخصات معامله</b>
🔹 نقطه ورود: <code>{format_price(entry)}</code>
🛑 حد ضرر (SL): <code>{format_price(stop)}</code>
🎯 حد سود (TP): <code>{format_price(target)}</code>
⚖️ نسبت R:R پیشنهادی: <b>1:{rr}</b>

━━━━━━━━━━━━━━
💡 <b>دلیل ورود:</b>
<i>{reason}</i>
""".strip()

# ---------------- MAIN SCANNER ----------------

def scan_markets():
    state = load_state()

    for symbol in SYMBOLS:
        try:
            logging.info("Checking %s ...", symbol)
            df_15m = fetch_candles(symbol, "15m")
            df_1h = fetch_candles(symbol, "1h")
            df_4h = fetch_candles(symbol, "4h")

            last_time = str(df_15m["timestamp"].iloc[-1])
            result = analyze_market(df_15m, df_1h, df_4h, symbol=symbol)
            signal = result.get("signal")

            if signal in ["LONG", "SHORT"]:
                last_saved_time = state.get(symbol, {}).get("last_candle_time")
                last_saved_signal = state.get(symbol, {}).get("signal")

                # اگر در همین کندل قبلاً پیام ارسال شده باشد، تکرار نشود
                if last_saved_time == last_time and last_saved_signal == signal:
                    logging.info("%s signal already reported. Skipping duplicate.", symbol)
                else:
                    report = create_signal_report(symbol, result)
                    if send_telegram(report):
                        state[symbol] = {
                            "last_candle_time": last_time,
                            "signal": signal,
                            "price": result.get("entry_price")
                        }
                        save_state(state)
                        logging.info(">>> SIGNAL SENT FOR %s (%s) <<<", symbol, signal)
            else:
                logging.info("%s: HOLD (%s)", symbol, result.get("reason"))

            # فاصله زمانی بین هر جفت‌ارز جهت حفظ محدودیت‌های صرافی
            time.sleep(0.5)

        except Exception as err:
            logging.exception("Failed processing %s: %s", symbol, err)


def main():
    logging.info("Starting SMC Bot for %d symbols...", len(SYMBOLS))
    try:
        exchange.load_markets()
    except Exception as exc:
        logging.critical("Exchange loading error: %s", exc)
        return

    scan_markets()


if __name__ == "__main__":
    main()
