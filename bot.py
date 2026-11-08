
import os
import time
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

TIMEFRAMES = ["15m", "1h", "4h"]
CANDLE_LIMIT = 250

# فعلاً فقط بیت‌کوین؛ بعداً می‌توانیم فهرست ارزها را گسترش دهیم.
SYMBOLS = [
    "BTC/USDT:USDT",
]

PAPER_MODE = True
INITIAL_BALANCE = 10.0

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

# ---------------- TELEGRAM ----------------

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials are missing.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        result = response.json()

        if not result.get("ok"):
            logging.error("Telegram error: %s", result)
            return False

        return True

    except Exception as error:
        logging.exception("Telegram connection error: %s", error)
        return False


# ---------------- MARKET DATA ----------------

def fetch_candles(symbol, timeframe):
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=CANDLE_LIMIT,
    )

    if not candles or len(candles) < 100:
        raise ValueError(
            f"Not enough candles for {symbol} {timeframe}"
        )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    # حذف کندل در حال تشکیل
    df = df.iloc[:-1].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
    )

    return df.reset_index(drop=True)


# ---------------- FORMAT HELPERS ----------------

def number(value):
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return "N/A"


def get_value(data, key, default="N/A"):
    value = data.get(key, default)

    if value is None:
        return default

    return value


def trend_text(value):
    trends = {
        "BULLISH": "🟢 صعودی",
        "BEARISH": "🔴 نزولی",
        "SIDEWAYS": "🟡 خنثی",
        "UP": "🟢 صعودی",
        "DOWN": "🔴 نزولی",
    }

    return trends.get(
        str(value).upper(),
        str(value)
    )


# ---------------- SIGNAL REPORT ----------------

def create_signal_report(symbol, result):

    signal = str(
        get_value(result, "signal", "WAIT")
    ).upper()

    emoji = {
        "LONG": "🟢",
        "SHORT": "🔴",
        "WAIT": "🟡",
    }.get(signal, "⚪")

    entry = get_value(result, "entry")
    stop = get_value(result, "stop_loss")

    tp1 = get_value(result, "tp1")
    tp2 = get_value(result, "tp2")
    tp3 = get_value(result, "tp3")

    confidence = get_value(result, "confidence")

    long_score = get_value(result, "long_score")
    short_score = get_value(result, "short_score")

    trend_15m = trend_text(
        get_value(result, "trend_15m")
    )

    trend_1h = trend_text(
        get_value(result, "trend_1h")
    )

    trend_4h = trend_text(
        get_value(result, "trend_4h")
    )

    reasons = get_value(result, "reasons", [])

    if not isinstance(reasons, (list, tuple)):
        reasons = [str(reasons)]

    reason_text = "\n".join(
        f"• {reason}" for reason in reasons[:6]
    )

    if not reason_text:
        reason_text = "• اطلاعات تکمیلی موجود نیست"

    report = f"""
{emoji} <b>XT FUTURES SIGNAL</b>

━━━━━━━━━━━━━━
<b>ارز:</b> {symbol}
<b>سیگنال:</b> {signal}
<b>حالت:</b> PAPER / NO REAL TRADE

━━━━━━━━━━━━━━
<b>تحلیل روند بازار</b>

15m: {trend_15m}
1h: {trend_1h}
4h: {trend_4h}

━━━━━━━━━━━━━━
<b>قدرت سیگنال</b>

LONG Score: {long_score}
SHORT Score: {short_score}
Confidence: {confidence}

━━━━━━━━━━━━━━
<b>اطلاعات معامله</b>

Entry: {number(entry)}
Stop Loss: {number(stop)}

TP1: {number(tp1)}
TP2: {number(tp2)}
TP3: {number(tp3)}

━━━━━━━━━━━━━━
<b>دلایل تحلیل</b>

{reason_text}

━━━━━━━━━━━━━━
<i>تحلیل الگوریتمی | XT</i>
"""

    return report.strip()


# ---------------- ANALYSIS ----------------

def analyze_symbol(symbol):

    logging.info("Analyzing %s", symbol)

    df_15m = fetch_candles(symbol, "15m")
    df_1h = fetch_candles(symbol, "1h")
    df_4h = fetch_candles(symbol, "4h")

    # استفاده مستقیم از الگوریتم جدید
    result = analyze_market(
        df_15m,
        df_1h,
        df_4h,
    )

    if not isinstance(result, dict):
        raise ValueError(
            "analyze_market must return a dictionary"
        )

    return result


# ---------------- MAIN ----------------

def main():

    logging.info("Bitcoin Futures Bot Started!")
    logging.info("Exchange: XT")
    logging.info("Paper mode: %s", PAPER_MODE)

    if not PAPER_MODE:
        raise RuntimeError(
            "Real trading is disabled in this bot version."
        )

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "Telegram credentials are missing."
        )

    try:
        exchange.load_markets()
        logging.info("XT markets loaded successfully.")

    except Exception as error:
        logging.exception("XT connection failed: %s", error)
        send_telegram(
            "⚠️ خطا در اتصال به بازار XT"
        )
        return

    send_telegram(
        "🚀 <b>Futures Signal Bot Started</b>\n\n"
        "Exchange: XT\n"
        "Mode: Paper Trading\n"
        "Timeframes: 15m / 1h / 4h\n"
        "Real orders: DISABLED"
    )

    for symbol in SYMBOLS:

        try:
            result = analyze_symbol(symbol)

            report = create_signal_report(
                symbol,
                result,
            )

            send_telegram(report)

            logging.info(
                "%s signal: %s",
                symbol,
                result.get("signal"),
            )

        except Exception as error:
            logging.exception(
                "Analysis failed for %s: %s",
                symbol,
                error,
            )

            send_telegram(
                f"⚠️ خطا در تحلیل {symbol}\n"
                f"<code>{str(error)[:500]}</code>"
            )


if __name__ == "__main__":
    main()
