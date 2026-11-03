
import os
import requests
import pandas as pd

from strategy import analyze_market
from chart import create_chart

XT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def get_market_data(interval):
    params = {
        "symbol": "btc_usdt",
        "interval": interval,
        "limit": 250
    }

    response = requests.get(
        XT_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()
    result = response.json()

    if result.get("returnCode") != 0:
        raise Exception(f"XT API Error: {result}")

    rows = result.get("result", [])

    if not rows:
        raise Exception(f"No data for {interval}")

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })

    columns = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    df = df[columns].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    # Remove the currently forming candle
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < 210:
        raise Exception(
            f"Not enough candles for {interval}: {len(df)}"
        )

    return df.reset_index(drop=True)


def send_photo(filename, caption):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendPhoto"
    )

    with open(filename, "rb") as photo:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption
            },
            files={"photo": photo},
            timeout=60
        )

    print("Telegram Photo Status:", response.status_code)
    print("Telegram Photo Response:", response.text)

    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise Exception(f"Telegram error: {result}")


def build_report(a):
    def show(value):
        return f"{value:.2f}" if value is not None else "N/A"

    signal_emoji = {
        "LONG": "🟢",
        "SHORT": "🔴",
        "NO SIGNAL": "⏳"
    }.get(a["signal"], "⏳")

    reasons = "\n".join(
        f"• {reason}" for reason in a["reasons"][:8]
    )

    if not reasons:
        reasons = "No strong confirmation"

    return f"""
{signal_emoji} BTC/USDT FUTURES ANALYSIS
━━━━━━━━━━━━━━━━━━

Exchange: XT
Timeframe: 15m
Price: {show(a['price'])} USDT

📊 MULTI-TIMEFRAME
━━━━━━━━━━━━━━━━━━
15m: {a['trend_15m']}
1h: {a['trend_1h']}
4h: {a['trend_4h']}

📈 TECHNICAL INDICATORS
━━━━━━━━━━━━━━━━━━
EMA20: {show(a['ema20'])}
EMA50: {show(a['ema50'])}
EMA200: {show(a['ema200'])}
RSI: {show(a['rsi'])}
MACD: {show(a['macd'])}
MACD Signal: {show(a['macd_signal'])}
ATR: {show(a['atr'])}
Volume Ratio: {show(a['volume_ratio'])}

🧠 SMART MONEY
━━━━━━━━━━━━━━━━━━
Bullish BOS: {a['bull_bos']}
Bearish BOS: {a['bear_bos']}
Bullish FVG: {a['bullish_fvg']}
Bearish FVG: {a['bearish_fvg']}
Bullish OB: {a['bullish_ob']}
Bearish OB: {a['bearish_ob']}
Sell-side Sweep: {a['bull_sweep']}
Buy-side Sweep: {a['bear_sweep']}
Market Zone: {a['zone']}

🎯 SIGNAL SCORING
━━━━━━━━━━━━━━━━━━
LONG Score: {a['long_score']}
SHORT Score: {a['short_score']}
Algorithm Score: {a['confidence']}%

🚦 FINAL RESULT
━━━━━━━━━━━━━━━━━━
Signal: {a['signal']}

Entry: {show(a['entry'])}
Stop Loss: {show(a['stop_loss'])}
TP1: {show(a['tp1'])}
TP2: {show(a['tp2'])}
TP3: {show(a['tp3'])}

📝 REASONS
━━━━━━━━━━━━━━━━━━
{reasons}

━━━━━━━━━━━━━━━━━━
Analysis only - No trade executed.
Algorithm score is not a probability of profit.
"""


def send_error(message):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "BTC Bot Error:\n" + str(message)[:3000]
        },
        timeout=20
    )


def main():
    print("BTC Futures Advanced Bot Started")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise Exception("Telegram secrets are missing")

    try:
        df_15m = get_market_data("15m")
        df_1h = get_market_data("1h")
        df_4h = get_market_data("4h")

        analysis = analyze_market(
            df_15m,
            df_1h,
            df_4h
        )

        print("Signal:", analysis["signal"])
        print("LONG:", analysis["long_score"])
        print("SHORT:", analysis["short_score"])

        chart_file = create_chart(
            df_15m,
            analysis
        )

        report = build_report(analysis)

        send_photo(chart_file, report)

        print("Analysis and chart sent successfully")

    except Exception as error:
        print("BOT ERROR:", str(error))
        send_error(error)
        raise


if __name__ == "__main__":
    main()
