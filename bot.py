
import os
import time
import requests
import pandas as pd

from strategy import analyze_market

XT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=20
    )

    print("Telegram Status:", response.status_code)
    print("Telegram Response:", response.text)

    response.raise_for_status()


def get_market_data(interval):
    params = {
        "symbol": "btc_usdt",
        "interval": interval,
        "limit": 200
    }

    response = requests.get(
        XT_URL,
        params=params,
        timeout=20
    )

    print(f"XT {interval} Status:", response.status_code)
    response.raise_for_status()

    result = response.json()

    if result.get("returnCode") != 0:
        raise Exception(f"XT API Error: {result}")

    rows = result.get("result", [])

    if not rows:
        raise Exception(f"No market data for {interval}")

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })

    required = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    df = df[required].copy()

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    # حذف کندل در حال تشکیل
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise Exception(f"Not enough candles for {interval}")

    return df.reset_index(drop=True)


def get_trend(df):
    close = df["close"]

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    if ema20 > ema50:
        return "Bullish"

    if ema20 < ema50:
        return "Bearish"

    return "Neutral"


def main():
    print("Bitcoin Futures Multi-Timeframe Bot Started")

    try:
        df_15m = get_market_data("15m")
        df_1h = get_market_data("1h")
        df_4h = get_market_data("4h")

        trend_15m = get_trend(df_15m)
        trend_1h = get_trend(df_1h)
        trend_4h = get_trend(df_4h)

        analysis = analyze_market(df_15m, df_1h, df_4h)

        print("15m Trend:", trend_15m)
        print("1h Trend:", trend_1h)
        print("4h Trend:", trend_4h)

        print("Market analysis completed")

        message = f"""
BTC/USDT FUTURES ANALYSIS
━━━━━━━━━━━━━━━━━━

Exchange: XT
Price: {analysis['price']:.2f} USDT

MULTI-TIMEFRAME ANALYSIS
━━━━━━━━━━━━━━━━━━
15 Minutes: {trend_15m}
1 Hour: {trend_1h}
4 Hours: {trend_4h}

15M STRATEGY RESULT
━━━━━━━━━━━━━━━━━━
Signal: {analysis['signal']}
LONG Score: {analysis['long_score']}
SHORT Score: {analysis['short_score']}
Confidence: {analysis['confidence']}%

━━━━━━━━━━━━━━━━━━
Analysis only - No trade executed.
"""

        send_message(message)

    except Exception as error:
        print("BOT ERROR:", str(error))

        try:
            send_message(
                "BTC Bot Error:\n" + str(error)[:3000]
            )
        except Exception as telegram_error:
            print("Telegram Error:", str(telegram_error))


if __name__ == "__main__":
    main()
