import os
import requests
import ccxt
import pandas as pd
from strategy import analyze_market

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

exchange = ccxt.binanceusdm({
    "enableRateLimit": True
})


def send_message(message):
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing")

    if not CHAT_ID:
        raise ValueError("TELEGRAM_CHAT_ID is missing")

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status()


def get_market_data():
    candles = exchange.fetch_ohlcv(
        "BTC/USDT",
        timeframe="15m",
        limit=200
    )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    return df.iloc[:-1].copy()


def main():
    print("Bitcoin Futures Bot Started!")

    df = get_market_data()

    signal = analyze_market(df)

    if signal is None:
        print("No signal found.")
        return

    message = (
        "BTC/USDT FUTURES SIGNAL\n\n"
        f"Direction: {signal['side']}\n"
        f"Entry: {signal['entry']:.2f}\n"
        f"Stop Loss: {signal['stop']:.2f}\n"
        f"Take Profit: {signal['target']:.2f}\n"
        f"RSI: {signal['rsi']}\n\n"
        "15 Minute Timeframe\n"
        "Signal Only - No Trade Executed"
    )

    send_message(message)


if __name__ == "__main__":
    main()
