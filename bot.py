```python
import os
import requests
import ccxt
import pandas as pd
from strategy import analyze_market

# Telegram settings
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Binance Futures
exchange = ccxt.binanceusdm({
    "enableRateLimit": True
})


def send_message(message):
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing!")

    if not CHAT_ID:
        raise ValueError("TELEGRAM_CHAT_ID is missing!")

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
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

        result = response.json()

        if not result.get("ok"):
            raise RuntimeError(
                f"Telegram error: {result}"
            )

        print("Telegram message sent successfully!")

    except requests.exceptions.RequestException as error:
        print("Telegram connection error:", error)
        raise


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

    # حذف کندل در حال تشکیل
    return df.iloc[:-1].copy()


def main():

    print("Bitcoin Futures Bot Started!")

    try:
        df = get_market_data()

        if len(df) < 60:
            print("Not enough market data!")
            return

        signal = analyze_market(df)

        if signal is None:
            print("No trading signal found.")
            return

        message = (
            "🚨 BTC/USDT FUTURES SIGNAL 🚨\n\n"
            f"Direction: {signal['side']}\n"
            f"Entry Price: {signal['entry']:.2f}\n"
            f"Stop Loss: {signal['stop']:.2f}\n"
            f"Take Profit: {signal['target']:.2f}\n"
            f"RSI: {signal['rsi']}\n\n"
            "Timeframe: 15 Minutes\n"
            "Signal Only - No Trade Executed"
        )

        send_message(message)

        print("Signal successfully processed.")

    except Exception as error:
        print("BOT ERROR:", str(error))
        raise


if __name__ == "__main__":
    main()
```
