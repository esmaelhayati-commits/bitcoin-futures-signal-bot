
import os
import time
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
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    ).raise_for_status()


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
    last_signal_candle = None

    send_message("Bitcoin Futures Bot Started!")

    while True:
        try:
            df = get_market_data()
            signal = analyze_market(df)

            candle_time = int(df.iloc[-1]["timestamp"])

            if signal and candle_time != last_signal_candle:

                message = (
                    f"BTC/USDT FUTURES\n"
                    f"Signal: {signal['side']}\n"
                    f"Entry: {signal['entry']:.2f}\n"
                    f"Stop Loss: {signal['stop']:.2f}\n"
                    f"Take Profit: {signal['target']:.2f}\n"
                    f"RSI: {signal['rsi']}\n\n"
                    f"Signal Only - No Trade Executed"
                )

                send_message(message)
                last_signal_candle = candle_time

        except Exception as error:
            print("ERROR:", error)

        time.sleep(60)


if __name__ == "__main__":
    main()
