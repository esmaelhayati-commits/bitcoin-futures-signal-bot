import os
import requests
import ccxt
import pandas as pd
from strategy import analyze_market

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

exchange = ccxt.binanceusdm({"enableRateLimit": True})


def send_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message},
        timeout=20
    )
    response.raise_for_status()


def main():
    candles = exchange.fetch_ohlcv(
        "BTC/USDT",
        timeframe="15m",
        limit=200
    )

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # آخرین کندل باز را حذف می‌کنیم
    df = df.iloc[:-1].copy()

    signal = analyze_market(df)

    if signal is None:
        print("No signal found.")
        return

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
    print("Signal sent to Telegram.")


if __name__ == "__main__":
    main()
