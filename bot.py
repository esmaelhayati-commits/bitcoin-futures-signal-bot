```python
import os
import requests
import pandas as pd
from strategy import analyze_market

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BYBIT_URL = "https://api.bybit.com/v5/market/kline"


def send_message(message):
    if not TOKEN or not CHAT_ID:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
        )

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message},
        timeout=20
    )

    print("Telegram response:", response.text)
    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")


def get_market_data():
    params = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "15",
        "limit": 200
    }

    response = requests.get(
        BYBIT_URL,
        params=params,
        timeout=20
    )
    response.raise_for_status()

    data = response.json()

    if data.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit API error: {data.get('retMsg')}"
        )

    candles = data["result"]["list"]

    if not candles:
        raise RuntimeError("Bybit returned no candles")

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover"
        ]
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df["timestamp"] = pd.to_numeric(df["timestamp"])

    # مرتب‌سازی از قدیمی به جدید
    df = df.sort_values("timestamp").reset_index(drop=True)

    # حذف کندل در حال تشکیل
    df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise RuntimeError("Not enough market data")

    print("Candles received:", len(df))
    print("Latest closed BTC candle:", df.iloc[-1]["close"])

    return df


def main():
    print("Bitcoin Futures Signal Bot Started")

    df = get_market_data()
    signal = analyze_market(df)

    if signal is None:
        print("No signal found.")
        return

    message = (
        "BTC/USDT FUTURES SIGNAL\n\n"
        f"Exchange: Bybit\n"
        f"Direction: {signal['side']}\n"
        f"Entry: {signal['entry']:.2f}\n"
        f"Stop Loss: {signal['stop']:.2f}\n"
        f"Take Profit: {signal['target']:.2f}\n"
        f"RSI: {signal['rsi']}\n\n"
        "Timeframe: 15 minutes\n"
        "Signal only - no trade executed"
    )

    send_message(message)
    print("Signal sent successfully.")


if __name__ == "__main__":
    main()
```
