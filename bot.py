import os
import requests
import pandas as pd
from strategy import analyze_market

# Telegram settings
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# XT Futures API
XT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"


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

    print("Telegram response:", response.text)
    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")


def get_market_data():

    params = {
        "symbol": "btc_usdt",
        "interval": "15m",
        "limit": 200
    }

    response = requests.get(
        XT_URL,
        params=params,
        timeout=20
    )

    print("XT HTTP Status:", response.status_code)
    print("XT Response:", response.text[:2000])

    response.raise_for_status()

    data = response.json()

    if data.get("returnCode") not in (0, "0", None):
        raise RuntimeError(
            f"XT API error: {data}"
        )

    # استخراج اطلاعات کندل‌ها
    result = data.get("result", data.get("data"))

    if isinstance(result, dict):
        candles = result.get("items", result.get("list", result.get("data")))
    else:
        candles = result

    if not candles:
        raise RuntimeError(
            f"No candle data received from XT: {data}"
        )

    # تبدیل اطلاعات به DataFrame
    if isinstance(candles[0], dict):

        df = pd.DataFrame(candles)

        column_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        }

        df = df.rename(columns=column_map)

    else:

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

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"Missing column from XT response: {column}"
            )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df["timestamp"] = pd.to_numeric(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # حذف آخرین کندل برای جلوگیری از تحلیل کندل باز
    df = df.iloc[:-1].copy()

    if len(df) < 60:
        raise RuntimeError("Not enough candle data")

    print("XT market data received successfully")
    print("Latest BTC price:", df.iloc[-1]["close"])

    return df


def main():

    print("Bitcoin Futures Bot Started")
    print("Exchange: XT")

    df = get_market_data()

    signal = analyze_market(df)

    if signal is None:
        print("No trading signal found")
        return

    message = (
        "BTC/USDT FUTURES SIGNAL\n\n"
        "Exchange: XT\n"
        f"Direction: {signal['side']}\n"
        f"Entry: {signal['entry']:.2f}\n"
        f"Stop Loss: {signal['stop']:.2f}\n"
        f"Take Profit: {signal['target']:.2f}\n"
        f"RSI: {signal['rsi']}\n\n"
        "Timeframe: 15 Minutes\n"
        "Signal Only - No Trade Executed"
    )

    send_message(message)

    print("Signal sent successfully")


if __name__ == "__main__":
    main()
