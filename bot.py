
import os
import requests
import pandas as pd

from strategy import analyze_market

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

XT_URL = "https://fapi.xt.com/future/market/v1/public/q/kline"


def send_message(message):
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing")

    if not CHAT_ID:
        raise ValueError("TELEGRAM_CHAT_ID is missing")

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message},
        timeout=20
    )

    print("Telegram HTTP Status:", response.status_code)
    print("Telegram Response:", response.text)

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")


def get_market_data():
    params = {
        "symbol": "btc_usdt",
        "interval": "15m",
        "limit": 250
    }

    response = requests.get(
        XT_URL,
        params=params,
        timeout=20
    )

    print("XT HTTP Status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("XT API Code:", data.get("returnCode"))
    print("XT API Message:", data.get("msgInfo"))

    if data.get("returnCode") not in (0, "0", None):
        raise RuntimeError(f"XT API error: {data}")

    candles = data.get("result")

    if not isinstance(candles, list) or not candles:
        raise RuntimeError(
            f"No candle data received from XT: {data}"
        )

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

    required = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(f"Missing column: {column}")

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=required)

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Remove the currently forming candle
    df = df.iloc[:-1].copy()

    if len(df) < 100:
        raise RuntimeError("Not enough closed candles")

    print("XT market data received successfully")
    print("Latest BTC price:", df.iloc[-1]["close"])
    print("Closed candles:", len(df))

    return df


def format_report(result):

    signal = result["signal"]

    if signal == "LONG":
        direction = "LONG 🟢"
    elif signal == "SHORT":
        direction = "SHORT 🔴"
    else:
        direction = "NO SIGNAL ⏳"

    message = (
        "BTC/USDT FUTURES ANALYSIS\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Exchange: XT\n"
        f"Timeframe: 15 Minutes\n"
        f"Price: {result['price']:.2f} USDT\n\n"

        "MARKET TREND\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"EMA Trend: {result['trend']}\n"
        f"Ichimoku: {result['ichimoku']}\n"
        f"RSI: {result['rsi']:.2f}\n"
        f"EMA20: {result['ema20']:.2f}\n"
        f"EMA50: {result['ema50']:.2f}\n"
        f"EMA200: {result['ema200']:.2f}\n"
        f"MACD: {result['macd']:.2f}\n"
        f"MACD Signal: {result['macd_signal']:.2f}\n"
        f"ATR: {result['atr']:.2f}\n\n"

        "SMART MONEY ANALYSIS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Bullish BOS: {result['bull_bos']}\n"
        f"Bearish BOS: {result['bear_bos']}\n"
        f"Bullish CHoCH: {result['bull_choch']}\n"
        f"Bearish CHoCH: {result['bear_choch']}\n"
        f"Bullish FVG: {result['bull_fvg']}\n"
        f"Bearish FVG: {result['bear_fvg']}\n"
        f"Sell-side Sweep: {result['bull_sweep']}\n"
        f"Buy-side Sweep: {result['bear_sweep']}\n"
        f"Market Zone: {result['zone']}\n"
        f"Volume Ratio: {result['volume_ratio']:.2f}\n\n"

        "SIGNAL SCORING\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"LONG Score: {result['long_score']}\n"
        f"SHORT Score: {result['short_score']}\n"
        f"Confidence Score: {result['confidence']}%\n\n"

        f"FINAL RESULT: {direction}\n"
    )

    if signal:
        message += (
            "\nTRADE PLAN\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"Entry: {result['price']:.2f}\n"
            f"Stop Loss: {result['stop']:.2f}\n"
            f"Take Profit: {result['target']:.2f}\n\n"
            "SIGNAL REASONS\n"
        )

        for reason in result["reasons"]:
            message += f"- {reason}\n"

    else:
        message += (
            "\nNo valid trading setup found.\n"
            "Waiting for stronger confirmation.\n"
        )

    message += (
        "\n━━━━━━━━━━━━━━━━━━\n"
        "Analysis only - No trade executed.\n"
        "Use risk management before trading."
    )

    return message


def main():
    print("Bitcoin Futures Bot Started")
    print("Exchange: XT")

    df = get_market_data()

    result = analyze_market(df)

    print("Market Trend:", result["trend"])
    print("LONG Score:", result["long_score"])
    print("SHORT Score:", result["short_score"])
    print("Signal:", result["signal"])

    message = format_report(result)

    send_message(message)

    print("Market report sent successfully")


if __name__ == "__main__":
    main()
