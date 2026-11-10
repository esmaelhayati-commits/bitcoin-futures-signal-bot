import ccxt
import pandas as pd
import numpy as np
import json
import time

from datetime import datetime, timezone

from strategy import (
    generate_signal,
    STRATEGY_VERSION,
    RR_RATIO
)

# ==========================================================
# BITCOIN & ALTCOINS SMC BACKTEST ENGINE
# NO REAL ORDERS
# Exchange: XT
# Entry: 15m | Confirmation: 1h + 4h
# ==========================================================

BACKTEST_VERSION = "SMC-3.0"

INITIAL_BALANCE = 10.0
LEVERAGE = 20
RISK_PER_TRADE = 0.01
MAX_MARGIN_FRACTION = 0.20
MAX_OPEN_POSITIONS = 3

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

LOOKBACK_DAYS = 90
CANDLE_LIMIT = 500
TIMEFRAME_MS = 15 * 60 * 1000

REPORT_FILE = "backtest_report_smc.json"
TRADES_FILE = "backtest_trades_smc.csv"

REQUESTED_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "LTC/USDT"
]

exchange = ccxt.xt({
    "enableRateLimit": True,
    "timeout": 30000,
    "options": {
        "defaultType": "swap"
    }
})


def log(message):
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] {message}",
        flush=True
    )


def resolve_swap_symbol(requested, markets):
    base, quote = requested.split("/")
    for market in markets.values():
        if (
            market.get("base") == base
            and market.get("quote") == quote
            and market.get("swap")
            and market.get("active", True)
            and market.get("settle") in (quote, None)
        ):
            return market["symbol"]
    return None


# ==========================================================
# MARKET DATA
# ==========================================================

def fetch_history(symbol):
    since = exchange.milliseconds() - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    all_candles = []
    log(f"Downloading history for {symbol}...")

    while True:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe="15m",
                since=since,
                limit=CANDLE_LIMIT
            )
        except Exception as exc:
            log(f"Download error {symbol}: {exc}")
            break

        if not batch:
            break

        all_candles.extend(batch)
        next_since = batch[-1][0] + TIMEFRAME_MS
        if next_since <= since:
            break
        since = next_since

        if len(batch) < CANDLE_LIMIT:
            break

        time.sleep(max(exchange.rateLimit, 250) / 1000)

    if not all_candles:
        return None

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=15) <= now]

    if len(df) < 500:
        log(f"Insufficient history {symbol}: {len(df)} candles")
        return None

    return df


def resample_ohlcv(df, timeframe, expected_bars):
    grouped = df.resample(timeframe, label="right", closed="left")
    result = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    counts = grouped["close"].count()
    return result[counts == expected_bars].dropna()


def prepare_market_data():
    log("Loading XT swap markets...")
    exchange.load_markets()
    market_data = {}

    for requested in REQUESTED_SYMBOLS:
        symbol = resolve_swap_symbol(requested, exchange.markets)
        if not symbol:
            log(f"Swap market not found: {requested}")
            continue

        df15 = fetch_history(symbol)
        if df15 is None:
            continue

        df1 = resample_ohlcv(df15, "1h", 4)
        df4 = resample_ohlcv(df15, "4h", 16)

        if min(len(df1), len(df4)) < 120:
            log(f"Insufficient higher timeframe history: {symbol}")
            continue

        market_data[symbol] = {
            "15m": df15,
            "1h": df1,
            "4h": df4
        }
        log(f"READY {symbol}: 15m={len(df15)}, 1h={len(df1)}, 4h={len(df4)}")

    return market_data


# ==========================================================
# POSITION ACCOUNTING
# ==========================================================

def unrealized_pnl(position, price):
    if position["side"] == "LONG":
        return (price - position["entry"]) * position["quantity"]
    return (position["entry"] - price) * position["quantity"]


def equity_value(balance, positions, prices):
    total = balance
    for symbol, position in positions.items():
        total += unrealized_pnl(position, prices.get(symbol, position["entry"]))
    return total


def margin_used(positions):
    return sum(p["margin"] for p in positions.values())


def update_excursions(position, candle):
    high = float(candle["high"])
    low = float(candle["low"])
    entry = position["entry"]
    quantity = position["quantity"]

    if position["side"] == "LONG":
        favorable = max(0.0, high - entry)
        adverse = max(0.0, entry - low)
    else:
        favorable = max(0.0, entry - low)
        adverse = max(0.0, high - entry)

    favorable_pnl = favorable * quantity
    adverse_pnl = adverse * quantity

    position["mfe_price"] = max(position["mfe_price"], favorable)
    position["mae_price"] = max(position["mae_price"], adverse)
    position["mfe_pnl"] = max(position["mfe_pnl"], favorable_pnl)
    position["mae_pnl"] = max(position["mae_pnl"], adverse_pnl)

    risk = position["initial_risk_amount"]
    if risk > 0:
        position["mfe_r"] = position["mfe_pnl"] / risk
        position["mae_r"] = position["mae_pnl"] / risk


def adverse_equity(balance, positions, market_data, timestamp, prices):
    total = balance
    for symbol, position in positions.items():
        df = market_data[symbol]["15m"]
        if timestamp in df.index:
            candle = df.loc[timestamp]
            mark = float(candle["low"] if position["side"] == "LONG" else candle["high"])
        else:
            mark = prices.get(symbol, position["entry"])
        total += unrealized_pnl(position, mark)
    return total


# ==========================================================
# EXIT EXECUTION
# ==========================================================

def check_exit(position, candle):
    side = position["side"]
    stop = position["stop_loss"]
    target = position["take_profit"]

    open_price = float(candle["open"])
    high = float(candle["high"])
    low = float(candle["low"])

    if side == "LONG":
        stop_hit = low <= stop
        target_hit = high >= target

        if stop_hit:
            raw_exit = min(open_price, stop)
            executed = raw_exit * (1 - SLIPPAGE_RATE)
            return executed, "STOP_LOSS", raw_exit

        if target_hit:
            raw_exit = max(open_price, target)
            executed = raw_exit * (1 - SLIPPAGE_RATE)
            return executed, "TAKE_PROFIT", raw_exit

    else:
        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit:
            raw_exit = max(open_price, stop)
            executed = raw_exit * (1 + SLIPPAGE_RATE)
            return executed, "STOP_LOSS", raw_exit

        if target_hit:
            raw_exit = min(open_price, target)
            executed = raw_exit * (1 + SLIPPAGE_RATE)
            return executed, "TAKE_PROFIT", raw_exit

    return None, None, None


# ==========================================================
# OPEN POSITION
# ==========================================================

def open_position(symbol, signal, candle, timestamp, signal_timestamp, equity, balance, positions, prices):
    side = signal.get("signal")
    if side not in ("LONG", "SHORT"):
        return balance
    if equity <= 0:
        return balance

    raw_entry = float(candle["open"])
    if not np.isfinite(raw_entry) or raw_entry <= 0:
        return balance

    entry = raw_entry * (1 + SLIPPAGE_RATE) if side == "LONG" else raw_entry * (1 - SLIPPAGE_RATE)

    # استفاده مستقیم از استاپ و تارگت حساب شده توسط اسمارت مانی
    stop = float(signal["stop_loss"])
    target = float(signal["take_profit"])

    if side == "LONG":
        stop_distance = max(entry - stop, entry * 0.002)
        estimated_exit = stop * (1 - SLIPPAGE_RATE)
    else:
        stop_distance = max(stop - entry, entry * 0.002)
        estimated_exit = stop * (1 + SLIPPAGE_RATE)

    estimated_cost_per_unit = (entry * FEE_RATE) + (estimated_exit * FEE_RATE)
    estimated_loss_per_unit = stop_distance + estimated_cost_per_unit

    if estimated_loss_per_unit <= 0:
        return balance

    risk_amount = equity * RISK_PER_TRADE
    quantity_by_risk = risk_amount / estimated_loss_per_unit

    max_margin = equity * MAX_MARGIN_FRACTION
    available_margin = max(0.0, equity - margin_used(positions))
    allowed_margin = min(max_margin, available_margin)
    max_notional = allowed_margin * LEVERAGE

    quantity = min(quantity_by_risk, max_notional / entry)
    if quantity <= 0:
        return balance

    notional = quantity * entry
    margin = notional / LEVERAGE
    entry_fee = notional * FEE_RATE

    if balance - entry_fee < 0:
        return balance

    balance -= entry_fee
    initial_risk_amount = quantity * estimated_loss_per_unit
    entry_slippage_cost = abs(entry - raw_entry) * quantity

    positions[symbol] = {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "raw_entry": raw_entry,
        "quantity": quantity,
        "stop_loss": stop,
        "take_profit": target,
        "stop_distance": stop_distance,
        "margin": margin,
        "entry_fee": entry_fee,
        "entry_slippage_cost": entry_slippage_cost,
        "initial_risk_amount": initial_risk_amount,
        "entry_time": timestamp.isoformat(),
        "signal_time": signal_timestamp.isoformat(),
        "entry_trend_1h": signal.get("trend_1h"),
        "entry_trend_4h": signal.get("trend_4h"),
        "entry_reason": signal.get("reason"),
        "mfe_price": 0.0,
        "mae_price": 0.0,
        "mfe_pnl": 0.0,
        "mae_pnl": 0.0,
        "mfe_r": 0.0,
        "mae_r": 0.0
    }

    prices[symbol] =
