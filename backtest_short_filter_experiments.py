"""
INDEPENDENT A/B/C BACKTEST EXPERIMENTS
Uses the exact strategy.py V2.3 and the V2.3.1 execution/accounting model.
Downloads market data once, then replays the same candles for all scenarios.
No live orders are placed.
"""

import ccxt
import pandas as pd
import numpy as np
import json
import time

from datetime import datetime, timezone

from strategy import (
    generate_signal,
    prepare_data,
    STRATEGY_VERSION,
    STOP_ATR_MULTIPLIER,
    TARGET_ATR_MULTIPLIER
)


# ==========================================================
# BITCOIN FUTURES BOT V2.3.1
# DIAGNOSTIC BACKTEST ONLY
# NO REAL ORDERS
# Exchange: XT
# Entry: 15m
# Confirmation: 1h + 4h
# ==========================================================

BACKTEST_VERSION = "2.3.2-SHORT-FILTERS"

INITIAL_BALANCE = 10.0
LEVERAGE = 20
RISK_PER_TRADE = 0.01
MAX_MARGIN_FRACTION = 0.20
MAX_OPEN_POSITIONS = 3
COOLDOWN_HOURS = 4
MAX_SAME_DIRECTION_POSITIONS = 2

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

LOOKBACK_DAYS = 90
CANDLE_LIMIT = 500
TIMEFRAME_MS = 15 * 60 * 1000

REPORT_FILE = "backtest_report_v231.json"
TRADES_FILE = "backtest_trades_v231.csv"

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
        f"[{datetime.now(timezone.utc).isoformat()}] "
        f"{message}",
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

    since = (
        exchange.milliseconds()
        - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    )

    all_candles = []

    log(f"Downloading {symbol}")

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

        time.sleep(
            max(exchange.rateLimit, 250) / 1000
        )

    if not all_candles:
        return None

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df = df.set_index("datetime")

    df = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Exclude unfinished candles.

    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df.index + pd.Timedelta(minutes=15) <= now
    ]

    if len(df) < 1000:

        log(
            f"Insufficient history {symbol}: "
            f"{len(df)} candles"
        )

        return None

    return df


def resample_ohlcv(df, timeframe, expected_bars):

    grouped = df.resample(
        timeframe,
        label="right",
        closed="left"
    )

    result = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    counts = grouped["close"].count()

    return result[
        counts == expected_bars
    ].dropna()


def prepare_market_data():

    log("Loading XT swap markets")

    exchange.load_markets()

    market_data = {}

    for requested in REQUESTED_SYMBOLS:

        symbol = resolve_swap_symbol(
            requested,
            exchange.markets
        )

        if not symbol:

            log(f"Swap market not found: {requested}")
            continue

        df15 = fetch_history(symbol)

        if df15 is None:
            continue

        df1 = resample_ohlcv(
            df15,
            "1h",
            4
        )

        df4 = resample_ohlcv(
            df15,
            "4h",
            16
        )

        if min(len(df1), len(df4)) < 210:

            log(
                f"Insufficient higher timeframe "
                f"history: {symbol}"
            )

            continue

        market_data[symbol] = {
            "15m": df15,
            "1h": df1,
            "4h": df4
        }

        log(
            f"READY {symbol}: "
            f"15m={len(df15)}, "
            f"1h={len(df1)}, "
            f"4h={len(df4)}"
        )

    return market_data


# ==========================================================
# TREND AGE
# ==========================================================

def calculate_trend_age(df):

    """
    Reproduces the trend definition used by strategy.py.

    Returns:
        trend
        consecutive_trend_bars

    BULLISH and BEARISH are calculated using:
        close vs EMA200
        EMA50 vs EMA200
        EMA50 vs EMA50 five bars earlier
    """

    try:

        data = prepare_data(df)

        if len(data) < 210:
            return "INSUFFICIENT_DATA", 0

        bullish = (
            (data["close"] > data["ema200"])
            & (data["ema50"] > data["ema200"])
            & (data["ema50"] > data["ema50"].shift(5))
        )

        bearish = (
            (data["close"] < data["ema200"])
            & (data["ema50"] < data["ema200"])
            & (data["ema50"] < data["ema50"].shift(5))
        )

        states = np.where(
            bullish,
            "BULLISH",
            np.where(
                bearish,
                "BEARISH",
                "SIDEWAYS"
            )
        )

        current = states[-1]

        if current == "SIDEWAYS":
            return current, 0

        age = 0

        for state in reversed(states):

            if state != current:
                break

            age += 1

        return current, age

    except Exception as exc:

        log(f"Trend age error: {exc}")

        return "UNKNOWN", 0


# ==========================================================
# POSITION ACCOUNTING
# ==========================================================

def unrealized_pnl(position, price):

    if position["side"] == "LONG":

        return (
            price - position["entry"]
        ) * position["quantity"]

    return (
        position["entry"] - price
    ) * position["quantity"]


def equity_value(balance, positions, prices):

    total = balance

    for symbol, position in positions.items():

        total += unrealized_pnl(
            position,
            prices.get(
                symbol,
                position["entry"]
            )
        )

    return total


def margin_used(positions):

    return sum(
        p["margin"]
        for p in positions.values()
    )


def update_excursions(position, candle):

    """
    OHLC-based MFE and MAE.

    These are candle-extreme estimates. Exact intrabar
    ordering cannot be recovered from OHLC data.
    """

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

    position["mfe_price"] = max(
        position["mfe_price"],
        favorable
    )

    position["mae_price"] = max(
        position["mae_price"],
        adverse
    )

    position["mfe_pnl"] = max(
        position["mfe_pnl"],
        favorable_pnl
    )

    position["mae_pnl"] = max(
        position["mae_pnl"],
        adverse_pnl
    )

    risk = position["initial_risk_amount"]

    if risk > 0:

        position["mfe_r"] = (
            position["mfe_pnl"] / risk
        )

        position["mae_r"] = (
            position["mae_pnl"] / risk
        )


def adverse_equity(
    balance,
    positions,
    market_data,
    timestamp,
    prices
):

    total = balance

    for symbol, position in positions.items():

        df = market_data[symbol]["15m"]

        if timestamp in df.index:

            candle = df.loc[timestamp]

            mark = float(
                candle["low"]
                if position["side"] == "LONG"
                else candle["high"]
            )

        else:

            mark = prices.get(
                symbol,
                position["entry"]
            )

        total += unrealized_pnl(
            position,
            mark
        )

    return total


# ==========================================================
# EXIT EXECUTION
# ==========================================================

def check_exit(position, candle):

    """
    Returns:
        executed_exit_price
        exit_reason
        raw_exit_price

    Conservative rule:
    If SL and TP are both touched in the same candle,
    SL is assumed to happen first.
    """

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

            raw_exit = min(
                open_price,
                stop
            )

            executed = (
                raw_exit * (1 - SLIPPAGE_RATE)
            )

            return executed, "STOP_LOSS", raw_exit

        if target_hit:

            raw_exit = max(
                open_price,
                target
            )

            executed = (
                raw_exit * (1 - SLIPPAGE_RATE)
            )

            return executed, "TAKE_PROFIT", raw_exit

    else:

        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit:

            raw_exit = max(
                open_price,
                stop
            )

            executed = (
                raw_exit * (1 + SLIPPAGE_RATE)
            )

            return executed, "STOP_LOSS", raw_exit

        if target_hit:

            raw_exit = min(
                open_price,
                target
            )

            executed = (
                raw_exit * (1 + SLIPPAGE_RATE)
            )

            return executed, "TAKE_PROFIT", raw_exit

    return None, None, None


# ==========================================================
# OPEN POSITION
# ==========================================================

def open_position(
    symbol,
    signal,
    candle,
    timestamp,
    signal_timestamp,
    signal_ema20,
    signal_close,
    trend_age_1h,
    trend_age_4h,
    equity,
    balance,
    positions,
    prices
):

    side = signal.get("signal")

    if side not in ("LONG", "SHORT"):
        return balance

    if equity <= 0:
        return balance

    raw_entry = float(candle["open"])

    if (
        not np.isfinite(raw_entry)
        or raw_entry <= 0
    ):
        return balance

    # Entry slippage.

    entry = (
        raw_entry * (1 + SLIPPAGE_RATE)
        if side == "LONG"
        else raw_entry * (1 - SLIPPAGE_RATE)
    )

    try:

        atr = float(signal["atr"])

    except (KeyError, TypeError, ValueError):

        log(f"Missing ATR: {symbol}")
        return balance

    if not np.isfinite(atr) or atr <= 0:
        return balance

    stop_distance = (
        STOP_ATR_MULTIPLIER * atr
    )

    target_distance = (
        TARGET_ATR_MULTIPLIER * atr
    )

    if side == "LONG":

        stop = entry - stop_distance
        target = entry + target_distance

        estimated_exit = (
            stop * (1 - SLIPPAGE_RATE)
        )

    else:

        stop = entry + stop_distance
        target = entry - target_distance

        estimated_exit = (
            stop * (1 + SLIPPAGE_RATE)
        )

    estimated_cost_per_unit = (
        entry * FEE_RATE
        + estimated_exit * FEE_RATE
    )

    estimated_loss_per_unit = (
        stop_distance
        + estimated_cost_per_unit
    )

    if estimated_loss_per_unit <= 0:
        return balance

    risk_amount = (
        equity * RISK_PER_TRADE
    )

    quantity_by_risk = (
        risk_amount / estimated_loss_per_unit
    )

    max_margin = (
        equity * MAX_MARGIN_FRACTION
    )

    available_margin = max(
        0.0,
        equity - margin_used(positions)
    )

    allowed_margin = min(
        max_margin,
        available_margin
    )

    max_notional = (
        allowed_margin * LEVERAGE
    )

    quantity = min(
        quantity_by_risk,
        max_notional / entry
    )

    if quantity <= 0:
        return balance

    notional = quantity * entry

    margin = (
        notional / LEVERAGE
    )

    entry_fee = (
        notional * FEE_RATE
    )

    if balance - entry_fee < 0:
        return balance

    balance -= entry_fee

    initial_risk_amount = (
        quantity * estimated_loss_per_unit
    )

    distance_ema_atr = (
        abs(entry - signal_ema20) / atr
        if signal_ema20 and atr > 0
        else None
    )

    atr_percent = (
        atr / signal_close * 100
        if signal_close > 0
        else None
    )

    stop_distance_percent = (
        stop_distance / entry * 100
    )

    entry_slippage_cost = (
        abs(entry - raw_entry) * quantity
    )

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

        "entry_rsi": signal.get("rsi"),
        "entry_adx": signal.get("adx"),
        "entry_body_ratio": signal.get("body_ratio"),

        "entry_trend_1h": signal.get("trend_1h"),
        "entry_trend_4h": signal.get("trend_4h"),

        "entry_reason": signal.get("reason"),

        "entry_ema20": signal_ema20,
        "entry_atr": atr,

        "entry_distance_ema20_atr": distance_ema_atr,
        "entry_atr_percent": atr_percent,
        "entry_stop_distance_percent": stop_distance_percent,

        "trend_age_1h_bars": trend_age_1h,
        "trend_age_4h_bars": trend_age_4h,

        "mfe_price": 0.0,
        "mae_price": 0.0,

        "mfe_pnl": 0.0,
        "mae_pnl": 0.0,

        "mfe_r": 0.0,
        "mae_r": 0.0
    }

    prices[symbol] = entry

    log(
        f"OPEN {symbol} {side} "
        f"entry={entry:.8f} "
        f"SL={stop:.8f} "
        f"TP={target:.8f} "
        f"ATR%={atr_percent:.4f} "
        f"EMA-distance-ATR={distance_ema_atr:.3f}"
    )

    return balance


# ==========================================================
# CLOSE POSITION
# ==========================================================

def close_position(
    position,
    exit_price,
    raw_exit_price,
    timestamp,
    reason,
    balance
):

    side = position["side"]

    quantity = position["quantity"]
    entry = position["entry"]
    raw_entry = position["raw_entry"]

    if side == "LONG":

        gross_pnl = (
            exit_price - entry
        ) * quantity

        gross_before_slippage = (
            raw_exit_price - raw_entry
        ) * quantity

    else:

        gross_pnl = (
            entry - exit_price
        ) * quantity

        gross_before_slippage = (
            raw_entry - raw_exit_price
        ) * quantity

    exit_fee = (
        exit_price * quantity * FEE_RATE
    )

    exit_slippage_cost = (
        abs(exit_price - raw_exit_price) * quantity
    )

    balance += gross_pnl - exit_fee

    net_pnl = (
        gross_pnl
        - exit_fee
        - position["entry_fee"]
    )

    total_fees = (
        position["entry_fee"] + exit_fee
    )

    total_slippage = (
        position["entry_slippage_cost"]
        + exit_slippage_cost
    )

    initial_risk = position["initial_risk_amount"]

    net_r = (
        net_pnl / initial_risk
        if initial_risk > 0
        else None
    )

    duration_minutes = (
        pd.Timestamp(timestamp)
        - pd.Timestamp(position["entry_time"])
    ).total_seconds() / 60

    trade = {

        "backtest_version": BACKTEST_VERSION,
        "strategy_version": STRATEGY_VERSION,

        "symbol": position["symbol"],
        "side": side,

        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_time": timestamp.isoformat(),

        "duration_minutes": round(
            duration_minutes, 2
        ),

        "raw_entry": round(raw_entry, 10),
        "entry": round(entry, 10),

        "raw_exit": round(raw_exit_price, 10),
        "exit": round(exit_price, 10),

        "stop_loss": round(
            position["stop_loss"], 10
        ),

        "take_profit": round(
            position["take_profit"], 10
        ),

        "quantity": round(quantity, 10),

        "margin": round(
            position["margin"], 6
        ),

        "initial_risk_amount": round(
            initial_risk, 8
        ),

        "gross_pnl_before_slippage": round(
            gross_before_slippage, 8
        ),

        "gross_pnl_after_slippage": round(
            gross_pnl, 8
        ),

        "entry_fee": round(
            position["entry_fee"], 8
        ),

        "exit_fee": round(
            exit_fee, 8
        ),

        "total_fees": round(
            total_fees, 8
        ),

        "entry_slippage_cost": round(
            position["entry_slippage_cost"], 8
        ),

        "exit_slippage_cost": round(
            exit_slippage_cost, 8
        ),

        "total_slippage_cost": round(
            total_slippage, 8
        ),

        "net_pnl": round(
            net_pnl, 8
        ),

        "net_r": (
            round(net_r, 6)
            if net_r is not None
            else None
        ),

        "entry_rsi": position["entry_rsi"],
        "entry_adx": position["entry_adx"],

        "entry_body_ratio": position["entry_body_ratio"],

        "entry_trend_1h": position["entry_trend_1h"],
        "entry_trend_4h": position["entry_trend_4h"],

        "entry_ema20": position["entry_ema20"],
        "entry_atr": position["entry_atr"],

        "entry_distance_ema20_atr":
            position["entry_distance_ema20_atr"],

        "entry_atr_percent":
            position["entry_atr_percent"],

        "entry_stop_distance_percent":
            position["entry_stop_distance_percent"],

        "trend_age_1h_bars":
            position["trend_age_1h_bars"],

        "trend_age_4h_bars":
            position["trend_age_4h_bars"],

        "mfe_price": round(
            position["mfe_price"], 10
        ),

        "mae_price": round(
            position["mae_price"], 10
        ),

        "mfe_pnl": round(
            position["mfe_pnl"], 8
        ),

        "mae_pnl": round(
            position["mae_pnl"], 8
        ),

        "mfe_r": round(
            position["mfe_r"], 6
        ),

        "mae_r": round(
            position["mae_r"], 6
        ),

        "entry_reason": position["entry_reason"],

        "reason": reason
    }

    return balance, trade


# ==========================================================
# STATISTICS
# ==========================================================

def profit_factor(items):

    profits = sum(
        t["net_pnl"]
        for t in items
        if t["net_pnl"] > 0
    )

    losses = abs(sum(
        t["net_pnl"]
        for t in items
        if t["net_pnl"] < 0
    ))

    if losses == 0:
        return None

    return round(
        profits / losses,
        4
    )


def calculate_streaks(trades):

    ordered = sorted(
        trades,
        key=lambda t: t["exit_time"]
    )

    current = 0
    longest = 0
    current_loss_sum = 0.0
    longest_loss_sum = 0.0

    for trade in ordered:

        if trade["net_pnl"] <= 0:

            current += 1
            current_loss_sum += trade["net_pnl"]

            if current > longest:

                longest = current
                longest_loss_sum = current_loss_sum

        else:

            current = 0
            current_loss_sum = 0.0

    return {
        "longest_losing_streak": longest,
        "loss_during_longest_streak": round(
            longest_loss_sum, 6
        )
    }


def group_statistics(trades, key):

    result = {}

    values = sorted(
        set(
            str(t.get(key))
            for t in trades
            if t.get(key) is not None
        )
    )

    for value in values:

        items = [
            t for t in trades
            if str(t.get(key)) == value
        ]

        wins = [
            t for t in items
            if t["net_pnl"] > 0
        ]

        result[value] = {

            "trades": len(items),

            "wins": len(wins),

            "win_rate_percent": round(
                100 * len(wins) / len(items),
                2
            ) if items else 0,

            "net_pnl": round(
                sum(t["net_pnl"] for t in items),
                6
            ),

            "average_net_r": round(
                float(np.mean([
                    t["net_r"]
                    for t in items
                    if t["net_r"] is not None
                ])),
                6
            ) if any(
                t["net_r"] is not None
                for t in items
            ) else None,

            "profit_factor": profit_factor(items)
        }

    return result


def side_statistics(trades, side):

    items = [
        t for t in trades
        if t["side"] == side
    ]

    winners = [
        t for t in items
        if t["net_pnl"] > 0
    ]

    losers = [
        t for t in items
        if t["net_pnl"] <= 0
    ]

    return {

        "trades": len(items),

        "wins": len(winners),
        "losses": len(losers),

        "win_rate_percent": round(
            100 * len(winners) / len(items),
            2
        ) if items else 0,

        "net_pnl": round(
            sum(t["net_pnl"] for t in items),
            6
        ),

        "gross_pnl_after_slippage": round(
            sum(
                t["gross_pnl_after_slippage"]
                for t in items
            ),
            6
        ),

        "total_fees": round(
            sum(t["total_fees"] for t in items),
            6
        ),

        "total_slippage_cost": round(
            sum(t["total_slippage_cost"] for t in items),
            6
        ),

        "average_win": round(
            float(np.mean([
                t["net_pnl"] for t in winners
            ])),
            6
        ) if winners else None,

        "average_loss": round(
            float(np.mean([
                t["net_pnl"] for t in losers
            ])),
            6
        ) if losers else None,

        "average_net_r": round(
            float(np.mean([
                t["net_r"]
                for t in items
                if t["net_r"] is not None
            ])),
            6
        ) if items else None,

        "profit_factor": profit_factor(items),

        "average_mfe_r": round(
            float(np.mean([
                t["mfe_r"] for t in items
            ])),
            6
        ) if items else None,

        "average_mae_r": round(
            float(np.mean([
                t["mae_r"] for t in items
            ])),
            6
        ) if items else None
    }


def summarize_trades(trades):

    winners = [
        t for t in trades
        if t["net_pnl"] > 0
    ]

    losers = [
        t for t in trades
        if t["net_pnl"] <= 0
    ]

    return {

        "closed_trades": len(trades),

        "wins": len(winners),
        "losses": len(losers),

        "win_rate_percent": round(
            100 * len(winners) / len(trades),
            2
        ) if trades else 0,

        "profit_factor": profit_factor(trades),

        "net_pnl": round(
            sum(t["net_pnl"] for t in trades),
            6
        ),

        "total_fees": round(
            sum(t["total_fees"] for t in trades),
            6
        ),

        "total_slippage_cost": round(
            sum(t["total_slippage_cost"] for t in trades),
            6
        ),

        "gross_pnl_before_slippage": round(
            sum(
                t["gross_pnl_before_slippage"]
                for t in trades
            ),
            6
        ),

        "gross_pnl_after_slippage": round(
            sum(
                t["gross_pnl_after_slippage"]
                for t in trades
            ),
            6
        ),

        "long": side_statistics(trades, "LONG"),

        "short": side_statistics(trades, "SHORT"),

        "by_symbol": group_statistics(
            trades,
            "symbol"
        ),

        "by_exit_reason": group_statistics(
            trades,
            "reason"
        ),

        "by_month": group_statistics(
            trades,
            "exit_month"
        ),

        **calculate_streaks(trades)
    }


# ==========================================================
# MAIN BACKTEST
# ==========================================================

def run_backtest(market_data_override=None, scenario_name="A_BASELINE"):

    if STRATEGY_VERSION != "2.3":

        raise RuntimeError(
            "strategy.py must remain version 2.3 "
            "for this diagnostic test."
        )

    market_data = (
        market_data_override
        if market_data_override is not None
        else prepare_market_data()
    )

    if not market_data:

        raise RuntimeError(
            "No market data downloaded from XT."
        )

    timeline = sorted(
        set().union(*[
            set(data["15m"].index[1:])
            for data in market_data.values()
        ])
    )

    if not timeline:

        raise RuntimeError(
            "No 15m timeline available."
        )

    balance = INITIAL_BALANCE

    positions = {}
    prices = {}
    cooldown_until = {}
    trades = []

    peak_equity = INITIAL_BALANCE
    max_drawdown = 0.0

    log(
        f"Starting diagnostic V{BACKTEST_VERSION}: "
        f"{len(timeline)} timestamps, "
        f"{len(market_data)} markets"
    )

    for timestamp in timeline:

        # --------------------------------------
        # 1. Update current candle opening prices
        # --------------------------------------

        for symbol, data in market_data.items():

            df = data["15m"]

            if timestamp in df.index:

                prices[symbol] = float(
                    df.loc[timestamp, "open"]
                )

        # --------------------------------------
        # 2. Update and exit existing positions
        # --------------------------------------

        for symbol in list(positions.keys()):

            df = market_data[symbol]["15m"]

            if timestamp not in df.index:
                continue

            candle = df.loc[timestamp]

            update_excursions(
                positions[symbol],
                candle
            )

            exit_price, reason, raw_exit = check_exit(
                positions[symbol],
                candle
            )

            if exit_price is not None:

                position = positions.pop(symbol)

                balance, trade = close_position(
                    position,
                    exit_price,
                    raw_exit,
                    timestamp,
                    reason,
                    balance
                )

                trade["exit_month"] = (
                    pd.Timestamp(timestamp).strftime("%Y-%m")
                )

                trades.append(trade)

                if scenario_name == "B_LOSS_COOLDOWN" and trade["net_pnl"] <= 0:
                    cooldown_until[symbol] = timestamp + pd.Timedelta(hours=COOLDOWN_HOURS)

                log(
                    f"CLOSE {symbol} {reason} "
                    f"net={trade['net_pnl']:.5f} "
                    f"MFE_R={trade['mfe_r']:.3f} "
                    f"MAE_R={trade['mae_r']:.3f}"
                )

        # --------------------------------------
        # 3. Calculate adverse drawdown
        # --------------------------------------

        adverse = adverse_equity(
            balance,
            positions,
            market_data,
            timestamp,
            prices
        )

        current_equity = equity_value(
            balance,
            positions,
            prices
        )

        peak_equity = max(
            peak_equity,
            current_equity
        )

        if peak_equity > 0:

            max_drawdown = max(
                max_drawdown,
                (peak_equity - adverse)
                / peak_equity * 100
            )

        # --------------------------------------
        # 4. Generate signals from closed candles
        # --------------------------------------

        if len(positions) < MAX_OPEN_POSITIONS:

            for symbol, data in market_data.items():

                if len(positions) >= MAX_OPEN_POSITIONS:
                    break

                if symbol in positions:
                    continue

                if (
                    scenario_name == "B_LOSS_COOLDOWN"
                    and timestamp < cooldown_until.get(symbol, pd.Timestamp.min.tz_localize("UTC"))
                ):
                    continue

                df15 = data["15m"]
                df1 = data["1h"]
                df4 = data["4h"]

                if timestamp not in df15.index:
                    continue

                idx = df15.index.get_loc(timestamp)

                if (
                    not isinstance(idx, (int, np.integer))
                    or idx < 1
                ):
                    continue

                # Only previously closed 15m candles.

                history15 = df15.iloc[
                    max(0, idx - 350):idx
                ]

                # Only completed 1h and 4h candles.

                end1 = df1.index.searchsorted(
                    timestamp,
                    side="right"
                )

                end4 = df4.index.searchsorted(
                    timestamp,
                    side="right"
                )

                history1 = df1.iloc[
                    max(0, end1 - 260):end1
                ]

                history4 = df4.iloc[
                    max(0, end4 - 260):end4
                ]

                if min(
                    len(history15),
                    len(history1),
                    len(history4)
                ) < 210:
                    continue

                try:

                    signal = generate_signal(
                        history15,
                        history1,
                        history4,
                        symbol
                    )

                except Exception as exc:

                    log(
                        f"Signal error {symbol}: {exc}"
                    )

                    continue

                if signal.get("signal") not in (
                    "LONG",
                    "SHORT"
                ):
                    continue

                # Experimental SHORT-only filters. LONG entries are
                # deliberately unchanged in every scenario.
                if signal.get("signal") == "SHORT":
                    signal_rsi = float(signal.get("rsi", np.nan))
                    signal_adx = float(signal.get("adx", np.nan))

                    if (
                        scenario_name == "D1_SHORT_RSI_MAX45"
                        and signal_rsi > 45
                    ):
                        continue

                    if (
                        scenario_name == "D2_SHORT_ADX_MAX20"
                        and signal_adx > 20
                    ):
                        continue

                    if (
                        scenario_name == "D3_SHORT_RSI45_ADX20"
                        and (signal_rsi > 45 or signal_adx > 20)
                    ):
                        continue

                # All scenarios retain the C direction cap.
                if scenario_name in (
                    "C_DIRECTION_CAP",
                    "D1_SHORT_RSI_MAX45",
                    "D2_SHORT_ADX_MAX20",
                    "D3_SHORT_RSI45_ADX20"
                ):
                    same_direction_open = sum(
                        1 for p in positions.values()
                        if p["side"] == signal["signal"]
                    )
                    if same_direction_open >= MAX_SAME_DIRECTION_POSITIONS:
                        continue

                # Diagnostic values from the last closed
                # 15m signal candle.

                signal_data = prepare_data(history15)

                signal_candle = signal_data.iloc[-1]

                signal_ema20 = float(
                    signal_candle["ema20"]
                )

                signal_close = float(
                    signal_candle["close"]
                )

                signal_timestamp = history15.index[-1]

                trend1, age1 = calculate_trend_age(
                    history1
                )

                trend4, age4 = calculate_trend_age(
                    history4
                )

                equity = equity_value(
                    balance,
                    positions,
                    prices
                )

                balance = open_position(
                    symbol,
                    signal,
                    df15.loc[timestamp],
                    timestamp,
                    signal_timestamp,
                    signal_ema20,
                    signal_close,
                    age1,
                    age4,
                    equity,
                    balance,
                    positions,
                    prices
                )

                # ----------------------------------
                # 5. IMPORTANT: Check the entry candle
                # ----------------------------------

                # OHLC cannot reveal exact intrabar order.
                # If SL and TP are both touched, SL wins.

                if symbol in positions:

                    entry_candle = df15.loc[timestamp]

                    update_excursions(
                        positions[symbol],
                        entry_candle
                    )

                    exit_price, reason, raw_exit = check_exit(
                        positions[symbol],
                        entry_candle
                    )

                    if exit_price is not None:

                        position = positions.pop(symbol)

                        balance, trade = close_position(
                            position,
                            exit_price,
                            raw_exit,
                            timestamp,
                            reason,
                            balance
                        )

                        trade["exit_month"] = (
                            pd.Timestamp(timestamp).strftime("%Y-%m")
                        )

                        trades.append(trade)

                        if scenario_name == "B_LOSS_COOLDOWN" and trade["net_pnl"] <= 0:
                            cooldown_until[symbol] = timestamp + pd.Timedelta(hours=COOLDOWN_HOURS)

                        log(
                            f"SAME-CANDLE CLOSE {symbol} "
                            f"{reason} "
                            f"net={trade['net_pnl']:.5f}"
                        )

        # --------------------------------------
        # 6. Update closing prices and drawdown
        # --------------------------------------

        for symbol, data in market_data.items():

            df = data["15m"]

            if timestamp in df.index:

                prices[symbol] = float(
                    df.loc[timestamp, "close"]
                )

        equity = equity_value(
            balance,
            positions,
            prices
        )

        peak_equity = max(
            peak_equity,
            equity
        )

        if peak_equity > 0:

            max_drawdown = max(
                max_drawdown,
                (peak_equity - equity)
                / peak_equity * 100
            )

    # ------------------------------------------
    # 7. Close remaining positions at test end
    # ------------------------------------------

    final_timestamp = (
        timeline[-1]
        + pd.Timedelta(minutes=15)
    )

    for symbol in list(positions.keys()):

        position = positions.pop(symbol)

        mark = prices.get(
            symbol,
            position["entry"]
        )

        exit_price = (
            mark * (1 - SLIPPAGE_RATE)
            if position["side"] == "LONG"
            else mark * (1 + SLIPPAGE_RATE)
        )

        balance, trade = close_position(
            position,
            exit_price,
            mark,
            final_timestamp,
            "END_OF_TEST",
            balance
        )

        trade["exit_month"] = (
            pd.Timestamp(final_timestamp).strftime("%Y-%m")
        )

        trades.append(trade)

    # ------------------------------------------
    # 8. Final report
    # ------------------------------------------

    final_equity = balance

    stats = summarize_trades(trades)

    report = {

        "backtest_version": BACKTEST_VERSION,
        "strategy_version": STRATEGY_VERSION,

        "mode": "BACKTEST_ONLY",

        "exchange": "XT",

        "symbols_tested": list(
            market_data.keys()
        ),

        "timeframe": "15m",

        "confirmation_timeframes": [
            "1h",
            "4h"
        ],

        "data_start": min(
            data["15m"].index[0]
            for data in market_data.values()
        ).isoformat(),

        "data_end": max(
            data["15m"].index[-1]
            for data in market_data.values()
        ).isoformat(),

        "first_signal_or_trade_time": (
            min(
                t["entry_time"]
                for t in trades
            )
            if trades
            else None
        ),

        "initial_balance": INITIAL_BALANCE,

        "final_balance": round(
            balance, 6
        ),

        "final_equity": round(
            final_equity, 6
        ),

        "net_profit": round(
            final_equity - INITIAL_BALANCE,
            6
        ),

        "return_percent": round(
            (final_equity / INITIAL_BALANCE - 1)
            * 100,
            3
        ),

        "max_drawdown_percent": round(
            max_drawdown, 3
        ),

        "leverage": LEVERAGE,

        "risk_per_trade": RISK_PER_TRADE,

        "max_margin_fraction_per_position":
            MAX_MARGIN_FRACTION,

        "max_open_positions": MAX_OPEN_POSITIONS,

        "fee_rate_per_side": FEE_RATE,

        "slippage_rate_per_side": SLIPPAGE_RATE,

        "stop_atr_multiplier":
            STOP_ATR_MULTIPLIER,

        "target_atr_multiplier":
            TARGET_ATR_MULTIPLIER,

        "funding_included": False,

        "liquidation_model_included": False,

        "mfe_mae_method": (
            "OHLC candle extremes; exact intrabar "
            "sequence is unknown"
        ),

        "same_candle_exit_check": True,

        "same_candle_sl_tp_rule": "STOP_LOSS_FIRST",

        "trend_age_method": (
            "Consecutive bars matching strategy.py "
            "EMA200 and EMA50 trend definition"
        ),

        "end_positions_closed": True,

        **stats,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report["experiment_scenario"] = scenario_name
    report["cooldown_hours_after_loss"] = (
        COOLDOWN_HOURS if scenario_name == "B_LOSS_COOLDOWN" else 0
    )
    report["max_same_direction_positions"] = (
        MAX_SAME_DIRECTION_POSITIONS
        if scenario_name in (
            "C_DIRECTION_CAP",
            "D1_SHORT_RSI_MAX45",
            "D2_SHORT_ADX_MAX20",
            "D3_SHORT_RSI45_ADX20"
        )
        else None
    )
    report["short_filter"] = {
        "D1_SHORT_RSI_MAX45": "SHORT RSI <= 45; LONG unchanged",
        "D2_SHORT_ADX_MAX20": "SHORT ADX <= 20; LONG unchanged",
        "D3_SHORT_RSI45_ADX20": "SHORT RSI <= 45 and ADX <= 20; LONG unchanged"
    }.get(scenario_name, "None; C direction-cap control")

    # ------------------------------------------
    # 9. Save new independent report files
    # ------------------------------------------

    report_file = f"backtest_report_{scenario_name.lower()}.json"
    trades_file = f"backtest_trades_{scenario_name.lower()}.csv"

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    pd.DataFrame(trades).to_csv(
        trades_file,
        index=False
    )

    print("\n========== BACKTEST V2.3.1 ==========")

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    print("=====================================")

    log(
        f"Saved {report_file} "
        f"and {trades_file}"
    )

    return report


if __name__ == "__main__":

    shared_market_data = prepare_market_data()

    if not shared_market_data:
        raise RuntimeError("No market data downloaded from XT.")

    # C is the control. D1-D3 test SHORT filters while
    # retaining the same two-position direction cap.
    scenarios = [
        "C_DIRECTION_CAP",
        "D1_SHORT_RSI_MAX45",
        "D2_SHORT_ADX_MAX20",
        "D3_SHORT_RSI45_ADX20"
    ]

    comparison = {}

    for scenario in scenarios:
        log(f"\n========== RUNNING {scenario} ==========")
        report = run_backtest(
            market_data_override=shared_market_data,
            scenario_name=scenario
        )
        comparison[scenario] = {
            "final_balance": report["final_balance"],
            "net_profit": report["net_profit"],
            "return_percent": report["return_percent"],
            "max_drawdown_percent": report["max_drawdown_percent"],
            "closed_trades": report["closed_trades"],
            "win_rate_percent": report["win_rate_percent"],
            "profit_factor": report["profit_factor"],
            "long_net_pnl": report["long"]["net_pnl"],
            "short_net_pnl": report["short"]["net_pnl"]
        }

    with open("backtest_short_filter_comparison.json", "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    pd.DataFrame(comparison).T.to_csv(
        "backtest_short_filter_comparison.csv",
        index=True
    )

    print("\n========== SHORT FILTER EXPERIMENT COMPARISON ==========")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("===========================================")
