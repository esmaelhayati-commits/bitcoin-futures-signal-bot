import ccxt
import pandas as pd
import numpy as np
import json
import time

from datetime import datetime, timezone

from strategy import generate_signal


# ==========================================
# FUTURES BOT V2.1 - BACKTEST ENGINE
# Exchange: XT
# Initial equity: 10 USDT
# Leverage: 20x
# ==========================================

INITIAL_BALANCE = 10.0
LEVERAGE = 20

RISK_PER_TRADE = 0.01
MAX_MARGIN_FRACTION = 0.20
MAX_OPEN_POSITIONS = 3

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

LOOKBACK_DAYS = 90
CANDLE_LIMIT = 500

REPORT_FILE = "backtest_report.json"
TRADES_FILE = "backtest_trades.csv"

SYMBOLS = [
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
        f"{message}"
    )


def resolve_swap_symbol(requested, markets):
    base, quote = requested.split("/")

    for market in markets.values():
        if (
            market.get("base") == base
            and market.get("quote") == quote
            and market.get("swap")
            and market.get("active", True)
        ):
            return market["symbol"]

    return None


def fetch_history(symbol):

    since = exchange.milliseconds() - (
        LOOKBACK_DAYS * 24 * 60 * 60 * 1000
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

        last_timestamp = batch[-1][0]

        next_since = last_timestamp + 15 * 60 * 1000

        if next_since <= since:
            break

        since = next_since

        if len(batch) < CANDLE_LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)

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

    df = df.drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp")

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df = df.set_index("datetime")

    df = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    # Remove currently forming 15m candles.
    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df.index + pd.Timedelta(minutes=15) <= now
    ]

    if len(df) < 1000:
        log(f"Insufficient history for {symbol}")
        return None

    return df


def resample_ohlcv(df, timeframe):

    result = df.resample(
        timeframe,
        label="right",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    return result.dropna()


def equity_value(balance, positions, prices):

    total = balance

    for position in positions.values():

        price = prices.get(
            position["symbol"],
            position["entry"]
        )

        if position["side"] == "LONG":
            pnl = (
                price - position["entry"]
            ) * position["quantity"]

        else:
            pnl = (
                position["entry"] - price
            ) * position["quantity"]

        total += pnl

    return total


def margin_used(positions):

    return sum(
        position["margin"]
        for position in positions.values()
    )


def close_position(
    position,
    exit_price,
    timestamp,
    reason,
    balance
):

    side = position["side"]
    quantity = position["quantity"]
    entry = position["entry"]

    if side == "LONG":
        gross_pnl = (
            exit_price - entry
        ) * quantity

    else:
        gross_pnl = (
            entry - exit_price
        ) * quantity

    exit_fee = (
        exit_price * quantity * FEE_RATE
    )

    # Entry fee was deducted at entry.
    balance += gross_pnl - exit_fee

    net_pnl = (
        gross_pnl
        - exit_fee
        - position["entry_fee"]
    )

    trade = {
        "symbol": position["symbol"],
        "side": side,

        "entry_time": position["entry_time"],
        "exit_time": timestamp.isoformat(),

        "entry": round(entry, 10),
        "exit": round(exit_price, 10),

        "stop_loss": position["stop_loss"],
        "take_profit": position["take_profit"],

        "quantity": round(quantity, 10),
        "margin": round(position["margin"], 6),

        "gross_pnl": round(gross_pnl, 6),
        "entry_fee": round(position["entry_fee"], 6),
        "exit_fee": round(exit_fee, 6),

        "net_pnl": round(net_pnl, 6),
        "reason": reason
    }

    return balance, trade


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

        # Conservative assumption if both are touched.
        if stop_hit:
            raw_exit = min(open_price, stop)

            return (
                raw_exit * (1 - SLIPPAGE_RATE),
                "STOP_LOSS"
            )

        if target_hit:
            raw_exit = max(open_price, target)

            return (
                raw_exit * (1 - SLIPPAGE_RATE),
                "TAKE_PROFIT"
            )

    else:

        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit:
            raw_exit = max(open_price, stop)

            return (
                raw_exit * (1 + SLIPPAGE_RATE),
                "STOP_LOSS"
            )

        if target_hit:
            raw_exit = min(open_price, target)

            return (
                raw_exit * (1 + SLIPPAGE_RATE),
                "TAKE_PROFIT"
            )

    return None, None


def open_position(
    symbol,
    signal,
    candle,
    timestamp,
    equity,
    balance,
    positions,
    prices
):

    side = signal.get("signal")

    if side not in ("LONG", "SHORT"):
        return balance

    raw_entry = float(candle["open"])

    if not np.isfinite(raw_entry) or raw_entry <= 0:
        return balance

    if side == "LONG":
        entry = raw_entry * (1 + SLIPPAGE_RATE)
    else:
        entry = raw_entry * (1 - SLIPPAGE_RATE)

    try:
        stop = float(signal["stop_loss"])
        target = float(signal["take_profit"])

    except (KeyError, TypeError, ValueError):
        log(f"Invalid stop/target: {symbol}")
        return balance

    if not np.isfinite(stop) or not np.isfinite(target):
        return balance

    if side == "LONG":
        if not (stop < entry < target):
            log(f"Invalid LONG levels: {symbol}")
            return balance

        stop_distance = entry - stop

    else:
        if not (target < entry < stop):
            log(f"Invalid SHORT levels: {symbol}")
            return balance

        stop_distance = stop - entry

    if stop_distance <= 0 or equity <= 0:
        return balance

    # Risk-based position sizing.
    risk_amount = equity * RISK_PER_TRADE

    quantity = risk_amount / stop_distance

    max_margin = equity * MAX_MARGIN_FRACTION

    available_margin = max(
        0,
        equity - margin_used(positions)
    )

    allowed_margin = min(
        max_margin,
        available_margin
    )

    max_notional = allowed_margin * LEVERAGE

    quantity = min(
        quantity,
        max_notional / entry
    )

    if quantity <= 0:
        return balance

    notional = quantity * entry
    margin = notional / LEVERAGE

    entry_fee = notional * FEE_RATE

    if balance - entry_fee < 0:
        return balance

    balance -= entry_fee

    positions[symbol] = {
        "symbol": symbol,
        "side": side,

        "entry": entry,
        "quantity": quantity,

        "stop_loss": stop,
        "take_profit": target,

        "margin": margin,
        "entry_fee": entry_fee,

        "entry_time": timestamp.isoformat()
    }

    prices[symbol] = entry

    log(
        f"OPEN {symbol} {side} "
        f"entry={entry:.8f} "
        f"SL={stop:.8f} "
        f"TP={target:.8f} "
        f"margin={margin:.4f}"
    )

    return balance


def prepare_market_data():

    log("Loading XT markets")

    exchange.load_markets()

    market_data = {}

    for requested in SYMBOLS:

        symbol = resolve_swap_symbol(
            requested,
            exchange.markets
        )

        if not symbol:
            log(f"Swap market not found: {requested}")
            continue

        df = fetch_history(symbol)

        if df is None:
            continue

        df_1h = resample_ohlcv(df, "1h")
        df_4h = resample_ohlcv(df, "4h")

        market_data[symbol] = {
            "15m": df,
            "1h": df_1h,
            "4h": df_4h
        }

        log(
            f"READY {symbol}: "
            f"15m={len(df)}, "
            f"1h={len(df_1h)}, "
            f"4h={len(df_4h)}"
        )

    return market_data


def run_backtest():

    market_data = prepare_market_data()

    if not market_data:
        raise RuntimeError("No market data downloaded")

    balance = INITIAL_BALANCE

    positions = {}
    prices = {}
    trades = []

    timeline = sorted(set().union(*[
        set(data["15m"].index[1:])
        for data in market_data.values()
    ]))

    log(
        f"Starting portfolio backtest: "
        f"{len(timeline)} timestamps"
    )

    peak_equity = INITIAL_BALANCE
    max_drawdown = 0.0

    for timestamp in timeline:

        # ----------------------------------
        # 1. Mark all markets at candle OPEN.
        # ----------------------------------

        for symbol, data in market_data.items():

            df = data["15m"]

            if timestamp in df.index:
                prices[symbol] = float(
                    df.loc[timestamp, "open"]
                )

        # ----------------------------------
        # 2. Manage existing positions.
        # ----------------------------------

        for symbol in list(positions.keys()):

            df = market_data[symbol]["15m"]

            if timestamp not in df.index:
                continue

            candle = df.loc[timestamp]

            exit_price, reason = check_exit(
                positions[symbol],
                candle
            )

            if exit_price is not None:

                balance, trade = close_position(
                    positions.pop(symbol),
                    exit_price,
                    timestamp,
                    reason,
                    balance
                )

                trades.append(trade)

                log(
                    f"CLOSE {symbol} "
                    f"{reason} "
                    f"net={trade['net_pnl']:.4f}"
                )

        # ----------------------------------
        # 3. Calculate equity using OPEN prices.
        # No current candle close is used here.
        # ----------------------------------

        equity = equity_value(
            balance,
            positions,
            prices
        )

        # ----------------------------------
        # 4. Find new signals.
        # ----------------------------------

        if len(positions) < MAX_OPEN_POSITIONS:

            for symbol, data in market_data.items():

                if len(positions) >= MAX_OPEN_POSITIONS:
                    break

                if symbol in positions:
                    continue

                df15 = data["15m"]

                if timestamp not in df15.index:
                    continue

                index = df15.index.get_loc(timestamp)

                if not isinstance(index, (int, np.integer)):
                    continue

                # Only previously completed 15m candles.
                history15 = df15.iloc[
                    max(0, index - 300):index
                ]

                # Higher timeframe candles must be closed.
                df1 = data["1h"]
                df4 = data["4h"]

                end1 = df1.index.searchsorted(
                    timestamp,
                    side="right"
                )

                end4 = df4.index.searchsorted(
                    timestamp,
                    side="right"
                )

                history1 = df1.iloc[
                    max(0, end1 - 250):end1
                ]

                history4 = df4.iloc[
                    max(0, end4 - 250):end4
                ]

                if (
                    len(history15) < 200
                    or len(history1) < 205
                    or len(history4) < 205
                ):
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

                # Recalculate equity at OPEN prices.
                equity = equity_value(
                    balance,
                    positions,
                    prices
                )

                balance = open_position(
                    symbol,
                    signal,
                    df15.iloc[index],
                    timestamp,
                    equity,
                    balance,
                    positions,
                    prices
                )

        # ----------------------------------
        # 5. Update prices to candle CLOSE.
        # Only after entry decisions are complete.
        # ----------------------------------

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

            drawdown = (
                (peak_equity - equity) / peak_equity
            ) * 100

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

    # ----------------------------------
    # 6. Final portfolio valuation.
    # ----------------------------------

    final_equity = equity_value(
        balance,
        positions,
        prices
    )

    closed = trades

    winners = [
        trade for trade in closed
        if trade["net_pnl"] > 0
    ]

    losers = [
        trade for trade in closed
        if trade["net_pnl"] <= 0
    ]

    gross_profit = sum(
        trade["net_pnl"]
        for trade in winners
    )

    gross_loss = abs(sum(
        trade["net_pnl"]
        for trade in losers
    ))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    win_rate = (
        len(winners) / len(closed) * 100
        if closed else 0
    )

    long_trades = [
        trade for trade in closed
        if trade["side"] == "LONG"
    ]

    short_trades = [
        trade for trade in closed
        if trade["side"] == "SHORT"
    ]

    def side_summary(items):

        return {
            "trades": len(items),

            "net_pnl": round(
                sum(
                    trade["net_pnl"]
                    for trade in items
                ),
                4
            ),

            "wins": sum(
                trade["net_pnl"] > 0
                for trade in items
            )
        }

    open_summary = []

    for symbol, position in positions.items():

        mark = prices.get(
            symbol,
            position["entry"]
        )

        if position["side"] == "LONG":

            unrealized = (
                mark - position["entry"]
            ) * position["quantity"]

        else:

            unrealized = (
                position["entry"] - mark
            ) * position["quantity"]

        open_summary.append({
            "symbol": symbol,
            "side": position["side"],

            "entry": position["entry"],
            "mark_price": mark,

            "unrealized_pnl": round(
                unrealized,
                6
            ),

            "margin": round(
                position["margin"],
                6
            )
        })

    report = {

        "version": "2.1",
        "exchange": "XT",

        "timeframe": "15m",
        "confirmation_timeframes": [
            "1h",
            "4h"
        ],

        "initial_balance": INITIAL_BALANCE,
        "final_balance": round(balance, 6),
        "final_equity": round(final_equity, 6),

        "net_profit": round(
            final_equity - INITIAL_BALANCE,
            6
        ),

        "return_percent": round(
            (final_equity / INITIAL_BALANCE - 1) * 100,
            3
        ),

        "closed_trades": len(closed),
        "wins": len(winners),
        "losses": len(losers),

        "win_rate": round(win_rate, 2),

        "profit_factor": (
            round(profit_factor, 4)
            if profit_factor is not None
            else None
        ),

        "max_drawdown_percent": round(
            max_drawdown,
            3
        ),

        "long": side_summary(long_trades),
        "short": side_summary(short_trades),

        "open_positions": open_summary,

        "leverage": LEVERAGE,

        "risk_per_trade": RISK_PER_TRADE,
        "max_margin_fraction": MAX_MARGIN_FRACTION,

        "fee_rate_per_side": FEE_RATE,
        "slippage_rate_per_side": SLIPPAGE_RATE,

        "funding_included": False,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    pd.DataFrame(closed).to_csv(
        TRADES_FILE,
        index=False
    )

    print("\n========== BACKTEST V2.1 ==========")

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ))

    print("===================================")

    return report


if __name__ == "__main__":
    run_backtest()
