
import ccxt
import pandas as pd
import numpy as np
import json
import time

from datetime import datetime, timezone
from strategy import (
    generate_signal,
    STRATEGY_VERSION,
    STOP_ATR_MULTIPLIER,
    TARGET_ATR_MULTIPLIER
)

# ==================================================
# BITCOIN FUTURES BOT V2.3
# BACKTEST ONLY - NO REAL ORDERS
# Exchange: XT
# Entry: 15m
# Confirmation: 1h + 4h
# ==================================================

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

REPORT_FILE = "backtest_report_v23.json"
TRADES_FILE = "backtest_trades_v23.csv"

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

# V2.2 reference values from the saved historical report.
# These values are NEVER used in the trading simulation.

V22_REFERENCE = {
    "final_equity": 2.782705,
    "return_percent": -72.173,
    "max_drawdown_percent": 82.059,
    "closed_trades": 529,
    "win_rate_percent": 34.22,
    "profit_factor": 0.7423,
    "long_net_pnl": -2.829383,
    "short_net_pnl": -4.387924
}

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
            and market.get("settle") in (quote, None)
        ):
            return market["symbol"]

    return None


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

    # Keep only complete higher-timeframe candles.

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


def margin_used(positions):

    return sum(
        p["margin"]
        for p in positions.values()
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

    balance += gross_pnl - exit_fee

    net_pnl = (
        gross_pnl
        - exit_fee
        - position["entry_fee"]
    )

    trade = {

        "strategy_version": STRATEGY_VERSION,

        "symbol": position["symbol"],
        "side": side,

        "entry_time": position["entry_time"],
        "exit_time": timestamp.isoformat(),

        "entry": round(entry, 10),
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

        "gross_pnl": round(
            gross_pnl, 6
        ),

        "entry_fee": round(
            position["entry_fee"], 6
        ),

        "exit_fee": round(
            exit_fee, 6
        ),

        "net_pnl": round(
            net_pnl, 6
        ),

        "entry_rsi": position.get("entry_rsi"),
        "entry_adx": position.get("entry_adx"),

        "entry_body_ratio": position.get(
            "entry_body_ratio"
        ),

        "entry_trend_1h": position.get(
            "entry_trend_1h"
        ),

        "entry_trend_4h": position.get(
            "entry_trend_4h"
        ),

        "entry_reason": position.get(
            "entry_reason"
        ),

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

        # Conservative rule:
        # If both levels hit in one candle,
        # assume stop loss happens first.

        if stop_hit:

            raw_exit = min(
                open_price,
                stop
            )

            return (
                raw_exit * (1 - SLIPPAGE_RATE),
                "STOP_LOSS"
            )

        if target_hit:

            raw_exit = max(
                open_price,
                target
            )

            return (
                raw_exit * (1 - SLIPPAGE_RATE),
                "TAKE_PROFIT"
            )

    else:

        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit:

            raw_exit = max(
                open_price,
                stop
            )

            return (
                raw_exit * (1 + SLIPPAGE_RATE),
                "STOP_LOSS"
            )

        if target_hit:

            raw_exit = min(
                open_price,
                target
            )

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

    if equity <= 0:
        return balance

    raw_entry = float(candle["open"])

    if (
        not np.isfinite(raw_entry)
        or raw_entry <= 0
    ):
        return balance

    # Simulated market execution with slippage.

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

    # Include both entry and estimated exit fees
    # in the risk calculation.

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

    positions[symbol] = {

        "symbol": symbol,
        "side": side,

        "entry": entry,
        "quantity": quantity,

        "stop_loss": stop,
        "take_profit": target,

        "margin": margin,
        "entry_fee": entry_fee,

        "entry_time": timestamp.isoformat(),

        "entry_rsi": signal.get("rsi"),
        "entry_adx": signal.get("adx"),

        "entry_body_ratio": signal.get(
            "body_ratio"
        ),

        "entry_trend_1h": signal.get(
            "trend_1h"
        ),

        "entry_trend_4h": signal.get(
            "trend_4h"
        ),

        "entry_reason": signal.get(
            "reason"
        )
    }

    prices[symbol] = entry

    log(
        f"OPEN {symbol} {side} "
        f"entry={entry:.8f} "
        f"SL={stop:.8f} "
        f"TP={target:.8f} "
        f"qty={quantity:.8f} "
        f"margin={margin:.4f}"
    )

    return balance


def profit_factor(items):

    profits = sum(
        t["net_pnl"]
        for t in items
        if t["net_pnl"] > 0
    )

    losses = abs(sum(
        t["net_pnl"]
        for t in items
        if t["net_pnl"] <= 0
    ))

    if losses == 0:
        return None

    return round(
        profits / losses,
        4
    )


def summarize_trades(trades):

    winners = [
        t for t in trades
        if t["net_pnl"] > 0
    ]

    losers = [
        t for t in trades
        if t["net_pnl"] <= 0
    ]

    def side_stats(side):

        items = [
            t for t in trades
            if t["side"] == side
        ]

        return {

            "trades": len(items),

            "wins": sum(
                t["net_pnl"] > 0
                for t in items
            ),

            "losses": sum(
                t["net_pnl"] <= 0
                for t in items
            ),

            "net_pnl": round(
                sum(t["net_pnl"] for t in items),
                6
            ),

            "profit_factor": profit_factor(items)
        }

    by_symbol = {}

    for symbol in sorted(
        set(t["symbol"] for t in trades)
    ):

        items = [
            t for t in trades
            if t["symbol"] == symbol
        ]

        by_symbol[symbol] = {

            "trades": len(items),

            "wins": sum(
                t["net_pnl"] > 0
                for t in items
            ),

            "net_pnl": round(
                sum(t["net_pnl"] for t in items),
                6
            ),

            "profit_factor": profit_factor(items)
        }

    by_exit_reason = {}

    for reason in sorted(
        set(t["reason"] for t in trades)
    ):

        items = [
            t for t in trades
            if t["reason"] == reason
        ]

        by_exit_reason[reason] = {

            "trades": len(items),

            "net_pnl": round(
                sum(t["net_pnl"] for t in items),
                6
            )
        }

    gross_profit = sum(
        t["net_pnl"] for t in winners
    )

    gross_loss = abs(sum(
        t["net_pnl"] for t in losers
    ))

    return {

        "closed_trades": len(trades),

        "wins": len(winners),
        "losses": len(losers),

        "win_rate_percent": round(
            100 * len(winners) / len(trades),
            2
        ) if trades else 0,

        "profit_factor": (
            round(gross_profit / gross_loss, 4)
            if gross_loss else None
        ),

        "gross_profit_net": round(
            gross_profit, 6
        ),

        "gross_loss_net": round(
            gross_loss, 6
        ),

        "long": side_stats("LONG"),
        "short": side_stats("SHORT"),

        "by_symbol": by_symbol,

        "by_exit_reason": by_exit_reason
    }


def compare_with_v22(report):

    comparison = {}

    current_values = {

        "final_equity": report["final_equity"],
        "return_percent": report["return_percent"],
        "max_drawdown_percent": report[
            "max_drawdown_percent"
        ],
        "closed_trades": report["closed_trades"],
        "win_rate_percent": report["win_rate_percent"],
        "profit_factor": report["profit_factor"],
        "long_net_pnl": report["long"]["net_pnl"],
        "short_net_pnl": report["short"]["net_pnl"]
    }

    for key, current in current_values.items():

        reference = V22_REFERENCE[key]

        if current is None:

            difference = None

        else:

            difference = round(
                current - reference,
                4
            )

        comparison[key] = {

            "v22_reference": reference,
            "v23_current": current,
            "difference": difference
        }

    return comparison


def run_backtest():

    if STRATEGY_VERSION != "2.3":

        raise RuntimeError(
            "strategy.py is not version 2.3"
        )

    market_data = prepare_market_data()

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
    trades = []

    peak_equity = INITIAL_BALANCE
    max_drawdown = 0.0

    log(
        f"Starting V2.3 backtest: "
        f"{len(timeline)} timestamps, "
        f"{len(market_data)} markets"
    )

    for timestamp in timeline:

        # ----------------------------------
        # 1. Update opening prices
        # ----------------------------------

        for symbol, data in market_data.items():

            df = data["15m"]

            if timestamp in df.index:

                prices[symbol] = float(
                    df.loc[timestamp, "open"]
                )

        # ----------------------------------
        # 2. Exit existing positions
        # ----------------------------------

        for symbol in list(positions.keys()):

            df = market_data[symbol]["15m"]

            if timestamp not in df.index:
                continue

            exit_price, reason = check_exit(
                positions[symbol],
                df.loc[timestamp]
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
                    f"net={trade['net_pnl']:.5f}"
                )

        # ----------------------------------
        # 3. Calculate adverse drawdown
        # ----------------------------------

        adverse = adverse_equity(
            balance,
            positions,
            market_data,
            timestamp,
            prices
        )

        peak_equity = max(
            peak_equity,
            equity_value(
                balance,
                positions,
                prices
            )
        )

        if peak_equity > 0:

            max_drawdown = max(
                max_drawdown,
                (peak_equity - adverse)
                / peak_equity * 100
            )

        # ----------------------------------
        # 4. Generate signals
        # ----------------------------------

        if len(positions) < MAX_OPEN_POSITIONS:

            for symbol, data in market_data.items():

                if len(positions) >= MAX_OPEN_POSITIONS:
                    break

                if symbol in positions:
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

                # Signal only uses candles closed
                # before the current 15m candle.

                history15 = df15.iloc[
                    max(0, idx - 350):idx
                ]

                # Higher timeframe candles are included
                # only after their closing timestamps.

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

                equity = equity_value(
                    balance,
                    positions,
                    prices
                )

                balance = open_position(
                    symbol,
                    signal,
                    df15.iloc[idx],
                    timestamp,
                    equity,
                    balance,
                    positions,
                    prices
                )

        # ----------------------------------
        # 5. Update closing prices
        # ----------------------------------

        for symbol, data in market_data.items():

            df = data["15m"]

            if timestamp in df.index:

                prices[symbol] = float(
                    df.loc[timestamp, "close"
                    ]
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

    # --------------------------------------
    # 6. Close remaining positions
    # --------------------------------------

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
            final_timestamp,
            "END_OF_TEST",
            balance
        )

        trades.append(trade)

    # --------------------------------------
    # 7. Build final report
    # --------------------------------------

    final_equity = balance

    stats = summarize_trades(trades)

    report = {

        "version": "2.3",
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

        "end_positions_closed": True,

        **stats,

        "v22_reference_comparison":
            compare_with_v22({
                **{
                    "final_equity": round(
                        final_equity, 6
                    ),
                    "return_percent": round(
                        (final_equity / INITIAL_BALANCE - 1)
                        * 100,
                        3
                    ),
                    "max_drawdown_percent": round(
                        max_drawdown, 3
                    )
                },
                **stats
            }),

        "comparison_note": (
            "V2.2 figures are historical reference values. "
            "Strict comparison requires replaying both "
            "strategies on identical saved candle data."
        ),

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    # --------------------------------------
    # 8. Save separate V2.3 files
    # --------------------------------------

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

    pd.DataFrame(trades).to_csv(
        TRADES_FILE,
        index=False
    )

    print("\n========== BACKTEST V2.3 ==========")

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    print("===================================")

    log(
        f"Saved {REPORT_FILE} "
        f"and {TRADES_FILE}"
    )

    return report


if __name__ == "__main__":

    run_backtest()
