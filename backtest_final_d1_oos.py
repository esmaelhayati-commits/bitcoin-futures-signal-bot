"""
FINAL OUT-OF-SAMPLE VALIDATION FOR D1.
Runs only D1_SHORT_RSI_MAX45 on an earlier, non-overlapping 30-day window.
The 36 days before the test window are indicator warm-up only.
Backtest only: no exchange orders or Telegram changes.
"""
import json
import time
from datetime import timedelta
import pandas as pd
import numpy as np

import backtest_short_filter_experiments as bt

VALIDATION_END_OFFSET_DAYS = 90
VALIDATION_DAYS = 30
WARMUP_DAYS = 36
SCENARIO = "D1_SHORT_RSI_MAX45"

def fetch_oos_history(symbol):
    now_ms = bt.exchange.milliseconds()
    sample_end_ms = now_ms - VALIDATION_END_OFFSET_DAYS * 86400000
    sample_start_ms = sample_end_ms - VALIDATION_DAYS * 86400000
    fetch_start_ms = sample_start_ms - WARMUP_DAYS * 86400000

    candles = []
    cursor = fetch_start_ms
    bt.log(
        f"OOS {symbol}: warmup_start={pd.to_datetime(fetch_start_ms, unit='ms', utc=True)} "
        f"test_start={pd.to_datetime(sample_start_ms, unit='ms', utc=True)} "
        f"test_end={pd.to_datetime(sample_end_ms, unit='ms', utc=True)}"
    )

    while cursor < sample_end_ms:
        try:
            batch = bt.exchange.fetch_ohlcv(
                symbol, timeframe="15m", since=cursor, limit=bt.CANDLE_LIMIT
            )
        except Exception as exc:
            bt.log(f"OOS download error {symbol}: {exc}")
            break
        if not batch:
            break
        candles.extend(batch)
        next_cursor = batch[-1][0] + bt.TIMEFRAME_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < bt.CANDLE_LIMIT:
            break
        time.sleep(max(bt.exchange.rateLimit, 250) / 1000)

    if not candles:
        return None

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    ).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Keep only completed candles and stop exactly at the independent test boundary.
    df = df[
        (df.index + pd.Timedelta(minutes=15) <= pd.to_datetime(sample_end_ms, unit="ms", utc=True))
    ]

    if len(df) < 3000:
        bt.log(f"Insufficient OOS + warmup history for {symbol}: {len(df)} candles")
        return None

    actual_start = int(df.index[0].timestamp() * 1000)
    if actual_start > fetch_start_ms + 6 * bt.TIMEFRAME_MS:
        bt.log(f"WARNING: {symbol} history starts later than requested warmup boundary")
    bt.log(f"DOWNLOADED {symbol}: {len(df)} candles")
    return df

def prepare_oos_market_data():
    bt.log("Loading XT swap markets for final D1 validation")
    bt.exchange.load_markets()
    market_data = {}
    for requested in bt.REQUESTED_SYMBOLS:
        symbol = bt.resolve_swap_symbol(requested, bt.exchange.markets)
        if not symbol:
            bt.log(f"Swap market not found: {requested}")
            continue
        df15 = fetch_oos_history(symbol)
        if df15 is None:
            continue
        df1 = bt.resample_ohlcv(df15, "1h", 4)
        df4 = bt.resample_ohlcv(df15, "4h", 16)
        if min(len(df1), len(df4)) < 210:
            bt.log(f"Insufficient warmup for {symbol}: 1h={len(df1)} 4h={len(df4)}")
            continue
        market_data[symbol] = {"15m": df15, "1h": df1, "4h": df4}
        bt.log(f"READY {symbol}: 15m={len(df15)} 1h={len(df1)} 4h={len(df4)}")
    return market_data

def main():
    bt.BACKTEST_VERSION = "2.3.3-D1-OOS"
    bt.fetch_history = fetch_oos_history
    market_data = prepare_oos_market_data()
    if len(market_data) < 5:
        raise RuntimeError(f"Not enough markets for validation: {len(market_data)}; need at least 5")

    report = bt.run_backtest(
        market_data_override=market_data,
        scenario_name=SCENARIO
    )

    now_ms = bt.exchange.milliseconds()
    sample_end = pd.to_datetime(now_ms - VALIDATION_END_OFFSET_DAYS * 86400000, unit="ms", utc=True)
    sample_start = sample_end - pd.Timedelta(days=VALIDATION_DAYS)
    report["validation_type"] = "OUT_OF_SAMPLE_HISTORICAL_WINDOW"
    report["validation_window_start"] = sample_start.isoformat()
    report["validation_window_end"] = sample_end.isoformat()
    report["warmup_days"] = WARMUP_DAYS
    report["validation_days"] = VALIDATION_DAYS
    report["markets_loaded"] = len(market_data)
    report["note"] = (
        "Only D1 was tested. The 36-day prefix is indicator warm-up; "
        "signals are expected only after the 4h history reaches 210 bars. "
        "Funding and liquidation are not modeled."
    )

    report_path = f"final_d1_oos_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    trades_path = "final_d1_oos_trades.csv"
    pd.read_csv(f"backtest_trades_{SCENARIO.lower()}.csv").to_csv(trades_path, index=False)

    summary = {
        "validation_window_start": report["validation_window_start"],
        "validation_window_end": report["validation_window_end"],
        "markets_loaded": len(market_data),
        "initial_balance": report["initial_balance"],
        "final_balance": report["final_balance"],
        "net_profit": report["net_profit"],
        "return_percent": report["return_percent"],
        "max_drawdown_percent": report["max_drawdown_percent"],
        "closed_trades": report["closed_trades"],
        "win_rate_percent": report["win_rate_percent"],
        "profit_factor": report["profit_factor"],
        "long": report.get("long"),
        "short": report.get("short"),
    }
    with open("final_d1_oos_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n========== FINAL D1 OUT-OF-SAMPLE RESULT ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===================================================")

if __name__ == "__main__":
    main()
