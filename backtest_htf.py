
import json
import pandas as pd
from datetime import datetime, timezone

from backtest import (
    prepare_market_data,
    run_backtest
)


def main():
    print("Starting HTF strategy evaluation...")

    # Run the unified V2 portfolio backtest.
    report = run_backtest()

    result = {
        "version": "2.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation": report,
        "note": (
            "This report uses the unified 15m entry "
            "and 1h/4h trend confirmation strategy. "
            "It is not an independent comparison "
            "of alternative entry timeframes."
        )
    }

    with open(
        "backtest_htf_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== HTF EVALUATION ==========")
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    ))
    print("====================================")


if __name__ == "__main__":
    main()
