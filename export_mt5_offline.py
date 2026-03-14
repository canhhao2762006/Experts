from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import TIMEFRAMES
from data_mt5 import ensure_symbol, get_rates, get_symbol_info, mt5_init, mt5_shutdown, timeframe_to_name


def parse_args():
    parser = argparse.ArgumentParser(description="Export MT5 rates to offline CSV files for Colab/offline training")
    parser.add_argument("--symbol", required=True, help="Broker symbol in MT5, for example XAUUSD or BTCUSDm")
    parser.add_argument(
        "--output-symbol",
        default="",
        help="Optional alias for output filenames, for example BTCUSD if broker symbol is BTCUSDm",
    )
    parser.add_argument("--years-back", type=int, default=1, help="How much history to export")
    parser.add_argument("--output-dir", default="offline_exports", help="Folder where CSV/spec files will be saved")
    return parser.parse_args()


def build_symbol_spec(symbol_info) -> dict:
    contract_size = getattr(symbol_info, "trade_contract_size", 0.0) or 0.0
    if contract_size <= 0:
        contract_size = 1.0

    trade_tick_size = getattr(symbol_info, "trade_tick_size", 0.0) or 0.0
    if trade_tick_size <= 0:
        trade_tick_size = float(symbol_info.point)

    trade_tick_value = getattr(symbol_info, "trade_tick_value", 0.0) or 0.0
    if trade_tick_value <= 0:
        trade_tick_value = contract_size * trade_tick_size

    return {
        "digits": int(symbol_info.digits),
        "point": float(symbol_info.point),
        "contract_size": float(contract_size),
        "trade_tick_size": float(trade_tick_size),
        "trade_tick_value": float(trade_tick_value),
        "volume_min": float(symbol_info.volume_min),
        "volume_max": float(symbol_info.volume_max),
        "volume_step": float(symbol_info.volume_step),
    }


def main():
    args = parse_args()
    symbol = args.symbol.strip().upper()
    output_symbol = (args.output_symbol.strip().upper() or symbol)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mt5_init()
    try:
        ensure_symbol(symbol)
        symbol_info = get_symbol_info(symbol)

        print(f"Exporting MT5 data | broker_symbol={symbol} output_symbol={output_symbol} years_back={args.years_back}")

        for tf_name, timeframe in TIMEFRAMES.items():
            df = get_rates(symbol, timeframe, years_back=args.years_back, verbose=True)
            export_path = output_dir / f"{output_symbol}_{timeframe_to_name(timeframe)}.csv"
            df.to_csv(export_path, index=False)
            print(f"[SAVED] {export_path} rows={len(df)}")

        spec = build_symbol_spec(symbol_info)
        spec_path = output_dir / f"{output_symbol.lower()}_spec.json"
        with spec_path.open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2)
        print(f"[SAVED] {spec_path}")

        print("\nDone. Upload these files to Colab or Google Drive:")
        for path in sorted(output_dir.glob(f"{output_symbol}*.csv")):
            print(f"  - {path}")
        print(f"  - {spec_path}")
    finally:
        mt5_shutdown()


if __name__ == "__main__":
    main()
