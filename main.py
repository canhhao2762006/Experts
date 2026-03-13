import json
import pandas as pd

from config import SYMBOL, YEARS_BACK, TRAIN_RATIO, VALID_RATIO, TIMEFRAMES
from utils import set_seed, StandardScaler3D
from data_mt5 import mt5_init, mt5_shutdown, ensure_symbol, get_symbol_info, get_rates
from features import add_indicators, merge_timeframes, add_cross_features, get_base_features
from filters import load_news_events
from sequence_dataset import build_sequence_bundle
from trainer import train_model, predict_proba, evaluate
from backtest import backtest_strategy, optimize_thresholds, run_walkforward
from save_load import save_outputs
from live import run_live


def prepare_dataset(symbol: str, years_back: int = 1) -> pd.DataFrame:
    raw_m1 = get_rates(symbol, TIMEFRAMES["M1"], years_back, verbose=True)
    raw_m5 = get_rates(symbol, TIMEFRAMES["M5"], years_back, verbose=True)
    raw_m15 = get_rates(symbol, TIMEFRAMES["M15"], years_back, verbose=True)
    raw_h1 = get_rates(symbol, TIMEFRAMES["H1"], years_back, verbose=True)

    base_m1 = raw_m1.copy()
    m1_feat = add_indicators(raw_m1, "M1")
    m5_feat = add_indicators(raw_m5, "M5")
    m15_feat = add_indicators(raw_m15, "M15")
    h1_feat = add_indicators(raw_h1, "H1")

    df = base_m1.merge(m1_feat, on="time", how="left")
    df = merge_timeframes(df, m5_feat, m15_feat, h1_feat)
    df = add_cross_features(df)
    return df.reset_index(drop=True)


def train_pipeline():
    symbol_info = get_symbol_info(SYMBOL)
    news_df = load_news_events("news_events.csv")

    df = prepare_dataset(SYMBOL, YEARS_BACK)
    print(f"Dataset rows: {len(df)}")
    print(df[["time", "open", "high", "low", "close"]].tail())

    bundle = build_sequence_bundle(df)

    n = len(bundle.x)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    x_train = bundle.x[:train_end]
    y_train = bundle.y[:train_end]
    x_valid = bundle.x[train_end:valid_end]
    y_valid = bundle.y[train_end:valid_end]
    x_test = bundle.x[valid_end:]
    y_test = bundle.y[valid_end:]

    row_df_valid = bundle.row_df.iloc[train_end:valid_end].reset_index(drop=True)
    row_df_test = bundle.row_df.iloc[valid_end:].reset_index(drop=True)

    scaler = StandardScaler3D()
    x_train_sc = scaler.fit_transform(x_train)
    x_valid_sc = scaler.transform(x_valid)
    x_test_sc = scaler.transform(x_test)

    model = train_model(x_train_sc, y_train, x_valid_sc, y_valid, input_size=x_train_sc.shape[-1])

    eval_train = evaluate(model, x_train_sc, y_train)
    eval_valid = evaluate(model, x_valid_sc, y_valid)
    eval_test = evaluate(model, x_test_sc, y_test)

    valid_probs = predict_proba(model, x_valid_sc)
    threshold_table, best = optimize_thresholds(row_df_valid, valid_probs, symbol_info, news_df)

    best_buy = float(best["buy_threshold"])
    best_sell = float(best["sell_threshold"])

    test_probs = predict_proba(model, x_test_sc)
    test_trades_df, test_equity_df, test_summary = backtest_strategy(
        row_df_test, test_probs, symbol_info, news_df, best_buy, best_sell
    )

    # ── FIX: Refit scaler on train+valid for deploy model ──
    x_deploy = bundle.x[:valid_end]
    y_deploy = bundle.y[:valid_end]
    deploy_scaler = StandardScaler3D()
    x_deploy_sc = deploy_scaler.fit_transform(x_deploy)
    deploy_model = train_model(x_deploy_sc, y_deploy, x_valid_sc, y_valid, input_size=x_train_sc.shape[-1])

    wf_results_df, wf_trades_df, wf_equity_df, wf_summary = run_walkforward(
        bundle=bundle,
        trainer_predict_fn=predict_proba,
        trainer_train_fn=train_model,
        symbol_info=symbol_info,
        news_df=news_df,
    )

    metrics = {
        "rows_total": int(n),
        "rows_train": int(len(x_train)),
        "rows_valid": int(len(x_valid)),
        "rows_test": int(len(x_test)),
        "eval_train": eval_train,
        "eval_valid": eval_valid,
        "eval_test": eval_test,
        "best_buy_threshold": best_buy,
        "best_sell_threshold": best_sell,
        "threshold_table": threshold_table.to_dict(orient="records"),
        "test_backtest_summary": test_summary,
        "walkforward_summary": wf_summary,
        "walkforward_windows": [] if len(wf_results_df) == 0 else wf_results_df.to_dict(orient="records"),
    }

    print("\n===== TEST REPORT =====")
    print(metrics["eval_test"]["report_text"])

    print("\n===== TEST BACKTEST SUMMARY =====")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    print("\n===== WALK-FORWARD SUMMARY =====")
    print(json.dumps(wf_summary, ensure_ascii=False, indent=2))

    save_outputs(
        model=deploy_model,
        scaler=deploy_scaler,
        feature_cols=list(get_base_features(df).columns),
        metrics=metrics,
        df_full=df,
        test_trades_df=test_trades_df,
        test_equity_df=test_equity_df,
        wf_results_df=wf_results_df,
        wf_trades_df=wf_trades_df,
        wf_equity_df=wf_equity_df,
        wf_summary=wf_summary,
    )


def main():
    set_seed()
    mt5_init()
    try:
        ensure_symbol(SYMBOL)

        print("1 = train + backtest + walk-forward")
        print("2 = live V7")
        mode = input("Chọn mode: ").strip()

        if mode == "1":
            train_pipeline()
        elif mode == "2":
            run_live()
        else:
            print("Mode không hợp lệ")
    finally:
        mt5_shutdown()


if __name__ == "__main__":
    main()
