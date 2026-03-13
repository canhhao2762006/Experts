import json
import joblib
import torch
import pandas as pd

from config import (
    MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH,
    DATASET_PATH, BT_TRADES_PATH, BT_SUMMARY_PATH, BT_EQUITY_PATH,
    WF_PATH, WF_TRADES_PATH, WF_EQUITY_PATH, SEQ_LEN,
)
from sequence_dataset import build_sequence_bundle
from features import get_base_features


def save_outputs(model, scaler, feature_cols: list, metrics: dict, df_full: pd.DataFrame,
                 test_trades_df: pd.DataFrame = None,
                 test_equity_df: pd.DataFrame = None,
                 wf_results_df: pd.DataFrame = None,
                 wf_trades_df: pd.DataFrame = None,
                 wf_equity_df: pd.DataFrame = None,
                 wf_summary: dict = None):
    torch.save(
        {"model_state_dict": model.state_dict(), "input_size": len(feature_cols)},
        MODEL_PATH,
    )

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    bundle = build_sequence_bundle(df_full, SEQ_LEN)
    row_df = bundle.row_df.copy()
    feature_df = get_base_features(df_full).dropna().reset_index(drop=True)
    feature_df = feature_df.iloc[-len(row_df):].reset_index(drop=True)
    save_df = pd.concat([row_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    save_df.to_csv(DATASET_PATH, index=False)

    if test_trades_df is not None:
        test_trades_df.to_csv(BT_TRADES_PATH, index=False)
    if test_equity_df is not None:
        test_equity_df.to_csv(BT_EQUITY_PATH, index=False)

    if metrics.get("test_backtest_summary") is not None:
        with open(BT_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics["test_backtest_summary"], f, ensure_ascii=False, indent=2)

    if wf_results_df is not None or wf_trades_df is not None or wf_summary is not None:
        payload = {
            "walkforward_windows": [] if wf_results_df is None else wf_results_df.to_dict(orient="records"),
            "walkforward_summary": {} if wf_summary is None else wf_summary,
        }
        with open(WF_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if wf_trades_df is not None and len(wf_trades_df) > 0:
            wf_trades_df.to_csv(WF_TRADES_PATH, index=False)
        if wf_equity_df is not None and len(wf_equity_df) > 0:
            wf_equity_df.to_csv(WF_EQUITY_PATH, index=False)
