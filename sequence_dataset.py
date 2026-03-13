import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from config import SEQ_LEN
from labels import build_labels_no_lookahead
from features import get_base_features


@dataclass
class SequenceBundle:
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    row_df: pd.DataFrame


def build_sequence_bundle(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> SequenceBundle:
    df = df.copy()
    df["target"] = build_labels_no_lookahead(df)

    feat_df = get_base_features(df)
    full_df = pd.concat(
        [df[["time", "open", "high", "low", "close", "tick_volume", "spread", "target"]], feat_df],
        axis=1,
    )
    full_df = full_df.dropna().reset_index(drop=True)

    feature_cols = list(feat_df.columns)
    arr = full_df[feature_cols].values.astype(np.float32)
    target = full_df["target"].values.astype(np.int64)
    time_arr = full_df["time"].values

    n = len(arr)
    if n < seq_len:
        raise ValueError(f"Not enough data: {n} rows < {seq_len} seq_len")

    # ── Memory-efficient sliding window (no intermediate list) ──
    windows = sliding_window_view(arr, window_shape=seq_len, axis=0)
    # shape: (n - seq_len + 1, n_features, seq_len)
    x = np.ascontiguousarray(windows.transpose(0, 2, 1))
    # shape: (n - seq_len + 1, seq_len, n_features)

    start = seq_len - 1
    y = target[start:]
    times = time_arr[start:]
    row_idx = list(range(start, n))

    return SequenceBundle(
        x=x,
        y=y,
        times=times,
        row_df=full_df.iloc[row_idx].reset_index(drop=True).copy(),
    )
