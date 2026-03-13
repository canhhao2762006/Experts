import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pctb = (series - lower) / (upper - lower).replace(0, np.nan)
    return width, pctb


def add_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()

    out[f"{prefix}_ret_1"] = out["close"].pct_change(1)
    out[f"{prefix}_ret_3"] = out["close"].pct_change(3)
    out[f"{prefix}_ret_5"] = out["close"].pct_change(5)
    out[f"{prefix}_ret_10"] = out["close"].pct_change(10)

    out[f"{prefix}_ema_9"] = ema(out["close"], 9)
    out[f"{prefix}_ema_20"] = ema(out["close"], 20)
    out[f"{prefix}_ema_50"] = ema(out["close"], 50)
    out[f"{prefix}_ema_200"] = ema(out["close"], 200)

    out[f"{prefix}_ema_spread_9_20"] = (out[f"{prefix}_ema_9"] - out[f"{prefix}_ema_20"]) / out["close"]
    out[f"{prefix}_ema_spread_20_50"] = (out[f"{prefix}_ema_20"] - out[f"{prefix}_ema_50"]) / out["close"]
    out[f"{prefix}_ema_spread_50_200"] = (out[f"{prefix}_ema_50"] - out[f"{prefix}_ema_200"]) / out["close"]

    out[f"{prefix}_rsi_14"] = rsi(out["close"], 14)
    out[f"{prefix}_atr_14"] = atr(out, 14)
    out[f"{prefix}_atr_pct"] = out[f"{prefix}_atr_14"] / out["close"]

    out[f"{prefix}_vol_10"] = out[f"{prefix}_ret_1"].rolling(10).std()
    out[f"{prefix}_vol_30"] = out[f"{prefix}_ret_1"].rolling(30).std()

    out[f"{prefix}_hh_20"] = out["high"].rolling(20).max()
    out[f"{prefix}_ll_20"] = out["low"].rolling(20).min()
    out[f"{prefix}_breakout_pos"] = (
        (out["close"] - out[f"{prefix}_ll_20"]) /
        (out[f"{prefix}_hh_20"] - out[f"{prefix}_ll_20"]).replace(0, np.nan)
    )

    out[f"{prefix}_range_1"] = (out["high"] - out["low"]) / out["close"]
    out[f"{prefix}_body_1"] = (out["close"] - out["open"]) / out["close"]

    out[f"{prefix}_trend_up_20_50"] = (out[f"{prefix}_ema_20"] > out[f"{prefix}_ema_50"]).astype(int)
    out[f"{prefix}_trend_up_50_200"] = (out[f"{prefix}_ema_50"] > out[f"{prefix}_ema_200"]).astype(int)

    # ── NEW: MACD ──
    m_line, m_sig, m_hist = macd(out["close"])
    out[f"{prefix}_macd"] = m_line / out["close"]
    out[f"{prefix}_macd_signal"] = m_sig / out["close"]
    out[f"{prefix}_macd_hist"] = m_hist / out["close"]

    # ── NEW: Bollinger Bands ──
    bb_w, bb_pb = bollinger_bands(out["close"])
    out[f"{prefix}_bb_width"] = bb_w
    out[f"{prefix}_bb_pctb"] = bb_pb

    # ── NEW: Volume features ──
    if "tick_volume" in out.columns:
        vol_ma = out["tick_volume"].rolling(20).mean().replace(0, np.nan)
        out[f"{prefix}_vol_ratio"] = out["tick_volume"] / vol_ma
        out[f"{prefix}_vol_delta"] = out["tick_volume"].diff(1) / vol_ma

    keep_cols = ["time"] + [c for c in out.columns if c.startswith(prefix + "_")]
    return out[keep_cols].copy()


def merge_timeframes(base_m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    df = base_m1.sort_values("time").copy()
    df = pd.merge_asof(df, m5.sort_values("time"), on="time", direction="backward")
    df = pd.merge_asof(df, m15.sort_values("time"), on="time", direction="backward")
    df = pd.merge_asof(df, h1.sort_values("time"), on="time", direction="backward")
    return df


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["dist_close_to_M15_ema20"] = (out["close"] - out["M15_ema_20"]) / out["close"]
    out["dist_close_to_H1_ema50"] = (out["close"] - out["H1_ema_50"]) / out["close"]
    out["dist_close_to_H1_ema200"] = (out["close"] - out["H1_ema_200"]) / out["close"]

    out["trend_align_m15_h1"] = (out["M15_trend_up_20_50"] == out["H1_trend_up_20_50"]).astype(int)
    out["trend_align_all"] = (
        (out["M5_trend_up_20_50"] == out["M15_trend_up_20_50"]) &
        (out["M15_trend_up_20_50"] == out["H1_trend_up_20_50"])
    ).astype(int)

    out["momentum_gap_m1_m15"] = out["M1_ret_3"] - out["M15_ret_3"]
    out["momentum_gap_m5_h1"] = out["M5_ret_3"] - out["H1_ret_3"]

    out["m1_vs_h1_breakout_gap"] = out["M1_breakout_pos"] - out["H1_breakout_pos"]
    out["m1_vs_m15_breakout_gap"] = out["M1_breakout_pos"] - out["M15_breakout_pos"]

    out["buy_context_score"] = (
        out["M15_trend_up_20_50"] +
        out["H1_trend_up_20_50"] +
        out["H1_trend_up_50_200"] +
        out["trend_align_all"]
    )
    out["sell_context_score"] = (
        (1 - out["M15_trend_up_20_50"]) +
        (1 - out["H1_trend_up_20_50"]) +
        (1 - out["H1_trend_up_50_200"]) +
        out["trend_align_all"]
    )

    # ── NEW: Time cyclical encoding ──
    if "time" in out.columns:
        hour = out["time"].dt.hour + out["time"].dt.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        dow = out["time"].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # ── NEW: Cross-timeframe MACD alignment ──
    out["macd_align_m1_h1"] = np.sign(out["M1_macd_hist"]) * np.sign(out["H1_macd_hist"])
    out["macd_align_m5_m15"] = np.sign(out["M5_macd_hist"]) * np.sign(out["M15_macd_hist"])

    return out


def get_base_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "M1_ret_1", "M1_ret_3", "M1_ret_5",
        "M1_ema_spread_9_20", "M1_ema_spread_20_50",
        "M1_rsi_14", "M1_atr_pct", "M1_vol_10",
        "M1_breakout_pos", "M1_range_1", "M1_body_1",
        "M1_trend_up_20_50", "M1_trend_up_50_200",
        "M1_macd", "M1_macd_signal", "M1_macd_hist",
        "M1_bb_width", "M1_bb_pctb",
        "M1_vol_ratio", "M1_vol_delta",

        "M5_ret_3", "M5_ema_spread_20_50", "M5_rsi_14", "M5_atr_pct", "M5_breakout_pos",
        "M5_macd_hist", "M5_bb_pctb", "M5_vol_ratio",

        "M15_ret_3", "M15_ema_spread_20_50", "M15_rsi_14", "M15_atr_pct", "M15_breakout_pos",
        "M15_macd_hist", "M15_bb_pctb",

        "H1_ret_3", "H1_ema_spread_20_50", "H1_rsi_14", "H1_atr_pct", "H1_breakout_pos",
        "H1_macd_hist", "H1_bb_pctb",

        "dist_close_to_M15_ema20",
        "dist_close_to_H1_ema50",
        "dist_close_to_H1_ema200",
        "trend_align_m15_h1",
        "trend_align_all",
        "momentum_gap_m1_m15",
        "momentum_gap_m5_h1",
        "m1_vs_h1_breakout_gap",
        "m1_vs_m15_breakout_gap",
        "buy_context_score",
        "sell_context_score",

        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "macd_align_m1_h1", "macd_align_m5_m15",
    ]
    return df[cols].copy()
