"""
features.py - Feature engineering for the Stock Price Movement Predictor
=========================================================================
All indicators are computed directly in pandas (no pandas-ta or TA-Lib).
Every value at row t uses ONLY data available up to and including day t's
close, so the full feature matrix is strictly free of look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load daily OHLCV CSV, sort chronologically, and normalize column names."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    # Put Open/High/Low on the same split- and dividend-adjusted scale as AdjClose,
    # otherwise gap, range and close-position features jump at every stock split.
    factor = df["AdjClose"] / df["Close"]
    for col in ("Open", "High", "Low"):
        df[col] = df[col] * factor
    return df


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI via exponential smoothing. NaN rows filled with 50 (neutral)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram - computed by hand via EMA."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    """Bollinger Bands. Returns (upper, lower, %B, bandwidth)."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_rng = (upper - lower).replace(0.0, np.nan)
    pct_b = (series - lower) / band_rng
    bandwidth = band_rng / mid
    return upper, lower, pct_b, bandwidth


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder exponential smoothing."""
    prev_close = df["AdjClose"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def _stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
    """Stochastic Oscillator %K and %D."""
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    denom = (high_max - low_min).replace(0.0, np.nan)
    pct_k = (100.0 * (df["AdjClose"] - low_min) / denom).fillna(50.0)
    pct_d = pct_k.rolling(d_window).mean()
    return pct_k, pct_d


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume - cumulative volume signed by daily price direction."""
    return (np.sign(df["AdjClose"].diff()).fillna(0.0) * df["Volume"]).cumsum()


def _williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]. Neutral filled with -50."""
    high_max = df["High"].rolling(window).max()
    low_min = df["Low"].rolling(window).min()
    denom = (high_max - low_min).replace(0.0, np.nan)
    return (-100.0 * (high_max - df["AdjClose"]) / denom).fillna(-50.0)


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Technical-indicator feature set (25 features).
    Every feature at row t uses ONLY data available as of day t's close.
    """
    out = pd.DataFrame(index=df.index)
    close = df["AdjClose"]

    # Trend (SMA ratios)
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    out["sma_10_ratio"] = close / sma_10 - 1.0
    out["sma_20_ratio"] = close / sma_20 - 1.0
    out["sma_50_ratio"] = close / sma_50 - 1.0
    out["sma_cross"] = sma_10 / sma_50 - 1.0

    # Momentum (RSI)
    out["rsi_14"] = _rsi(close, window=14)

    # Momentum (MACD)
    macd_line, signal_line, histogram = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = histogram

    # Volatility (Bollinger Bands)
    _, _, pct_b, bandwidth = _bollinger(close, window=20)
    out["bb_pct_b"] = pct_b
    out["bb_bandwidth"] = bandwidth

    # Volatility (Normalized ATR)
    out["atr_norm"] = _atr(df, window=14) / close

    # Oscillators
    pct_k, pct_d = _stochastic(df)
    out["stoch_k"] = pct_k
    out["stoch_d"] = pct_d
    out["williams_r"] = _williams_r(df, window=14)

    # Volume
    avg_vol = df["Volume"].rolling(20).mean().replace(0.0, np.nan)
    out["obv_delta"] = _obv(df).diff(5) / avg_vol
    out["volume_change"] = df["Volume"].pct_change(1)
    out["volume_ratio"] = df["Volume"] / avg_vol - 1.0

    # Returns
    ret_1d = close.pct_change(1)
    out["return_1d"] = ret_1d
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    # Volatility of returns
    out["volatility_10d"] = ret_1d.rolling(10).std()
    out["volatility_20d"] = ret_1d.rolling(20).std()

    # Intraday structure
    hl = (df["High"] - df["Low"]).replace(0.0, np.nan)
    out["hl_range"] = (df["High"] - df["Low"]) / close
    out["gap"] = df["Open"] / close.shift(1) - 1.0
    out["close_position"] = (close - df["Low"]) / hl

    # Zero-volume or zero-range days (common in older data) produce +/-inf; treat them as missing.
    return out.replace([np.inf, -np.inf], np.nan)


def build_raw_price_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Raw-price feature set (10 features).
    Lagged price returns plus raw OHLCV levels.
    """
    out = pd.DataFrame(index=df.index)
    close = df["AdjClose"]

    for k in range(1, n_lags + 1):
        out[f"close_lag_{k}"] = close.shift(k - 1) / close.shift(k) - 1.0

    out["open_close_gap"] = df["Open"] / close.shift(1) - 1.0
    out["volume_raw"] = df["Volume"]
    out["high_raw"] = df["High"]
    out["low_raw"] = df["Low"]
    out["close_raw"] = close

    return out
