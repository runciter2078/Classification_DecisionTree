#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPYV3 Dataset Generator
========================

Downloads real SPY (S&P 500 ETF) daily price data via yfinance and
builds an SPYV3.csv-compatible dataset using the exact numbered
column names referenced across the SPY Random Forest, Decision Tree
and PCA repos, so those scripts can be run end-to-end without access
to the original proprietary dataset.

IMPORTANT: this does NOT reconstruct the original SPYV3.csv, whose
exact feature definitions are unknown. It generates an independently
engineered set of real technical indicators and maps each one to a
numbered column those scripts expect (e.g. '1', '31', '42', ...),
purely so the pipelines can be exercised end-to-end on real market
data.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_spy_dataset")

TICKER = "SPY"

# Columna numerada -> indicador técnico que representa en este dataset.
# Cubre la unión de todas las columnas usadas en los repos de Random
# Forest, Decision Tree y PCA.
NUMERIC_FEATURE_MAP = {
    "1": "return_1d",
    "2": "return_5d",
    "31": "sma_10_ratio",
    "42": "sma_20_ratio",
    "45": "sma_50_ratio",
    "46": "ema_12_ratio",
    "47": "ema_26_ratio",
    "48": "macd",
    "60": "macd_signal",
    "68": "macd_hist",
    "75": "rsi_14",
    "76": "stoch_k",
    "77": "stoch_d",
    "88": "atr_14",
    "93": "bollinger_pct_b",
    "139": "bollinger_bandwidth",
    "171": "volatility_20",
    "173": "momentum_10",
    "179": "roc_10",
    "187": "obv_zscore",
    "191": "volume_change_20",
    "218": "high_low_range",
    "221": "sma10_50_cross",
    "223": "dist_52w_high",
    "225": "dist_52w_low",
    "231": "cci_20",
    "237": "williams_r",
}

# Columnas categóricas (nominales), usadas por el repo PCA.
CATEGORICAL_FEATURE_MAP = {
    "39": "trend_regime",
    "41": "volatility_regime",
    "43": "rsi_zone",
    "168": "volume_regime",
    "172": "weekday_bucket",
}


def download_price_data(months: int, buffer_months: int) -> pd.DataFrame:
    """
    Descarga datos diarios de SPY. Se pide un buffer adicional de meses
    antes del periodo final, porque indicadores como SMA_50, SMA_200 o
    el máximo/mínimo de 52 semanas necesitan histórico previo para no
    generar NaN al principio de la ventana finalmente solicitada.
    """
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=months + buffer_months)
    logger.info(
        "Descargando %s desde %s hasta %s (incluye buffer de %d meses)",
        TICKER, start_date.date(), end_date.date(), buffer_months,
    )
    df = yf.download(
        TICKER,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No se pudieron descargar datos para {TICKER}. Revisa la conexión o el ticker.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    logger.info("Datos descargados: %d filas (%s a %s)", len(df), df.index.min().date(), df.index.max().date())
    return df


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # --- Retornos y medias móviles ---
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    out["sma_10_ratio"] = close / sma_10 - 1
    out["sma_20_ratio"] = close / sma_20 - 1
    out["sma_50_ratio"] = close / sma_50 - 1
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_12_ratio"] = close / ema_12 - 1
    out["ema_26_ratio"] = close / ema_26 - 1

    # --- MACD ---
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Oscilador estocástico ---
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3).mean()

    # --- ATR ---
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()

    # --- Bandas de Bollinger ---
    std_20 = close.rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    out["bollinger_pct_b"] = (close - lower_band) / (upper_band - lower_band)
    out["bollinger_bandwidth"] = (upper_band - lower_band) / sma_20

    # --- Volatilidad y momentum ---
    out["volatility_20"] = close.pct_change().rolling(20).std()
    out["momentum_10"] = close - close.shift(10)
    out["roc_10"] = close.pct_change(10)

    # --- Volumen ---
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    out["obv_zscore"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
    out["volume_change_20"] = volume / volume.rolling(20).mean() - 1

    # --- Rango diario, cruce de medias y niveles de 52 semanas ---
    out["high_low_range"] = (high - low) / close
    out["sma10_50_cross"] = sma_10 / sma_50 - 1  # cruce dorado/de la muerte
    rolling_max_252 = close.rolling(252, min_periods=50).max()
    rolling_min_252 = close.rolling(252, min_periods=50).min()
    out["dist_52w_high"] = close / rolling_max_252 - 1
    out["dist_52w_low"] = close / rolling_min_252 - 1

    # --- CCI y Williams %R ---
    typical_price = (high + low + close) / 3
    sma_tp_20 = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["cci_20"] = (typical_price - sma_tp_20) / (0.015 * mean_dev)
    highest_14 = high.rolling(14).max()
    lowest_14 = low.rolling(14).min()
    out["williams_r"] = -100 * (highest_14 - close) / (highest_14 - lowest_14)

    return out


def compute_categorical_features(df: pd.DataFrame, numeric: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    sma_200 = close.rolling(200, min_periods=100).mean()

    out["trend_regime"] = np.select(
        [close > sma_200, close < sma_200], ["bull", "bear"], default="sideways"
    )
    close_to_sma200 = (close / sma_200 - 1).abs()
    out.loc[close_to_sma200 < 0.01, "trend_regime"] = "sideways"

    atr_pct_rank = numeric["atr_14"].rank(pct=True)
    out["volatility_regime"] = pd.cut(
        atr_pct_rank, bins=[0, 0.33, 0.66, 1.0], labels=["low", "medium", "high"], include_lowest=True
    ).astype(str)

    out["rsi_zone"] = pd.cut(
        numeric["rsi_14"], bins=[0, 30, 70, 100], labels=["oversold", "neutral", "overbought"]
    ).astype(str)

    vol_pct_rank = df["volume"].rank(pct=True)
    out["volume_regime"] = pd.cut(
        vol_pct_rank, bins=[0, 0.33, 0.66, 1.0], labels=["low", "normal", "high"], include_lowest=True
    ).astype(str)

    weekday = df.index.dayofweek  # 0=lunes ... 4=viernes
    out["weekday_bucket"] = pd.cut(
        weekday, bins=[-1, 1, 2, 4], labels=["start_week", "mid_week", "end_week"]
    ).astype(str)

    return out


def build_dataset(months: int, buffer_months: int) -> pd.DataFrame:
    raw = download_price_data(months, buffer_months)
    numeric_features = compute_technical_features(raw)
    categorical_features = compute_categorical_features(raw, numeric_features)

    df = pd.DataFrame(index=raw.index)
    df["FECHA"] = raw.index.strftime("%Y-%m-%d")
    df["OPEN"] = raw["open"].values
    df["MAX"] = raw["high"].values
    df["MIN"] = raw["low"].values
    df["CLOSE"] = raw["close"].values
    df["FECHA.year"] = raw.index.year
    df["FECHA.month"] = raw.index.month
    df["FECHA.day-of-month"] = raw.index.day
    df["FECHA.day-of-week"] = raw.index.dayofweek

    for col, source in NUMERIC_FEATURE_MAP.items():
        df[col] = numeric_features[source].values
    for col, source in CATEGORICAL_FEATURE_MAP.items():
        df[col] = categorical_features[source].values

    # CLASIFICADOR: día de entrada positivo = el cierre del día
    # siguiente es superior al de hoy. Se desplaza -1 (mira al futuro,
    # como corresponde a una variable objetivo) y se elimina la última
    # fila, que no tiene "día siguiente" conocido. Ajusta esta regla si
    # tu definición de "entrada positiva" es distinta (p.ej. un umbral
    # de retorno o una ventana de varios días).
    next_return = raw["close"].shift(-1) / raw["close"] - 1
    df["CLASIFICADOR"] = (next_return > 0).astype(int)
    df = df.iloc[:-1]

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Filas eliminadas por warm-up de indicadores: %d", n_before - len(df))
    return df


def trim_to_period(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Recorta el dataset a los últimos `months` meses, ya con los indicadores calculados."""
    df = df.copy()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    cutoff = df["FECHA"].max() - pd.DateOffset(months=months)
    trimmed = df[df["FECHA"] > cutoff].reset_index(drop=True)
    trimmed["FECHA"] = trimmed["FECHA"].dt.strftime("%Y-%m-%d")
    logger.info("Dataset recortado a los últimos %d meses: %d filas", months, len(trimmed))
    return trimmed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un SPYV3.csv con datos reales de SPY y features técnicas propias."
    )
    parser.add_argument("--months", type=int, default=12, help="Meses de datos finales a conservar.")
    parser.add_argument(
        "--buffer-months", type=int, default=14,
        help="Meses adicionales descargados antes del periodo final, para el warm-up de indicadores (SMA_200, 52w, etc.).",
    )
    parser.add_argument("--output", type=str, default="SPYV3.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(args.months, args.buffer_months)
    df = trim_to_period(df, args.months)
    df.to_csv(args.output, index=False)
    logger.info("SPYV3.csv generado en %s (%d filas, %d columnas)", args.output, *df.shape)


if __name__ == "__main__":
    main()