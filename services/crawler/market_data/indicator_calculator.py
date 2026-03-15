"""Technical indicators calculation engine.

Calculates 10 indicators from OHLCV data using pandas-ta.
Returns list of IndicatorRecord for persistence.
Only computes the most recent date (last row) from OHLCV data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pandas_ta as ta

from shared.logging import get_logger

logger = get_logger("indicator_calculator")

MIN_PERIODS = {
    "SMA_20": 20,
    "SMA_50": 50,
    "SMA_200": 200,
    "RSI_14": 15,
    "MACD": 35,
    "VOLUME": 20,
    "ATR_14": 15,
    "BOLLINGER": 20,
    "DONCHIAN": 20,
    "RS_VNINDEX": 20,
}


@dataclass
class IndicatorRecord:
    """Single indicator value to persist."""

    indicator_name: str
    indicator_value: Decimal | None
    data_as_of: datetime


def calculate_indicators(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    df_vnindex: pd.DataFrame | None = None,
) -> list[IndicatorRecord]:
    """Calculate 10 technical indicators from OHLCV data.

    Args:
        ticker: Stock ticker symbol.
        df_ohlcv: DataFrame with columns [time, open, high, low, close, volume] sorted by time ASC.
        df_vnindex: Optional VN-Index DataFrame for Relative Strength calculation.

    Returns:
        List of IndicatorRecord for the most recent date.
    """
    if df_ohlcv.empty:
        return []

    data_len = len(df_ohlcv)
    data_as_of = df_ohlcv["time"].iloc[-1]
    if isinstance(data_as_of, pd.Timestamp):
        data_as_of = data_as_of.to_pydatetime()

    records: list[IndicatorRecord] = []

    # SMA indicators
    for period, name in [(20, "SMA_20"), (50, "SMA_50"), (200, "SMA_200")]:
        if data_len >= MIN_PERIODS[name]:
            sma = ta.sma(df_ohlcv["close"], length=period)
            val = sma.iloc[-1]
            if pd.notna(val):
                records.append(IndicatorRecord(name, Decimal(str(round(val, 6))), data_as_of))
        else:
            logger.warning(
                "insufficient_data_for_indicator",
                ticker=ticker,
                indicator=name,
                required=MIN_PERIODS[name],
                available=data_len,
                component="technical_indicators",
            )

    # RSI
    if data_len >= MIN_PERIODS["RSI_14"]:
        rsi = ta.rsi(df_ohlcv["close"], length=14)
        val = rsi.iloc[-1]
        if pd.notna(val):
            records.append(IndicatorRecord("RSI_14", Decimal(str(round(val, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="RSI_14",
            required=MIN_PERIODS["RSI_14"],
            available=data_len,
            component="technical_indicators",
        )

    # MACD
    if data_len >= MIN_PERIODS["MACD"]:
        macd_df = ta.macd(df_ohlcv["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_map = {
                "MACD_12_26_9": "MACD_LINE",
                "MACDs_12_26_9": "MACD_SIGNAL",
                "MACDh_12_26_9": "MACD_HISTOGRAM",
            }
            for col, indicator_name in macd_map.items():
                if col in macd_df.columns:
                    val = macd_df[col].iloc[-1]
                    if pd.notna(val):
                        records.append(IndicatorRecord(indicator_name, Decimal(str(round(val, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="MACD",
            required=MIN_PERIODS["MACD"],
            available=data_len,
            component="technical_indicators",
        )

    # Volume Analysis (manual calculation)
    if data_len >= MIN_PERIODS["VOLUME"]:
        vol_avg_20 = df_ohlcv["volume"].rolling(window=20).mean().iloc[-1]
        vol_current = df_ohlcv["volume"].iloc[-1]
        if pd.notna(vol_avg_20) and vol_avg_20 > 0:
            vol_ratio = vol_current / vol_avg_20
            records.append(IndicatorRecord("VOLUME_AVG_20", Decimal(str(round(vol_avg_20, 6))), data_as_of))
            records.append(IndicatorRecord("VOLUME_CURRENT", Decimal(str(round(vol_current, 6))), data_as_of))
            records.append(IndicatorRecord("VOLUME_RATIO", Decimal(str(round(vol_ratio, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="VOLUME",
            required=MIN_PERIODS["VOLUME"],
            available=data_len,
            component="technical_indicators",
        )

    # ATR
    if data_len >= MIN_PERIODS["ATR_14"]:
        atr = ta.atr(df_ohlcv["high"], df_ohlcv["low"], df_ohlcv["close"], length=14)
        if atr is not None:
            val = atr.iloc[-1]
            if pd.notna(val):
                records.append(IndicatorRecord("ATR_14", Decimal(str(round(val, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="ATR_14",
            required=MIN_PERIODS["ATR_14"],
            available=data_len,
            component="technical_indicators",
        )

    # Bollinger Bands
    if data_len >= MIN_PERIODS["BOLLINGER"]:
        bb_df = ta.bbands(df_ohlcv["close"], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            # pandas-ta bbands column names vary by version; match dynamically
            bb_map = {}
            for col in bb_df.columns:
                if col.startswith("BBL_"):
                    bb_map[col] = "BB_LOWER"
                elif col.startswith("BBM_"):
                    bb_map[col] = "BB_MIDDLE"
                elif col.startswith("BBU_"):
                    bb_map[col] = "BB_UPPER"
            for col, indicator_name in bb_map.items():
                if col in bb_df.columns:
                    val = bb_df[col].iloc[-1]
                    if pd.notna(val):
                        records.append(IndicatorRecord(indicator_name, Decimal(str(round(val, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="BOLLINGER",
            required=MIN_PERIODS["BOLLINGER"],
            available=data_len,
            component="technical_indicators",
        )

    # Donchian Channel
    if data_len >= MIN_PERIODS["DONCHIAN"]:
        dc_df = ta.donchian(df_ohlcv["high"], df_ohlcv["low"], lower_length=20, upper_length=20)
        if dc_df is not None and not dc_df.empty:
            dc_map = {
                "DCL_20_20": "DC_LOWER",
                "DCM_20_20": "DC_MIDDLE",
                "DCU_20_20": "DC_UPPER",
            }
            for col, indicator_name in dc_map.items():
                if col in dc_df.columns:
                    val = dc_df[col].iloc[-1]
                    if pd.notna(val):
                        records.append(IndicatorRecord(indicator_name, Decimal(str(round(val, 6))), data_as_of))
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="DONCHIAN",
            required=MIN_PERIODS["DONCHIAN"],
            available=data_len,
            component="technical_indicators",
        )

    # Relative Strength vs VN-Index
    if data_len >= MIN_PERIODS["RS_VNINDEX"]:
        if df_vnindex is not None and len(df_vnindex) >= MIN_PERIODS["RS_VNINDEX"]:
            rs_val = _calculate_relative_strength(df_ohlcv, df_vnindex)
            if rs_val is not None:
                records.append(IndicatorRecord("RS_VNINDEX", Decimal(str(round(rs_val, 6))), data_as_of))
        else:
            logger.warning(
                "vnindex_data_unavailable",
                ticker=ticker,
                message="VN-Index data unavailable for RS calculation",
                component="technical_indicators",
            )
    else:
        logger.warning(
            "insufficient_data_for_indicator",
            ticker=ticker,
            indicator="RS_VNINDEX",
            required=MIN_PERIODS["RS_VNINDEX"],
            available=data_len,
            component="technical_indicators",
        )

    logger.info(
        "indicators_calculated",
        ticker=ticker,
        count=len(records),
        data_as_of=str(data_as_of),
        component="technical_indicators",
    )
    return records


def _calculate_relative_strength(
    df_ticker: pd.DataFrame,
    df_vnindex: pd.DataFrame,
    periods: int = 20,
) -> float | None:
    """Calculate Relative Strength of ticker vs VN-Index.

    RS = (ticker_close / ticker_close[n_periods_ago]) / (vnindex_close / vnindex_close[n_periods_ago])
    """
    if len(df_ticker) < periods + 1 or len(df_vnindex) < periods + 1:
        return None

    ticker_current = df_ticker["close"].iloc[-1]
    ticker_past = df_ticker["close"].iloc[-(periods + 1)]
    vnindex_current = df_vnindex["close"].iloc[-1]
    vnindex_past = df_vnindex["close"].iloc[-(periods + 1)]

    if ticker_past == 0 or vnindex_past == 0 or vnindex_current == 0:
        return None

    ticker_return = ticker_current / ticker_past
    vnindex_return = vnindex_current / vnindex_past
    return ticker_return / vnindex_return
