"""Data persistence for stock market data.

Saves OHLCV prices and financial ratios to the market_data table
using shared async DB sessions and the MarketData ORM model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from sqlalchemy import func, select

from shared.db.client import get_async_session
from shared.db.orm.market_data import MarketData
from shared.logging import get_logger
from shared.models.market_data import MarketDataCreate

logger = get_logger("stock_data_repo")


async def save_stock_prices(
    ticker: str,
    df: pd.DataFrame,
    data_source: str,
) -> int:
    """Save OHLCV price rows to market_data table.

    Maps DataFrame columns: time→data_as_of, open→open_price, high→high_price,
    low→low_price, close→close_price, volume→volume.

    Implements upsert: skips if (ticker_symbol, data_type, data_as_of) already exists.

    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    # Build all candidate records with Pydantic validation
    candidates: list[tuple[datetime, MarketDataCreate]] = []
    for _, row in df.iterrows():
        data_as_of = _to_aware_datetime(row["time"])
        validated = MarketDataCreate(
            ticker_symbol=ticker,
            data_type="stock_price",
            open_price=Decimal(str(row["open"])),
            high_price=Decimal(str(row["high"])),
            low_price=Decimal(str(row["low"])),
            close_price=Decimal(str(row["close"])),
            volume=int(row["volume"]),
            data_as_of=data_as_of,
            data_source=data_source,
        )
        candidates.append((data_as_of, validated))

    inserted = 0
    async with get_async_session() as session:
        # Batch-fetch existing records to avoid N+1 queries
        all_dates = [c[0] for c in candidates]
        existing_result = await session.execute(
            select(MarketData.data_as_of).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "stock_price",
                MarketData.data_as_of.in_(all_dates),
            )
        )
        existing_dates = {row[0] for row in existing_result.all()}

        for data_as_of, validated in candidates:
            if data_as_of in existing_dates:
                continue
            record = MarketData(**validated.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "stock_prices_saved",
        ticker=ticker,
        inserted=inserted,
        total_rows=len(df),
        data_source=data_source,
        component="stock_data_repo",
    )
    return inserted


async def count_stock_prices_batch(tickers: list[str]) -> dict[str, int]:
    """Batch count stock_price rows for multiple tickers."""
    if not tickers:
        return {}
    async with get_async_session() as session:
        result = await session.execute(
            select(
                MarketData.ticker_symbol,
                func.count(MarketData.id),
            )
            .where(MarketData.ticker_symbol.in_(tickers))
            .where(MarketData.data_type == "stock_price")
            .group_by(MarketData.ticker_symbol)
        )
        counts = {row[0]: row[1] for row in result.all()}
        return {t: counts.get(t, 0) for t in tickers}


async def save_financial_ratios(
    ticker: str,
    df: pd.DataFrame,
    data_source: str,
) -> int:
    """Save financial ratio rows to market_data table.

    Extracts P/E, P/B, ROE, EPS from ratio DataFrame.
    Implements upsert: skips if (ticker_symbol, data_type, data_as_of) already exists.

    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    # Parse ratio data — KBS format: rows=items, columns=periods
    ratio_map = _extract_ratios_from_df(df)
    if not ratio_map:
        return 0

    # Build candidates with Pydantic validation
    candidates: list[tuple[datetime, MarketDataCreate]] = []
    for period_str, ratios in ratio_map.items():
        data_as_of = _period_to_datetime(period_str)
        validated = MarketDataCreate(
            ticker_symbol=ticker,
            data_type="financial_ratio",
            pe_ratio=ratios.get("pe_ratio"),
            pb_ratio=ratios.get("pb_ratio"),
            roe=ratios.get("roe"),
            eps=ratios.get("eps"),
            eps_growth_yoy=ratios.get("eps_growth_yoy"),
            data_as_of=data_as_of,
            data_source=data_source,
        )
        candidates.append((data_as_of, validated))

    inserted = 0
    async with get_async_session() as session:
        # Batch-fetch existing records to avoid N+1 queries
        all_dates = [c[0] for c in candidates]
        existing_result = await session.execute(
            select(MarketData.data_as_of).where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "financial_ratio",
                MarketData.data_as_of.in_(all_dates),
            )
        )
        existing_dates = {row[0] for row in existing_result.all()}

        for data_as_of, validated in candidates:
            if data_as_of in existing_dates:
                continue
            record = MarketData(**validated.model_dump())
            session.add(record)
            inserted += 1

        await session.commit()

    logger.info(
        "financial_ratios_saved",
        ticker=ticker,
        inserted=inserted,
        data_source=data_source,
        component="stock_data_repo",
    )
    return inserted


async def get_sector_average_ratios(
    tickers: list[str],
    exclude_ticker: str = "",
) -> dict[str, Decimal | None]:
    """Compute average financial ratios across sector tickers.

    For each ticker, takes the most recent financial_ratio record,
    then averages across tickers.

    Args:
        tickers: List of sector tickers.
        exclude_ticker: Ticker to exclude from averaging.

    Returns:
        Dict with keys: pe, pb, roe, eps. Values are None when no data.
    """
    filtered = [t for t in tickers if t != exclude_ticker]
    if not filtered:
        return {"pe": None, "pb": None, "roe": None, "eps": None}

    # Subquery: latest data_as_of per ticker
    latest_sub = (
        select(
            MarketData.ticker_symbol,
            func.max(MarketData.data_as_of).label("max_as_of"),
        )
        .where(
            MarketData.ticker_symbol.in_(filtered),
            MarketData.data_type == "financial_ratio",
        )
        .group_by(MarketData.ticker_symbol)
        .subquery()
    )

    async with get_async_session() as session:
        result = await session.execute(
            select(
                func.avg(MarketData.pe_ratio).label("avg_pe"),
                func.avg(MarketData.pb_ratio).label("avg_pb"),
                func.avg(MarketData.roe).label("avg_roe"),
                func.avg(MarketData.eps).label("avg_eps"),
            )
            .join(
                latest_sub,
                (MarketData.ticker_symbol == latest_sub.c.ticker_symbol)
                & (MarketData.data_as_of == latest_sub.c.max_as_of),
            )
            .where(MarketData.data_type == "financial_ratio")
        )
        row = result.first()

    if row is None:
        return {"pe": None, "pb": None, "roe": None, "eps": None}

    mapping = row._mapping
    return {
        "pe": mapping["avg_pe"],
        "pb": mapping["avg_pb"],
        "roe": mapping["avg_roe"],
        "eps": mapping["avg_eps"],
    }


async def get_peer_ratios(
    tickers: list[str],
    exclude_ticker: str = "",
) -> list[dict]:
    """Fetch latest financial ratios for each peer ticker.

    Returns list of dicts sorted by ticker, max 5 peers.
    Each dict: {"ticker": str, "pe": float|None, "pb": float|None, "roe": float|None}
    """
    filtered = [t for t in tickers if t != exclude_ticker]
    if not filtered:
        return []

    # Subquery: latest data_as_of per ticker
    latest_sub = (
        select(
            MarketData.ticker_symbol,
            func.max(MarketData.data_as_of).label("max_as_of"),
        )
        .where(
            MarketData.ticker_symbol.in_(filtered),
            MarketData.data_type == "financial_ratio",
        )
        .group_by(MarketData.ticker_symbol)
        .subquery()
    )

    async with get_async_session() as session:
        result = await session.execute(
            select(MarketData)
            .join(
                latest_sub,
                (MarketData.ticker_symbol == latest_sub.c.ticker_symbol)
                & (MarketData.data_as_of == latest_sub.c.max_as_of),
            )
            .where(MarketData.data_type == "financial_ratio")
            .order_by(MarketData.ticker_symbol)
        )
        rows = result.all()

    peers: list[dict] = []
    for row in rows:
        record = row[0] if isinstance(row, tuple) else row
        peers.append({
            "ticker": record.ticker_symbol,
            "pe": float(record.pe_ratio) if record.pe_ratio is not None else None,
            "pb": float(record.pb_ratio) if record.pb_ratio is not None else None,
            "roe": float(record.roe) if record.roe is not None else None,
        })

    return peers[:5]


async def get_latest_financial_ratios(
    ticker: str,
) -> tuple[dict[str, Decimal | None], datetime | None]:
    """Fetch the most recent financial_ratio record for a ticker.

    Returns:
        Tuple of (ratio_dict, data_as_of).
        Empty dict and None when no data exists.
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(MarketData)
            .where(
                MarketData.ticker_symbol == ticker,
                MarketData.data_type == "financial_ratio",
            )
            .order_by(MarketData.data_as_of.desc())
            .limit(1)
        )
        record = result.scalars().first()

    if record is None:
        return {}, None

    ratios: dict[str, Decimal | None] = {
        field: getattr(record, field, None)
        for field in ("pe_ratio", "pb_ratio", "roe", "eps", "eps_growth_yoy")
    }

    return ratios, record.data_as_of


def _extract_ratios_from_df(
    df: pd.DataFrame,
) -> dict[str, dict[str, Decimal | None]]:
    """Extract ratio values from KBS-format DataFrame.

    KBS format: rows have 'item' column with ratio names,
    period columns like '2025-Q4', '2024', etc.
    """
    if "item" not in df.columns:
        return {}

    # Get period columns (exclude 'item' and 'item_id')
    period_cols = [c for c in df.columns if c not in ("item", "item_id")]

    # Build ratio name mapping (case-insensitive)
    ratio_key_map = {
        "p/e": "pe_ratio",
        "pe": "pe_ratio",
        "p/b": "pb_ratio",
        "pb": "pb_ratio",
        "roe": "roe",
        "eps": "eps",
        "eps growth yoy": "eps_growth_yoy",
    }

    result: dict[str, dict[str, Decimal | None]] = {}

    for period in period_cols:
        ratios: dict[str, Decimal | None] = {}
        for _, row in df.iterrows():
            item_name = str(row["item"]).strip().lower()
            orm_field = ratio_key_map.get(item_name)
            if orm_field is not None:
                val = row.get(period)
                if pd.notna(val):
                    ratios[orm_field] = Decimal(str(val))
        if ratios:
            result[period] = ratios

    return result


def _period_to_datetime(period_str: str) -> datetime:
    """Convert period string like '2025-Q4' or '2024' to datetime."""
    if "-Q" in period_str:
        year, quarter = period_str.split("-Q")
        month = int(quarter) * 3
        return datetime(int(year), month, 1, tzinfo=timezone.utc)
    # Year-only format
    return datetime(int(period_str), 12, 31, tzinfo=timezone.utc)


def _to_aware_datetime(value: object) -> datetime:
    """Convert a value to timezone-aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    # Fallback: try parsing string
    dt = pd.Timestamp(value).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
