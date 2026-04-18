"""Tests for _extract_bctc_latest helper in fundamental_analysis.node.

Verifies KBS Finance DataFrame parsing for latest-period BCTC values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from services.app.agents.fundamental_analysis.node import _extract_bctc_latest


def _make_income_df(
    net_revenue: float | None = 10000.0,
    net_profit: float | None = 2000.0,
    latest_period: str = "2025-Q4",
    previous_period: str = "2025-Q3",
) -> pd.DataFrame:
    """Build a KBS-format income_statement DataFrame."""
    rows = [
        {
            "item": "Doanh thu thuần",
            "item_en": "Net revenue",
            "item_id": "net_revenue",
            "unit": "VND",
            "levels": 0,
            "row_number": 1,
            previous_period: 9000.0,
            latest_period: net_revenue,
        },
        {
            "item": "Lợi nhuận sau thuế",
            "item_en": "Profit after tax",
            "item_id": "profit_after_tax",
            "unit": "VND",
            "levels": 0,
            "row_number": 2,
            previous_period: 1800.0,
            latest_period: net_profit,
        },
    ]
    return pd.DataFrame(rows)


def _make_balance_df(
    total_liabilities: float | None = 4000.0,
    total_equity: float | None = 8000.0,
    current_assets: float | None = 3000.0,
    current_liabilities: float | None = 1500.0,
    latest_period: str = "2025-Q4",
) -> pd.DataFrame:
    rows = [
        {
            "item": "Tổng nợ phải trả",
            "item_en": "Total liabilities",
            "item_id": "total_liabilities",
            "unit": "VND",
            "levels": 0,
            "row_number": 1,
            "2025-Q3": 3800.0,
            latest_period: total_liabilities,
        },
        {
            "item": "Vốn chủ sở hữu",
            "item_en": "Owners equity",
            "item_id": "total_equity",
            "unit": "VND",
            "levels": 0,
            "row_number": 2,
            "2025-Q3": 7800.0,
            latest_period: total_equity,
        },
        {
            "item": "Tài sản ngắn hạn",
            "item_en": "Current assets",
            "item_id": "current_assets",
            "unit": "VND",
            "levels": 0,
            "row_number": 3,
            "2025-Q3": 2900.0,
            latest_period: current_assets,
        },
        {
            "item": "Nợ ngắn hạn",
            "item_en": "Current liabilities",
            "item_id": "current_liabilities",
            "unit": "VND",
            "levels": 0,
            "row_number": 4,
            "2025-Q3": 1400.0,
            latest_period: current_liabilities,
        },
    ]
    return pd.DataFrame(rows)


class TestExtractBctcLatest:
    def test_extracts_all_fields_from_full_dataframes(self):
        income = _make_income_df(net_revenue=12000.0, net_profit=2400.0)
        balance = _make_balance_df(
            total_liabilities=5000.0,
            total_equity=10000.0,
            current_assets=4000.0,
            current_liabilities=2000.0,
        )
        result = _extract_bctc_latest(income, balance)
        assert result["revenue"] == 12000.0
        assert result["net_profit"] == 2400.0
        assert result["debt_to_equity"] == 0.5
        assert result["current_ratio"] == 2.0

    def test_picks_latest_period_by_label_sort(self):
        income = _make_income_df(
            net_revenue=99999.0,
            latest_period="2025-Q4",
            previous_period="2025-Q1",
        )
        result = _extract_bctc_latest(income, Exception("no balance"))
        # 2025-Q4 > 2025-Q1 → latest column selected
        assert result["revenue"] == 99999.0

    def test_exception_input_returns_all_none(self):
        result = _extract_bctc_latest(
            Exception("income fetch failed"),
            Exception("balance fetch failed"),
        )
        assert result == {
            "revenue": None,
            "net_profit": None,
            "debt_to_equity": None,
            "current_ratio": None,
        }

    def test_empty_dataframe_returns_all_none(self):
        empty = pd.DataFrame()
        result = _extract_bctc_latest(empty, empty)
        assert result["revenue"] is None
        assert result["net_profit"] is None
        assert result["debt_to_equity"] is None
        assert result["current_ratio"] is None

    def test_partial_data_income_only(self):
        income = _make_income_df(net_revenue=5000.0, net_profit=800.0)
        result = _extract_bctc_latest(income, Exception("no balance"))
        assert result["revenue"] == 5000.0
        assert result["net_profit"] == 800.0
        assert result["debt_to_equity"] is None
        assert result["current_ratio"] is None

    def test_zero_equity_yields_none_debt_ratio(self):
        balance = _make_balance_df(total_liabilities=100.0, total_equity=0.0)
        result = _extract_bctc_latest(Exception("x"), balance)
        assert result["debt_to_equity"] is None

    def test_zero_current_liab_yields_none_current_ratio(self):
        balance = _make_balance_df(current_assets=500.0, current_liabilities=0.0)
        result = _extract_bctc_latest(Exception("x"), balance)
        assert result["current_ratio"] is None

    def test_missing_metric_row_returns_none(self):
        # DataFrame with only unrelated rows
        df = pd.DataFrame([
            {
                "item": "Unrelated row",
                "item_en": "Unrelated",
                "item_id": "unrelated_metric",
                "unit": "VND",
                "levels": 0,
                "row_number": 1,
                "2025-Q4": 500.0,
            },
        ])
        result = _extract_bctc_latest(df, df)
        assert result["revenue"] is None
        assert result["net_profit"] is None

    def test_vietnamese_fallback_matches_via_item_column(self):
        """When item_id is missing/non-standard, Vietnamese `item` column should still match."""
        df = pd.DataFrame([
            {
                "item": "Doanh thu thuần từ bán hàng",
                "item_en": "",
                "item_id": "",
                "unit": "VND",
                "levels": 0,
                "row_number": 1,
                "2025-Q4": 7777.0,
            },
        ])
        result = _extract_bctc_latest(df, Exception("x"))
        assert result["revenue"] == 7777.0
