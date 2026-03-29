"""Unit tests for ticker_config — config loading, dedup, validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from services.crawler.market_data.ticker_config import (
    TickerConfig,
    get_sector_for_ticker,
    load_ticker_config,
)


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a temporary stock_tickers.yaml for testing."""
    config = {
        "groups": {
            "vn30": {
                "enabled": True,
                "description": "VN30",
                "tickers": ["VNM", "VHM", "HPG", "VCB"],
            },
            "banking": {
                "enabled": True,
                "description": "Banking",
                "tickers": ["VCB", "BID", "CTG"],
            },
            "steel": {
                "enabled": False,
                "description": "Steel",
                "tickers": ["HPG", "HSG"],
            },
        },
        "holidays": {
            2026: [
                "2026-02-16",
                "2026-02-17",
                "2026-02-18",
            ],
        },
    }
    config_file = tmp_path / "stock_tickers.yaml"
    config_file.write_text(yaml.dump(config))
    return config_file


@pytest.fixture
def empty_config_path(tmp_path: Path) -> Path:
    """Config with no groups."""
    config_file = tmp_path / "stock_tickers.yaml"
    config_file.write_text(yaml.dump({"groups": {}}))
    return config_file


class TestLoadTickerConfig:
    """Tests for load_ticker_config."""

    def test_loads_and_deduplicates_tickers(self, sample_config_path: Path) -> None:
        """Enabled groups are merged and deduplicated."""
        result = load_ticker_config(config_path=sample_config_path)
        assert isinstance(result, TickerConfig)
        # VCB appears in vn30 + banking, HPG in vn30 + steel (disabled)
        # steel is disabled, so only vn30 + banking
        expected = {"VNM", "VHM", "HPG", "VCB", "BID", "CTG"}
        assert set(result.tickers) == expected

    def test_disabled_groups_excluded(self, sample_config_path: Path) -> None:
        """Disabled groups are not included."""
        result = load_ticker_config(config_path=sample_config_path)
        # HSG only in disabled steel group
        assert "HSG" not in result.tickers

    def test_empty_config_returns_empty_list(self, empty_config_path: Path) -> None:
        """Empty config returns empty ticker list."""
        result = load_ticker_config(config_path=empty_config_path)
        assert result.tickers == []

    def test_validates_ticker_format(self, tmp_path: Path) -> None:
        """Invalid ticker format (lowercase, too long) is filtered out."""
        config = {
            "groups": {
                "test": {
                    "enabled": True,
                    "description": "Test",
                    "tickers": ["VNM", "invalid", "TOOLONG5", "AB", "FPT"],
                },
            },
        }
        config_file = tmp_path / "stock_tickers.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_ticker_config(config_path=config_file)
        # Only 3-4 char uppercase tickers are valid
        assert set(result.tickers) == {"VNM", "FPT"}

    def test_ticker_count_metadata(self, sample_config_path: Path) -> None:
        """TickerConfig has correct count metadata."""
        result = load_ticker_config(config_path=sample_config_path)
        assert result.total_count == len(result.tickers)
        assert result.enabled_groups == 2  # vn30 + banking (steel disabled)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_ticker_config(config_path=tmp_path / "nonexistent.yaml")

    def test_loads_holidays(self, sample_config_path: Path) -> None:
        """Holidays are loaded from config."""
        result = load_ticker_config(config_path=sample_config_path)
        from datetime import date

        assert date(2026, 2, 16) in result.holidays
        assert date(2026, 2, 17) in result.holidays
        assert date(2026, 2, 18) in result.holidays


@pytest.fixture
def sector_config_path(tmp_path: Path) -> Path:
    """Config with multiple enabled sector groups for get_sector_for_ticker tests."""
    config = {
        "groups": {
            "vn30": {
                "enabled": True,
                "description": "VN30",
                "tickers": ["VNM", "VHM", "HPG", "VCB", "SSI", "GAS"],
            },
            "steel": {
                "enabled": True,
                "description": "Thep",
                "tickers": ["HPG", "HSG", "NKG", "TLH"],
            },
            "banking": {
                "enabled": True,
                "description": "Ngan hang",
                "tickers": ["VCB", "BID", "CTG"],
            },
            "oil_gas": {
                "enabled": True,
                "description": "Dau khi",
                "tickers": ["GAS", "PLX"],
            },
            "securities": {
                "enabled": True,
                "description": "Chung khoan",
                "tickers": ["SSI", "VND"],
            },
        },
    }
    config_file = tmp_path / "stock_tickers.yaml"
    config_file.write_text(yaml.dump(config))
    return config_file


class TestGetSectorForTicker:
    """Tests for get_sector_for_ticker."""

    def test_ticker_in_sector_group(self, sector_config_path: Path) -> None:
        """Test ticker found in a sector group (not vn30)."""
        sector_name, tickers = get_sector_for_ticker("HPG", config_path=sector_config_path)
        assert sector_name == "Thép"
        assert "HPG" in tickers
        assert "HSG" in tickers

    def test_ticker_only_in_vn30(self, sector_config_path: Path) -> None:
        """Test ticker only in vn30 returns VN30 sector."""
        sector_name, tickers = get_sector_for_ticker("VNM", config_path=sector_config_path)
        assert sector_name == "VN30"
        assert "VNM" in tickers

    def test_ticker_not_found(self, sector_config_path: Path) -> None:
        """Test ticker not in any group returns unknown."""
        sector_name, tickers = get_sector_for_ticker("XYZ", config_path=sector_config_path)
        assert sector_name == "Không xác định"
        assert tickers == ["XYZ"]

    def test_banking_sector(self, sector_config_path: Path) -> None:
        """Test banking sector detection."""
        sector_name, tickers = get_sector_for_ticker("VCB", config_path=sector_config_path)
        assert sector_name == "Ngân hàng"
        assert "BID" in tickers

    def test_securities_sector(self, sector_config_path: Path) -> None:
        """Test securities sector detection."""
        sector_name, tickers = get_sector_for_ticker("SSI", config_path=sector_config_path)
        assert sector_name == "Chứng khoán"
        assert "VND" in tickers
