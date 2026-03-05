"""Integration tests for Alembic migrations and pgvector extension.

These tests require a running PostgreSQL instance with pgvector extension available.
Run via: docker-compose run --rm app pytest tests/integration/test_migrations.py

Environment variable required: TEST_DATABASE_URL
"""
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


@pytest.fixture
def test_db_url():
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set — skipping integration tests")
    return url


async def test_rev001_migration_creates_all_tables(test_db_url):
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        )
        tables = {row[0] for row in result}
        assert {"articles", "market_data", "recommendations", "alerts"}.issubset(tables)
    await engine.dispose()


async def test_pgvector_extension_enabled(test_db_url):
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname='vector'")
        )
        assert result.fetchone() is not None
    await engine.dispose()


async def test_hnsw_index_exists(test_db_url):
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE indexname='idx_articles_embedding_hnsw'"
            )
        )
        assert result.fetchone() is not None
    await engine.dispose()


async def test_articles_table_has_embedding_column(test_db_url):
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name='articles' AND column_name='embedding'"
            )
        )
        row = result.fetchone()
        assert row is not None
    await engine.dispose()


async def test_alembic_version_table_exists(test_db_url):
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT version_num FROM alembic_version")
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "rev_001"
    await engine.dispose()


def test_applying_migration_head_twice_is_idempotent(test_db_url):
    """Running 'alembic upgrade head' when already at head must be a no-op — not raise.

    AC3: Existing data is preserved and schema is updated without docker-compose down -v.
    """
    alembic_ini = (
        Path(__file__).parent.parent.parent / "shared" / "db" / "alembic.ini"
    )
    cfg = Config(str(alembic_ini))
    cfg.set_main_option("sqlalchemy.url", test_db_url)
    # DB is already at rev_001 (applied by prior tests / docker-compose setup).
    # Second upgrade head must be a silent no-op — not raise any exception.
    command.upgrade(cfg, "head")
