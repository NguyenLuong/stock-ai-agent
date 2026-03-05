"""Unit tests for shared.db.client — connection pool and session factory."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


@pytest.fixture(autouse=True)
def reset_db_client():
    """Reset module-level singletons between tests."""
    from shared.db import client as db_client
    db_client._engine = None
    db_client._session_factory = None
    yield
    db_client._engine = None
    db_client._session_factory = None


class TestGetEngine:
    def test_returns_async_engine(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_engine
        engine = get_engine()
        assert isinstance(engine, AsyncEngine)

    def test_singleton_same_instance(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_engine
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2

    def test_raises_if_database_url_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            from shared.db import client as db_client
            db_client._engine = None
            with pytest.raises(KeyError):
                db_client.get_engine()

    def test_pool_pre_ping_enabled(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_engine
        engine = get_engine()
        assert engine.pool._pre_ping is True


class TestGetSessionFactory:
    def test_returns_async_sessionmaker(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_session_factory
        factory = get_session_factory()
        assert isinstance(factory, async_sessionmaker)

    def test_singleton_same_instance(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_session_factory
        f1 = get_session_factory()
        f2 = get_session_factory()
        assert f1 is f2

    def test_session_class_is_async_session(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_session_factory
        factory = get_session_factory()
        assert factory.class_ is AsyncSession

    def test_expire_on_commit_false(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_session_factory
        factory = get_session_factory()
        assert factory.kw.get("expire_on_commit") is False


class TestResetEngine:
    def test_reset_clears_singletons(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db import client as db_client
        db_client.get_engine()
        db_client.get_session_factory()
        assert db_client._engine is not None
        assert db_client._session_factory is not None

        db_client.reset_engine()
        assert db_client._engine is None
        assert db_client._session_factory is None

    def test_new_engine_created_after_reset(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_engine, reset_engine
        e1 = get_engine()
        reset_engine()
        e2 = get_engine()
        assert e1 is not e2


class TestGetAsyncSession:
    @pytest.mark.asyncio
    async def test_yields_async_session(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
        from shared.db.client import get_async_session

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session

        with patch("shared.db.client.get_session_factory", return_value=mock_factory):
            async with get_async_session() as session:
                assert session is mock_session
