"""Initial schema — all 4 tables + pgvector extension + HNSW index

Revision ID: rev_001
Revises:
Create Date: 2026-03-05
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "rev_001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension — must run BEFORE creating vector columns
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # articles table
    op.create_table(
        "articles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("ticker_symbol", sa.String(20), nullable=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False, unique=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_content", sa.Text, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "embedded",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # market_data table
    op.create_table(
        "market_data",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ticker_symbol", sa.String(20), nullable=True),
        sa.Column("data_type", sa.String(50), nullable=False),
        sa.Column("open_price", sa.Numeric(15, 2), nullable=True),
        sa.Column("high_price", sa.Numeric(15, 2), nullable=True),
        sa.Column("low_price", sa.Numeric(15, 2), nullable=True),
        sa.Column("close_price", sa.Numeric(15, 2), nullable=True),
        sa.Column("volume", sa.BigInteger, nullable=True),
        sa.Column("indicator_name", sa.String(100), nullable=True),
        sa.Column("indicator_value", sa.Numeric(20, 6), nullable=True),
        sa.Column("pe_ratio", sa.Numeric(10, 4), nullable=True),
        sa.Column("pb_ratio", sa.Numeric(10, 4), nullable=True),
        sa.Column("roe", sa.Numeric(10, 4), nullable=True),
        sa.Column("eps", sa.Numeric(15, 2), nullable=True),
        sa.Column("eps_growth_yoy", sa.Numeric(10, 4), nullable=True),
        sa.Column("data_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data_source", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # recommendations table
    op.create_table(
        "recommendations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("ticker_symbol", sa.String(20), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("bull_case", sa.Text, nullable=True),
        sa.Column("bear_case", sa.Text, nullable=True),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("risk_level", sa.String(20), nullable=True),
        sa.Column("agents_used", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("agents_failed", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("data_sources", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # alerts table
    op.create_table(
        "alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ticker_symbol", sa.String(20), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_data", postgresql.JSONB, nullable=False),
        sa.Column("analysis", sa.Text, nullable=True),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=True),
        sa.Column(
            "telegram_sent",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column("telegram_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # HNSW index for semantic search — must be created AFTER articles table exists
    op.execute("""
        CREATE INDEX idx_articles_embedding_hnsw
        ON articles USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Standard indexes for articles
    op.create_index("idx_articles_ticker", "articles", ["ticker_symbol"])
    op.create_index("idx_articles_published_at", "articles", ["published_at"])
    op.create_index(
        "idx_articles_embedded",
        "articles",
        ["embedded"],
        postgresql_where=sa.text("embedded = FALSE"),
    )

    # Indexes for market_data — data_as_of DESC for "latest data" queries
    op.create_index("idx_market_data_ticker", "market_data", ["ticker_symbol"])
    op.create_index("idx_market_data_type", "market_data", ["data_type"])
    op.create_index(
        "idx_market_data_as_of", "market_data", [sa.text("data_as_of DESC")]
    )

    # Indexes for recommendations — created_at DESC for "latest recommendations" queries
    op.create_index("idx_recommendations_type", "recommendations", ["type"])
    op.create_index("idx_recommendations_ticker", "recommendations", ["ticker_symbol"])
    op.create_index(
        "idx_recommendations_created_at",
        "recommendations",
        [sa.text("created_at DESC")],
    )

    # Indexes for alerts — detected_at DESC for "latest alerts" queries
    op.create_index("idx_alerts_ticker", "alerts", ["ticker_symbol"])
    op.create_index(
        "idx_alerts_detected_at", "alerts", [sa.text("detected_at DESC")]
    )
    op.create_index(
        "idx_alerts_telegram_sent",
        "alerts",
        ["telegram_sent"],
        postgresql_where=sa.text("telegram_sent = FALSE"),
    )


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("recommendations")
    op.drop_table("market_data")
    op.drop_table("articles")
    op.execute("DROP EXTENSION IF EXISTS vector")
