"""Add category column to articles table

Revision ID: rev_002
Revises: rev_001
Create Date: 2026-03-21
"""
import sqlalchemy as sa
from alembic import op

revision = "rev_002"
down_revision = "rev_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column(
            "category", sa.String(20), nullable=False, server_default="stock"
        ),
    )
    op.create_index("idx_articles_category", "articles", ["category"])


def downgrade() -> None:
    op.drop_index("idx_articles_category", table_name="articles")
    op.drop_column("articles", "category")
