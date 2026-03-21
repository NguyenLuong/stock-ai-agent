"""Article model — news and research articles from crawlers."""
import uuid
from datetime import datetime

from typing import Literal

from pydantic import BaseModel


class ArticleBase(BaseModel):
    source: str
    ticker_symbol: str | None = None
    title: str
    url: str
    published_at: datetime
    raw_content: str | None = None
    summary: str | None = None
    embedding: list[float] | None = None
    embedded: bool = False
    category: Literal["stock", "macro"] = "stock"


class ArticleCreate(ArticleBase):
    pass


class Article(ArticleBase):
    id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}
