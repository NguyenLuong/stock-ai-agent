FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
# Build context = project root — shared package accessible
COPY shared /shared
RUN uv pip install --system -e /shared
COPY services/crawler/pyproject.toml .
RUN uv pip install --system \
    "scrapy>=2.11" \
    "httpx>=0.27" \
    "beautifulsoup4>=4.12" \
    pytest pytest-asyncio
COPY services/crawler .
CMD ["python", "main.py"]
