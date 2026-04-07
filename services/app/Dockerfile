FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
# Build context = project root — shared package accessible
COPY shared /shared-pkg/shared
COPY shared/pyproject.toml /shared-pkg/
RUN uv pip install --system -e /shared-pkg
COPY services/app/pyproject.toml .
RUN uv pip install --system \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "langgraph>=0.1" \
    "python-telegram-bot>=21.0" \
    pytest pytest-asyncio "httpx>=0.27"
COPY services/crawler /crawler
RUN uv pip install --system \
    "httpx>=0.27" \
    "beautifulsoup4>=4.12" \
    "vnstock>=3.0" \
    "pandas>=2.0" \
    "tenacity>=8.0" \
    "pyrate-limiter>=4.0" \
    "lxml>=5.0" \
    "defusedxml>=0.7" \
    "pandas-ta>=0.3.14b"
COPY services/app .
# Create namespace so "from services.app.*" and "from services.crawler.*" resolve.
# /pylib/services/ acts as the 'services' package; symlinks point to real dirs.
RUN mkdir -p /pylib/services \
    && touch /pylib/services/__init__.py \
    && ln -s /app /pylib/services/app \
    && ln -s /crawler /pylib/services/crawler
ENV PYTHONPATH="/pylib:/crawler:${PYTHONPATH}"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
