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
COPY services/app .
ENV PYTHONPATH="/crawler:${PYTHONPATH}"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
