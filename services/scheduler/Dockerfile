FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
# Build context = project root — shared package accessible
COPY shared /shared-pkg/shared
COPY shared/pyproject.toml /shared-pkg/
RUN uv pip install --system -e /shared-pkg
COPY services/scheduler/pyproject.toml .
RUN uv pip install --system \
    "prefect>=2.19" \
    "httpx>=0.27" \
    pytest pytest-asyncio
COPY services/scheduler .
CMD ["python", "main.py"]
