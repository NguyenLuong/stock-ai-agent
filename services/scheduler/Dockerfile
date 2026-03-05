FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
# Build context = project root — shared package accessible
COPY shared /shared
RUN uv pip install --system -e /shared
COPY services/scheduler/pyproject.toml .
RUN uv pip install --system \
    "prefect>=2.19" \
    "httpx>=0.27" \
    pytest pytest-asyncio
COPY services/scheduler .
# TODO: Story 2.6 — replace with: prefect worker start --pool default-agent-pool
CMD ["python", "main.py"]
