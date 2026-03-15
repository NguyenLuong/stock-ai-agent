FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
# Build context = project root — shared package accessible
COPY shared /shared
RUN uv pip install --system -e /shared
COPY services/scheduler/pyproject.toml .
RUN uv pip install --system -e . && \
    uv pip install --system pytest pytest-asyncio
COPY services/scheduler .
CMD ["python", "main.py"]
