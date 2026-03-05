# Stock AI Agent

AI-powered Vietnamese stock market analysis system delivering daily morning briefings and real-time alerts via Telegram.

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
3. Start all services:
   ```bash
   docker-compose up -d
   ```
4. Verify health:
   ```bash
   curl http://localhost:8000/health
   ```

## Services

| Service     | Description                          |
|-------------|--------------------------------------|
| `postgres`  | PostgreSQL with pgvector extension   |
| `crawler`   | News & market data ingestion         |
| `scheduler` | Prefect-based task scheduling        |
| `app`       | FastAPI + LangGraph AI agent         |

## Project Structure

```
stock-ai-agent/
├── config/          # YAML configuration (models, prompts, schedules)
├── services/        # Microservices (app, crawler, scheduler)
├── shared/          # Shared Python package
├── tests/           # Unit and integration tests
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Development

Stop all services:
```bash
docker-compose down
```

Restart cleanly:
```bash
docker-compose down && docker-compose up -d
```
