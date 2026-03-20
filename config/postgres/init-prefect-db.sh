#!/bin/bash
# Create a separate database for Prefect server to avoid Alembic migration
# conflicts with the application database.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE prefect_db OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect_db')\gexec
EOSQL
