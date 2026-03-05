"""Central API router — includes all v1 routes."""
from fastapi import APIRouter
from api.v1.health import router as health_router

# TODO: Story 4.x — add chat, internal trigger routers here
router = APIRouter()
router.include_router(health_router, tags=["health"])
