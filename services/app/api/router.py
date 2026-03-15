"""Central API router — includes all v1 routes."""
from fastapi import APIRouter
from api.v1.health import router as health_router
from services.app.routers.internal import router as internal_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(internal_router)
