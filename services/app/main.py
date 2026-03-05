from fastapi import FastAPI
from api.router import router

app = FastAPI(title="Stock AI Agent")
app.include_router(router)
