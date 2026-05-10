import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.getenv("LOG_FILE", os.path.join(BASE_DIR, "app.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, sessions, places
from config.settings import get_settings


ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

settings = get_settings()


app = FastAPI(
    title="Warsaw Spot Guide API",
    version="1.0.0",
    docs_url=None if ENVIRONMENT == "prod" else "/docs",
    redoc_url=None if ENVIRONMENT == "prod" else "/redoc",
    openapi_url=None if ENVIRONMENT == "prod" else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["server"] = "spotguide"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(places.router, prefix="/places", tags=["places"])


@app.get("/health")
def health():
    return {"status": "ok"}
