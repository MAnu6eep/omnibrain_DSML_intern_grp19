import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from omnibrain.app.api.routes.pdf import router as pdf_router
from omnibrain.app.api.routes.chat import router as chat_router
from omnibrain.app.api.routes.ingestion import router as ingestion_router
from omnibrain.app.core.constants import APP_NAME, APP_VERSION
from omnibrain.app.core.logging import logger

load_dotenv()

print("=" * 80)
print("ENV CHECK")
print("GEMINI =", "Present" if os.getenv("GEMINI_API_KEY") else "NOT SET")
print("=" * 80)


def create_app() -> FastAPI:
    app = FastAPI(
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(
    "/static/pdf_images",
    StaticFiles(directory="static/pdf_images"),
    name="pdf_images",
)

    app.mount(
    "/static/pdfs",
    StaticFiles(directory="static/pdfs"),
    name="pdfs",
    )

    @app.get("/")
    async def root():
        return {
            "status": "success",
            "message": "OmniBrain Backend Running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    # -------------------------
    # Register Routers
    # -------------------------

    app.include_router(
        ingestion_router,
        prefix="/api/v1/ingestion",
        tags=["Ingestion"],
    )

    app.include_router(
        chat_router,
        prefix="/api/v1/chat",
        tags=["Chat"],
    )
    app.include_router(
    pdf_router,
    prefix="/api/v1/pdf",
    tags=["PDF"],
    )

    logger.info(
        "Registered %s routes",
        len([route for route in app.routes if isinstance(route, APIRoute)]),
    )

    return app


app = create_app()
