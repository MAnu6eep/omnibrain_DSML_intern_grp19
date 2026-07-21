from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omnibrain.app.api.routes.ingestion import router as ingestion_router
from omnibrain.app.core.constants import APP_NAME, APP_VERSION


def create_app() -> FastAPI:
    # 1. Initialize core application using repository constants
    app = FastAPI(
        title=APP_NAME, description="Backend API for OmniBrain", version=APP_VERSION
    )

    # 2. Inject your teammate's required CORS middleware configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Teammate's root landing endpoint
    @app.get("/")
    def root_check():
        return {"status": "success", "message": "OmniBrain Backend Running", "timestamp": datetime.now(timezone.utc).isoformat(),}


    
    # 4. Original standardized health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}
    app.include_router(ingestion_router)

    return app


app = create_app()
