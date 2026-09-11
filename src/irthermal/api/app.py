"""The ASGI application.

Cameras start when the app starts and stop when it stops, so there is exactly
one capture thread per camera for the process lifetime however many workers or
requests arrive. That also means this app must run single-process: several
uvicorn workers would each open the same device.

Metrics are served from this same app rather than a second HTTP server on
another port, which is what v1 did. One port is one thing to expose, one thing
to proxy and one thing to firewall.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.registry import Collector

from irthermal.api.routes import router
from irthermal.camera.manager import CameraManager
from irthermal.config import Settings
from irthermal.logging import get_logger
from irthermal.metrics import CameraCollector

log = get_logger(__name__)

API_PREFIX = "/api"


def create_app(
    settings: Settings,
    manager: CameraManager | None = None,
    *,
    extra_collectors: list[Collector] | None = None,
) -> FastAPI:
    """Build the application.

    `manager` can be supplied to inject emulated cameras; otherwise one is built
    from the settings.
    """
    camera_manager = manager if manager is not None else CameraManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("server.starting", cameras=camera_manager.ids)
        await camera_manager.start()
        try:
            yield
        finally:
            await camera_manager.stop()
            log.info("server.stopped")

    app = FastAPI(
        title="IRThermal",
        summary="Thermal camera streaming, measurement and metrics.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.manager = camera_manager

    registry = CollectorRegistry()
    registry.register(CameraCollector(camera_manager, settings))
    for collector in extra_collectors or []:
        registry.register(collector)
    app.state.registry = registry

    app.include_router(router, prefix=API_PREFIX)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Prometheus exposition, sampled at scrape time."""
        return PlainTextResponse(
            generate_latest(registry).decode("utf-8"), media_type=CONTENT_TYPE_LATEST
        )

    @app.get("/health", include_in_schema=False)
    async def health_alias() -> RedirectResponse:
        """Convenience alias; the canonical path is under the API prefix."""
        return RedirectResponse(f"{API_PREFIX}/health")

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        """Until the web UI lands, send people somewhere useful."""
        return RedirectResponse("/docs")

    return app
