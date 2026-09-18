"""Running the HTTP server, and stopping it without a wall of tracebacks.

Uvicorn shuts down in a fixed order: stop accepting, wait for in-flight
requests, then run the application's lifespan shutdown. That order does not suit
this server. An MJPEG response and a telemetry socket both end only when their
subscription does, and the thing that would end those subscriptions is the
lifespan shutdown -- which has not run yet. So the wait always reaches its
timeout, every live response is cancelled, and uvicorn logs each cancellation it
caused as an error.

Ending the subscriptions first turns that into an ordinary shutdown: the
responses complete, the connections close, and the wait finishes on its own.
"""

from __future__ import annotations

import socket

import uvicorn

from irthermal.api import create_app
from irthermal.camera.manager import CameraManager
from irthermal.config import Settings

SHUTDOWN_TIMEOUT_S = 5
"""How long to wait for in-flight requests before cutting them off.

Uvicorn waits forever by default. With streams that only end when the client
hangs up, a single open browser tab would leave Ctrl+C hanging with the listener
already closed and the capture threads still running. This is the backstop for
anything `close_streams` does not cover.
"""


class StreamAwareServer(uvicorn.Server):
    """A server that releases its subscribers before waiting on connections."""

    def __init__(self, config: uvicorn.Config, manager: CameraManager) -> None:
        super().__init__(config)
        self._manager = manager

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Before `super()` starts waiting for connections that would never end.
        self._manager.close_streams()
        await super().shutdown(sockets)


def run(settings: Settings, *, host: str | None = None, port: int | None = None) -> None:
    """Serve until interrupted."""
    app = create_app(settings)
    manager: CameraManager = app.state.manager

    config = uvicorn.Config(
        app,
        host=host or settings.server.host,
        port=port or settings.server.port,
        log_config=None,
        timeout_graceful_shutdown=SHUTDOWN_TIMEOUT_S,
    )
    StreamAwareServer(config, manager).run()
