"""Every configured camera, started and stopped together.

Cameras are independent: one failing to open must not stop the others from
running, so startup is best-effort and the failure surfaces through that
camera's status rather than as an exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Self

from irthermal.camera.base import SourceFactory
from irthermal.camera.runtime import CameraRuntime, CameraStatus
from irthermal.config import Settings
from irthermal.logging import get_logger

log = get_logger(__name__)


class UnknownCameraError(KeyError):
    """No camera is configured under that id."""


class CameraManager:
    """Owns the lifetime of every camera runtime."""

    def __init__(
        self, settings: Settings, factories: dict[str, SourceFactory] | None = None
    ) -> None:
        overrides = factories or {}
        self._runtimes: dict[str, CameraRuntime] = {
            camera.id: CameraRuntime(camera, overrides.get(camera.id))
            for camera in settings.cameras
        }

    def __len__(self) -> int:
        return len(self._runtimes)

    def __iter__(self) -> Iterator[CameraRuntime]:
        return iter(self._runtimes.values())

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._runtimes

    @property
    def ids(self) -> list[str]:
        return list(self._runtimes)

    def get(self, camera_id: str) -> CameraRuntime:
        """Look up a runtime by id."""
        try:
            return self._runtimes[camera_id]
        except KeyError:
            msg = f"unknown camera {camera_id!r}; configured: {sorted(self._runtimes)}"
            raise UnknownCameraError(msg) from None

    async def start(self) -> None:
        """Start every camera. A camera that fails to start is logged, not raised."""
        results = await asyncio.gather(
            *(runtime.start() for runtime in self._runtimes.values()),
            return_exceptions=True,
        )
        for camera_id, result in zip(self._runtimes, results, strict=True):
            if isinstance(result, BaseException):
                log.error("camera.start_failed", camera=camera_id, error=str(result))

        log.info("cameras.started", count=len(self._runtimes))

    async def stop(self) -> None:
        """Stop every camera, whatever any individual one does on the way down."""
        await asyncio.gather(
            *(runtime.stop() for runtime in self._runtimes.values()),
            return_exceptions=True,
        )
        log.info("cameras.stopped", count=len(self._runtimes))

    def statuses(self) -> list[CameraStatus]:
        """Current status of every camera."""
        return [runtime.status() for runtime in self._runtimes.values()]

    @property
    def healthy(self) -> bool:
        """Whether every configured camera is currently connected."""
        return all(status.connected for status in self.statuses())

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
