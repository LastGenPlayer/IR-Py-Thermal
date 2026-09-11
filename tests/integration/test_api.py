"""The HTTP surface, driven against emulated cameras."""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from conftest import FrameSpec, build_capture
from irthermal.api import create_app
from irthermal.camera import CameraManager, EmulatorSource
from irthermal.camera.base import CameraSource, CameraUnavailableError
from irthermal.config import CameraConfig, PointConfig, Settings
from irthermal.core.resolutions import params_for_width

REPLAY_FPS = 200.0
PARAMS = params_for_width(256)
SERVER_START_TIMEOUT_S = 10.0
SERVER_STOP_TIMEOUT_S = 10.0


def emulator(spec: FrameSpec | None = None):
    capture = build_capture(PARAMS, spec)

    def build() -> CameraSource:
        return EmulatorSource.from_array(capture, fps=REPLAY_FPS)

    return build


def unavailable() -> CameraSource:
    raise CameraUnavailableError("unplugged")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        cameras=[
            CameraConfig(
                id="rack",
                device="/dev/video0",
                colormap="inferno",
                points=[
                    PointConfig(name="hiend-1", x=43, y=125, labels={"group": "hiend"}),
                    PointConfig(name="ind-1", x=132, y=120, labels={"group": "ind"}),
                    PointConfig(name="off-frame", x=5_000, y=5),
                ],
            ),
            CameraConfig(id="bench", device="/dev/video1"),
        ]
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    manager = CameraManager(settings, {"rack": emulator(), "bench": emulator()})
    app = create_app(settings, manager)

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        await manager.get("rack").wait_for_frame(timeout=5.0)
        await manager.get("bench").wait_for_frame(timeout=5.0)
        yield http_client


@pytest.fixture
async def degraded_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """One working camera, one that cannot be opened."""
    manager = CameraManager(settings, {"rack": emulator(), "bench": unavailable})
    app = create_app(settings, manager)

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        await manager.get("rack").wait_for_frame(timeout=5.0)
        yield http_client


@asynccontextmanager
async def running_server(settings: Settings) -> AsyncIterator[str]:
    """A real uvicorn server on an ephemeral port, for as long as the block runs.

    Two reasons this is a context manager used inside the test rather than a
    fixture. httpx's ASGITransport buffers a response to completion before
    handing it over, so an endless MJPEG stream can only be exercised over a
    real socket. And an async fixture can end up owning a different event loop
    than the test body, which leaves the server's sockets unserviced and the
    request hanging forever -- entering it here keeps both in one loop.
    """
    import uvicorn

    manager = CameraManager(settings, {"rack": emulator(), "bench": emulator()})
    app = create_app(settings, manager)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, access_log=False)
    )
    task = asyncio.create_task(server.serve())

    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("server task exited before it finished starting")
        if time.monotonic() > deadline:
            task.cancel()
            raise RuntimeError("server never reported itself started")
        await asyncio.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=SERVER_STOP_TIMEOUT_S)


# --- service ----------------------------------------------------------------


async def test_health_reports_all_cameras_connected(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "cameras": 2, "connected": 2}


async def test_health_is_degraded_but_still_ok_when_a_camera_is_down(
    degraded_client: AsyncClient,
) -> None:
    """Liveness must not fail just because hardware is missing."""
    response = await degraded_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["connected"] == 1


async def test_ready_requires_every_camera(degraded_client: AsyncClient) -> None:
    response = await degraded_client.get("/api/ready")

    assert response.status_code == 503
    assert "bench" in response.json()["detail"]


async def test_ready_succeeds_when_all_are_up(client: AsyncClient) -> None:
    assert (await client.get("/api/ready")).status_code == 204


async def test_colormaps_are_listed(client: AsyncClient) -> None:
    names = (await client.get("/api/colormaps")).json()

    assert "inferno" in names
    assert "gray" in names


async def test_openapi_schema_is_served(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/cameras/{camera_id}/stream.mjpg" in response.json()["paths"]


# --- cameras ----------------------------------------------------------------


async def test_lists_every_configured_camera(client: AsyncClient) -> None:
    body = (await client.get("/api/cameras")).json()

    assert [c["id"] for c in body] == ["rack", "bench"]
    assert all(c["connected"] for c in body)
    assert body[0]["width"] == 256


async def test_camera_detail_includes_telemetry(client: AsyncClient) -> None:
    body = (await client.get("/api/cameras/rack")).json()

    assert body["id"] == "rack"
    assert body["sequence"] >= 1
    assert body["telemetry"]["serial"] == "SN0042"
    assert body["adjustments"]["user_offset"] == 0.0


async def test_unknown_camera_is_a_404_naming_the_real_ones(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/nope")

    assert response.status_code == 404
    assert "rack" in response.json()["detail"]


async def test_requests_to_a_disconnected_camera_report_503(
    degraded_client: AsyncClient,
) -> None:
    response = await degraded_client.get("/api/cameras/bench/temperatures")

    assert response.status_code == 503
    assert "not connected" in response.json()["detail"]


# --- imagery ----------------------------------------------------------------


async def test_snapshot_returns_a_jpeg(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/rack/snapshot.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8\xff")  # JPEG SOI
    assert int(response.headers["x-frame-sequence"]) >= 1


async def test_snapshot_returns_a_png(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/rack/snapshot.png")

    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n")


async def test_snapshot_honours_upscaling(client: AsyncClient) -> None:
    import cv2

    plain = await client.get("/api/cameras/rack/snapshot.png")
    scaled = await client.get("/api/cameras/rack/snapshot.png?upscale=4")

    decoded_plain = cv2.imdecode(np.frombuffer(plain.content, np.uint8), cv2.IMREAD_COLOR)
    decoded_scaled = cv2.imdecode(np.frombuffer(scaled.content, np.uint8), cv2.IMREAD_COLOR)

    assert decoded_plain.shape[:2] == (192, 256)
    assert decoded_scaled.shape[:2] == (192 * 4, 256 * 4)


async def test_colormap_changes_the_image(client: AsyncClient) -> None:
    inferno = await client.get("/api/cameras/rack/snapshot.png?colormap=inferno")
    gray = await client.get("/api/cameras/rack/snapshot.png?colormap=gray")

    assert inferno.content != gray.content


async def test_unknown_colormap_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/rack/snapshot.jpg?colormap=chartreuse")

    assert response.status_code == 400
    assert "unknown colormap" in response.json()["detail"]


async def _collect_parts(base_url: str, path: str, parts: int) -> bytearray:
    """Read an MJPEG stream until `parts` boundaries have arrived, then stop."""
    seen = bytearray()
    async with (
        AsyncClient(base_url=base_url, timeout=10.0) as client,
        client.stream("GET", path) as response,
    ):
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]

        async for chunk in response.aiter_bytes():
            seen.extend(chunk)
            if seen.count(b"--frame") >= parts:
                break
    return seen


async def test_stream_delivers_multipart_jpeg(settings: Settings) -> None:
    """The MJPEG contract: a boundary, then JPEGs, indefinitely."""
    async with running_server(settings) as base_url:
        chunks = await _collect_parts(base_url, "/api/cameras/rack/stream.mjpg?fps=60", 3)

    assert b"Content-Type: image/jpeg" in chunks
    assert b"Content-Length:" in chunks
    assert chunks.count(b"\xff\xd8\xff") >= 2  # several JPEG frames


async def test_stream_is_shared_between_viewers(settings: Settings) -> None:
    """Two viewers must not mean two captures; in v1 every consumer read()."""
    async with running_server(settings) as base_url:
        first, second = await asyncio.gather(
            _collect_parts(base_url, "/api/cameras/rack/stream.mjpg", 2),
            _collect_parts(base_url, "/api/cameras/rack/stream.mjpg", 2),
        )

        async with AsyncClient(base_url=base_url, timeout=10.0) as client:
            status = (await client.get("/api/cameras/rack")).json()

    assert first.count(b"--frame") >= 2
    assert second.count(b"--frame") >= 2
    # One capture thread served both, and both have since disconnected.
    assert status["frames"] >= 2
    assert status["subscribers"] == 0


async def test_raw_frame_round_trips_through_numpy(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/rack/frame.npy")

    assert response.status_code == 200
    array = np.load(io.BytesIO(response.content), allow_pickle=False)
    assert array.shape == (192, 256)
    assert array.dtype == np.uint16


# --- measurement ------------------------------------------------------------


async def test_temperatures_return_configured_points(client: AsyncClient) -> None:
    body = (await client.get("/api/cameras/rack/temperatures")).json()

    assert set(body["points"]) == {"hiend-1", "ind-1"}
    assert body["points"]["hiend-1"]["x"] == 43
    assert body["minimum"]["celsius"] <= body["maximum"]["celsius"]
    assert body["center"]["x"] == 128


async def test_temperature_at_a_pixel(client: AsyncClient) -> None:
    body = (await client.get("/api/cameras/rack/temperature?x=10&y=20")).json()

    assert (body["x"], body["y"]) == (10, 20)
    assert isinstance(body["celsius"], float)


async def test_temperature_outside_the_frame_is_a_400(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/rack/temperature?x=9999&y=0")

    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


async def test_region_stats(client: AsyncClient) -> None:
    response = await client.post(
        "/api/cameras/rack/region", json={"x1": 10, "y1": 10, "x2": 50, "y2": 50}
    )
    body = response.json()

    assert response.status_code == 200
    assert 10 <= body["maximum"]["x"] < 50
    assert body["minimum"]["celsius"] <= body["mean_celsius"] <= body["maximum"]["celsius"]


async def test_zero_area_region_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/cameras/rack/region", json={"x1": 10, "y1": 10, "x2": 10, "y2": 50}
    )

    assert response.status_code == 422


# --- control ----------------------------------------------------------------


async def test_calibrate_is_accepted(client: AsyncClient) -> None:
    response = await client.post("/api/cameras/rack/calibrate")

    assert response.status_code == 202
    assert response.json()["camera"] == "rack"


async def test_switching_to_the_high_range_applies_its_gain(client: AsyncClient) -> None:
    response = await client.post("/api/cameras/rack/range", json={"mode": "high"})
    assert response.status_code == 202

    detail = (await client.get("/api/cameras/rack")).json()
    assert detail["adjustments"]["gain_m"] == pytest.approx(1.17)


async def test_invalid_range_mode_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/api/cameras/rack/range", json={"mode": "sideways"})

    assert response.status_code == 422


async def test_patching_parameters_updates_the_offset(client: AsyncClient) -> None:
    response = await client.patch(
        "/api/cameras/rack/parameters", json={"temperature_offset": -12.3}
    )

    assert response.status_code == 200
    assert response.json()["user_offset"] == pytest.approx(-12.3)


async def test_patching_device_parameters_is_accepted(client: AsyncClient) -> None:
    response = await client.patch(
        "/api/cameras/rack/parameters", json={"emissivity": 0.9, "distance_m": 2}
    )

    assert response.status_code == 200


async def test_empty_parameter_patch_is_rejected(client: AsyncClient) -> None:
    response = await client.patch("/api/cameras/rack/parameters", json={})

    assert response.status_code == 422


async def test_out_of_range_emissivity_is_rejected(client: AsyncClient) -> None:
    response = await client.patch("/api/cameras/rack/parameters", json={"emissivity": 1.5})

    assert response.status_code == 422


# --- metrics ----------------------------------------------------------------


async def test_metrics_expose_labelled_points(client: AsyncClient) -> None:
    """v1 hardcoded three metric names and dropped anything past fifteen points."""
    body = (await client.get("/metrics")).text

    assert 'thermal_point_celsius{camera="rack",group="hiend",point="hiend-1"}' in body
    assert 'thermal_point_celsius{camera="rack",group="ind",point="ind-1"}' in body


async def test_metrics_report_camera_health(client: AsyncClient) -> None:
    body = (await client.get("/metrics")).text

    assert 'thermal_camera_up{camera="rack"} 1.0' in body
    assert 'thermal_camera_up{camera="bench"} 1.0' in body
    assert "thermal_camera_frames_total" in body


async def test_metrics_show_a_disconnected_camera_as_down(
    degraded_client: AsyncClient,
) -> None:
    body = (await degraded_client.get("/metrics")).text

    assert 'thermal_camera_up{camera="rack"} 1.0' in body
    assert 'thermal_camera_up{camera="bench"} 0.0' in body


async def test_metrics_include_frame_extremes(client: AsyncClient) -> None:
    body = (await client.get("/metrics")).text

    assert 'thermal_frame_min_celsius{camera="rack"}' in body
    assert 'thermal_frame_max_celsius{camera="rack"}' in body
    assert 'thermal_frame_age_seconds{camera="rack"}' in body


async def test_metrics_are_sampled_at_scrape_time(client: AsyncClient) -> None:
    """A shifted offset must show up on the very next scrape, not five seconds later."""
    before = (await client.get("/metrics")).text
    await client.patch("/api/cameras/rack/parameters", json={"temperature_offset": 50.0})

    # Wait for a frame carrying the new table.
    for _ in range(50):
        after = (await client.get("/metrics")).text
        if after != before:
            break
    else:  # pragma: no cover
        pytest.fail("metrics never reflected the new offset")

    assert after != before
