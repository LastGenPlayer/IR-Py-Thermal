"""The HTTP surface, driven against emulated cameras."""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import InvalidStatus

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


LOCKIN_FREQUENCY_HZ = 4.0
LOCKIN_AMPLITUDE = 8_000.0
LOCKIN_BASE = 8_000.0
"""Equal to the amplitude, so the signal swings down to zero and the pedestal is
as small as a non-negative raw value allows.

That ratio is what bounds this test's accuracy. Nothing subtracts the mean --
v1 did not, and parity is the point -- so the pedestal leaks into any window
that is not a whole number of evenly spaced periods, and the error scales with
the pedestal rather than with the signal. A live test cannot land on such a
window, however evenly it tries to pace itself.
`test_an_incomplete_window_leaks_the_background` pins the effect down where the
sampling can be controlled exactly."""

LOCKIN_FPS = 60.0
"""A 16.7 ms interval, chosen to sit just above this platform's 15.6 ms timer
granularity. Asking for 120 fps gets clumps of frames rather than evenly spaced
ones, and uneven spacing is precisely what stops the pedestal cancelling."""

LOCKIN_INTEGRATION_S = 2.0
"""Eight periods. The leak falls as the window lengthens, so this buys accuracy
that a shorter run cannot have; the tests that do not measure amplitude use a
much shorter one."""


class ModulatedSource:
    """A capture whose visible pixels swing at a known frequency and amplitude.

    The whole point of a lock-in run is to pull a known signal out of a scene,
    so the fixture has to put one in. What goes in here is what the result has
    to come back with.
    """

    def __init__(self, capture: np.ndarray) -> None:
        self._params = PARAMS
        self._words = np.ascontiguousarray(capture, dtype=np.uint16).ravel().copy()
        self._interval = 1.0 / LOCKIN_FPS
        self._started = time.monotonic()
        self._next_due = self._started
        self._closed = False
        self.commands: list[int] = []

    @property
    def params(self):
        return self._params

    @property
    def camera_raw(self) -> bool:
        return False

    @property
    def description(self) -> str:
        return f"modulated({LOCKIN_FREQUENCY_HZ} Hz)"

    def read_words(self) -> np.ndarray:
        if self._closed:
            raise CameraUnavailableError("modulated source is closed")

        delay = self._next_due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_due = max(self._next_due + self._interval, time.monotonic())

        phase = 2 * math.pi * LOCKIN_FREQUENCY_HZ * (time.monotonic() - self._started)
        self._words[: self._params.pixel_count] = round(
            LOCKIN_BASE + LOCKIN_AMPLITUDE * math.sin(phase)
        )
        return self._words

    def send_command(self, payload: int) -> bool:
        self.commands.append(payload)
        return True

    def close(self) -> None:
        self._closed = True


def modulated() -> CameraSource:
    return ModulatedSource(build_capture(PARAMS))


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


@pytest.fixture
async def lockin_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """A camera carrying a planted modulation, for the lock-in endpoints."""
    manager = CameraManager(settings, {"rack": modulated, "bench": emulator()})
    app = create_app(settings, manager)

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        await manager.get("rack").wait_for_frame(timeout=5.0)
        yield http_client


async def finished_job(client: AsyncClient, job_id: str, timeout: float = 20.0) -> dict:
    """Poll a job until it stops running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = (await client.get(f"/api/jobs/{job_id}")).json()
        if body["state"] != "running":
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")  # pragma: no cover


@asynccontextmanager
async def running_server(settings: Settings) -> AsyncGenerator[str, None]:
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


# --- telemetry socket -------------------------------------------------------
#
# Over a real socket rather than Starlette's TestClient, which now wants a
# different HTTP library than the rest of this suite uses. The same reasoning as
# the MJPEG tests: a streaming protocol is worth exercising as one.


def socket_url(base_url: str, path: str) -> str:
    return base_url.replace("http://", "ws://", 1) + path


async def test_telemetry_socket_pushes_measurements(settings: Settings) -> None:
    async with (
        running_server(settings) as base_url,
        websocket_connect(socket_url(base_url, "/api/cameras/rack/telemetry?fps=30")) as socket,
    ):
        message = json.loads(await socket.recv())

    assert message["camera"] == "rack"
    assert set(message["points"]) == {"hiend-1", "ind-1"}
    assert message["points"]["hiend-1"]["x"] == 43
    assert message["minimum"]["celsius"] <= message["maximum"]["celsius"]
    assert message["telemetry"]["serial"] == "SN0042"
    assert message["adjustments"]["user_offset"] == 0.0


async def test_telemetry_socket_keeps_pushing(settings: Settings) -> None:
    """Each message must be a new frame, not the same one repeated."""
    async with (
        running_server(settings) as base_url,
        websocket_connect(socket_url(base_url, "/api/cameras/rack/telemetry?fps=30")) as socket,
    ):
        first = json.loads(await socket.recv())
        second = json.loads(await socket.recv())

    assert second["sequence"] > first["sequence"]
    assert second["captured_at"] >= first["captured_at"]


async def test_telemetry_socket_rejects_an_unknown_camera(settings: Settings) -> None:
    async with running_server(settings) as base_url:
        with pytest.raises(InvalidStatus):
            async with websocket_connect(socket_url(base_url, "/api/cameras/nope/telemetry")):
                pass  # pragma: no cover - the handshake never completes


async def test_telemetry_socket_unsubscribes_on_disconnect(settings: Settings) -> None:
    """A closed socket must not leave a subscriber attached to the hub."""
    async with running_server(settings) as base_url, AsyncClient(base_url=base_url) as client:
        async with websocket_connect(
            socket_url(base_url, "/api/cameras/rack/telemetry?fps=30")
        ) as socket:
            await socket.recv()
            assert (await client.get("/api/cameras/rack")).json()["subscribers"] == 1

        for _ in range(100):
            if (await client.get("/api/cameras/rack")).json()["subscribers"] == 0:
                break
            await asyncio.sleep(0.02)
        else:  # pragma: no cover
            pytest.fail("the subscription outlived the socket")


# --- web ui -----------------------------------------------------------------


async def test_web_ui_is_served_at_the_root(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "IRThermal" in response.text


async def test_web_ui_assets_are_served(client: AsyncClient) -> None:
    script = await client.get("/app.js")
    stylesheet = await client.get("/style.css")

    assert "stream.mjpg" in script.text
    assert "#stream" in stylesheet.text


async def test_docs_survive_the_static_mount(client: AsyncClient) -> None:
    """The mount answers for `/`, so the routes registered before it must win."""
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/metrics")).status_code == 200


async def test_unknown_page_is_a_404(client: AsyncClient) -> None:
    assert (await client.get("/nope.html")).status_code == 404


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


# --- lock-in ----------------------------------------------------------------


async def start_run(client: AsyncClient, **body) -> dict:
    request = {"frequency_hz": LOCKIN_FREQUENCY_HZ, "integration_s": 1.0, **body}
    response = await client.post("/api/cameras/rack/lockin", json=request)

    assert response.status_code == 202
    return response.json()


async def test_a_run_recovers_the_planted_amplitude(lockin_client: AsyncClient) -> None:
    """The measurement end to end: plant a signal, ask for it back over HTTP."""
    job = await start_run(lockin_client, integration_s=LOCKIN_INTEGRATION_S)
    assert job["state"] == "running"
    assert job["modulator"] == "external"

    finished = await finished_job(lockin_client, job["id"])
    assert finished["state"] == "succeeded", finished["error"]
    assert finished["progress"] == 1.0
    assert finished["frames"] > 20
    assert finished["aliased"] is False
    assert finished["samples_per_period"] > 10

    arrays = await lockin_client.get(f"/api/jobs/{job['id']}/result.npz")
    assert arrays.status_code == 200
    assert int(arrays.headers["x-lockin-frames"]) == finished["frames"]

    with np.load(io.BytesIO(arrays.content)) as loaded:
        magnitude = np.hypot(loaded["in_phase"], loaded["quadrature"])

    assert magnitude.shape == (192, 256)
    assert float(np.median(magnitude)) == pytest.approx(LOCKIN_AMPLITUDE, rel=0.1)


async def magnitude_of(client: AsyncClient, job_id: str) -> float:
    response = await client.get(f"/api/jobs/{job_id}/result.npz")
    assert response.status_code == 200

    with np.load(io.BytesIO(response.content)) as loaded:
        return float(np.median(np.hypot(loaded["in_phase"], loaded["quadrature"])))


async def test_an_unmodulated_camera_reads_far_lower(lockin_client: AsyncClient) -> None:
    """The discrimination the whole method rests on, measured rather than assumed."""
    modulated_job = await start_run(lockin_client, integration_s=LOCKIN_INTEGRATION_S)
    await finished_job(lockin_client, modulated_job["id"])

    quiet = await lockin_client.post(
        "/api/cameras/bench/lockin",
        json={
            "frequency_hz": LOCKIN_FREQUENCY_HZ,
            "integration_s": LOCKIN_INTEGRATION_S,
        },
    )
    assert quiet.status_code == 202
    await finished_job(lockin_client, quiet.json()["id"])

    driven = await magnitude_of(lockin_client, modulated_job["id"])
    still = await magnitude_of(lockin_client, quiet.json()["id"])

    assert still < driven / 10.0


async def test_a_celsius_run_reports_degrees(lockin_client: AsyncClient) -> None:
    """The same swing, demodulated through the calibration curve instead."""
    job = await start_run(lockin_client, signal="celsius")
    finished = await finished_job(lockin_client, job["id"])

    assert finished["state"] == "succeeded", finished["error"]
    assert finished["signal"] == "celsius"

    # Degrees. The exact figure depends on the calibration curve across the
    # swing, but it is nothing like the several thousand counts the raw run
    # reports for the very same modulation.
    magnitude = await magnitude_of(lockin_client, job["id"])
    assert 0.0 < magnitude < LOCKIN_AMPLITUDE / 10.0


@pytest.mark.parametrize("component", ["magnitude", "phase", "in_phase", "quadrature"])
async def test_every_component_renders(lockin_client: AsyncClient, component: str) -> None:
    job = await start_run(lockin_client, integration_s=0.3)
    await finished_job(lockin_client, job["id"])

    response = await lockin_client.get(
        f"/api/jobs/{job['id']}/result.png?component={component}&colormap=turbo&upscale=2"
    )

    assert response.status_code == 200
    assert response.content.startswith(b"\x89PNG\r\n")
    assert response.headers["x-lockin-component"] == component

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (192 * 2, 256 * 2)


async def test_an_unknown_component_is_rejected(lockin_client: AsyncClient) -> None:
    job = await start_run(lockin_client, integration_s=0.3)
    await finished_job(lockin_client, job["id"])

    response = await lockin_client.get(f"/api/jobs/{job['id']}/result.png?component=sideways")

    assert response.status_code == 422


async def test_a_result_is_not_available_while_the_run_is_going(
    lockin_client: AsyncClient,
) -> None:
    job = await start_run(lockin_client, integration_s=20.0)

    response = await lockin_client.get(f"/api/jobs/{job['id']}/result.npz")

    assert response.status_code == 409
    assert "running" in response.json()["detail"]
    await lockin_client.delete(f"/api/jobs/{job['id']}")


async def test_a_second_run_on_one_camera_is_refused(lockin_client: AsyncClient) -> None:
    """One load, one switch; two runs would drive it to opposite states."""
    job = await start_run(lockin_client, integration_s=20.0)

    response = await lockin_client.post(
        "/api/cameras/rack/lockin", json={"frequency_hz": 2.0, "integration_s": 1.0}
    )

    assert response.status_code == 409
    assert job["id"] in response.json()["detail"]
    await lockin_client.delete(f"/api/jobs/{job['id']}")


async def test_cancelling_a_run(lockin_client: AsyncClient) -> None:
    job = await start_run(lockin_client, integration_s=30.0)

    response = await lockin_client.delete(f"/api/jobs/{job['id']}")

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    assert (await lockin_client.get(f"/api/jobs/{job['id']}/result.npz")).status_code == 422


async def test_jobs_are_listed(lockin_client: AsyncClient) -> None:
    job = await start_run(lockin_client, integration_s=0.3)
    await finished_job(lockin_client, job["id"])

    body = (await lockin_client.get("/api/jobs")).json()

    assert [entry["id"] for entry in body] == [job["id"]]
    assert body[0]["camera"] == "rack"


async def test_an_unknown_job_is_a_404(lockin_client: AsyncClient) -> None:
    response = await lockin_client.get("/api/jobs/nope")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"frequency_hz": 0.0, "integration_s": 1.0},
        {"frequency_hz": 2.0, "integration_s": 0.0},
        {"frequency_hz": 2.0, "integration_s": 1.0, "signal": "kelvin"},
        {"frequency_hz": 2.0, "integration_s": 1.0, "port": "/dev/ttyUSB0"},
    ],
    ids=["zero-frequency", "zero-integration", "unknown-signal", "port-in-request"],
)
async def test_invalid_run_requests_are_rejected(lockin_client: AsyncClient, body: dict) -> None:
    """The port case matters: the device comes from config, never from a caller."""
    response = await lockin_client.post("/api/cameras/rack/lockin", json=body)

    assert response.status_code == 422


async def test_shutdown_cancels_a_run_in_flight(settings: Settings) -> None:
    """A lock-in run must not outlive the app, nor hold the load on."""
    manager = CameraManager(settings, {"rack": modulated, "bench": emulator()})
    app = create_app(settings, manager)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        async with app.router.lifespan_context(app):
            await manager.get("rack").wait_for_frame(timeout=5.0)
            job = await start_run(http_client, integration_s=60.0)

        assert app.state.jobs.get(job["id"]).state == "cancelled"


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
