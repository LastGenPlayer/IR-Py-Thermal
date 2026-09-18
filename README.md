# IRThermal

A server for XTherm / HT301 / InfiRay USB thermal cameras. One process, one
port: live video, measurements, camera control, Prometheus metrics, a web UI
and lock-in thermography, all over HTTP.

Supported cameras:

- Hti HT-301
- Xtherm T3S, thanks to Angel-1024
- Xtherm T2L, T2S+, thanks to Christoph Mair and DiminDDL

Resolutions 240×180, 256×192, 384×288 and 640×512 are recognised. Cameras are
matched by resolution, so a device path is optional.

> **This is version 2.** The desktop viewers (`pyplot.py`, `opencv.py`) are
> gone; everything they did is now an HTTP endpoint. See
> [Upgrading from v1](#upgrading-from-v1).

## What it does

- **Streams.** MJPEG over HTTP, so `<img src="...">` and `curl` both work with
  no plugin, no negotiation and no client-side code.
- **Measures.** Temperature at a pixel, over a rectangle, or at named points
  fixed in the configuration.
- **Controls.** Shutter calibration, temperature range, emissivity, distance,
  humidity and a software offset.
- **Exports.** Prometheus metrics with your own labels on every point.
- **Demodulates.** Lock-in thermography as a background job, with the drive
  switched over a serial port.
- **Runs several cameras** in one process, each with its own capture thread and
  its own settings.

One capture thread per camera feeds every consumer. Ten browsers watching one
camera cost one capture and one JPEG encode per frame.

## Quick start

### Docker

```bash
cp config.example.toml config.toml   # then edit it
docker compose up -d
curl localhost:8000/api/health
```

Open <http://localhost:8000> for the web UI, or `/docs` for the generated API
reference.

The image is built on `python:3.12-slim` with `opencv-python-headless`, so it
carries no X11 stack at all. It runs as a non-root user that belongs to `video`
and `dialout`, and it needs no writable filesystem.

### From source

```bash
pip install -e ".[dev]"              # or: uv sync --extra dev
irthermal check-config -c config.example.toml
irthermal serve -c config.toml
```

Python 3.12 or newer. On Windows and macOS add the `ffmpeg` extra -- see
[Windows and macOS](#windows-and-macos).

```bash
pip install -e ".[dev,ffmpeg]"
```

### Stopping it

Ctrl+C, and it is gone in well under a second with viewers still attached.

Getting there took some care. The MJPEG stream and the telemetry WebSocket both
run until the *client* hangs up, and uvicorn waits for in-flight requests before
it runs the application's shutdown -- which is the very thing that would end
those subscriptions. Left alone it waits forever, so one open browser tab hangs
Ctrl+C with the listener already closed and the capture threads still running.

`StreamAwareServer` in [`server.py`](src/irthermal/server.py) closes the
subscriptions first, so the responses complete by themselves and the wait ends
on its own. `SHUTDOWN_TIMEOUT_S` stays as a five-second backstop for anything
that misses; reaching it cancels live requests, and uvicorn logs each
cancellation as an error with a full traceback, which is what a clean shutdown
now avoids entirely.

`--reload` is refused. It needs the app as an import string and would run a
second copy of the server, and the two would fight over the same camera.

### With no camera to hand

Point a camera at a saved capture and the whole thing runs on a laptop:

```toml
[[cameras]]
id = "replay"
replay_file = "captures/2024-06-21_23-02-55.npy"
```

`GET /api/cameras/{id}/frame.npy` writes captures in that format.

## Configuration

A TOML file, by default `/etc/irthermal/config.toml`; `--config` points
somewhere else. Every value can be overridden by an environment variable, using
a double underscore to descend:

```bash
IRTHERMAL__SERVER__PORT=9000 IRTHERMAL__SERVER__LOG_FORMAT=console irthermal serve
```

Explicit arguments beat environment variables, which beat the file. See
[`config.example.toml`](config.example.toml) for the annotated version; the
shape is:

```toml
[server]
host = "0.0.0.0"
port = 8000
log_level = "info"     # debug | info | warning | error
log_format = "json"    # json | console

[[cameras]]
id = "pi-rack"                 # used in URLs and metric labels
device = "/dev/video0"         # or auto_detect = true, or replay_file = "..."
transport = "auto"             # auto | v4l2 | ffmpeg
raw_mode = true                # camera sends unprocessed sensor data
temperature_offset = 8.5       # constant trim, found once per camera
colormap = "inferno"
emissivity = 0.95

[[cameras.points]]             # measured and exported every frame
name = "hiend-1"
x = 43
y = 125
labels = { group = "hiend", slot = "1" }
```

`irthermal check-config -c config.toml` validates a file and prints what it
declares. It names the offending field rather than dumping a stack trace.

### Raw mode

Some cameras -- the T2S+ V2 among them -- send unprocessed sensor data, which
looks like a pixelated mess until it is corrected. `raw_mode = true` does that
correction in software: a flat-field calibration on connect, then dead-pixel
inpainting on every frame. `POST .../calibrate` recalibrates.

Raw mode often leaves a constant offset. Find it once for your camera and put
it in `temperature_offset`.

## Windows and macOS

The camera is a UVC device, but it does not send video: it sends 16-bit sensor
words wearing a `yuyv422` format tag. OpenCV's Windows and macOS backends decode
that to BGR before anything can stop them -- and they accept
`CAP_PROP_CONVERT_RGB = 0` while ignoring it, so the loss is silent. What arrives
is 196x256x3 bytes where the driver needs 196x256x2, and the measurement is gone.

ffmpeg reads the capture pin at its native format and copies it out untouched,
which works on every platform. Install it one of these ways:

```bash
pip install 'irthermal[ffmpeg]'      # bundled binary, nothing else to do
winget install Gyan.FFmpeg           # or apt install ffmpeg / brew install ffmpeg
```

Set `IRTHERMAL_FFMPEG` to use a specific binary. `transport = "auto"` picks it
automatically off Linux, so a config that works on a Pi works on a laptop:

```toml
[[cameras]]
id = "bench"
auto_detect = true
raw_mode = true
```

Naming the device explicitly takes the transport's own spelling -- a DirectShow
name on Windows, an avfoundation index on macOS:

```toml
device = "T2SPro"        # Windows;  `ffmpeg -list_devices true -f dshow -i dummy`
```

### What an ffmpeg pipe cannot do

A pipe carries pixels and nothing else. These cameras have no control endpoint
-- commands are smuggled through the UVC zoom control -- so anything that writes
to the camera needs a V4L2 handle, which means Linux.

| | ffmpeg | V4L2 |
|---|---|---|
| Streaming, snapshots, measurement, metrics, lock-in | yes | yes |
| `temperature_offset`, colormap, scaling | yes | yes |
| Shutter calibration, `/range`, `/parameters` | **501** | yes |

Raw-mode cameras are the one real limitation. Flat-field correction needs a
reference frame taken with the shutter closed, and the shutter cannot be closed
over a pipe. Rather than capture the scene and subtract it from every later
frame, the server logs `camera.calibration_skipped` and streams uncorrected --
temperatures will be wrong until it is calibrated. Cover the lens and ask
explicitly:

```bash
curl -X POST localhost:8000/api/cameras/bench/calibrate
```

Set `calibration_file` and that reference is kept, so this is a one-time chore
rather than something to repeat after every restart:

```toml
[[cameras]]
id = "bench"
auto_detect = true
raw_mode = true
calibration_file = "/var/lib/irthermal/bench.npz"
```

It is written whenever a calibration succeeds and reloaded on connect, and
refused if it does not match the sensor it is loaded against. A camera whose
shutter *can* be driven ignores the file and calibrates fresh, which is always
better.

The stored pattern keeps its value as the camera warms. Measured on a T2S+ over
two minutes, the per-column pattern held to within 1% while the overall level
moved about 59 counts -- so what drifts is a single scalar, and that is what
`temperature_offset` is for. Recalibrate when the striping becomes visible
again, not on a schedule.

These sensors do not fire their own shutter in raw mode: 3014 consecutive
frames showed no shutter event at all, so there is nothing for the server to
detect and adopt on its own.

## HTTP API

Everything lives under `/api`, and `/docs` is the generated reference. For an
API client -- Bruno, Insomnia, Postman -- write the schema to a file and import
it. No camera is opened and no port is bound, so a config is optional:

```bash
irthermal openapi -o docs/openapi.json          # base URL: http://localhost:8000
irthermal openapi -c config.toml                # takes the configured host:port
irthermal openapi --server-url http://pi-rack.local:8000
```

A checked-in copy lives at [`docs/openapi.json`](docs/openapi.json). It records
a `servers` entry, which FastAPI itself omits and without which an imported
collection has no host to send anything to. The telemetry WebSocket is absent
because OpenAPI cannot describe one; `/metrics` and `/health` are deliberately
kept out of the schema.

| | |
|---|---|
| `GET /api/health` | Liveness. 200 even when a camera is down. |
| `GET /api/ready` | Readiness. 503 unless every camera is connected. |
| `GET /api/colormaps` | Palette names the rendering endpoints accept. |
| `GET /api/cameras` | Every configured camera and its capture health. |
| `GET /api/cameras/{id}` | One camera, with its latest telemetry. |
| `GET /api/cameras/{id}/stream.mjpg` | Live MJPEG. |
| `GET /api/cameras/{id}/snapshot.{jpg,png}` | One rendered frame. |
| `GET /api/cameras/{id}/frame.npy` | Raw 16-bit pixels, for analysis. |
| `WS /api/cameras/{id}/telemetry` | Measurements pushed per frame. |
| `GET /api/cameras/{id}/temperatures` | Configured points plus frame extremes. |
| `GET /api/cameras/{id}/temperature?x=&y=` | One pixel. |
| `POST /api/cameras/{id}/region` | Min, max and mean over a rectangle. |
| `POST /api/cameras/{id}/calibrate` | Trigger a shutter calibration. |
| `POST /api/cameras/{id}/range` | Switch normal / high temperature range. |
| `PATCH /api/cameras/{id}/parameters` | Emissivity, distance, humidity, offset. |
| `POST /api/cameras/{id}/lockin` | Start a lock-in run. |
| `GET /api/jobs`, `GET /api/jobs/{job}` | Lock-in runs and their progress. |
| `DELETE /api/jobs/{job}` | Stop a run and release its load. |
| `GET /api/jobs/{job}/result.npz` | Both demodulated components. |
| `GET /api/jobs/{job}/result.png` | One component, rendered. |
| `GET /metrics` | Prometheus exposition. |

The rendering endpoints take `colormap`, `min_celsius`, `max_celsius`,
`equalize` and `upscale`; the stream also takes `fps` and `quality`.

```bash
# live video, scaled to a fixed 20-45 C window
curl -N 'localhost:8000/api/cameras/rack/stream.mjpg?min_celsius=20&max_celsius=45'

# what is the hottest pixel right now
curl -s localhost:8000/api/cameras/rack/temperatures | jq .maximum

# a still, four times larger
curl -o frame.png 'localhost:8000/api/cameras/rack/snapshot.png?upscale=4'

# emissivity for painted metal, and a trim
curl -X PATCH localhost:8000/api/cameras/rack/parameters \
     -H 'content-type: application/json' \
     -d '{"emissivity": 0.95, "temperature_offset": -12.3}'
```

Temperatures are `null`, never a number, where the calibration model cannot
produce one.

## Web UI

<http://localhost:8000> serves a single page: the live stream, a camera
selector, palette and scaling controls, calibration and range switching, the
measurement parameters, and a table of readings that updates over a WebSocket.
No build step and no framework -- it is three static files.

## Lock-in thermography

Modulate whatever is being heated at a known frequency, correlate the video
against that frequency, and everything not moving at it averages away. Hotspots
far below the camera's noise floor become visible: silicon dies, circuit board
shorts, delamination under a surface, even a pulse.

The load is switched over a serial port, one ASCII digit and a newline -- `1`
on, `0` off -- which is the same protocol v1 used, so existing firmware still
works. The port is **configuration, not a request field**: taking it from the
request body would let anyone who can reach the API make the server open and
write to an arbitrary device.

```toml
[[cameras]]
id = "die"
device = "/dev/video2"
lockin_port = "/dev/ttyUSB0"
lockin_baudrate = 115200
lockin_invert = false          # true for a load wired active-low
```

```bash
# 1 Hz, integrating for 60 seconds
curl -X POST localhost:8000/api/cameras/die/lockin \
     -H 'content-type: application/json' \
     -d '{"frequency_hz": 1.0, "integration_s": 60}'
# -> {"id": "0f139d501331", "state": "running", ...}

curl -s localhost:8000/api/jobs/0f139d501331 | jq '{state, progress, frames}'
curl -o lockin.png 'localhost:8000/api/jobs/0f139d501331/result.png?component=magnitude'
curl -O localhost:8000/api/jobs/0f139d501331/result.npz
```

`result.npz` holds `in_phase` and `quadrature`; `result.png` renders
`magnitude`, `phase`, `in_phase` or `quadrature`. Magnitude is how strongly a
pixel responds, phase is how far it lags the drive -- which grows with how far
the heat travelled, so phase finds buried features that magnitude misses.

Three things to get right:

- **Keep the frequency well below the frame rate.** Ten times below is a good
  rule. Each result reports `samples_per_period` and sets `aliased` when the
  frame rate did not cover the modulation.
- **Integrate over a whole number of periods.** Nothing subtracts the mean, so
  a partial period leaks the background into the answer. Each result reports
  `periods`.
- **Integrate for longer to see less.** Noise falls with the number of cycles
  averaged; a faint signal is a matter of patience.

Omit `lockin_port` when something else drives the load -- a function generator,
or the device under test itself. The demodulation runs either way, and the job
reports its modulator as `external`.

Only one run per camera at a time: the load is physical, and two runs would
drive the same switch to opposite states.

![lockin output](docs/lock-in.png)

*A v1 capture, kept because the physics has not changed.*

## Metrics

`GET /metrics`, sampled at scrape time rather than on a timer, so a change
shows up on the very next scrape.

```
thermal_point_celsius{camera="rack",point="hiend-1",group="hiend",slot="1"}
thermal_frame_min_celsius{camera="rack"}
thermal_frame_max_celsius{camera="rack"}
thermal_frame_mean_celsius{camera="rack"}
thermal_frame_age_seconds{camera="rack"}
thermal_camera_up{camera="rack"}
thermal_camera_fps{camera="rack"}
thermal_camera_frames_total{camera="rack"}
thermal_camera_read_errors_total{camera="rack"}
thermal_camera_reconnects_total{camera="rack"}
```

Whatever you put in a point's `labels` becomes a label on
`thermal_point_celsius`, so group your points however you intend to query them.

## Deployment

Run one process. Several uvicorn workers would each try to open the same
devices, so `serve` does not offer a worker count.

### Compose

[`docker-compose.yml`](docker-compose.yml) passes the devices through, mounts
the config read-only and keeps the root filesystem read-only. List every camera
device you have configured. If your host numbers the `video` or `dialout`
groups differently from Debian's 44 and 20, add them with `group_add` --
`getent group video` will say.

### systemd

[`deploy/irthermal.service`](deploy/irthermal.service) runs it without a
container, under a dedicated user with access to nothing but the devices it
uses. Installation is in the comments at the top of the file.

### Behind a proxy

The stream is long-lived and must not be buffered. It sets
`X-Accel-Buffering: no` for nginx; other proxies need telling in their own way.
WebSocket upgrades need forwarding for the web UI's readings to update.

## Development

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest tests -q
ruff check src tests && ruff format --check src tests
mypy src/irthermal
```

The suite needs no camera: `EmulatorSource` replays captures, and a source is
anything satisfying a protocol. `tests/reference_legacy.py` is a transcription
of v1's temperature pipeline and lock-in kernel, and the parity tests diff the
new code against it, so the radiometry cannot drift from the original by
accident.

## Upgrading from v1

The desktop viewers are gone. Their features map onto endpoints:

| v1 | v2 |
|---|---|
| `pyplot.py`, `opencv.py` | The web UI, and the API under `/api`. |
| `-d /dev/video0` | `device` in the configuration. |
| `-r` (raw mode) | `raw_mode = true`. |
| `-o 8.5` (offset) | `temperature_offset`, or `PATCH .../parameters`. |
| `-c` (custom coordinates) | `[[cameras.points]]`, any number, with labels. |
| `--headless` | The only mode there is. |
| `-l`, `-p`, `-i` (lock-in) | `POST .../lockin` plus `lockin_port`. |
| `u` key (calibrate) | `POST .../calibrate`. |
| `k` / `l` keys (range) | `POST .../range`. |
| `r` key (save raw) | `GET .../frame.npy`. |
| `w` key (save image) | `GET .../snapshot.png`. |
| Prometheus on a second port | `/metrics` on the same port. |

Captures written by v1's `r` key replay unchanged.

Two behavioural differences worth knowing. v1's reported temperatures depended
on the installed NumPy version, because the calibration table's dtype was
decided by scalar promotion rules that changed in NumPy 2.0; v2 pins float64,
so a given frame gives one answer everywhere. And v1 raised an `IndexError` on
384×288 and 640×512 sensors in raw mode, reading a shutter trim past the end of
the frame; v2 skips that trim and says so in `telemetry.shutter_trim_applied`.

## Related projects

- https://gitlab.com/netman69/inficam
- https://github.com/MCMH2000/OpenHD_HT301_Driver
- https://github.com/sumpster/ht301_viewer
- https://github.com/cmair/ht301_hacklib
- https://github.com/stawel/ht301_hacklib
- https://github.com/mcguire-steve/ht301_ircam

## Related materials

- https://www.mdpi.com/1424-8220/17/8/1718
- https://www-old.mpi-halle.mpg.de/mpi/publi/pdf/540_02.pdf

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
