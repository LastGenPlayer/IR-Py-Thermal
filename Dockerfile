# syntax=docker/dockerfile:1

# --- build ------------------------------------------------------------------
# Built in a throwaway stage so that nothing used to compile or install wheels
# is present in the image that actually runs.

FROM python:3.12-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src

# The pip cache is mounted rather than copied, so a source-only change reuses
# the downloaded wheels instead of fetching sixty megabytes of OpenCV again.
RUN --mount=type=cache,target=/root/.cache/pip pip install .


# --- runtime ----------------------------------------------------------------

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="IRThermal" \
      org.opencontainers.image.description="Thermal camera server: HTTP API, MJPEG streaming and Prometheus metrics" \
      org.opencontainers.image.source="https://github.com/diminDDL/IR-Py-Thermal" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

# No libgl1, and no glib either. That is the whole reason this depends on
# opencv-python-headless: the desktop build of OpenCV drags in an X11 stack
# that a server with no display has no use for. v1's image installed both.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IRTHERMAL__SERVER__HOST=0.0.0.0 \
    IRTHERMAL__SERVER__PORT=8000

COPY --from=build /opt/venv /opt/venv

# `video` already exists on Debian at GID 44 and owns /dev/video*; `dialout` at
# GID 20 owns the serial ports a lock-in run switches its load through. A host
# that numbers either differently needs `group_add` in compose.
RUN groupadd --gid 10001 irthermal \
    && useradd --no-create-home --uid 10001 --gid 10001 \
       --groups video,dialout --shell /usr/sbin/nologin irthermal

USER irthermal
WORKDIR /etc/irthermal
EXPOSE 8000

# Python is already here, so this costs nothing; installing curl for it would
# add a package and an attack surface for one HTTP GET. Liveness deliberately
# rather than readiness: a camera being unplugged should not restart a process
# that is otherwise working, which is what /api/ready would do.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

ENTRYPOINT ["irthermal"]
CMD ["serve"]
