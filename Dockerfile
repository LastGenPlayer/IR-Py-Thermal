FROM python:3.12-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip pip install .


FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="IRThermal" \
      org.opencontainers.image.description="Thermal camera server: HTTP API, MJPEG streaming and Prometheus metrics" \
      org.opencontainers.image.source="https://github.com/diminDDL/IR-Py-Thermal" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IRTHERMAL__SERVER__HOST=0.0.0.0 \
    IRTHERMAL__SERVER__PORT=8000

COPY --from=build /opt/venv /opt/venv

RUN groupadd --gid 10001 irthermal \
    && useradd --no-create-home --uid 10001 --gid 10001 \
       --groups video,dialout --shell /usr/sbin/nologin irthermal

USER irthermal
WORKDIR /etc/irthermal
EXPOSE 8000

ENTRYPOINT ["irthermal"]
CMD ["serve"]
