"""Prometheus metrics, collected on scrape.

Values are pulled from the cameras when Prometheus asks, rather than pushed on a
timer. v1 exported on a five-second loop, which meant every reading was up to
five seconds stale and the work happened whether or not anyone was scraping.

Point metrics carry whatever labels the operator put in the config. v1 hardcoded
three families of five -- `temp_pi_hend`, `temp_pi_ind`, `temp_pi_ultra` -- and
silently dropped any coordinate past the fifteenth, because the exporter mapped
list position to metric name.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from irthermal.camera.manager import CameraManager
from irthermal.config import Settings

CAMERA_LABEL = "camera"
POINT_LABEL = "point"


def point_label_names(settings: Settings) -> list[str]:
    """Every custom label name used by any configured point.

    Prometheus requires a fixed label set per metric, so the union is computed
    once at startup; a point that omits one reports an empty string for it.
    """
    names: set[str] = set()
    for camera in settings.cameras:
        for point in camera.points:
            names.update(point.labels)
    return sorted(names)


class CameraCollector(Collector):
    """Reports the state of every camera at scrape time."""

    def __init__(self, manager: CameraManager, settings: Settings) -> None:
        self._manager = manager
        self._label_names = point_label_names(settings)
        self._points = {
            camera.id: {point.name: point for point in camera.points} for camera in settings.cameras
        }

    def collect(self) -> Iterator[Metric]:
        yield from self._camera_health()
        yield from self._temperatures()

    def _camera_health(self) -> Iterable[Metric]:
        labels = [CAMERA_LABEL]

        up = GaugeMetricFamily(
            "thermal_camera_up", "Whether the camera is currently connected.", labels=labels
        )
        fps = GaugeMetricFamily("thermal_camera_fps", "Frames captured per second.", labels=labels)
        frames = CounterMetricFamily(
            "thermal_camera_frames", "Frames captured since startup.", labels=labels
        )
        errors = CounterMetricFamily(
            "thermal_camera_read_errors", "Failed frame reads since startup.", labels=labels
        )
        reconnects = CounterMetricFamily(
            "thermal_camera_reconnects", "Camera reconnections since startup.", labels=labels
        )

        for status in self._manager.statuses():
            key = [status.id]
            up.add_metric(key, 1.0 if status.connected else 0.0)
            fps.add_metric(key, status.fps)
            frames.add_metric(key, status.frames)
            errors.add_metric(key, status.read_errors)
            reconnects.add_metric(key, status.reconnects)

        return (up, fps, frames, errors, reconnects)

    def _temperatures(self) -> Iterable[Metric]:
        point_labels = [CAMERA_LABEL, POINT_LABEL, *self._label_names]

        point_temperature = GaugeMetricFamily(
            "thermal_point_celsius",
            "Temperature at a configured measurement point.",
            labels=point_labels,
        )
        frame_min = GaugeMetricFamily(
            "thermal_frame_min_celsius", "Coldest pixel in the frame.", labels=[CAMERA_LABEL]
        )
        frame_max = GaugeMetricFamily(
            "thermal_frame_max_celsius", "Hottest pixel in the frame.", labels=[CAMERA_LABEL]
        )
        frame_mean = GaugeMetricFamily(
            "thermal_frame_mean_celsius", "Mean pixel temperature.", labels=[CAMERA_LABEL]
        )
        frame_age = GaugeMetricFamily(
            "thermal_frame_age_seconds",
            "Age of the most recent frame at scrape time.",
            labels=[CAMERA_LABEL],
        )

        for runtime in self._manager:
            frame = runtime.latest
            if frame is None:
                continue

            camera_id = runtime.config.id
            configured = self._points.get(camera_id, {})
            measured = frame.measure((p.name, p.x, p.y) for p in configured.values())

            for name, measurement in measured.items():
                extra = configured[name].labels
                point_temperature.add_metric(
                    [camera_id, name, *(extra.get(label, "") for label in self._label_names)],
                    measurement.celsius,
                )

            stats = frame.stats()
            frame_min.add_metric([camera_id], stats.minimum.celsius)
            frame_max.add_metric([camera_id], stats.maximum.celsius)
            frame_mean.add_metric([camera_id], stats.mean_celsius)
            frame_age.add_metric([camera_id], max(0.0, _age_seconds(frame.captured_at)))

        return (point_temperature, frame_min, frame_max, frame_mean, frame_age)


def _age_seconds(captured_at: float) -> float:
    return time.monotonic() - captured_at
