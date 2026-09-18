"""Storing a flat-field reference between runs.

A camera reached over a pipe cannot have its shutter closed, so its reference is
taken by hand. Keeping it on disk is what stops that being a per-restart chore.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irthermal.camera.processing import CALIBRATION_VERSION, RawCorrector

SHAPE = (192, 256)


def _reference(seed: int = 0) -> np.ndarray:
    """A frame with per-column offsets, which is what the real pattern looks like."""
    rng = np.random.default_rng(seed)
    columns = rng.integers(4000, 6000, size=SHAPE[1], dtype=np.uint16)
    return np.tile(columns, (SHAPE[0], 1)).astype(np.uint16)


def _scene(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3000, size=SHAPE, dtype=np.uint16)


def test_a_stored_calibration_corrects_exactly_as_the_original_did(tmp_path: Path) -> None:
    reference = _reference()
    frame = (_scene() + reference).astype(np.uint16)

    original = RawCorrector()
    original.calibrate(reference)
    expected = original.apply(frame)

    path = tmp_path / "cal.npz"
    original.save(path, fpa_temp_c=27.5)

    restored = RawCorrector()
    record = restored.load(path, shape=SHAPE)

    assert record is not None
    assert record.fpa_temp_c == pytest.approx(27.5)
    assert restored.calibrated
    np.testing.assert_array_equal(restored.apply(frame), expected)


def test_dead_pixels_survive_the_round_trip(tmp_path: Path) -> None:
    reference = _reference()
    reference[:, :4] = 0  # a few columns reading as dead

    original = RawCorrector()
    original.calibrate(reference)
    assert original.dead_pixel_count > 0

    path = tmp_path / "cal.npz"
    original.save(path)

    restored = RawCorrector()
    record = restored.load(path, shape=SHAPE)

    assert record is not None
    assert record.dead_pixels == original.dead_pixel_count
    assert restored.dead_pixel_count == original.dead_pixel_count


def test_a_reference_without_dead_pixels_round_trips_too(tmp_path: Path) -> None:
    # A flat reference has no spread, so nothing falls under the dead threshold.
    original = RawCorrector()
    original.calibrate(np.full(SHAPE, 5000, dtype=np.uint16))
    assert original.dead_pixel_count == 0

    path = tmp_path / "cal.npz"
    original.save(path)

    restored = RawCorrector()
    assert restored.load(path, shape=SHAPE) is not None
    assert restored.dead_pixel_count == 0


def test_saving_before_calibrating_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="calibrate first"):
        RawCorrector().save(tmp_path / "cal.npz")


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert RawCorrector().load(tmp_path / "absent.npz", shape=SHAPE) is None


def test_a_reference_for_a_different_sensor_is_refused(tmp_path: Path) -> None:
    """Loading it would subtract one sensor's pattern from another's frames."""
    original = RawCorrector()
    original.calibrate(_reference())
    path = tmp_path / "cal.npz"
    original.save(path)

    restored = RawCorrector()
    assert restored.load(path, shape=(180, 240)) is None
    assert not restored.calibrated


def test_a_file_from_an_older_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cal.npz"
    with path.open("wb") as handle:
        np.savez(
            handle,
            version=np.asarray(CALIBRATION_VERSION + 1),
            reference=_reference().astype(np.float32),
            offset_mean=np.asarray(0.0),
            has_dead=np.asarray(False),
            dead_pixels=np.zeros((1, 1), dtype=np.uint8),
            saved_at=np.asarray(0.0),
            fpa_temp_c=np.asarray(float("nan")),
        )

    assert RawCorrector().load(path, shape=SHAPE) is None


def test_a_corrupt_file_is_refused_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "cal.npz"
    path.write_bytes(b"this is not an npz")

    assert RawCorrector().load(path, shape=SHAPE) is None


def test_a_failed_load_leaves_an_existing_calibration_alone(tmp_path: Path) -> None:
    corrector = RawCorrector()
    corrector.calibrate(_reference())
    frame = (_scene() + _reference()).astype(np.uint16)
    expected = corrector.apply(frame)

    assert corrector.load(tmp_path / "absent.npz", shape=SHAPE) is None

    assert corrector.calibrated
    np.testing.assert_array_equal(corrector.apply(frame), expected)


def test_saving_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    corrector = RawCorrector()
    corrector.calibrate(_reference())
    corrector.save(tmp_path / "cal.npz")

    assert [p.name for p in sorted(tmp_path.iterdir())] == ["cal.npz"]


def test_the_directory_is_created_if_it_does_not_exist(tmp_path: Path) -> None:
    corrector = RawCorrector()
    corrector.calibrate(_reference())
    path = tmp_path / "nested" / "deeper" / "cal.npz"

    corrector.save(path)

    assert path.is_file()


def test_resetting_discards_a_loaded_calibration(tmp_path: Path) -> None:
    corrector = RawCorrector()
    corrector.calibrate(_reference())
    path = tmp_path / "cal.npz"
    corrector.save(path)

    restored = RawCorrector()
    restored.load(path, shape=SHAPE)
    restored.reset()

    assert not restored.calibrated
