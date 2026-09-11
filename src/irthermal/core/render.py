"""Turning a thermal frame into something a browser can display.

Two decisions matter here. The first is how temperatures map onto the 0-255
range a colormap expects: a fixed window makes frames comparable over time,
while auto-scaling to each frame's own range makes detail visible but means the
same colour denotes different temperatures from one frame to the next. Both are
offered; neither is right in general.

The second is that this is the only place OpenCV's imaging half is used. v1
pulled in scikit-image for histogram equalisation alone -- a dependency it never
actually declared, so `opencv.py` died on import wherever it was installed
cleanly. `cv2.equalizeHist` does the same job with no new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from irthermal.core.frame import Frame

BgrImage = npt.NDArray[np.uint8]

COLORMAPS: dict[str, int | None] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "gray": None,
}
"""Friendly name -> OpenCV colormap. `gray` means no colormap at all."""

DEFAULT_COLORMAP = "inferno"
DEFAULT_JPEG_QUALITY = 85

_AUTO_PERCENTILES = (1.0, 99.0)
"""Auto-scaling ignores the top and bottom 1%, so one dead pixel cannot flatten
the whole image."""


class UnknownColormapError(ValueError):
    """The requested colormap is not one we have."""


def resolve_colormap(name: str) -> int | None:
    """Look up a colormap by name."""
    try:
        return COLORMAPS[name]
    except KeyError:
        msg = f"unknown colormap {name!r}; available: {', '.join(sorted(COLORMAPS))}"
        raise UnknownColormapError(msg) from None


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """How to turn temperatures into pixels."""

    colormap: str = DEFAULT_COLORMAP
    min_celsius: float | None = None
    max_celsius: float | None = None
    equalize: bool = False
    """Spread the histogram over the full range. Reveals detail in a narrow
    temperature band, at the cost of any fixed relationship between colour and
    temperature."""
    upscale: int = 1

    @property
    def fixed_range(self) -> tuple[float, float] | None:
        """The requested temperature window, if both ends were given."""
        if self.min_celsius is None or self.max_celsius is None:
            return None
        return (self.min_celsius, self.max_celsius)


def normalize(temperatures: npt.NDArray[np.float64], options: RenderOptions) -> BgrImage:
    """Map temperatures onto 0-255."""
    window = options.fixed_range
    if window is None:
        finite = temperatures[np.isfinite(temperatures)]
        if finite.size == 0:
            return np.zeros(temperatures.shape, dtype=np.uint8)
        low, high = (float(v) for v in np.percentile(finite, _AUTO_PERCENTILES))
    else:
        low, high = window

    if high <= low:
        high = low + 1.0

    # NaN pixels sit at the bottom of the scale rather than propagating.
    filled = np.nan_to_num(temperatures, nan=low, posinf=high, neginf=low)
    scaled = (filled - low) * (255.0 / (high - low))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _u8(image: object) -> BgrImage:
    """cv2's stubs widen the dtype; every call below preserves uint8."""
    return cast("BgrImage", image)


def colorize(gray: BgrImage, options: RenderOptions) -> BgrImage:
    """Apply the colormap, and any upscaling, to an 8-bit image."""
    if options.equalize:
        gray = _u8(cv2.equalizeHist(gray))

    colormap = resolve_colormap(options.colormap)
    image = _u8(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if colormap is None
        else cv2.applyColorMap(gray, colormap)
    )

    if options.upscale > 1:
        # Nearest-neighbour keeps sensor pixels legible as pixels. v1 used
        # np.kron here, which allocates the whole outer product.
        image = _u8(
            cv2.resize(
                image,
                (image.shape[1] * options.upscale, image.shape[0] * options.upscale),
                interpolation=cv2.INTER_NEAREST,
            )
        )

    return image


def render(frame: Frame, options: RenderOptions | None = None) -> BgrImage:
    """Render a frame to a BGR image."""
    options = options or RenderOptions()
    return colorize(normalize(frame.temperatures(), options), options)


def encode_jpeg(image: BgrImage, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """JPEG-encode a BGR image."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - cv2 does not fail on well-formed input
        raise RuntimeError("JPEG encoding failed")
    return bytes(buffer.tobytes())


def encode_png(image: BgrImage) -> bytes:
    """PNG-encode a BGR image."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return bytes(buffer.tobytes())
