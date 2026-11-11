"""Layout helpers shared by the gallery figures: dark theme, labelled panels, crops and splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage

BACKGROUND = "#0b0b0e"
TEXT = "#f4f4f6"
MUTED = "#9b9ba8"
ZOOM_COLOR = "#ffd166"


def use_style() -> None:
    """Set the dark gallery theme, using the Inter typeface when it is installed."""
    for path in font_manager.findSystemFonts():
        if Path(path).name.startswith("Inter"):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": ["Inter", "DejaVu Sans"],
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "image.interpolation": "antialiased",
        }
    )


def bbox(mask: np.ndarray, margin: int = 0) -> tuple[slice, slice]:
    """Return the row and column slices of the bounding box of ``mask``, grown by ``margin``."""
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, mask.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, mask.shape[1])
    return slice(r0, r1), slice(c0, c1)


def gray(img: np.ndarray) -> np.ndarray:
    """Return a [0, 1] grayscale image as RGB."""
    return np.repeat(np.clip(img, 0, 1)[..., None], 3, axis=-1)


def isotropic(img: np.ndarray, row_mm: float, col_mm: float) -> np.ndarray:
    """Resample an image (2D, or RGB with channels last) so that rows are as fine as columns."""
    factors = (row_mm / col_mm, 1.0) + (1.0,) * (img.ndim - 2)
    return ndimage.zoom(img, factors, order=1, grid_mode=True, mode="nearest")


def grid(
    ncols: int,
    row_aspects: list[float],
    width: float = 21.0,
    labelled_rows: tuple[int, ...] = (0,),
    header: float = 1.05,
    label: float = 0.8,
    left: float = 0.15,
    gap: float = 0.12,
    bottom: float = 0.32,
) -> tuple[Figure, np.ndarray]:
    """Create a figure of image panels laid out in inches, so panels fit their images exactly.

    Args:
        ncols: Panels per row.
        row_aspects: Height/width ratio of the panels in each row.
        width: Figure width in inches; the height follows from the panels.
        labelled_rows: Rows that get room above them for a title and a note.
        header: Space for the figure title and subtitle, in inches.
        label: Space above labelled rows, in inches.
        left: Left margin in inches (the right margin is fixed).
        gap: Space between panels, in inches.
        bottom: Bottom margin (for the data credit), in inches.

    Returns:
        The figure and a ``(len(row_aspects), ncols)`` array of axes.
    """
    panel = (width - left - 0.15 - gap * (ncols - 1)) / ncols
    above = [label if i in labelled_rows else gap for i in range(len(row_aspects))]
    height = header + sum(above) + panel * sum(row_aspects) + bottom
    fig = plt.figure(figsize=(width, height))
    axes = np.empty((len(row_aspects), ncols), dtype=object)
    top = height - header
    for i, aspect in enumerate(row_aspects):
        top -= above[i] + panel * aspect
        for j in range(ncols):
            x = left + j * (panel + gap)
            axes[i, j] = fig.add_axes((x / width, top / height, panel / width, panel * aspect / height))
    return fig, axes


def header(fig: Figure, title: str, subtitle: str = "", left: float = 0.15) -> None:
    """Write the figure title and an optional muted subtitle in the top margin."""
    w, h = fig.get_size_inches()
    fig.text(left / w, 1 - 0.28 / h, title, fontsize=22, weight="bold", va="top")
    if subtitle:
        fig.text(left / w, 1 - 0.7 / h, subtitle, fontsize=11, color=MUTED, va="top")


def show(ax: Axes, img: np.ndarray, title: str = "", note: str = "", size: float = 13) -> None:
    """Draw an RGB image without axes, with a bold title and up to two lines of muted note above it.

    Titles sit at the same height in every panel; the note hangs below the title.
    """
    ax.imshow(np.clip(img, 0, 1))
    ax.set_axis_off()
    labels: list[tuple[str, float, str, dict[str, Any]]] = [
        (title, 40, "bottom", {"fontsize": size, "weight": "bold"}),
        (note, 37, "top", {"fontsize": 0.72 * size, "color": MUTED, "linespacing": 1.3}),
    ]
    for text, offset, va, style in labels:
        if text:
            ax.annotate(
                text,
                (0, 1),
                xycoords="axes fraction",
                xytext=(0, offset),
                textcoords="offset points",
                ha="left",
                va=va,
                **style,
            )


def zoom_box(ax: Axes, rows: slice, cols: slice) -> None:
    """Outline the region (rows, cols) that another panel magnifies."""
    ax.add_patch(
        Rectangle(
            (cols.start, rows.start),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor=ZOOM_COLOR,
            linewidth=1.4,
        )
    )


def diagonal_strip(
    images: list[np.ndarray], slope: float = 0.35, line: float = 2.5, span: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Join same-size RGB images side by side along parallel slanted lines, with soft white seams.

    The part of the width given by ``span`` (fractions of the width, so the
    seams can be spread over the subject only) is cut into ``len(images)``
    equal bands; band ``k`` shows ``images[k]``. ``slope`` is the horizontal
    shift of the seams per row (as a fraction of a pixel) and ``line`` the
    seams' half-width in pixels.
    """
    h, w = images[0].shape[:2]
    rows, cols = np.mgrid[0:h, 0:w]
    out = images[0]
    light = np.zeros((h, w, 1))
    for k in range(1, len(images)):
        # Signed horizontal distance of each pixel from seam k.
        dist = (cols - w * (span[0] + (span[1] - span[0]) * k / len(images))) + slope * (rows - h / 2)
        alpha = np.clip(dist + 0.5, 0, 1)[..., None]
        out = (1 - alpha) * out + alpha * images[k]
        seam = np.exp(-0.5 * (dist / line) ** 2) + 0.35 * np.exp(-0.5 * (dist / (6 * line)) ** 2)
        light += seam[..., None]
    return np.clip(out + (1 - out) * np.clip(light, 0, 1), 0, 1)


def diagonal_split(
    before: np.ndarray, after: np.ndarray, slope: float = 0.35, line: float = 2.5
) -> np.ndarray:
    """Join two RGB images along a slanted line through the centre, with a soft white seam.

    Pixels left of the line come from ``before``, pixels right of it from ``after``.
    """
    return diagonal_strip([before, after], slope, line)


def credit(fig: Figure, text: str) -> None:
    """Write a data credit along the bottom edge of the figure."""
    fig.text(0.995, 0.006, text, ha="right", va="bottom", fontsize=8, color=MUTED)


def save(fig: Figure, path: Path, dpi: int = 150) -> None:
    """Save ``fig`` as a high-quality JPEG (no chroma subsampling) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, pil_kwargs={"quality": 93, "subsampling": 0})
    plt.close(fig)
