"""Microscopy and pathology renderings for radiology images.

Every renderer maps maps of the *same* voxels to RGB in [0, 1]: no structure
is invented. The only cosmetic extra is the optional glow in ``fluorescence``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from matplotlib import colormaps
from scipy import ndimage
from skimage.segmentation import find_boundaries, slic

Color = Sequence[float] | np.ndarray
Layer = tuple[np.ndarray, Color]


def absorbance(rgb: Color) -> np.ndarray:
    """Return the optical density of a stain whose unit amount transmits ``rgb``."""
    return -np.log10(np.asarray(rgb, dtype=np.float64))


# Brightfield stains as optical density per unit amount (R, G, B).
# Hematoxylin and eosin are measured with ``measure_stains`` on the public-domain
# H&E slide ``skimage.data.skin()``; the hematoxylin vector agrees with Ruifrok &
# Johnston's (2001) to within 0.05. DAB is Ruifrok & Johnston's measurement.
HEMATOXYLIN = np.array([0.654, 0.678, 0.337])
EOSIN = np.array([0.207, 0.926, 0.317])
DAB = np.array([0.27, 0.57, 0.78])
# Kluver-Barrera myelin stain: Luxol fast blue turns myelin turquoise-blue,
# cresyl violet turns Nissl substance (neuron cell bodies) violet.
LUXOL_FAST_BLUE = absorbance((0.10, 0.50, 0.78))
CRESYL_VIOLET = absorbance((0.62, 0.28, 0.72))
# Masson's trichrome: Weigert's hematoxylin (nuclei, near black), Biebrich
# scarlet-acid fuchsin (muscle and cytoplasm, red), aniline blue (collagen, bone).
WEIGERT = absorbance((0.30, 0.25, 0.30))
FUCHSIN = absorbance((0.85, 0.12, 0.22))
ANILINE_BLUE = absorbance((0.18, 0.42, 0.85))

# Display colours conventionally used for common fluorophores.
FLUOROPHORES = {
    "DAPI": (0.20, 0.42, 1.00),
    "CFP": (0.00, 0.80, 1.00),
    "GFP": (0.20, 1.00, 0.35),
    "YFP": (1.00, 0.88, 0.10),
    "mCherry": (1.00, 0.28, 0.18),
    "Cy5": (1.00, 0.10, 0.85),
}


def measure_stains(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measure hematoxylin and eosin optical densities from an H&E image (Macenko et al., 2009).

    Stained pixels are converted to optical density and projected on the plane of
    their two main directions; the extreme angles in that plane (1st and 99th
    percentiles) are the two pure stains.

    Returns:
        ``(hematoxylin, eosin)`` unit vectors of optical density in R, G, B.
    """
    od = -np.log10((rgb.reshape(-1, 3).astype(np.float64) + 1) / 256)
    od = od[od.sum(axis=1) > 0.45]  # drop the bright, unstained background
    _, vectors = np.linalg.eigh(np.cov(od.T))
    plane = vectors[:, [2, 1]].T
    plane[0] *= np.sign(plane[0].sum())
    proj = od @ plane.T
    angles = np.percentile(np.arctan2(proj[:, 1], proj[:, 0]), [1, 99])
    stains = [np.array([np.cos(a), np.sin(a)]) @ plane for a in angles]
    stains = [v / np.linalg.norm(v) * np.sign(v.sum()) for v in stains]
    # Hematoxylin (blue-purple) absorbs more red light than eosin (pink).
    hematoxylin, eosin = sorted(stains, key=lambda v: -v[0])
    return hematoxylin, eosin


def brightfield(stains: Sequence[Layer], light: Color = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Render stain amounts as light transmitted through a stained slide (Beer-Lambert law).

    Args:
        stains: ``(amount, optical_density)`` pairs; ``amount`` is a map of stain
            per pixel (0 = unstained, 1 = one unit of the stain's density).
        light: Colour of the empty slide (the microscope's illumination).
    """
    density = sum(amount[..., None] * np.asarray(od) for amount, od in stains)
    return np.asarray(light) * 10.0 ** -np.asarray(density)


def fluorescence(
    channels: Sequence[Layer], exposure: float = 1.0, glow: float = 0.0, glow_sigma: float = 6.0
) -> np.ndarray:
    """Render channels as a multiplexed fluorescence image: coloured light added on black.

    Args:
        channels: ``(signal, colour)`` pairs, signals roughly in [0, 1].
        exposure: Gain applied before the camera's soft saturation.
        glow: Strength of a blurred copy added back, like out-of-focus light.
        glow_sigma: Blur radius of the glow in pixels.
    """
    light = sum(signal[..., None] * np.asarray(color) for signal, color in channels) * exposure
    light = np.asarray(light, dtype=np.float32)
    if glow:
        light = light + glow * ndimage.gaussian_filter(light, (glow_sigma, glow_sigma, 0))
    # Soft saturation: hot spots burn out to white, as on a real camera.
    return 1.0 - np.exp(-light)


def depth_color_code(volume: np.ndarray, axis: int, lut: str = "turbo") -> np.ndarray:
    """Project a volume with depth encoded as colour, like Fiji's "Temporal-Color Code".

    Each slice along ``axis`` is tinted by its position in the colour map ``lut``,
    then each RGB channel is maximum-intensity projected.
    """
    n = volume.shape[axis]
    colors = colormaps[lut](np.linspace(0, 1, n))[:, :3]
    shape = [1] * volume.ndim
    shape[axis] = n
    return np.stack([(volume * colors[:, c].reshape(shape)).max(axis=axis) for c in range(3)], axis=-1)


def brainbow(
    intensity: np.ndarray, mask: np.ndarray, n_segments: int = 2500, seed: int = 7, **fluo_kwargs: float
) -> np.ndarray:
    """Render an image in the style of Brainbow, where each neuron gets a random colour.

    The image is cut into superpixels (SLIC); each one gets a random mix of
    CFP, YFP and mCherry, as Brainbow cells do from random fluorescent-protein
    expression, and glows with the image's own intensity. Dark seams between
    superpixels separate them like cell membranes.
    """
    with warnings.catch_warnings():
        # SLIC seeds masked images with k-means, which warns when a seed cluster ends up empty (harmless).
        warnings.filterwarnings("ignore", message="One of the clusters is empty")
        labels = slic(
            intensity, n_segments=n_segments, compactness=0.05, mask=mask, channel_axis=None, start_label=1
        )
    rng = np.random.default_rng(seed)
    primaries = np.array([FLUOROPHORES["CFP"], FLUOROPHORES["YFP"], FLUOROPHORES["mCherry"]])
    weights = rng.random((labels.max() + 1, 3)) ** 2
    colors = weights @ primaries
    colors /= colors.max(axis=1, keepdims=True)  # vivid hues without overexposure
    colors[0] = 0.0  # label 0 is outside the mask
    seams = find_boundaries(labels, mode="inner")
    rgb = ndimage.gaussian_filter(colors[labels] * (intensity * (1 - 0.8 * seams))[..., None], (1.0, 1.0, 0))
    return fluorescence([(rgb[..., c], np.eye(3)[c]) for c in range(3)], **fluo_kwargs)
