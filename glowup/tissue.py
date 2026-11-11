"""Turn scan intensities into soft tissue maps that the renderers can stain."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.special import expit
from skimage.filters import threshold_multiotsu


def window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map ``[lo, hi]`` linearly to ``[0, 1]`` and clip, like a radiology window."""
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Monotone S-curve from 0 (at ``lo``) to 1 (at ``hi``); sharpens soft tissue edges."""
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def band(x: np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    """Smooth indicator of ``lo < x < hi`` with logistic edges ``soft`` wide."""
    return expit((x - lo) / soft) * expit((hi - x) / soft)


def local_mean(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-weighted mean of ``img`` over the pixels selected by ``weight`` (normalized convolution).

    Unlike a plain blur, the result is not pulled down by the background, so it
    estimates a smooth field (such as MRI receive-coil shading) from one tissue.
    """
    num = ndimage.gaussian_filter(img * weight, sigma)
    den = ndimage.gaussian_filter(weight.astype(np.float32), sigma)
    return num / np.maximum(den, 1e-3 * den.max())


def brain_tissues(img: np.ndarray, sigma: float = 50.0) -> dict[str, np.ndarray]:
    """Return soft gray- and white-matter maps of an MRI slice where gray matter is brighter.

    Receive-coil shading makes the same tissue brighter near the coil, so the slice
    is first divided by the local white-matter level (estimated over ``sigma``
    pixels). Otsu thresholds then split background, white and gray matter. The
    gray-matter fraction runs linearly from the white-matter level to a typical
    gray-matter level (30th percentile of the gray-matter class, so that thin,
    dimmer cortex still counts as gray), keeping gradations inside the cortex.

    Returns:
        ``tissue`` (brain vs background), ``gm`` and ``wm`` maps in [0, 1], and
        ``density``: the shading-corrected image with typical gray matter at 1
        (white matter near 0.7), clipped to [0, 1.4] and zero outside the brain.
    """
    background, _ = threshold_multiotsu(img, classes=3)
    tissue = smoothstep(img, 0.5 * background, 1.2 * background)

    def flatten(mask: np.ndarray) -> np.ndarray:
        # The floor keeps the far background (where the estimate falls to 0) finite.
        return img / np.maximum(local_mean(img, mask, sigma), background)

    # First pass: flatten with all brain tissue, then refine with white matter only.
    flat = flatten(tissue > 0.5)
    for _ in range(2):
        brain = flat[tissue > 0.5]
        flat = flatten((tissue > 0.5) & (flat < threshold_multiotsu(brain, classes=2)[0]))
    brain = flat[tissue > 0.5]
    gm_level = float(np.percentile(brain[brain > threshold_multiotsu(brain, classes=2)[0]], 30))
    gm_fraction = window(flat, 1.0, gm_level)
    return {
        "tissue": tissue,
        "gm": tissue * gm_fraction,
        "wm": tissue * (1 - gm_fraction),
        "density": np.clip(flat / gm_level, 0, 1.4) * tissue,
    }


def ct_tissues(hu: np.ndarray) -> dict[str, np.ndarray]:
    """Return soft memberships in [0, 1] of CT voxels to tissue classes by Hounsfield range.

    The ranges are the textbook ones for a portal-venous contrast CT:
    aerated lung, fat, soft tissue (muscle, unenhanced organs, fluid),
    contrast-enhanced tissue (vessels, liver, spleen, kidneys) and bone.
    """
    return {
        "lung": band(hu, -950, -500, 40),
        "fat": band(hu, -190, -30, 15),
        "soft": band(hu, -10, 90, 12),
        "enhanced": band(hu, 90, 260, 15),
        "bone": expit((hu - 260) / 40),
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected region of a boolean mask."""
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def body_mask(hu: np.ndarray) -> np.ndarray:
    """Return the patient's body in a CT slice or axial volume, dropping the table and outside air.

    The body is the largest connected region above -500 HU after a small
    opening (which detaches the table), with holes such as the lungs filled
    (slice by slice for a volume).
    """
    mask = largest_component(ndimage.binary_opening(hu > -500, iterations=2))
    if mask.ndim == 2:
        return ndimage.binary_fill_holes(mask)
    for i in range(mask.shape[0]):
        mask[i] = ndimage.binary_fill_holes(mask[i])
    return mask
