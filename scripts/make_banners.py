"""Build LinkedIn profile banners (1584 x 396 px) in ``gallery/linkedin_banners/``.

Brain MRI makes up about two thirds of the banners; the rest show CT and
PET/CT. LinkedIn's profile photo covers part of the lower left on desktop, so
the main subjects sit in the centre or to the right. An overview sheet of all
banners is written as ``_overview.jpg``.

Usage:
    python scripts/make_banners.py              # all banners
    python scripts/make_banners.py brain body   # banners whose names contain an argument
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.transform import resize

from glowup import recipes
from glowup.data import load_brain_slice, load_ct_planes, load_petct
from glowup.figure import bbox, diagonal_strip, gray, isotropic
from glowup.tissue import body_mask, smoothstep
from make_gallery import (
    BODY_TOP,
    BRAIN_ZOOM,
    CT_AXIAL,
    CT_CORONAL_ROW,
    GALLERY,
    body_projections,
    brain_before,
    ct_columns,
    upsample,
)

OUT = GALLERY / "linkedin_banners"
H, W = 396, 1584  # LinkedIn's recommended background photo size
STYLES = ("before", "he", "kb", "if", "brainbow")


def to_size(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image to ``shape`` (rows, cols), with anti-aliasing when shrinking."""
    return np.clip(resize(img, shape, order=3, anti_aliasing=True), 0, 1)


def frame(img: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """Crop a banner-sized window centred at ``center`` (row, col), at the image's own resolution.

    Where the window leaves the image, it is filled with the image's background
    colour (its top-left pixel).
    """
    canvas = np.empty((img.shape[0] + 2 * H, img.shape[1] + 2 * W, 3), np.float32)
    canvas[:] = img[0, 0]
    canvas[H:-H, W:-W] = img
    r0, c0 = center[0] - H // 2 + H, center[1] - W // 2 + W
    return canvas[r0 : r0 + H, c0 : c0 + W]


def place(
    images: list[np.ndarray], height: int, gap: int = 24, x: float = 0.5, bg: np.ndarray | None = None
) -> np.ndarray:
    """Scale images to ``height`` and lay them out in a row, centred near fraction ``x`` of the banner width.

    The height shrinks if the row would not fit with a 24 px margin. The banner
    is filled with ``bg``, by default the first image's background colour.
    """
    margin = 24
    aspect = sum(im.shape[1] / im.shape[0] for im in images)
    height = min(height, int((W - 2 * margin - gap * (len(images) - 1)) / aspect))
    scaled = [to_size(im, (height, round(im.shape[1] * height / im.shape[0]))) for im in images]
    row_width = sum(s.shape[1] for s in scaled) + gap * (len(scaled) - 1)
    c = int(np.clip(round(x * W - row_width / 2), margin, W - margin - row_width))
    r = (H - height) // 2
    out = np.empty((H, W, 3))
    out[:] = images[0][0, 0] if bg is None else bg
    for s in scaled:
        out[r : r + height, c : c + s.shape[1]] = s
        c += s.shape[1] + gap
    return out


def fade(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Blend smoothly from ``before`` on the left to ``after`` on the right, across the middle."""
    alpha = smoothstep(np.linspace(0, 1, W), 0.3, 0.7)[None, :, None]
    return (1 - alpha) * before + alpha * after


@lru_cache(maxsize=2)
def brain_maps(index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return coronal brain slice ``index`` and its tissue maps (see ``recipes.brain_maps``)."""
    raw = load_brain_slice(index)
    return raw, recipes.brain_maps(ndimage.gaussian_filter(raw, 0.6))


@lru_cache(maxsize=8)
def brain(index: int, style: str) -> np.ndarray:
    """Return coronal brain slice ``index`` in one of ``STYLES``."""
    raw, maps = brain_maps(index)
    renderers = {
        "he": recipes.brain_he,
        "kb": recipes.brain_kluver_barrera,
        "if": recipes.brain_fluorescence,
        "brainbow": recipes.brain_brainbow,
    }
    img = brain_before(raw) if style == "before" else renderers[style](maps)
    return img.astype(np.float32)


def whole_slice(index: int, style: str) -> np.ndarray:
    """Return a brain slice in one style, cropped to the brain, with debris around the specimen erased."""
    labels, _ = ndimage.label(brain_maps(index)[1]["tissue"] > 0.5)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    brain_mask = ndimage.binary_dilation(sizes[labels] > 0.01 * sizes.sum(), iterations=4)
    img = brain(index, style)
    img = np.where(brain_mask[..., None], img, img[0, 0])  # erase the debris
    rows, cols = bbox(brain_mask, margin=40)
    return img[rows, cols]


def fade_end(img: np.ndarray, length: int = 40) -> np.ndarray:
    """Fade the last ``length`` columns into the background, softening where the scan was cut off."""
    w = img.shape[1]
    ramp = 1 - smoothstep(np.arange(w), w - length, w - 1)
    return img[0, 0] + (img - img[0, 0]) * ramp[None, :, None]


def strip_on_black(images: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Step through banner-sized ``images`` left to right across the subject, black outside ``mask``.

    Brightfield styles have white backgrounds and fluorescence black ones, so
    blacking out everything outside the subject gives every band the same
    background. The seams are spread over the columns where the subject fills
    at least 5% of the height.
    """
    cols = np.nonzero(mask[..., 0].mean(axis=0) > 0.05)[0]
    strip = diagonal_strip([img * mask for img in images], span=(cols[0] / W, cols[-1] / W))
    # Fade the seams out just beyond the subject's edge, so they do not cut across the empty background.
    return strip * np.clip(2 * ndimage.gaussian_filter(mask, (8, 8, 0)), 0, 1)


def lie(img: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    """Place a lying body (see ``lying_body``) on a banner, a little right of centre."""
    return place([img], 372, x=0.56, bg=bg)


@lru_cache(maxsize=1)
def lying_body() -> dict[str, np.ndarray]:
    """Return whole-body projections lying down (feet left, head right), cropped to the body."""
    bone_mip, depth, suv_mip = body_projections()
    rows, cols = bbox((bone_mip > 0.05) | (suv_mip > 0.4), margin=8)
    views = {
        "before": gray(bone_mip),
        "depth": depth,
        "petct": recipes.petct_fluorescence(bone_mip, suv_mip),
        "ihc": recipes.petct_ihc_projection(bone_mip, suv_mip),
    }
    return {name: fade_end(np.rot90(img[rows, cols], k=-1)) for name, img in views.items()}


@lru_cache(maxsize=1)
def lying_coronal() -> tuple[list[np.ndarray], np.ndarray]:
    """Return a whole-body coronal CT plane lying down in every CT style, and its body mask.

    A flat coronal plane misses the legs, so this one bends with the body like a
    curved planar reformat: in each axial slice it passes through the mean depth
    of the bone (smoothed along the body), following the spine, pelvis and leg
    bones. Styles, in order: before, H&E, trichrome, fluorescence, PET as IHC.
    """
    ct, pet = load_petct()
    hu = ct.data[:BODY_TOP]
    bone = recipes.ct_bone(hu)
    y = np.arange(hu.shape[1])[None, :, None]
    depth = (bone * y).sum(axis=(1, 2)) / np.maximum(bone.sum(axis=(1, 2)), 1e-6)
    depth = np.round(ndimage.gaussian_filter1d(depth, 6)).astype(int)
    z = np.arange(len(depth))
    dz, _, dx = ct.spacing
    plane_hu = isotropic(hu[z, depth], dz, dx)[::-1]
    plane_suv = isotropic(pet.data[:BODY_TOP][z, depth], dz, dx)[::-1]
    rows, cols = bbox(recipes.ct_maps(plane_hu)["window"] > 0, margin=10)  # the crop ct_columns uses
    body = ndimage.gaussian_filter(body_mask(plane_hu).astype(np.float32), 1.0)[rows, cols]
    panels = [fade_end(np.rot90(img, k=-1)) for img, _, _ in ct_columns(plane_hu, plane_suv, 6)]
    return panels, fade_end(np.rot90(gray(body), k=-1))


@lru_cache(maxsize=1)
def axial() -> list[list[np.ndarray]]:
    """Return the abdomen and chest axial CT panels (same styles and order as ``lying_coronal``)."""
    planes, _ = load_ct_planes(CT_AXIAL, CT_CORONAL_ROW)
    _, pet = load_petct()
    return [
        [img for img, _, _ in ct_columns(hu, upsample(pet.data[z], (2, 2)), 8)]
        for hu, z in zip(planes, CT_AXIAL, strict=True)
    ]


# Brain bands at the scan's native 100 micron resolution: (slice, centre row, centre column).
THALAMUS, CORTEX, TEMPORAL = (850, 600, 860), (850, 420, 860), (850, 760, 860)
CEREBELLUM, BASAL_GANGLIA, FRONTAL = (1300, 920, 950), (700, 560, 900), (500, 640, 910)
UPPER_CORTEX, PARIETAL, OCCIPITAL = (1000, 420, 900), (1150, 420, 900), (1150, 860, 900)


def band(where: tuple[int, int, int], style: str) -> np.ndarray:
    """Return a banner-sized band of a brain slice in one style, at native resolution."""
    index, row, col = where
    return frame(brain(index, style), (row, col))


def stains(where: tuple[int, int, int]) -> np.ndarray:
    """Return a brain band that steps through every style, left to right."""
    index, row, col = where
    tissue = frame(gray(brain_maps(index)[1]["tissue"]), (row, col))
    return strip_on_black([band(where, s) for s in STYLES], tissue)


BLACK = np.zeros(3)

# Every banner, in order; the files are numbered in this order.
BANNERS: dict[str, Callable[[], np.ndarray]] = {
    "brain_stains_thalamus": lambda: stains(THALAMUS),
    "brain_split_cortex": lambda: diagonal_strip([band(CORTEX, "before"), band(CORTEX, "if")]),
    "brain_if_temporal": lambda: band(TEMPORAL, "if"),
    "brain_stains_cerebellum": lambda: stains(CEREBELLUM),
    "brain_split_cerebellum": lambda: diagonal_strip([band(CEREBELLUM, "before"), band(CEREBELLUM, "kb")]),
    "brain_if_cerebellum": lambda: band(CEREBELLUM, "if"),
    "brain_brainbow_cerebellum": lambda: band(CEREBELLUM, "brainbow"),
    "brain_kb_cerebellum": lambda: band(CEREBELLUM, "kb"),
    "brain_folia_tiles": lambda: place([brain(1300, s)[BRAIN_ZOOM] for s in STYLES], 290, gap=12, bg=BLACK),
    "brain_fade_basal_ganglia": lambda: fade(band(BASAL_GANGLIA, "before"), band(BASAL_GANGLIA, "if")),
    "brain_he_basal_ganglia": lambda: band(BASAL_GANGLIA, "he"),
    "brain_if_frontal": lambda: band(FRONTAL, "if"),
    "brain_kb_cortex": lambda: band(UPPER_CORTEX, "kb"),
    "brain_brainbow_parietal": lambda: band(PARIETAL, "brainbow"),
    "brain_he_occipital": lambda: band(OCCIPITAL, "he"),
    "brain_series_if": lambda: place([whole_slice(i, "if") for i in (700, 850, 1300)], 370, gap=40, x=0.56),
    "brain_series_kb": lambda: place([whole_slice(i, "kb") for i in (500, 1000, 1150)], 370, gap=40, x=0.56),
    "body_depth": lambda: lie(lying_body()["depth"]),
    "body_petct": lambda: lie(lying_body()["petct"]),
    "body_ihc": lambda: lie(lying_body()["ihc"]),
    "body_split": lambda: diagonal_strip([lie(lying_body()[k]) for k in ("before", "depth")]),
    "ct_coronal_stains": lambda: strip_on_black(
        [lie(img) for img in lying_coronal()[0]], lie(lying_coronal()[1], bg=BLACK)
    ),
    "ct_coronal_if": lambda: lie(lying_coronal()[0][3]),
    "ct_coronal_trichrome": lambda: lie(lying_coronal()[0][2]),
    "ct_axial_if_pair": lambda: place([axial()[0][3], axial()[1][3]], 360, x=0.58),
    "ct_axial_abdomen_trio": lambda: place([axial()[0][i] for i in (0, 2, 3)], 360, gap=20, bg=BLACK),
    "mixed_brain_ct_if": lambda: place([whole_slice(850, "if"), axial()[0][3], axial()[1][3]], 360, x=0.53),
}


def overview(names: list[str]) -> Image.Image:
    """Return a contact sheet of the saved banners at a third of their size, three per row, labelled."""
    cols, w, h, label = 3, W // 3, H // 3, 26
    sheet = Image.new(
        "RGB", (cols * (w + 12) + 12, -(-len(names) // cols) * (h + label + 12) + 12), "#0b0b0e"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for k, name in enumerate(names):
        x, y = 12 + (k % cols) * (w + 12), 12 + (k // cols) * (h + label + 12)
        sheet.paste(Image.open(OUT / f"{name}.jpg").resize((w, h), Image.Resampling.LANCZOS), (x, y + label))
        draw.text((x, y + 2), name, fill="#f4f4f6", font=font)
    return sheet


def main() -> None:
    """Build the banners whose names contain any command-line argument (all by default), then the overview."""
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"{k:02d}_{name}" for k, name in enumerate(BANNERS, start=1)]
    for numbered, (name, build) in zip(names, BANNERS.items(), strict=True):
        if sys.argv[1:] and not any(arg in name for arg in sys.argv[1:]):
            continue
        img = (np.clip(build(), 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(img).save(OUT / f"{numbered}.jpg", quality=95, subsampling=0)
        print(f"wrote {numbered}")
    overview([n for n in names if (OUT / f"{n}.jpg").exists()]).save(OUT / "_overview.jpg", quality=92)


if __name__ == "__main__":
    main()
