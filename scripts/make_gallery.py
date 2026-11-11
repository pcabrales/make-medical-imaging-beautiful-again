"""Build every figure in ``gallery/`` from the public datasets (downloaded on first run).

Usage:
    python scripts/make_gallery.py            # all figures
    python scripts/make_gallery.py brain ct   # only some of them
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import skimage.data  # noqa: E402
from scipy import ndimage  # noqa: E402

from glowup import recipes  # noqa: E402
from glowup.data import load_brain_slice, load_ct_planes, load_petct  # noqa: E402
from glowup.figure import (  # noqa: E402
    MUTED,
    bbox,
    credit,
    diagonal_split,
    gray,
    grid,
    header,
    isotropic,
    save,
    show,
    use_style,
    zoom_box,
)
from glowup.tissue import body_mask, window  # noqa: E402

GALLERY = Path(__file__).resolve().parents[1] / "gallery"
BRAIN_CREDIT = "MRI: Edlow et al. 2019, 7 T ex vivo brain at 100 µm, OpenNeuro ds002179 (CC0)"
PETCT_CREDIT = "PET/CT: Gatidis & Kuestner 2022, TCIA FDG-PET-CT-Lesions, negative control (CC BY 4.0)"

# Slices and regions shown (indices into the arrays returned by glowup.data).
HERO_SLICE = 850  # coronal: thalamus, hippocampi, insula
BRAIN_SLICE = 1300  # coronal: occipital lobes and cerebellum
BRAIN_ZOOM = (slice(830, 1150), slice(560, 880))  # cerebellar folia, 32 mm wide
CT_AXIAL = (520, 605)  # kidneys and liver; heart and lungs
CT_CORONAL_ROW = 320  # through the aorta, the lumbar spine and the hips
CT_TORSO = slice(300, 656)  # coronal z range: upper thighs to neck
BODY_TOP = 656  # the head was blanked out (defaced) above this slice


def brain_before(img: np.ndarray) -> np.ndarray:
    """Standard grayscale display of an MRI slice (0 to the 99.9th percentile)."""
    return gray(window(img, 0, np.percentile(img, 99.9)))


def ct_before(hu: np.ndarray) -> np.ndarray:
    """Standard soft-tissue CT window (level 40 HU, width 400 HU)."""
    return gray(window(hu, -160, 240))


def upsample(plane: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Interpolate a PET plane from the half-resolution grid of ``load_petct`` to full resolution."""
    return ndimage.zoom(plane, factors, order=1, grid_mode=True, mode="nearest")


def coronal_plane(z: slice) -> tuple[np.ndarray, np.ndarray]:
    """Return the coronal CT (HU) and PET (SUV) planes over ``z``, isotropic and head up."""
    _, coronal = load_ct_planes(CT_AXIAL, CT_CORONAL_ROW)
    _, pet = load_petct()
    dz = pet.spacing[0]
    dx = pet.spacing[2] / 2  # full in-plane resolution
    hu = isotropic(coronal[z], dz, dx)[::-1]
    suv = isotropic(upsample(pet.data[z, CT_CORONAL_ROW // 2], (1, 2)), dz, dx)[::-1]
    return hu, suv


def body_projections() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return whole-body coronal projections, isotropic and head up: bone MIP, bone depth code, SUV MIP."""
    ct_volume, pet = load_petct()
    dz, _, dx = ct_volume.spacing
    bone = recipes.ct_bone(ct_volume.data[:BODY_TOP])
    bone_mip = isotropic(bone.max(axis=1), dz, dx)[::-1]
    depth = isotropic(recipes.bone_depth_code(bone), dz, dx)[::-1]
    suv_mip = isotropic(pet.data[:BODY_TOP].max(axis=1), dz, dx)[::-1]
    return bone_mip, depth, suv_mip


def hero() -> None:
    """One coronal brain slice, half as radiology shows it, half as immunofluorescence."""
    raw = load_brain_slice(HERO_SLICE)
    maps = recipes.brain_maps(ndimage.gaussian_filter(raw, 0.6))
    rows, cols = bbox(maps["tissue"] > 0.5, margin=60)
    image = diagonal_split(brain_before(raw)[rows, cols], recipes.brain_fluorescence(maps)[rows, cols])

    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / 150, h / 150))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    for x, ha, label in [(0.03, "left", "BEFORE"), (0.97, "right", "AFTER")]:
        fig.text(x, 0.035, label, ha=ha, va="bottom", fontsize=15, weight="bold")
    fig.text(0.03, 0.012, "7 T MRI, as radiology shows it", ha="left", va="bottom", fontsize=9, color=MUTED)
    fig.text(
        0.97, 0.012, "same voxels, as immunofluorescence", ha="right", va="bottom", fontsize=9, color=MUTED
    )
    save(fig, GALLERY / "00_hero.jpg")


def the_problem() -> None:
    """Radiology as it is usually shown, next to microscopy as it is usually shown."""
    mri = brain_before(load_brain_slice(BRAIN_SLICE))[700:1240, 420:960]
    (abdomen, _), _ = load_ct_planes(CT_AXIAL, CT_CORONAL_ROW)
    rows, cols = bbox(body_mask(abdomen), margin=12)
    ct = ct_before(abdomen)[rows, cols]
    _, pet = load_petct()
    dz, _, dx = pet.spacing
    mip = isotropic(pet.data[:BODY_TOP].max(axis=1), dz, dx)[::-1]
    pet_mip = gray(1 - window(mip, 0, 7))

    kidney = skimage.data.kidney().max(axis=0).astype(np.float32)
    lo, hi = np.percentile(kidney, [1, 99.7], axis=(0, 1))
    kidney = window(kidney, lo, hi)[..., ::-1]  # 450/515/605 nm channels shown as blue/green/red
    tiles = [
        (mri, "MRI", "7 T brain, coronal"),
        (ct, "CT", "abdomen, soft-tissue window"),
        (pet_mip, "PET", "FDG, maximum intensity projection"),
        (skimage.data.skin()[80:880, 240:1040] / 255, "H&E", "skin with a benign nevus"),
        (kidney, "Fluorescence", "mouse kidney, confocal"),
        (skimage.data.immunohistochemistry() / 255, "Immunohistochemistry", "colon, DAB and hematoxylin"),
    ]

    fig, axes = grid(3, [1, 1], width=13, labelled_rows=(0, 1), label=0.95, left=0.75, gap=0.15, bottom=0.5)
    for ax, (img, title, note) in zip(axes.flat, tiles, strict=True):
        h, w = img.shape[:2]
        s = min(h, w)  # square tiles: the top (head end) or centre of the image
        r0 = 0 if title == "PET" else (h - s) // 2
        show(ax, img[r0 : r0 + s, (w - s) // 2 : (w + s) // 2], title, note)
    for ax, label in zip(axes[:, 0], ["Radiology", "Microscopy"], strict=True):
        ax.annotate(
            label,
            (0, 0.5),
            xycoords="axes fraction",
            xytext=(-14, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="center",
            fontsize=17,
            weight="bold",
            color=MUTED,
        )
    header(
        fig,
        "Exhibit A: the beauty gap",
        "Two fields image the same tissues. Guess which one gets the journal covers.",
        left=0.75,
    )
    credit(
        fig,
        "Microscopy: scikit-image sample data (public domain, CC0, no known restrictions).\n"
        f"{BRAIN_CREDIT}. {PETCT_CREDIT}.",
    )
    save(fig, GALLERY / "01_the_problem.jpg")


def brain() -> None:
    """A coronal brain slice in four stains, whole and magnified."""
    raw = load_brain_slice(BRAIN_SLICE)
    maps = recipes.brain_maps(ndimage.gaussian_filter(raw, 0.6))
    columns = [
        (brain_before(raw), "Before", "7 T MRI, 100 µm, grayscale"),
        (recipes.brain_he(maps), "H&E", "cell-rich gray matter purple,\nwhite matter pink"),
        (
            recipes.brain_kluver_barrera(maps),
            "Klüver-Barrera",
            "myelin in Luxol fast blue,\nneurons in cresyl violet",
        ),
        (
            recipes.brain_fluorescence(maps),
            "Immunofluorescence",
            "neurons, myelin, borders\nand nuclei in four colours",
        ),
        (recipes.brain_brainbow(maps), "Brainbow", "every cell its own colour"),
    ]
    rows, cols = bbox(maps["tissue"] > 0.5, margin=30)
    zr, zc = BRAIN_ZOOM
    box = (
        slice(zr.start - rows.start, zr.stop - rows.start),
        slice(zc.start - cols.start, zc.stop - cols.start),
    )

    fig, axes = grid(5, [(rows.stop - rows.start) / (cols.stop - cols.start), 1])
    for j, (img, title, note) in enumerate(columns):
        show(axes[0, j], img[rows, cols], title, note)
        zoom_box(axes[0, j], *box)
        show(axes[1, j], img[zr, zc])
    header(fig, "Brain MRI", "Same slice, same voxels. The bottom row magnifies the cerebellar folia (box).")
    credit(fig, BRAIN_CREDIT)
    save(fig, GALLERY / "02_brain.jpg")


def ct_columns(hu: np.ndarray, suv: np.ndarray, glow_sigma: float) -> list[tuple[np.ndarray, str, str]]:
    """Return the CT panels (before and four stains) of one plane, cropped to the body."""
    maps = recipes.ct_maps(hu)
    rows, cols = bbox(maps["window"] > 0, margin=10)
    panels = [
        (ct_before(hu), "Before", "CT, soft-tissue window"),
        (recipes.ct_he(maps), "H&E", "muscle and bone pink,\ncell-dense organs purple"),
        (recipes.ct_trichrome(maps), "Masson's trichrome", "muscle red, bone blue,\nfat unstained"),
        (
            recipes.ct_fluorescence(maps, glow_sigma),
            "Immunofluorescence",
            "fat, muscle, organs\nand bone in four colours",
        ),
        (recipes.petct_ihc(maps, suv), "PET as IHC", "glucose uptake in brown DAB,\nanatomy as counterstain"),
    ]
    return [(img[rows, cols], title, note) for img, title, note in panels]


def ct() -> None:
    """Two axial CT slices in four stains, with PET as immunohistochemistry."""
    axial, _ = load_ct_planes(CT_AXIAL, CT_CORONAL_ROW)
    _, pet = load_petct()
    planes = [ct_columns(hu, upsample(pet.data[z], (2, 2)), 8) for hu, z in zip(axial, CT_AXIAL, strict=True)]

    fig, axes = grid(5, [p[0][0].shape[0] / p[0][0].shape[1] for p in planes])
    for i, panels in enumerate(planes):
        for j, (img, title, note) in enumerate(panels):
            show(axes[i, j], img, *((title, note) if i == 0 else ()))
    header(
        fig,
        "CT, axial",
        "Abdomen (top) and chest (bottom) of a contrast-enhanced PET/CT. "
        "Each stain follows textbook Hounsfield ranges.",
    )
    credit(fig, PETCT_CREDIT)
    save(fig, GALLERY / "03_ct_axial.jpg")


def ct_coronal() -> None:
    """A coronal CT plane through the torso in four stains."""
    panels = ct_columns(*coronal_plane(CT_TORSO), 10)

    fig, axes = grid(5, [panels[0][0].shape[0] / panels[0][0].shape[1]])
    for ax, (img, title, note) in zip(axes[0], panels, strict=True):
        show(ax, img, title, note)
    header(
        fig,
        "CT, coronal",
        "Neck to upper thighs, through the heart, the aorta, the lumbar spine and the hips.",
    )
    credit(fig, PETCT_CREDIT)
    save(fig, GALLERY / "04_ct_coronal.jpg")


def whole_body() -> None:
    """Whole-body projections: skeleton as a depth-coded z-stack, PET/CT as two-colour fluorescence."""
    bone_mip, depth, suv_mip = body_projections()
    panels = [
        (gray(bone_mip), "Before", "maximum intensity projection\nof the CT, bone window"),
        (depth, "Depth colour code", "front blue, back red, as for\na confocal z-stack"),
        (gray(1 - window(suv_mip, 0, 7)), "Before", "maximum intensity projection\nof the FDG PET"),
        (
            recipes.petct_fluorescence(bone_mip, suv_mip),
            "Two-colour fluorescence",
            "skeleton in DAPI, glucose\nuptake in mCherry and YFP",
        ),
    ]
    rows, cols = bbox((bone_mip > 0.05) | (suv_mip > 0.4), margin=12)

    fig, axes = grid(4, [(rows.stop - rows.start) / (cols.stop - cols.start)], width=16)
    for ax, (img, title, note) in zip(axes[0], panels, strict=True):
        show(ax, img[rows, cols], title, note)
    header(fig, "Whole body", "Neck to toes (the head was removed from the public data for privacy).")
    credit(fig, PETCT_CREDIT)
    save(fig, GALLERY / "05_whole_body.jpg")


FIGURES = {
    "hero": hero,
    "problem": the_problem,
    "brain": brain,
    "ct": ct,
    "coronal": ct_coronal,
    "body": whole_body,
}


def main() -> None:
    """Build the figures named on the command line, or all of them."""
    use_style()
    for name in sys.argv[1:] or FIGURES:
        print(f"building {name}")
        FIGURES[name]()


if __name__ == "__main__":
    main()
