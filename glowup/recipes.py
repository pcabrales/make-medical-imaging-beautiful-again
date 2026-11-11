"""Stain recipes: which tissue map gets which stain, for each modality and style.

The amounts are hand-tuned to look like real slides; every one of them is a
fixed function of the scan's own intensities.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from glowup.styles import (
    ANILINE_BLUE,
    CRESYL_VIOLET,
    DAB,
    EOSIN,
    FLUOROPHORES,
    FUCHSIN,
    HEMATOXYLIN,
    LUXOL_FAST_BLUE,
    WEIGERT,
    brainbow,
    brightfield,
    depth_color_code,
    fluorescence,
)
from glowup.tissue import body_mask, brain_tissues, ct_tissues, window

# ---------------------------------------------------------------- brain MRI


def brain_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    """Return the maps the brain recipes stain, from an MRI slice where gray matter is brighter.

    Keys: ``tissue`` (brain vs background), ``density`` (shading-corrected signal,
    1 at the typical gray-matter level), ``gray`` and ``white`` (gray and white
    matter weighted by their signal, so texture inside each is kept) and
    ``edge`` (tissue boundaries: pial surface and the gray-white junction).
    """
    t = brain_tissues(img)
    density = t["density"]
    edge = ndimage.gaussian_gradient_magnitude(density, 1.2)
    edge = window(edge, 0, np.percentile(edge[t["tissue"] > 0.5], 99)) * t["tissue"]
    return {
        "tissue": t["tissue"],
        "density": density,
        "gray": t["gm"] * density,
        "white": t["wm"] * np.clip(1.4 * density, 0, 1.4),
        "edge": edge,
    }


def brain_he(m: dict[str, np.ndarray]) -> np.ndarray:
    """H&E: gray matter, rich in cell nuclei, takes more hematoxylin; white matter stays pink."""
    return brightfield(
        [
            (0.06 * m["density"] + 0.28 * m["gray"], HEMATOXYLIN),
            (0.30 * m["density"] + 0.06 * m["white"], EOSIN),
        ]
    )


def brain_kluver_barrera(m: dict[str, np.ndarray]) -> np.ndarray:
    """Kluver-Barrera: Luxol fast blue on myelin (white matter), cresyl violet on neurons (gray matter)."""
    return brightfield(
        [(0.85 * m["white"], LUXOL_FAST_BLUE), (0.45 * m["gray"] + 0.05 * m["density"], CRESYL_VIOLET)]
    )


def brain_fluorescence(m: dict[str, np.ndarray], glow_sigma: float = 10.0) -> np.ndarray:
    """Four-channel immunofluorescence: DAPI haze, neurons in Cy5, myelin in CFP, borders in YFP."""
    return fluorescence(
        [
            (0.22 * m["density"], FLUOROPHORES["DAPI"]),
            (0.8 * m["gray"] ** 1.4, FLUOROPHORES["Cy5"]),
            (0.5 * m["white"] ** 1.6, FLUOROPHORES["CFP"]),
            (0.7 * m["edge"] ** 1.5, FLUOROPHORES["YFP"]),
        ],
        exposure=1.15,
        glow=0.4,
        glow_sigma=glow_sigma,
    )


def brain_brainbow(m: dict[str, np.ndarray], n_segments: int = 2500) -> np.ndarray:
    """Brainbow: the cortical ribbon cut into "cells", each in its own random colour."""
    return brainbow(m["gray"], m["gray"] > 0.3, n_segments=n_segments, exposure=1.8, glow=0.35, glow_sigma=10)


# ---------------------------------------------------------------- CT


def ct_maps(hu: np.ndarray) -> dict[str, np.ndarray]:
    """Return CT tissue memberships (see ``ct_tissues``) inside the body, plus ``window``.

    ``window`` is the standard soft-tissue window (level 40 HU, width 400 HU)
    mapped to [0, 1]. The body edge is softened by one pixel.
    """
    body = ndimage.gaussian_filter(body_mask(hu).astype(np.float32), 1.0)
    maps = {name: m * body for name, m in ct_tissues(hu).items()}
    maps["window"] = window(hu, -160, 240) * body
    return maps


def ct_trichrome(m: dict[str, np.ndarray]) -> np.ndarray:
    """Masson's trichrome: muscle and organs red, bone blue, fat nearly white, lung lace pale blue."""
    return brightfield(
        [
            (0.55 * m["soft"] + 0.75 * m["enhanced"], FUCHSIN),
            (0.9 * m["bone"] + 0.18 * m["lung"] + 0.06 * m["fat"], ANILINE_BLUE),
            (0.25 * m["enhanced"] + 0.15 * m["bone"], WEIGERT),
        ]
    )


def ct_he(m: dict[str, np.ndarray]) -> np.ndarray:
    """H&E: muscle and bone matrix pink (eosinophilic), cell-dense organs purple, fat white."""
    return brightfield(
        [
            (0.08 * m["soft"] + 0.32 * m["enhanced"] + 0.08 * m["bone"] + 0.06 * m["lung"], HEMATOXYLIN),
            (
                0.45 * m["soft"]
                + 0.30 * m["enhanced"]
                + 0.75 * m["bone"]
                + 0.12 * m["lung"]
                + 0.03 * m["fat"],
                EOSIN,
            ),
        ]
    )


def ct_fluorescence(m: dict[str, np.ndarray], glow_sigma: float = 6.0) -> np.ndarray:
    """Multiplexed fluorescence: fat in DAPI, muscle in Cy5, enhanced organs in YFP, bone in CFP.

    Lungs stay dark, so the vessels inside them glow like a dendritic tree.
    """
    return fluorescence(
        [
            (0.30 * m["fat"], FLUOROPHORES["DAPI"]),
            (0.9 * m["soft"] * m["window"], FLUOROPHORES["Cy5"]),
            (m["enhanced"], FLUOROPHORES["YFP"]),
            (1.2 * m["bone"], FLUOROPHORES["CFP"]),
        ],
        exposure=1.2,
        glow=0.4,
        glow_sigma=glow_sigma,
    )


def ct_bone(hu: np.ndarray) -> np.ndarray:
    """Return the bone signal in [0, 1] (150 to 1050 HU) inside the body, dropping the table."""
    return window(hu, 150, 1050) * body_mask(hu)


def bone_depth_code(bone: np.ndarray, axis: int = 1) -> np.ndarray:
    """Depth colour-coded projection of the skeleton, like a confocal z-stack in Fiji.

    Args:
        bone: Bone signal (see ``ct_bone``) of a volume indexed (z, y, x).
        axis: Projection axis; 1 projects anterior to posterior (a coronal view),
            from blue (front) through green and yellow to red (back).
    """
    # Spread the colours over the skeleton's own depth, not the whole field of view.
    other = tuple(a for a in range(bone.ndim) if a != axis)
    present = np.nonzero(bone.max(axis=other) > 0.2)[0]
    bone = np.take(bone, np.arange(present[0], present[-1] + 1), axis=axis)
    return np.clip(1.6 * depth_color_code(bone, axis=axis), 0, 1)


def petct_fluorescence(bone: np.ndarray, uptake: np.ndarray, glow_sigma: float = 4.0) -> np.ndarray:
    """Two-colour fluorescence: skeleton (CT) in DAPI, glucose uptake (PET) in mCherry and YFP.

    Args:
        bone: Bone signal in [0, 1], e.g. a projection of ``ct_bone``.
        uptake: PET SUV on the same grid. Uptake above SUV 4 adds yellow, so hot
            organs burn from red to gold to white.
    """
    return fluorescence(
        [
            (bone, FLUOROPHORES["DAPI"]),
            (window(uptake, 0, 5), FLUOROPHORES["mCherry"]),
            (window(uptake, 4, 12), FLUOROPHORES["YFP"]),
        ],
        exposure=1.4,
        glow=0.5,
        glow_sigma=glow_sigma,
    )


def petct_ihc(m: dict[str, np.ndarray], suv: np.ndarray) -> np.ndarray:
    """Immunohistochemistry: brown DAB where glucose uptake is high, on a hematoxylin counterstain.

    In IHC, DAB marks where an antibody found its target; here it marks FDG
    uptake (SUV 1.5 to 8), and the anatomy from CT is the blue counterstain.
    """
    return brightfield(
        [
            (0.10 * m["soft"] + 0.16 * m["enhanced"] + 0.35 * m["bone"] + 0.05 * m["lung"], HEMATOXYLIN),
            (1.1 * window(suv, 1.5, 8) ** 0.8 * (m["soft"] + m["enhanced"] + m["bone"]), DAB),
        ]
    )


def petct_ihc_projection(bone: np.ndarray, uptake: np.ndarray) -> np.ndarray:
    """Immunohistochemistry of whole-body projections: glucose uptake in DAB, skeleton as counterstain.

    Args:
        bone: Bone signal in [0, 1], e.g. a projection of ``ct_bone``.
        uptake: PET SUV projection on the same grid. The body silhouette
            (SUV above 0.3) takes a trace of hematoxylin, so the outline shows.
    """
    body = ndimage.gaussian_filter((uptake > 0.3).astype(np.float32), 1.5)
    return brightfield([(0.08 * body + 0.7 * bone, HEMATOXYLIN), (1.2 * window(uptake, 2, 8) ** 0.8, DAB)])
