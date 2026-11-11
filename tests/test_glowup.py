"""Unit tests on synthetic images (no downloads needed)."""

from __future__ import annotations

import numpy as np
import pydicom
import pytest
from matplotlib import colormaps
from scipy import ndimage

from glowup.data import suv_factor
from glowup.figure import bbox, diagonal_split, diagonal_strip
from glowup.styles import EOSIN, HEMATOXYLIN, brightfield, depth_color_code, fluorescence, measure_stains
from glowup.tissue import body_mask, brain_tissues, ct_tissues, local_mean


def test_brightfield_follows_beer_lambert() -> None:
    """No stain transmits the light; stain densities add, and transmission is 10**-density."""
    zero, one = np.zeros((2, 2)), np.ones((2, 2))
    assert np.allclose(brightfield([(zero, HEMATOXYLIN)], light=(0.9, 0.95, 1.0)), (0.9, 0.95, 1.0))
    assert np.allclose(brightfield([(one, HEMATOXYLIN)]), 10.0**-HEMATOXYLIN)
    assert np.allclose(brightfield([(one, HEMATOXYLIN), (one, EOSIN)]), 10.0 ** -(HEMATOXYLIN + EOSIN))


def test_fluorescence_adds_light_and_saturates() -> None:
    """Black without signal, the fluorophore's hue for weak signal, white-capped for strong signal."""
    signal = np.array([[0.0, 0.01, 1.0, 100.0]])
    rgb = fluorescence([(signal, (1.0, 0.5, 0.0))])
    assert np.allclose(rgb[0, 0], 0)
    assert np.allclose(rgb[0, 1] / rgb[0, 1, 0], (1.0, 0.5, 0.0), atol=0.01)
    assert np.all(np.diff(rgb[0, :, 0]) > 0) and rgb.max() <= 1


def test_measure_stains_recovers_known_vectors() -> None:
    """Macenko estimation finds the two stains used to synthesize an H&E image."""
    rng = np.random.default_rng(0)
    h, e = HEMATOXYLIN / np.linalg.norm(HEMATOXYLIN), EOSIN / np.linalg.norm(EOSIN)
    amounts = rng.uniform(0, 1.2, (2, 200, 200))
    amounts[0, :, :50] = 0  # pure eosin
    amounts[1, :, 50:100] = 0  # pure hematoxylin
    rgb = (255 * brightfield([(amounts[0], h), (amounts[1], e)])).astype(np.uint8)
    found_h, found_e = measure_stains(rgb)
    assert found_h @ h > 0.99 and found_e @ e > 0.99


def test_suv_factor_decay_corrects_the_dose() -> None:
    """SUV factor = weight / dose decayed from injection to the series start."""
    ds = pydicom.Dataset()
    ds.Units, ds.DecayCorrection, ds.SeriesTime, ds.PatientWeight = "BQML", "START", "100000", 70
    info = pydicom.Dataset()
    info.RadiopharmaceuticalStartTime, info.RadionuclideTotalDose, info.RadionuclideHalfLife = (
        "090000",
        370e6,
        6586.2,
    )
    ds.RadiopharmaceuticalInformationSequence = [info]
    assert suv_factor(ds) == pytest.approx(70_000 / (370e6 * 2 ** (-3600 / 6586.2)))
    ds.Units = "CNTS"
    with pytest.raises(ValueError):
        suv_factor(ds)


def test_local_mean_ignores_pixels_outside_the_weight() -> None:
    """The weighted mean of a constant region stays constant up to its edge."""
    img = np.zeros((60, 60))
    img[20:40, 20:40] = 5.0
    assert np.allclose(local_mean(img, img > 0, sigma=10)[20:40, 20:40], 5.0)


def test_brain_tissues_separate_gray_and_white_matter_under_shading() -> None:
    """A bright cortical ring around darker white matter, under strong shading.

    The shading makes white matter on the right (0.7 x 1.3) brighter than gray
    matter on the left (1.0 x 0.7), so only a shading-corrected split passes.
    """
    yy, xx = np.mgrid[-100:100, -100:100]
    r = np.hypot(yy, xx)
    img = np.where(r < 60, 0.7, 0.0) + np.where((r >= 60) & (r < 80), 1.0, 0.0)
    img = ndimage.gaussian_filter(img, 1.5) * np.linspace(0.7, 1.3, 200)[None, :]  # partial volume, shading
    img = np.abs(img + np.random.default_rng(0).normal(0, 0.03, img.shape))  # magnitude noise
    t = brain_tissues(img, sigma=20)
    ring, core = (r > 64) & (r < 76), r < 50
    for side in (xx < 0, xx > 0):
        assert t["gm"][ring & side].mean() > 0.85 and t["wm"][core & side].mean() > 0.85
    assert t["tissue"][r > 90].max() < 0.01


def test_ct_tissues_follow_hounsfield_ranges() -> None:
    """Each textbook HU value belongs mostly to its own class."""
    hu = np.array([-750.0, -100, 40, 150, 1000])
    classes = ct_tissues(hu)
    for i, name in enumerate(["lung", "fat", "soft", "enhanced", "bone"]):
        assert classes[name][i] > 0.9
        assert all(classes[other][i] < 0.1 for other in classes if other != name)


def test_body_mask_fills_lungs_and_drops_the_table() -> None:
    """The body is the largest region with holes filled; a separate table strip is dropped."""
    hu = np.full((120, 120), -1000.0)
    yy, xx = np.mgrid[0:120, 0:120]
    hu[np.hypot(yy - 50, xx - 60) < 40] = 40
    hu[np.hypot(yy - 50, xx - 60) < 15] = -850  # lung inside the body
    hu[100:106, 10:110] = 200  # table
    mask = body_mask(hu)
    assert mask[50, 60] and mask[50, 30]
    assert not mask[103, 60] and not mask[5, 5]


def test_bbox_adds_margin_within_the_image() -> None:
    """The box is grown by the margin but clipped at the image border."""
    mask = np.zeros((10, 10), bool)
    mask[1, 4:6] = True
    assert bbox(mask, margin=2) == (slice(0, 4), slice(2, 8))


def test_diagonal_split_keeps_both_images_away_from_the_seam() -> None:
    """Far from the seam, the left comes from ``before`` and the right from ``after``."""
    before, after = np.zeros((50, 200, 3)), np.full((50, 200, 3), 0.5)
    out = diagonal_split(before, after)
    assert np.allclose(out[:, :20], 0, atol=1e-3) and np.allclose(out[:, -20:], 0.5, atol=1e-3)
    assert out[25, 100].min() > 0.9  # the bright seam passes through the centre


def test_diagonal_strip_spreads_bands_over_the_span() -> None:
    """With ``span=(0.5, 1)``, the first image fills the left half and the others share the right half."""
    images = [np.full((40, 600, 3), value) for value in (0.0, 0.25, 0.5)]
    out = diagonal_strip(images, span=(0.5, 1.0))  # seams at columns 400 and 500
    assert np.allclose(out[:, :300], 0, atol=1e-3)
    assert np.allclose(out[:, 450], 0.25, atol=0.01) and np.allclose(out[:, 560:], 0.5, atol=0.01)


def test_depth_color_code_tints_by_depth() -> None:
    """A voxel at the front takes the first colour of the LUT, one at the back the last."""
    volume = np.zeros((2, 5, 3))
    volume[0, 0, 0] = volume[1, 4, 0] = 1.0
    rgb = depth_color_code(volume, axis=1)
    lut = colormaps["turbo"]
    assert np.allclose(rgb[0, 0], lut(0.0)[:3]) and np.allclose(rgb[1, 0], lut(1.0)[:3])
