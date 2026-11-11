"""Download and load the public datasets used by the gallery.

Both sources are public and de-identified:

- OpenNeuro ds002179 (CC0): 7 T MRI of an ex vivo human brain at 100 micron
  resolution (Edlow et al., 2019). Single coronal slices are read straight out
  of the published 2 GB zip of TIFFs with HTTP range requests.
- TCIA FDG-PET-CT-Lesions v2 (CC BY 4.0, defaced images): one negative-control
  whole-body FDG PET/CT study, i.e. a scan with no PET-positive lesions.
"""

from __future__ import annotations

import io
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage

DATA_DIR = Path(os.environ.get("GLOWUP_DATA", Path(__file__).resolve().parents[1] / "data"))

BRAIN_URL = "https://s3.amazonaws.com/openneuro.org/ds002179/derivatives/sub-EXC004/"
BRAIN_CORONAL_ZIP = BRAIN_URL + (
    "synthesized_FLASH25_100um_tiff_images/Synthesized_FLASH25_100um_TIFF_Coronal_Images.zip"
)

NBIA_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
# Public (defaced) series of one negative-control study in TCIA FDG-PET-CT-Lesions v2.
PETCT_SERIES = {
    "ct": "1.3.6.1.4.1.14519.5.2.1.4219.6651.316972198471893103321834187720146616413",
    "pet": "1.3.6.1.4.1.14519.5.2.1.4219.6651.210515093446179705696292174352008985416",
}


@dataclass
class Volume:
    """A 3D image on an axis-aligned grid in DICOM patient coordinates.

    Attributes:
        data: Voxel values, indexed (z, y, x): z runs feet to head, y anterior to
            posterior, x patient right to left.
        spacing: Voxel size in mm, (z, y, x).
        origin: Patient position in mm of the centre of voxel (0, 0, 0), (z, y, x).
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists.

    The file is written under a temporary name first, so an interrupted
    download is never mistaken for a finished one.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out)
        part.rename(dest)
    return dest


def fetch_zip(url: str, folder: Path) -> Path:
    """Download a zip archive once and extract it into ``folder``."""
    if not (folder / ".extracted").exists():
        archive = download(url, folder / "archive.zip")
        if not zipfile.is_zipfile(archive):
            # TCIA answers HTTP 200 with a short text message for non-public series.
            message = archive.read_text(errors="replace")[:200]
            archive.unlink()
            raise RuntimeError(f"{url} did not return a zip archive: {message}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()
        (folder / ".extracted").touch()
    return folder


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file that fetches bytes with HTTP range requests.

    Wrapped in ``io.BufferedReader`` it lets ``zipfile`` read one member of a
    large remote archive without downloading the rest.
    """

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def load_brain_slice(index: int) -> np.ndarray:
    """Return coronal slice ``index`` (1 to 1760) of the 100 micron ex vivo brain, as float32.

    Rows run superior to inferior. The slice is cached as a TIFF on first use
    (about 1.3 MB is downloaded from the 2 GB archive).
    """
    path = DATA_DIR / "exvivo_brain" / f"coronal_{index:04d}.tiff"
    if not path.exists():
        remote = io.BufferedReader(HttpRangeFile(BRAIN_CORONAL_ZIP), buffer_size=1 << 20)
        with zipfile.ZipFile(remote) as zf:
            # Member names start with "../", so never extract; read the bytes only.
            (member,) = [n for n in zf.namelist() if n.endswith(f"_Coronal_{index:04d}.tiff")]
            data = zf.read(member)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return np.asarray(Image.open(path), dtype=np.float32)


def _patient_z(ds: pydicom.Dataset) -> float:
    """Return the patient z position (mm) of a DICOM slice."""
    return float(ds.ImagePositionPatient[2])


def suv_factor(ds: pydicom.Dataset) -> float:
    """Return the factor that converts PET activity (Bq/ml) to body-weight SUV.

    The injected dose is decay-corrected to the series start time, which is the
    reference time for images with ``DecayCorrection == START``.
    """
    if ds.Units != "BQML" or ds.DecayCorrection != "START":
        raise ValueError(f"Unsupported PET units/decay correction: {ds.Units}/{ds.DecayCorrection}")
    info = ds.RadiopharmaceuticalInformationSequence[0]
    elapsed = (
        datetime.strptime(ds.SeriesTime[:6], "%H%M%S")
        - datetime.strptime(info.RadiopharmaceuticalStartTime[:6], "%H%M%S")
    ).total_seconds()
    dose_bq = float(info.RadionuclideTotalDose) * 2 ** (-elapsed / float(info.RadionuclideHalfLife))
    return float(ds.PatientWeight) * 1000.0 / dose_bq


def _sorted_headers(folder: Path) -> list[pydicom.Dataset]:
    """Return the headers of the ``*.dcm`` files in ``folder``, sorted feet to head."""
    return sorted((pydicom.dcmread(f, stop_before_pixels=True) for f in folder.glob("*.dcm")), key=_patient_z)


def _pixels(header: pydicom.Dataset) -> np.ndarray:
    """Read the image of a slice with its rescale slope and intercept applied.

    Slope and intercept can change from slice to slice (they do for PET).
    """
    ds = pydicom.dcmread(header.filename)
    return ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)


def read_dicom_series(folder: Path, step: int = 1, suv: bool = False) -> Volume:
    """Read an axial DICOM series into a ``Volume`` with rescale slope/intercept applied.

    Args:
        folder: Directory with one series of ``*.dcm`` files.
        step: In-plane block-averaging factor (2 halves the in-plane resolution),
            which keeps whole-body CT within a few hundred MB of memory.
        suv: Convert PET activity to body-weight SUV.
    """
    headers = _sorted_headers(folder)
    first = headers[0]
    if not np.allclose([float(v) for v in first.ImageOrientationPatient], [1, 0, 0, 0, 1, 0]):
        raise ValueError("Only axial series without gantry tilt are supported.")
    z = np.array([_patient_z(h) for h in headers])
    dz = np.diff(z)
    if not np.allclose(dz, dz[0]):
        raise ValueError("Slices are not evenly spaced.")

    rows, cols = int(first.Rows) // step, int(first.Columns) // step
    data = np.empty((len(headers), rows, cols), np.float32)
    for i, h in enumerate(headers):
        img = _pixels(h)[: rows * step, : cols * step]
        data[i] = img.reshape(rows, step, cols, step).mean(axis=(1, 3))
    if suv:
        data *= suv_factor(first)

    dy, dx = (float(v) for v in first.PixelSpacing)
    x0, y0 = (float(v) for v in first.ImagePositionPatient[:2])
    # A block-averaged voxel is centred halfway across the block it replaces.
    shift = (step - 1) / 2
    return Volume(data, (float(dz[0]), dy * step, dx * step), (float(z[0]), y0 + shift * dy, x0 + shift * dx))


def resample_like(src: Volume, ref: Volume) -> Volume:
    """Trilinearly resample ``src`` onto the grid of ``ref`` (both axis-aligned)."""
    scale = np.array(ref.spacing) / np.array(src.spacing)
    offset = (np.array(ref.origin) - np.array(src.origin)) / np.array(src.spacing)
    data = ndimage.affine_transform(
        src.data, scale, offset=offset, output_shape=ref.data.shape, order=1, mode="constant", cval=0.0
    )
    return Volume(data, ref.spacing, ref.origin)


def load_petct(step: int = 2) -> tuple[Volume, Volume]:
    """Return the whole-body CT (HU) and the PET (SUV) resampled onto the CT grid.

    The first call downloads about 600 MB of DICOM from TCIA and caches the
    loaded volumes as ``.npz`` files next to them.
    """
    root = DATA_DIR / "tcia_fdg_pet_ct"
    cache = root / f"petct_step{step}.npz"
    if not cache.exists():
        ct = read_dicom_series(fetch_zip(NBIA_IMAGE_URL + PETCT_SERIES["ct"], root / "ct"), step=step)
        pet = read_dicom_series(fetch_zip(NBIA_IMAGE_URL + PETCT_SERIES["pet"], root / "pet"), suv=True)
        pet = resample_like(pet, ct)
        np.savez(cache, ct=ct.data, pet=pet.data, spacing=ct.spacing, origin=ct.origin)
    with np.load(cache) as f:
        spacing, origin = tuple(f["spacing"].tolist()), tuple(f["origin"].tolist())
        return Volume(f["ct"], spacing, origin), Volume(f["pet"], spacing, origin)


def load_ct_planes(axial: tuple[int, ...], coronal: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full-resolution CT planes (HU) of the PET/CT study.

    Args:
        axial: Slice indices (feet to head, as in ``load_petct``) of the axial planes.
        coronal: Row index (anterior to posterior, 512 rows) of the coronal plane.

    Returns:
        ``(axial_slices, coronal_plane)``: arrays of shape ``(len(axial), 512, 512)``
        and ``(n_slices, 512)``, cached as ``.npz`` after the first call.
    """
    root = DATA_DIR / "tcia_fdg_pet_ct"
    cache = root / f"ct_planes_ax{'-'.join(map(str, axial))}_cor{coronal}.npz"
    if not cache.exists():
        headers = _sorted_headers(fetch_zip(NBIA_IMAGE_URL + PETCT_SERIES["ct"], root / "ct"))
        slices, rows = {}, []
        for i, h in enumerate(headers):
            img = _pixels(h).astype(np.float32)
            rows.append(img[coronal])
            if i in axial:
                slices[i] = img
        np.savez(cache, axial=np.stack([slices[i] for i in axial]), coronal=np.stack(rows))
    with np.load(cache) as f:
        return f["axial"], f["coronal"]
