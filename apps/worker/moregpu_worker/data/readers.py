"""Array readers (ADR-0110). NumPy is built in; NIfTI, DICOM, PNG/JPEG and TIFF use the optional ``vision`` extra
(``pip install 'moregpu-worker[vision]'``) and are advertised per worker by :func:`capabilities`.

Axis conventions: volumes are returned as ``(Z, Y, X)`` (slowest axis first), 2-D images as ``(H, W)`` or
``(H, W, C)`` as stored. ``.npy`` files are memory-mapped read-only so a slice touches only the bytes it needs.
"""
from __future__ import annotations

import importlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np

_IMAGE = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
_TIFF = (".tif", ".tiff")
_FORMATS = {"npy", "npz", "nifti", "dicom", "image", "tiff"}


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def capabilities() -> dict:
    return {
        "numpy": True,
        "nifti": _has("nibabel"),
        "dicom": _has("pydicom") or _has("SimpleITK"),
        "image": _has("PIL"),
        "tiff": _has("tifffile"),
        "s5cmd": shutil.which("s5cmd") is not None,
        "gsutil": shutil.which("gsutil") is not None,
    }


def _import(name: str):
    return importlib.import_module(name)


def _need(name: str, what: str):
    try:
        return _import(name)
    except ImportError as e:
        raise RuntimeError(f"reading {what} needs '{name}' — install the worker's vision extra "
                           f"(pip install 'moregpu-worker[vision]')") from e


def _detect(path: Path) -> str:
    if path.is_dir():
        return "dicom"
    name = path.name.lower()
    if name.endswith(".npy"):
        return "npy"
    if name.endswith(".npz"):
        return "npz"
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "nifti"
    if name.endswith(".dcm"):
        return "dicom"
    if name.endswith(_IMAGE):
        return "image"
    if name.endswith(_TIFF):
        return "tiff"
    raise ValueError(f"cannot tell the format of {path.name!r}; pass fmt= (one of {sorted(_FORMATS)})")


def read_array(path, fmt: str | None = None) -> np.ndarray:
    """Read one array. ``fmt`` overrides detection by suffix; for ``.npz`` it may also be an array key
    (``"mask"`` or ``"npz:mask"``) — without a key the first array in the archive is returned."""
    path = Path(path)
    key = None
    if fmt is not None and fmt.startswith("npz:"):
        fmt, key = "npz", fmt[4:]
    if fmt is not None and fmt not in _FORMATS:
        if path.name.lower().endswith(".npz"):
            fmt, key = "npz", fmt
        else:
            raise ValueError(f"unknown format {fmt!r} (one of {sorted(_FORMATS)})")
    kind = fmt or _detect(path)
    if kind == "npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if kind == "npz":
        with np.load(path, allow_pickle=False) as z:
            if key is None:
                key = z.files[0]
            if key not in z.files:
                raise KeyError(f"{key!r} not in {path.name} (has {z.files})")
            return z[key]
    if kind == "nifti":
        return read_nifti(path)[0]
    if kind == "dicom":
        return read_dicom(path)
    if kind == "image":
        Image = _need("PIL.Image", "PNG/JPEG")
        with Image.open(path) as im:
            return np.asarray(im)
    tifffile = _need("tifffile", "TIFF")
    return np.asarray(tifffile.imread(path))


def read_nifti(path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(array, affine)``: the array in ``(Z, Y, X[, ...])`` order (NIfTI stores ``(X, Y, Z, ...)``) and the
    4x4 voxel-to-world affine exactly as stored — note it maps ``(i, j, k) = (x, y, z)`` voxel indices."""
    nib = _need("nibabel", "NIfTI")
    name = Path(path).name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        img = nib.load(str(path))
    else:  # content-addressed cache / pushed blob without a suffix: parse from bytes
        raw = Path(path).read_bytes()
        if raw[:2] == b"\x1f\x8b":
            import gzip
            raw = gzip.decompress(raw)
        try:
            img = nib.Nifti1Image.from_bytes(raw)
        except Exception:
            img = nib.Nifti2Image.from_bytes(raw)
    data = np.asanyarray(img.dataobj)
    if data.ndim >= 3:
        data = np.swapaxes(data, 0, 2)
    return data, np.asarray(img.affine, dtype=np.float64)


def _slice_key(ds, fallback: int) -> tuple:
    pos = getattr(ds, "ImagePositionPatient", None)
    ori = getattr(ds, "ImageOrientationPatient", None)
    if pos is not None:
        pos = np.asarray([float(v) for v in pos])
        if ori is not None and len(ori) == 6:
            o = np.asarray([float(v) for v in ori])
            normal = np.cross(o[:3], o[3:])
            return (0, float(pos @ normal), fallback)
        return (0, float(pos[2]), fallback)
    return (1, float(getattr(ds, "InstanceNumber", fallback) or fallback), fallback)


def _pixels(ds) -> np.ndarray:
    arr = ds.pixel_array
    slope = getattr(ds, "RescaleSlope", None)
    inter = getattr(ds, "RescaleIntercept", None)
    if slope is not None or inter is not None:
        arr = arr.astype(np.float32) * float(slope if slope is not None else 1.0) + float(inter or 0.0)
    return arr


def read_dicom(path) -> np.ndarray:
    """A single DICOM file → its (rescaled) pixels; a directory → the series stacked as ``(Z, Y, X)``, sorted by
    ``ImagePositionPatient`` projected on the slice normal (``InstanceNumber`` as fallback). Non-DICOM files in the
    directory are skipped."""
    path = Path(path)
    try:
        pydicom = _import("pydicom")
    except ImportError:
        return _read_dicom_sitk(path)
    if path.is_file():
        return _pixels(pydicom.dcmread(str(path)))
    slices = []
    for i, f in enumerate(sorted(p for p in path.iterdir() if p.is_file())):
        try:
            ds = pydicom.dcmread(str(f))
        except Exception:  # InvalidDicomError and friends: not part of the series
            continue
        if "PixelData" not in ds:
            continue
        slices.append((_slice_key(ds, i), ds))
    if not slices:
        raise ValueError(f"no DICOM slices with pixel data in {path}")
    slices.sort(key=lambda t: t[0])
    return np.stack([_pixels(ds) for _, ds in slices])


def _read_dicom_sitk(path: Path) -> np.ndarray:  # pragma: no cover - only when pydicom is absent
    sitk = _need("SimpleITK", "DICOM")
    if path.is_file():
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))[0]
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(path))
    if not names:
        raise ValueError(f"no DICOM series in {path}")
    reader.SetFileNames(names)
    return sitk.GetArrayFromImage(reader.Execute())  # SimpleITK already sorts by position and returns (Z, Y, X)
