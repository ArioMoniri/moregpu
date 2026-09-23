"""Readers: numpy/memmap, NIfTI geometry, DICOM series, PNG/JPEG/TIFF — synthetic data only (ADR-0110)."""
import numpy as np
import pytest

from moregpu_worker.data import readers as R


def test_capabilities_shape():
    c = R.capabilities()
    assert set(c) >= {"numpy", "nifti", "dicom", "image", "tiff", "s5cmd", "gsutil"}
    assert c["numpy"] is True and all(isinstance(v, bool) for v in c.values())


def test_npy_is_memmapped(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.save(tmp_path / "a.npy", a)
    out = R.read_array(tmp_path / "a.npy")
    assert isinstance(out, np.memmap) and np.array_equal(out, a)


def test_npz_first_and_key(tmp_path):
    np.savez(tmp_path / "z.npz", img=np.ones((2, 2)), mask=np.zeros((2, 2), dtype=np.uint8))
    assert np.array_equal(R.read_array(tmp_path / "z.npz"), np.ones((2, 2)))
    assert R.read_array(tmp_path / "z.npz", fmt="mask").dtype == np.uint8
    assert R.read_array(tmp_path / "z.npz", fmt="npz:mask").dtype == np.uint8
    with pytest.raises(KeyError):
        R.read_array(tmp_path / "z.npz", fmt="nope")


def test_fmt_overrides_suffix(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(3))
    (tmp_path / "blob").write_bytes((tmp_path / "a.npy").read_bytes())
    assert np.array_equal(R.read_array(tmp_path / "blob", fmt="npy"), np.ones(3))


def test_unknown_format(tmp_path):
    (tmp_path / "x.weird").write_bytes(b"?")
    with pytest.raises(ValueError):
        R.read_array(tmp_path / "x.weird")
    with pytest.raises(ValueError):
        R.read_array(tmp_path / "x.npy", fmt="nonsense-format")


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_geometry_round_trip(tmp_path, suffix):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    vol_zyx = rng.standard_normal((5, 6, 7)).astype(np.float32)   # Z, Y, X
    aff = np.array([[0.8, 0, 0, -10.0], [0, 0.7, 0, 5.0], [0, 0, 2.5, 3.0], [0, 0, 0, 1]])
    p = tmp_path / f"vol{suffix}"
    nib.save(nib.Nifti1Image(vol_zyx.transpose(2, 1, 0), aff), str(p))
    arr, a2 = R.read_nifti(p)
    assert arr.shape == (5, 6, 7)
    np.testing.assert_allclose(arr, vol_zyx)
    np.testing.assert_allclose(a2, aff)
    np.testing.assert_allclose(R.read_array(p), vol_zyx)


@pytest.mark.parametrize("gz", [True, False])
def test_nifti_without_suffix_parsed_from_bytes(tmp_path, gz):
    nib = pytest.importorskip("nibabel")
    vol = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    src = tmp_path / ("v.nii.gz" if gz else "v.nii")
    nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), np.diag([2.0, 3.0, 4.0, 1.0])), str(src))
    blob = tmp_path / "0123abcd"                      # content-addressed name, no suffix
    blob.write_bytes(src.read_bytes())
    arr, aff = R.read_nifti(blob)
    np.testing.assert_array_equal(arr, vol)
    assert aff[2, 2] == 4.0
    np.testing.assert_array_equal(R.read_array(blob, fmt="nifti"), vol)


def test_nifti_4d_keeps_trailing_axis(tmp_path):
    nib = pytest.importorskip("nibabel")
    xyzt = np.zeros((4, 3, 2, 5), dtype=np.int16); xyzt[1, 2, 0, 3] = 7
    p = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(xyzt, np.eye(4)), str(p))
    arr, _ = R.read_nifti(p)
    assert arr.shape == (2, 3, 4, 5) and arr[0, 2, 1, 3] == 7


def _write_dicom_slice(path, pixels, z, inst, slope=None, intercept=None):
    pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "OT"
    ds.InstanceNumber = inst
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    if slope is not None:
        ds.RescaleSlope = slope; ds.RescaleIntercept = intercept
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_dicom_series_sorted_by_position(tmp_path):
    pytest.importorskip("pydicom")
    d = tmp_path / "series"; d.mkdir()
    # write out of order, instance numbers scrambled; position decides the order
    for fname, z, inst in (("c.dcm", 2.0, 1), ("a.dcm", 0.0, 3), ("b.dcm", 1.0, 2)):
        _write_dicom_slice(d / fname, np.full((4, 5), int(z * 10) + 1), z, inst)
    (d / "README").write_text("not dicom")
    arr = R.read_array(d)
    assert arr.shape == (3, 4, 5)
    assert [int(arr[i, 0, 0]) for i in range(3)] == [1, 11, 21]


def test_dicom_rescale_and_single_file(tmp_path):
    pytest.importorskip("pydicom")
    d = tmp_path / "s"; d.mkdir()
    _write_dicom_slice(d / "0.dcm", np.full((2, 2), 10), 0.0, 1, slope=2.0, intercept=-5.0)
    arr = R.read_array(d)
    assert arr.dtype == np.float32 and arr.shape == (1, 2, 2) and float(arr[0, 0, 0]) == 15.0
    single = R.read_array(d / "0.dcm")
    assert single.shape == (2, 2) and float(single[0, 0]) == 15.0


def test_dicom_empty_dir(tmp_path):
    pytest.importorskip("pydicom")
    (tmp_path / "e").mkdir()
    with pytest.raises(ValueError):
        R.read_array(tmp_path / "e")


def test_png_and_jpeg(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    g = (np.arange(48, dtype=np.uint8).reshape(6, 8))
    Image.fromarray(g).save(tmp_path / "g.png")
    assert np.array_equal(R.read_array(tmp_path / "g.png"), g)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8); rgb[..., 1] = 200
    Image.fromarray(rgb).save(tmp_path / "c.jpg", quality=100)
    out = R.read_array(tmp_path / "c.jpg")
    assert out.shape == (6, 8, 3) and abs(int(out[0, 0, 1]) - 200) <= 2


def test_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    a = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    tifffile.imwrite(tmp_path / "s.tif", a, photometric="minisblack")
    assert np.array_equal(R.read_array(tmp_path / "s.tif"), a)
    tifffile.imwrite(tmp_path / "s2.tiff", a[0])
    assert np.array_equal(R.read_array(tmp_path / "s2.tiff"), a[0])


def test_missing_optional_reader_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_import", lambda name: (_ for _ in ()).throw(ImportError(name)))
    (tmp_path / "x.png").write_bytes(b"")
    with pytest.raises(RuntimeError, match="vision"):
        R.read_array(tmp_path / "x.png")
