from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from CryoNetRefine import assets


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, content_length: int | None = None, *, status: int = 200, content_range: str | None = None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data) if content_length is None else content_length)}
        if content_range is not None:
            self.headers["Content-Range"] = content_range


class AssetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_uses_existing_verified_asset(self):
        destination = self.directory / "0775.mrc"
        destination.write_bytes(b"verified")
        with patch.dict(assets.ASSET_SHA256, {"0775.mrc": hashlib.sha256(b"verified").hexdigest()}):
            with patch.object(assets.urllib.request, "urlopen") as urlopen:
                self.assertEqual(assets.download_asset("0775.mrc", destination), destination)
                urlopen.assert_not_called()

    def test_falls_back_after_first_source_fails(self):
        destination = self.directory / "0775.mrc"
        with patch.dict(os.environ, {"CRYONET_ASSET_BASE_URLS": "https://first.invalid,https://second.invalid"}):
            with patch.dict(assets.ASSET_SHA256, {"0775.mrc": hashlib.sha256(b"good").hexdigest()}):
                with patch.object(
                    assets.urllib.request,
                    "urlopen",
                    side_effect=[urllib.error.URLError("offline"), FakeResponse(b"good")],
                ) as urlopen:
                    self.assertEqual(assets.download_asset("0775.mrc", destination, retries_per_url=1), destination)
        self.assertEqual(destination.read_bytes(), b"good")
        self.assertEqual([call.args[0] for call in urlopen.call_args_list], [
            "https://first.invalid/0775.mrc",
            "https://second.invalid/0775.mrc",
        ])
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_rejects_incomplete_download_before_fallback(self):
        destination = self.directory / "0775_af3.cif"
        with patch.dict(os.environ, {"CRYONET_ASSET_BASE_URLS": "https://first.invalid,https://second.invalid"}):
            with patch.dict(assets.ASSET_SHA256, {"0775_af3.cif": hashlib.sha256(b"good").hexdigest()}):
                with patch.object(
                    assets.urllib.request,
                    "urlopen",
                    side_effect=[FakeResponse(b"cut", content_length=9), FakeResponse(b"good")],
                ):
                    assets.download_asset("0775_af3.cif", destination, retries_per_url=1)
        self.assertEqual(destination.read_bytes(), b"good")
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_rejects_wrong_hash_and_preserves_existing_file_on_failure(self):
        destination = self.directory / "0775.mrc"
        destination.write_bytes(b"old corrupt content")
        with patch.dict(os.environ, {"CRYONET_ASSET_BASE_URLS": "https://only.invalid"}):
            with patch.dict(assets.ASSET_SHA256, {"0775.mrc": hashlib.sha256(b"good").hexdigest()}):
                with patch.object(assets.urllib.request, "urlopen", return_value=FakeResponse(b"bad")):
                    with self.assertRaisesRegex(RuntimeError, "Could not download"):
                        assets.download_asset("0775.mrc", destination, retries_per_url=1)
        self.assertEqual(destination.read_bytes(), b"old corrupt content")
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_resumes_partial_download_on_same_source(self):
        destination = self.directory / "0775.mrc"
        with patch.dict(os.environ, {"CRYONET_ASSET_BASE_URLS": "https://only.invalid"}):
            with patch.dict(assets.ASSET_SHA256, {"0775.mrc": hashlib.sha256(b"good").hexdigest()}):
                with patch.object(
                    assets.urllib.request,
                    "urlopen",
                    side_effect=[
                        FakeResponse(b"go", content_length=4),
                        FakeResponse(b"od", status=206, content_range="bytes 2-3/4"),
                    ],
                ) as urlopen:
                    assets.download_asset("0775.mrc", destination, retries_per_url=2)
        self.assertEqual(destination.read_bytes(), b"good")
        request = urlopen.call_args_list[1].args[0]
        self.assertEqual(request.headers["Range"], "bytes=2-")

    def test_mols_urls_can_be_overridden(self):
        with patch.dict(os.environ, {"CRYONET_MOLS_URLS": "https://first.invalid/mols.tar,https://second.invalid/mols.tar"}):
            self.assertEqual(assets.asset_urls("mols.tar"), (
                "https://first.invalid/mols.tar",
                "https://second.invalid/mols.tar",
            ))

    def test_checkpoint_uses_the_same_fallback_mechanism(self):
        destination = self.directory / "CryoNet.Refine_model.pt"
        with patch.dict(os.environ, {"CRYONET_ASSET_BASE_URLS": "https://first.invalid,https://second.invalid"}):
            with patch.dict(assets.ASSET_SHA256, {"CryoNet.Refine_model.pt": hashlib.sha256(b"model").hexdigest()}):
                with patch.object(
                    assets.urllib.request,
                    "urlopen",
                    side_effect=[urllib.error.URLError("offline"), FakeResponse(b"model")],
                ) as urlopen:
                    assets.download_asset("CryoNet.Refine_model.pt", destination, retries_per_url=1)
        self.assertEqual(destination.read_bytes(), b"model")
        self.assertEqual([call.args[0] for call in urlopen.call_args_list], [
            "https://first.invalid/CryoNet.Refine_model.pt",
            "https://second.invalid/CryoNet.Refine_model.pt",
        ])


if __name__ == "__main__":
    unittest.main()
