"""Download the public assets used by refinement and its example run.

The source lists can be overridden without editing application code. Downloads
are checked before being moved into place, so a failed source cannot leave a
partial file that a later run mistakes for a valid asset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable


DEFAULT_ASSET_BASE_URLS = (
    "https://zenodo.org/records/23184057/files",
    "https://huggingface.co/FuyaoHuang/cryonet-refine-assets/resolve/main",
    "https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine",
)
DEFAULT_MOLS_URLS = (
    "https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/mols.tar",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar",
)
ASSET_SHA256 = {
    "CryoNet.Refine_model.pt": "6ec94feeb76fc99549665e9cb25e7e66d3791d094b57164d3ba49d60c42b4301",
    "0775_af3.cif": "7421c6d4e33274fb30d84439c357305efd645495010ef7b66ed5d0dcff8d26ca",
    "0775.mrc": "95703206787e7fb93f56422e49d665125633a2c3b33f29943a0453956c87630d",
}


def asset_urls(name: str) -> tuple[str, ...]:
    """Return ordered URLs for a known asset (first successful source wins)."""
    if name == "mols.tar":
        configured = os.environ.get("CRYONET_MOLS_URLS")
        defaults = DEFAULT_MOLS_URLS
    elif name in ASSET_SHA256:
        configured = os.environ.get("CRYONET_ASSET_BASE_URLS")
        defaults = DEFAULT_ASSET_BASE_URLS
    else:
        raise ValueError(f"Unknown asset: {name}")

    sources = tuple(part.strip().rstrip("/") for part in configured.split(",")) if configured is not None else defaults
    if not sources or any(not source for source in sources):
        raise ValueError(f"No valid download sources configured for {name}")
    return sources if name == "mols.tar" else tuple(f"{base}/{name}" for base in sources)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_file(path: Path, expected_hash: str | None, validate: Callable[[Path], None] | None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_hash is not None and _sha256(path) != expected_hash:
        return False
    if validate is not None:
        try:
            validate(path)
        except Exception:
            return False
    return True


def download_asset(
    name: str,
    destination: Path,
    *,
    validate: Callable[[Path], None] | None = None,
    retries_per_url: int = 3,
) -> Path:
    """Download an asset with ordered fallback, integrity checks and atomic replacement."""
    destination = Path(destination)
    expected_hash = ASSET_SHA256.get(name)
    if retries_per_url < 1:
        raise ValueError("retries_per_url must be positive")
    if _valid_file(destination, expected_hash, validate):
        print(f"Using verified asset: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in asset_urls(name):
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
        try:
            for attempt in range(1, retries_per_url + 1):
                try:
                    offset = temporary.stat().st_size
                    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"}) if offset else url
                    print(f"Downloading {name} from {url} (attempt {attempt}/{retries_per_url}, offset {offset})")
                    with urllib.request.urlopen(request, timeout=60) as response:
                        status = getattr(response, "status", 200)
                        if offset and status == 206:
                            content_range = response.headers.get("Content-Range", "")
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                            if match is None or int(match.group(1)) != offset:
                                raise RuntimeError(f"invalid Content-Range: {content_range}")
                            expected_size = int(match.group(3))
                            mode = "ab"
                        elif status == 200:
                            expected_size = int(response.headers.get("Content-Length", 0))
                            mode = "wb"
                        else:
                            raise RuntimeError(f"unexpected HTTP status: {status}")
                        with temporary.open(mode) as output:
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                output.write(chunk)
                    received = temporary.stat().st_size
                    if expected_size and received != expected_size:
                        raise RuntimeError(f"incomplete download: expected {expected_size} bytes, received {received}")
                    if not _valid_file(temporary, expected_hash, validate):
                        temporary.write_bytes(b"")
                        raise RuntimeError("downloaded file failed integrity validation")
                    os.replace(temporary, destination)
                    print(f"Downloaded and verified: {destination}")
                    return destination
                except Exception as error:
                    failures.append(f"{url} (attempt {attempt}): {error}")
                    print(f"Download failed: {failures[-1]}")
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {name} from any source:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a CryoNet.Refine asset")
    parser.add_argument("name", choices=(*ASSET_SHA256, "mols.tar"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    download_asset(args.name, args.destination)


if __name__ == "__main__":
    main()
