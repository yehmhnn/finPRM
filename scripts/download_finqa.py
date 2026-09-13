#!/usr/bin/env python3
"""Download the official, pinned FinQA JSON splits and record checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

FINQA_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLITS = ("train.json", "dev.json", "test.json", "private_test.json")
BASE_URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REVISION}/dataset"
ARCHIVE_URL = f"https://codeload.github.com/czyssrs/FinQA/zip/{FINQA_REVISION}"
EXPECTED_SHA256 = {
    "train.json": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dev.json": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test.json": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
    "private_test.json": "94dfd3f82aeeb91835da7a161c5d64b9223ed3c630a27457cf5695da2ee24756",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/finqa"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--transport", choices=("raw", "archive"), default="raw")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_revision": FINQA_REVISION,
        "transport": args.transport,
        "files": {},
    }
    archive = None
    archive_path = args.output / ".finqa-source.zip.part"
    if args.transport == "archive" and (
        args.force
        or any(
            not (args.output / filename).exists()
            or sha256(args.output / filename) != EXPECTED_SHA256[filename]
            for filename in SPLITS
        )
    ):
        print("downloading pinned FinQA source archive")
        try:
            urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
            archive = zipfile.ZipFile(archive_path)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
    for filename in SPLITS:
        destination = args.output / filename
        expected = EXPECTED_SHA256[filename]
        needs_download = args.force or not destination.exists()
        if destination.exists() and not args.force:
            actual = sha256(destination)
            if actual != expected:
                if args.transport == "archive":
                    needs_download = True
                else:
                    raise RuntimeError(
                        f"checksum mismatch for existing {destination}: "
                        f"expected {expected}, got {actual}; rerun with --force"
                    )
            else:
                print(f"verified existing {destination}")
        if needs_download:
            print(f"downloading {filename}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                if archive is None:
                    urllib.request.urlretrieve(f"{BASE_URL}/{filename}", temporary)
                else:
                    member = f"FinQA-{FINQA_REVISION}/dataset/{filename}"
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                actual = sha256(temporary)
                if actual != expected:
                    raise RuntimeError(
                        f"checksum mismatch for downloaded {filename}: "
                        f"expected {expected}, got {actual}"
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        manifest["files"][filename] = {
            "bytes": destination.stat().st_size,
            "sha256": expected,
            "url": f"{BASE_URL}/{filename}",
        }
        if args.transport == "archive":
            manifest["archive_url"] = ARCHIVE_URL
    if archive is not None:
        archive.close()
        archive_path.unlink(missing_ok=True)

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
