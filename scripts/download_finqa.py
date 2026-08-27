#!/usr/bin/env python3
"""Download the official, pinned FinQA JSON splits and record checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

FINQA_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLITS = ("train.json", "dev.json", "test.json", "private_test.json")
BASE_URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REVISION}/dataset"


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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {"source_revision": FINQA_REVISION, "files": {}}
    for filename in SPLITS:
        destination = args.output / filename
        if destination.exists() and not args.force:
            print(f"using existing {destination}")
        else:
            print(f"downloading {filename}")
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)
        manifest["files"][filename] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

