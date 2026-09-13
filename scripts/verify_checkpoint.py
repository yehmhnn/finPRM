#!/usr/bin/env python3
"""Verify the frozen Qwen PRM checkpoint copied from the official mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "config.json": (835, "70d5ef2f21af53e6c95252d8838f4898fbac6c83708a67c9c7c3f83f75318bbe"),
    "configuration_qwen2_rm.py": (6685, "c50ec249c0ab452765dd6acc9825448dd11a268ff62ddc9b2dbdeab25a1008e2"),
    "model-00001-of-00004.safetensors": (3971153412, "6351eb3fdb31ff4d98b7c34080665e925dfa515d5f44fd5a972dbc2612839fcc"),
    "model-00002-of-00004.safetensors": (3864726352, "9ce988e446c3666bfde723cf53fde432ebe2ec76a5f368fd1578cd46d8474242"),
    "model-00003-of-00004.safetensors": (3864726424, "4d9eb85ee50decd0682b32e9d351c1e8b53c583903bdf225fc44c5fbe24b06d9"),
    "model-00004-of-00004.safetensors": (3556377672, "97ba1ba0ea27ba56d4c3340601cba7d2a011bf0c957a0ac4a85eda46e0e3a183"),
    "model.safetensors.index.json": (27980, "5b61f9273d8fe706ec119b3f960bfb07b07e28097646baeee4da64b64ba6fdc7"),
    "modeling_qwen2_rm.py": (74658, "2cb5fa464fa15b3e5ebcb9dc5e9136f09354113f5095adb9dd8384ac64ef4eb0"),
    "tokenizer.json": (7029106, "b145a1b9700174ec9a0617ba1bd0edb819ba861ab0d939b0c2b1d77d93cd76ac"),
    "tokenizer_config.json": (2439, "f2c303ece66df80e620fe0127864885249d4100de824e3b50dc72ff98ce11442"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    verified = {}
    for filename, (expected_size, expected_hash) in EXPECTED.items():
        path = args.checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {filename}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        verified[filename] = {"bytes": actual_size, "sha256": actual_hash}
    print(json.dumps({"verified": verified}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
