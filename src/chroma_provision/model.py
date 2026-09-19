"""Validation for pinned model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class ModelValidationError(ValueError):
    """A model artifact did not match its pinned identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: str | Path, expected_size: int, expected_sha256: str) -> dict:
    path = Path(path)
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise ModelValidationError(f"size validation failed: {error}") from error
    if actual_size != expected_size:
        raise ModelValidationError(
            f"size mismatch: expected {expected_size}, got {actual_size}"
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise ModelValidationError(
            f"sha256 mismatch: expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    try:
        from safetensors import safe_open

        with safe_open(path, framework="numpy", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise ValueError("tensor table is empty")
    except Exception as error:
        raise ModelValidationError(f"format validation failed: {error}") from error

    return {"bytes": actual_size, "sha256": actual_sha256, "tensors": len(keys)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("sha256")
    args = parser.parse_args()
    print(json.dumps(verify_model(args.path, args.size, args.sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
