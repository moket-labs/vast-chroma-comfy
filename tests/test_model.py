import hashlib
import json
import struct

import pytest

from chroma_provision.model import ModelValidationError, verify_model


def make_safetensors(path):
    header = json.dumps(
        {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")


def test_verify_model_accepts_exact_valid_safetensors(tmp_path):
    path = tmp_path / "model.safetensors"
    make_safetensors(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = verify_model(path, path.stat().st_size, digest)

    assert result == {"bytes": path.stat().st_size, "sha256": digest, "tensors": 1}


@pytest.mark.parametrize("field", ["size", "sha256", "format"])
def test_verify_model_fails_closed(field, tmp_path):
    path = tmp_path / "model.safetensors"
    make_safetensors(path)
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if field == "size":
        size += 1
    elif field == "sha256":
        digest = "0" * 64
    else:
        path.write_bytes(b"not a safetensors file")
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ModelValidationError, match=field):
        verify_model(path, size, digest)
