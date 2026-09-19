"""Patch the canonical Chroma workflow for the verified RTX 3090 setup."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import urllib.request
from pathlib import Path

WORKFLOW_REVISION = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
WORKFLOW_URL = f"https://huggingface.co/lodestones/Chroma1-HD/resolve/{WORKFLOW_REVISION}/ComfyUI_Chroma1-HD_T2I-workflow.json"
WORKFLOW_SHA256 = "c5cc4059bf24e14c9936ae6b3979481ce3fb3dae832f95bd21e2d563ad64105b"
MODEL_BASE = "https://huggingface.co"

MODEL_SETTINGS = {
    "UNETLoader": (
        ["Chroma1-HD.safetensors", "default"],
        {
            "name": "Chroma1-HD.safetensors",
            "url": f"{MODEL_BASE}/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors",
            "directory": "diffusion_models",
        },
    ),
    "CLIPLoader": (
        ["t5xxl_fp16.safetensors", "chroma", "default"],
        {
            "name": "t5xxl_fp16.safetensors",
            "url": f"{MODEL_BASE}/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
            "directory": "text_encoders",
        },
    ),
    "VAELoader": (
        ["ae.safetensors"],
        {
            "name": "ae.safetensors",
            "url": f"{MODEL_BASE}/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
            "directory": "vae",
        },
    ),
}


def patch_workflow(source: dict, width: int = 768, height: int = 768) -> dict:
    if width not in (512, 768) or height != width:
        raise ValueError("workflow dimensions must be 512x512 or 768x768")
    result = copy.deepcopy(source)
    nodes = result.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("workflow has no nodes list")
    found: set[str] = set()
    required = set(MODEL_SETTINGS) | {"EmptySD3LatentImage"}
    for node in nodes:
        node_type = node.get("type")
        if node_type in required and node_type in found:
            raise ValueError(f"workflow has duplicate required node: {node_type}")
        if node_type in MODEL_SETTINGS:
            widgets, model = MODEL_SETTINGS[node_type]
            node["widgets_values"] = widgets
            node.setdefault("properties", {})["models"] = [model]
            found.add(node_type)
        elif node_type == "EmptySD3LatentImage":
            node["widgets_values"] = [width, height, 1]
            found.add(node_type)
    missing = sorted(required - found)
    if missing:
        raise ValueError("workflow missing required node(s): " + ", ".join(missing))
    return result


def download_and_patch(output: Path, width: int, timeout: float = 60) -> None:
    request = urllib.request.Request(
        WORKFLOW_URL, headers={"User-Agent": "vast-chroma-comfy/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != WORKFLOW_SHA256:
        raise ValueError(
            f"workflow sha256 mismatch: expected {WORKFLOW_SHA256}, got {actual_sha256}"
        )
    source = json.loads(content)
    patched = patch_workflow(source, width, width)
    output.write_text(json.dumps(patched, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", choices=(512, 768), default=768, type=int)
    parser.add_argument("--timeout", default=60, type=float)
    args = parser.parse_args()
    download_and_patch(args.output, args.size, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
