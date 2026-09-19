"""Submit and validate a bounded Chroma ComfyUI smoke generation."""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageStat


def build_prompt(text: str, width: int, height: int, seed: int) -> dict:
    if width not in (512, 768) or height != width:
        raise ValueError("dimensions must be 512x512 or 768x768")
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Chroma1-HD.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "t5xxl_fp16.safetensors",
                "type": "chroma",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": text, "clip": ["13", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, broken, malformed, watermark",
                "clip": ["13", 0],
            },
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["12", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 3.8,
            },
        },
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {
            "class_type": "BetaSamplingScheduler",
            "inputs": {"model": ["12", 0], "steps": 26, "alpha": 0.45, "beta": 0.45},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["8", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["6", 0],
            },
        },
        "12": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 1.0},
        },
        "13": {
            "class_type": "T5TokenizerOptions",
            "inputs": {"clip": ["2", 0], "min_length": 0, "min_padding": 0},
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {"images": ["14", 0], "filename_prefix": "chroma-smoke"},
        },
    }


def validate_png(content: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError("output is not PNG")
            rgb = image.convert("RGB")
            extrema = [list(pair) for pair in rgb.getextrema()]
            means = [round(value, 6) for value in ImageStat.Stat(rgb).mean]
            size = list(rgb.size)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid PNG: {error}") from error
    if max(high for _, high in extrema) == 0 or max(means) == 0:
        raise ValueError("output PNG is all-black")
    return {"size": size, "extrema": extrema, "means": means}


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def _json(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def submit(self, graph: dict) -> str:
        response = self._json("/prompt", {"prompt": graph, "client_id": self.client_id})
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI rejected prompt: {response}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict:
        return self._json("/history/" + urllib.parse.quote(prompt_id, safe=""))

    def image(self, descriptor: dict) -> bytes:
        query = urllib.parse.urlencode(
            {key: descriptor[key] for key in ("filename", "subfolder", "type")}
        )
        with urllib.request.urlopen(
            self.base_url + "/view?" + query, timeout=self.timeout
        ) as response:
            return response.read()


def run_smoke(
    client,
    text: str,
    width: int,
    height: int,
    seed: int,
    timeout: float,
    poll_interval: float,
    accepted_at: float | None = None,
) -> dict:
    started_wall = time.time()
    started = time.monotonic()
    prompt_id = client.submit(build_prompt(text, width, height, seed))
    submitted = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        history = client.history(prompt_id)
        record = history.get(prompt_id)
        if record:
            status = record.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI generation failed: {status}")
            images = [
                image
                for output in record.get("outputs", {}).values()
                for image in output.get("images", [])
            ]
            if images:
                image = validate_png(client.image(images[0]))
                finished = time.monotonic()
                timings = {
                    "submit": round(submitted - started, 3),
                    "generation": round(finished - submitted, 3),
                    "smoke_total": round(finished - started, 3),
                }
                if accepted_at is not None:
                    timings["rent_accepted_to_valid_image"] = round(
                        time.time() - accepted_at, 3
                    )
                return {
                    "ok": True,
                    "prompt_id": prompt_id,
                    "image": image,
                    "timing_seconds": timings,
                    "started_at_epoch": started_wall,
                }
        time.sleep(poll_interval)
    raise TimeoutError(f"no valid image within {timeout:.1f}s for prompt {prompt_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:18188")
    parser.add_argument(
        "--prompt",
        default="A red kite flying over green hills, natural light, sharp focus",
    )
    parser.add_argument("--size", choices=(512, 768), type=int, default=512)
    parser.add_argument(
        "--seed", type=int, default=random.SystemRandom().randrange(2**63)
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--accepted-at", type=float)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        result = run_smoke(
            ComfyClient(args.url),
            args.prompt,
            args.size,
            args.size,
            args.seed,
            args.timeout,
            args.poll_interval,
            args.accepted_at,
        )
    except Exception as error:  # noqa: BLE001 - CLI must fail closed with JSON
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    encoded = json.dumps(result, sort_keys=True)
    if args.output_json:
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
