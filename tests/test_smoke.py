import io
import json

import pytest
from PIL import Image

from chroma_provision.smoke import build_prompt, run_smoke, validate_png


def png_bytes(color, size=(8, 8)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "PNG")
    return output.getvalue()


def test_validate_png_rejects_all_black():
    with pytest.raises(ValueError, match="all-black"):
        validate_png(png_bytes((0, 0, 0)))


def test_validate_png_reports_extrema_and_means():
    result = validate_png(png_bytes((5, 10, 15)))
    assert result["size"] == [8, 8]
    assert result["extrema"] == [[5, 5], [10, 10], [15, 15]]
    assert result["means"] == [5.0, 10.0, 15.0]


def test_build_prompt_uses_full_models_and_requested_dimensions():
    graph = build_prompt("a red kite", 512, 512, 123)
    assert graph["1"]["inputs"]["unet_name"] == "Chroma1-HD.safetensors"
    assert graph["2"]["inputs"]["clip_name"] == "t5xxl_fp16.safetensors"
    assert graph["6"]["inputs"]["width"] == 512
    assert graph["6"]["inputs"]["height"] == 512
    assert graph["4"]["inputs"]["text"] == "a red kite"


class FakeClient:
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.request_timeouts = []

    def submit(self, graph, timeout=None):
        self.request_timeouts.append(timeout)
        self.graph = graph
        return "prompt-1"

    def history(self, prompt_id, timeout=None):
        self.request_timeouts.append(timeout)
        return {
            prompt_id: {
                "status": {"completed": True},
                "outputs": {
                    "15": {
                        "images": [
                            {
                                "filename": "result.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        }

    def image(self, descriptor, timeout=None):
        self.request_timeouts.append(timeout)
        return png_bytes((5, 10, 15), self.image_size)


def test_run_smoke_emits_machine_readable_timing():
    client = FakeClient()
    result = run_smoke(client, "test", 512, 512, 123, timeout=1, poll_interval=0)
    encoded = json.dumps(result)
    assert result["ok"] is True
    assert result["prompt_id"] == "prompt-1"
    assert result["image"]["means"] == [5.0, 10.0, 15.0]
    assert set(result["timing_seconds"]) >= {"submit", "generation", "smoke_total"}
    assert all(
        value is not None and 0 < value <= 1 for value in client.request_timeouts
    )
    assert "prompt-1" in encoded


def test_run_smoke_rejects_output_with_unrequested_dimensions():
    with pytest.raises(ValueError, match="expected 768x768"):
        run_smoke(
            FakeClient(image_size=(512, 512)),
            "test",
            768,
            768,
            123,
            timeout=1,
            poll_interval=0,
        )
