import io
import json

import pytest
from PIL import Image

from chroma_provision.smoke import build_prompt, run_smoke, validate_png


def png_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, "PNG")
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
    def submit(self, graph):
        self.graph = graph
        return "prompt-1"

    def history(self, prompt_id):
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

    def image(self, descriptor):
        return png_bytes((5, 10, 15))


def test_run_smoke_emits_machine_readable_timing():
    result = run_smoke(FakeClient(), "test", 512, 512, 123, timeout=1, poll_interval=0)
    encoded = json.dumps(result)
    assert result["ok"] is True
    assert result["prompt_id"] == "prompt-1"
    assert result["image"]["means"] == [5.0, 10.0, 15.0]
    assert set(result["timing_seconds"]) >= {"submit", "generation", "smoke_total"}
    assert "prompt-1" in encoded
