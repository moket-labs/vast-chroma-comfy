import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_img2img_workflow_uses_reference_image_and_partial_sigma_schedule():
    workflow = json.loads((ROOT / "workflows/Chroma1-HD-Img2Img.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert "LoadImage" in by_type
    assert "VAEEncode" in by_type
    assert "SplitSigmas" in by_type
    assert "EmptySD3LatentImage" not in by_type
    sampler = by_type["SamplerCustomAdvanced"]
    sigmas = next(item for item in sampler["inputs"] if item["name"] == "sigmas")
    latent = next(item for item in sampler["inputs"] if item["name"] == "latent_image")
    links = {link[0]: link for link in workflow["links"]}
    assert links[sigmas["link"]][1] == by_type["SplitSigmas"]["id"]
    assert links[sigmas["link"]][2] == 1
    assert links[latent["link"]][1] == by_type["VAEEncode"]["id"]
    assert nodes[by_type["LoadImage"]["id"]]["title"].startswith("REFERENCE IMAGE")


def test_provision_installs_img2img_workflow():
    script = (ROOT / "provision.sh").read_text()
    assert "Chroma1-HD-Img2Img.json" in script
