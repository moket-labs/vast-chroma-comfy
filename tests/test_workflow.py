import pytest

from chroma_provision.workflow import (
    MODEL_SETTINGS,
    WORKFLOW_SHA256,
    WORKFLOW_URL,
    patch_workflow,
)


def node(node_type, values, models=None):
    properties = {}
    if models is not None:
        properties["models"] = models
    return {"type": node_type, "widgets_values": values, "properties": properties}


def test_canonical_workflow_source_is_revision_and_hash_pinned():
    assert "0e0c60ece1e82b17cb7f77342d765ba5024c40c0" in WORKFLOW_URL
    assert (
        WORKFLOW_SHA256
        == "c5cc4059bf24e14c9936ae6b3979481ce3fb3dae832f95bd21e2d563ad64105b"
    )
    assert set(MODEL_SETTINGS) == {"UNETLoader", "CLIPLoader", "VAELoader"}


def test_patch_workflow_selects_full_precision_models_and_768_square():
    source = {
        "nodes": [
            node("UNETLoader", ["old-unet.safetensors", "default"], [{}]),
            node("CLIPLoader", ["old-t5.safetensors", "chroma", "default"], [{}]),
            node("VAELoader", ["old-vae.safetensors"], [{}]),
            node("EmptySD3LatentImage", [1152, 1152, 1]),
        ]
    }

    patched = patch_workflow(source, 768, 768)
    by_type = {item["type"]: item for item in patched["nodes"]}

    assert by_type["UNETLoader"]["widgets_values"] == [
        "Chroma1-HD.safetensors",
        "default",
    ]
    assert by_type["CLIPLoader"]["widgets_values"] == [
        "t5xxl_fp16.safetensors",
        "chroma",
        "default",
    ]
    assert by_type["VAELoader"]["widgets_values"] == ["ae.safetensors"]
    assert by_type["EmptySD3LatentImage"]["widgets_values"] == [768, 768, 1]
    assert by_type["UNETLoader"]["properties"]["models"][0]["url"].endswith(
        "/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors"
    )
    assert by_type["CLIPLoader"]["properties"]["models"][0]["url"].endswith(
        "/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
    )


def test_patch_workflow_rejects_missing_required_node():
    try:
        patch_workflow({"nodes": []}, 768, 768)
    except ValueError as error:
        assert "UNETLoader" in str(error)
    else:
        raise AssertionError("missing nodes must fail closed")


@pytest.mark.parametrize(
    "duplicate_type",
    ["UNETLoader", "CLIPLoader", "VAELoader", "EmptySD3LatentImage"],
)
def test_patch_workflow_rejects_duplicate_required_nodes(duplicate_type):
    required = [
        node("UNETLoader", ["old-unet.safetensors", "default"]),
        node("CLIPLoader", ["old-t5.safetensors", "chroma", "default"]),
        node("VAELoader", ["old-vae.safetensors"]),
        node("EmptySD3LatentImage", [1152, 1152, 1]),
    ]
    duplicate = next(item.copy() for item in required if item["type"] == duplicate_type)

    with pytest.raises(ValueError, match=rf"duplicate.*{duplicate_type}"):
        patch_workflow({"nodes": required + [duplicate]}, 768, 768)
