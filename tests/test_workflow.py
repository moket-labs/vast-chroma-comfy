from chroma_provision.workflow import patch_workflow


def node(node_type, values, models=None):
    properties = {}
    if models is not None:
        properties["models"] = models
    return {"type": node_type, "widgets_values": values, "properties": properties}


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
