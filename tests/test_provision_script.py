from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_provision_script_pins_models_and_verified_metadata():
    text = (ROOT / "provision.sh").read_text()
    assert "17800038288" in text
    assert "d446d9695d08276f61e53653e025289dd96f7a489c27982a1fa54ecabc06642a" in text
    assert "9787841024" in text
    assert "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635" in text
    assert "335304388" in text
    assert "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38" in text


def test_provision_script_is_strict_parallel_atomic_and_idempotent():
    text = (ROOT / "provision.sh").read_text()
    assert "set -Eeuo pipefail" in text
    assert "wait" in text
    assert "&" in text
    assert "mktemp" in text
    assert "mv" in text
    assert "verify_model" in text
    assert "already valid" in text


def test_exact_workflow_destination_is_configured():
    text = (ROOT / "provision.sh").read_text()
    assert "/workspace/ComfyUI/user/default/workflows/Chroma1-HD-RTX3090.json" in text
