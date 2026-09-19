import os
import subprocess
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


def test_provision_script_uses_the_official_venv_hf_binary():
    text = (ROOT / "provision.sh").read_text()
    assert "/venv/main/bin/hf" in text
    assert 'readonly HF_BIN="${HF_BIN:-$default_hf}"' in text


def test_exact_workflow_destination_is_configured():
    text = (ROOT / "provision.sh").read_text()
    assert (
        "${WORKFLOW_DEST:-${COMFYUI_ROOT}/user/default/workflows/"
        "Chroma1-HD-RTX3090.json}" in text
    )


def test_provision_fails_before_download_without_force_upcast_attention():
    env = os.environ.copy()
    env["COMFYUI_ARGS"] = "--listen 127.0.0.1 --port 18188"
    result = subprocess.run(
        ["bash", str(ROOT / "provision.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "COMFYUI_ARGS must include --force-upcast-attention" in result.stderr


def test_readme_uses_localhost_comfyui_args_with_required_flags():
    text = (ROOT / "README.md").read_text()
    comfyui_args = next(
        line for line in text.splitlines() if line.startswith("COMFYUI_ARGS=")
    )
    assert "--listen 127.0.0.1" in comfyui_args
    assert "--listen 0.0.0.0" not in comfyui_args
    for flag in (
        "--disable-auto-launch",
        "--disable-xformers",
        "--port 18188",
        "--enable-cors-header",
        "--force-upcast-attention",
    ):
        assert flag in comfyui_args
    assert "Vast Instance Portal" in text
    assert "authentication" in text


def test_bootstrap_checkout_is_restart_safe_and_verifies_exact_commit():
    text = (ROOT / "bootstrap.sh").read_text()
    assert 'if [[ ! -d "$CHECKOUT/.git" ]]' in text
    assert '[[ "$REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]' in text
    assert '[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$REPO_COMMIT" ]]' in text
    assert '"$CHECKOUT/provision.sh"' in text
    assert "entrypoint.sh" not in text


def test_repository_has_mit_license_and_complete_ci_checks():
    license_text = (ROOT / "LICENSE").read_text()
    assert "MIT License" in license_text
    assert "Permission is hereby granted, free of charge" in license_text

    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    for command in (
        "pytest",
        "ruff check",
        "ruff format --check",
        "bash -n provision.sh",
        "shellcheck provision.sh",
    ):
        assert command in ci
