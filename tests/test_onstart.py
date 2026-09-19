from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_onstart_runs_pinned_provisioner_then_official_entrypoint():
    text = (ROOT / "onstart.sh").read_text()
    provision = '"$CHECKOUT/provision.sh"'
    entrypoint = "exec entrypoint.sh"
    assert provision in text
    assert entrypoint in text
    assert text.index(provision) < text.index(entrypoint)
    assert "readonly REPO_COMMIT=" in text
