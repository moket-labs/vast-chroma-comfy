from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_bootstrap_checks_out_exact_commit_and_only_provisions():
    text = (ROOT / "bootstrap.sh").read_text()
    assert 'readonly REPO_COMMIT="${REPO_COMMIT:?' in text
    assert '[[ "$REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]' in text
    assert '"$CHECKOUT/provision.sh"' in text
    assert "entrypoint.sh" not in text
