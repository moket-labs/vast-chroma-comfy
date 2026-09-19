#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_URL="${REPO_URL:-https://github.com/moket-labs/vast-chroma-comfy.git}"
readonly REPO_COMMIT="${REPO_COMMIT:?REPO_COMMIT must be set to an exact commit}"
readonly CHECKOUT="${CHECKOUT:-/opt/vast-chroma-comfy}"

[[ "$REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'invalid pinned commit\n' >&2
  exit 1
}

if [[ ! -d "$CHECKOUT/.git" ]]; then
  rm -rf -- "$CHECKOUT"
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$CHECKOUT"
else
  git -C "$CHECKOUT" remote set-url origin "$REPO_URL"
fi

git -C "$CHECKOUT" fetch --depth=1 origin "$REPO_COMMIT"
git -C "$CHECKOUT" checkout --detach "$REPO_COMMIT"
[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$REPO_COMMIT" ]]

"$CHECKOUT/provision.sh"
