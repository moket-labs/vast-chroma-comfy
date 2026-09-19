# Reproducible Vast.ai Chroma1-HD + ComfyUI

Provision the verified full-precision Chroma1-HD stack on an RTX 3090 before ComfyUI starts, then submit a bounded API smoke generation and reject black output.

## Pinned runtime

Use the official Vast interactive ComfyUI template with these exact values:

```text
Docker image: vastai/comfy:v0.35.0-cuda-13.2-py312
Disk: at least 60 GB
GPU: RTX 3090 (24 GB)
COMFYUI_ARGS=--listen 127.0.0.1 --disable-auto-launch --disable-xformers --port 18188 --enable-cors-header --force-upcast-attention
```

`--force-upcast-attention` is required for this tested 3090 configuration. Without it, ComfyUI can report success while writing an all-black image.
Binding to `127.0.0.1` keeps port 18188 behind Vast Instance Portal authentication instead of exposing ComfyUI directly on every network interface.

No secret is needed: all three Hugging Face repositories are public. Do not add an HF token, Vast API key, or other credential to this repository or to an on-start command.

## Install before ComfyUI startup

Make a pinned revision of this repository available in the instance, then use this as the Vast **On-start Script**. Replace the two public repository coordinates only after publishing; pin a commit rather than a moving branch.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
readonly REPO_URL='https://github.com/OWNER/vast-chroma-comfy.git'
readonly REPO_COMMIT='FULL_40_CHARACTER_COMMIT_SHA'
[[ "$REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'REPO_COMMIT must be an exact 40-character lowercase commit SHA\n' >&2
  exit 1
}
if [[ ! -d /opt/vast-chroma-comfy/.git ]]; then
  git clone --filter=blob:none --no-checkout "$REPO_URL" /opt/vast-chroma-comfy
else
  git -C /opt/vast-chroma-comfy remote set-url origin "$REPO_URL"
fi
git -C /opt/vast-chroma-comfy fetch --depth=1 origin "$REPO_COMMIT"
git -C /opt/vast-chroma-comfy checkout --detach "$REPO_COMMIT"
checked_out_commit="$(git -C /opt/vast-chroma-comfy rev-parse HEAD)"
[[ "$checked_out_commit" == "$REPO_COMMIT" ]] || {
  printf 'checked-out commit does not match REPO_COMMIT\n' >&2
  exit 1
}
exec /opt/vast-chroma-comfy/provision.sh
```

The placeholders are intentional because this repository is created and committed locally only. Do not use the example until it is published and both values are replaced.

When the repository already exists in an image or mounted volume, the complete pre-start command is simply:

```bash
/workspace/vast-chroma-comfy/provision.sh
```

The script uses `/venv/main/bin/python` in the official image, falling back to `python3`. It requires `hf`, NumPy, Pillow, and safetensors from the template. Missing tools, failed downloads, metadata mismatches, malformed safetensors, or workflow errors stop startup.

## What provisioning guarantees

Three `hf download` jobs run concurrently. Every artifact lands in a temporary directory on the destination filesystem, is checked for exact byte length, SHA-256, and safetensors readability, and is then moved atomically into place. A valid existing artifact is reused; an invalid existing artifact is never treated as a cache hit.

| ComfyUI destination | Bytes | SHA-256 |
|---|---:|---|
| `models/diffusion_models/Chroma1-HD.safetensors` | 17,800,038,288 | `d446d9695d08276f61e53653e025289dd96f7a489c27982a1fa54ecabc06642a` |
| `models/text_encoders/t5xxl_fp16.safetensors` | 9,787,841,024 | `6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635` |
| `models/vae/ae.safetensors` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

After all models pass, the script fetches the canonical Hugging Face UI workflow, patches it to full Chroma, FP16 T5, the pinned VAE, and a 768×768 latent, validates its JSON, and atomically installs:

```text
/workspace/ComfyUI/user/default/workflows/Chroma1-HD-RTX3090.json
```

Set `WORKFLOW_SIZE=512` before provisioning if a 512×512 UI workflow is preferred. Other sizes fail closed.

## Smoke test

After ComfyUI is listening locally:

```bash
/venv/main/bin/python ./smoke_test.py \
  --url http://localhost:18188 \
  --size 512 \
  --timeout 600 \
  --prompt 'A red kite flying over green hills, natural light, sharp focus'
```

Only `--size 512` and `--size 768` are accepted. The test submits an API-format workflow to `/prompt`, polls `/history/<prompt_id>` until the bounded deadline, downloads the resulting PNG through `/view`, and checks Pillow channel extrema and means. Missing output, API errors, timeouts, malformed images, and all-black images return nonzero. Success prints one JSON object.

## Cold/warm five-minute measurement

The target metric is **Vast rent accepted → valid non-black PNG**, not container start, provisioning-only time, or ComfyUI's successful history status. Record the Unix timestamp immediately when Vast accepts the rental, then pass it to the smoke test:

```bash
ACCEPTED_AT='1760000000.000' # replace with the create-acceptance timestamp
/venv/main/bin/python ./smoke_test.py --size 512 --timeout 300 --accepted-at "$ACCEPTED_AT"
```

Read `timing_seconds.rent_accepted_to_valid_image` from the emitted JSON. The sub-five-minute criterion is that value `< 300.000`.

Measure both cases separately:

- **Cold:** the persistent volume has none of the three target model files before rental acceptance.
- **Warm:** the same volume contains all three files and each passes full size, SHA-256, and safetensors validation. Hash verification time is included.

`provision.sh` also emits `provision_seconds`, while the smoke result emits `submit`, `generation`, and `smoke_total`. These are diagnostic subdivisions only and must not replace the end-to-end acceptance metric. Keep the raw JSON from each run; do not report a sub-five-minute result unless `ok` is true and `rent_accepted_to_valid_image` is below 300 seconds.

## Development

```bash
uv venv --seed .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/pytest -q
bash -n provision.sh
shellcheck provision.sh
```
