#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
readonly WORKFLOW_DEST="${WORKFLOW_DEST:-${COMFYUI_ROOT}/user/default/workflows/Chroma1-HD-RTX3090.json}"
readonly WORKFLOW_SIZE="${WORKFLOW_SIZE:-768}"
if [[ -x /venv/main/bin/python ]]; then
  default_python="/venv/main/bin/python"
else
  default_python="python3"
fi
readonly PYTHON_BIN="${PYTHON_BIN:-$default_python}"
unset default_python
if [[ -x /venv/main/bin/hf ]]; then
  default_hf="/venv/main/bin/hf"
else
  default_hf="hf"
fi
readonly HF_BIN="${HF_BIN:-$default_hf}"
unset default_hf
START_EPOCH="$(date +%s)"
readonly START_EPOCH

# repo|remote path|Comfy model subdir|destination name|exact bytes|sha256
readonly MODELS=(
  "lodestones/Chroma1-HD|Chroma1-HD.safetensors|diffusion_models|Chroma1-HD.safetensors|17800038288|d446d9695d08276f61e53653e025289dd96f7a489c27982a1fa54ecabc06642a"
  "comfyanonymous/flux_text_encoders|t5xxl_fp16.safetensors|text_encoders|t5xxl_fp16.safetensors|9787841024|6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"
  "Comfy-Org/Lumina_Image_2.0_Repackaged|split_files/vae/ae.safetensors|vae|ae.safetensors|335304388|afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
)

log() { printf '[chroma-provision] %s\n' "$*" >&2; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || { log "required command missing: $1"; return 1; }
}

verify_model() {
  "$PYTHON_BIN" -m chroma_provision.model "$1" "$2" "$3" >/dev/null
}

download_model() {
  local record="$1" repo remote subdir name size sha target_dir target temp downloaded
  IFS='|' read -r repo remote subdir name size sha <<<"$record"
  target_dir="${COMFYUI_ROOT}/models/${subdir}"
  target="${target_dir}/${name}"
  mkdir -p -- "$target_dir"

  if [[ -f "$target" ]] && verify_model "$target" "$size" "$sha"; then
    log "already valid: $target"
    return 0
  fi

  temp="$(mktemp -d "${target_dir}/.chroma-download.XXXXXX")"
  log "downloading ${repo}/${remote}"
  if ! "$HF_BIN" download "$repo" "$remote" --local-dir "$temp" --quiet >/dev/null; then
    rm -rf -- "$temp"
    return 1
  fi
  downloaded="${temp}/${remote}"
  if [[ ! -f "$downloaded" ]]; then
    log "hf did not produce $downloaded"
    rm -rf -- "$temp"
    return 1
  fi
  if ! verify_model "$downloaded" "$size" "$sha"; then
    rm -rf -- "$temp"
    return 1
  fi
  mv -f -- "$downloaded" "$target"
  rm -rf -- "$temp"
  log "installed: $target"
}

install_workflow() {
  local workflow_dir temp
  workflow_dir="$(dirname -- "$WORKFLOW_DEST")"
  mkdir -p -- "$workflow_dir"
  temp="$(mktemp "${workflow_dir}/.Chroma1-HD-RTX3090.XXXXXX.json")"
  if ! "$PYTHON_BIN" "$SCRIPT_DIR/prepare_workflow.py" --output "$temp" --size "$WORKFLOW_SIZE" ||
     ! "$PYTHON_BIN" -m json.tool "$temp" >/dev/null; then
    rm -f -- "$temp"
    return 1
  fi
  mv -f -- "$temp" "$WORKFLOW_DEST"
  log "installed workflow: $WORKFLOW_DEST"
}

main() {
  local -a pids=()
  local failed=0 pid elapsed
  if [[ " ${COMFYUI_ARGS:-} " != *" --force-upcast-attention "* ]]; then
    log "COMFYUI_ARGS must include --force-upcast-attention"
    return 1
  fi
  require_command "$HF_BIN"
  require_command "$PYTHON_BIN"
  export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

  for record in "${MODELS[@]}"; do
    download_model "$record" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    log "one or more model downloads failed; refusing to install workflow"
    return 1
  fi

  install_workflow
  elapsed=$(( $(date +%s) - START_EPOCH ))
  printf '{"ok":true,"provision_seconds":%d,"workflow":"%s"}\n' "$elapsed" "$WORKFLOW_DEST"
}

main "$@"
