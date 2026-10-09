#!/usr/bin/env bash
# Fetch the initialisation weights a from-scratch training run needs.
#
#   bash scripts/download_pretrained.sh            # vits only (config default)
#   bash scripts/download_pretrained.sh vits vitb vitl
#
# These are the two things main.py loads before fit():
#   checkpointing.pretrained_mvdepth   <- gmflow-scale1-things (UniMatch)
#   checkpointing.pretrained_monodepth <- depth_anything_v2_vit{s,b,l}
#
# checkpointing.pretrained_model in scripts/re10k_depthsplat_train.sh points at a
# prior in-house surfsplat checkpoint, not a public artefact. Leave it unset for a
# from-scratch run.
#
# Note: encoder.unimatch_weights_path in config/model/encoder/depthsplat.yaml is a
# dead field -- nothing reads it -- so that scannet checkpoint is not needed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PRETRAINED_DIR:-$REPO/pretrained}"
mkdir -p "$DEST"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(vits)
fi

fetch() {
  local url="$1" out="$2"
  if [[ -s "$DEST/$out" ]]; then
    echo "  have    $out"
    return
  fi
  echo "  getting $out"
  # Depth-Anything-V2 Base/Large are CC-BY-NC-4.0 and may require acceptance;
  # export HF_TOKEN if a download comes back as HTML instead of a checkpoint.
  local auth=()
  [[ -n "${HF_TOKEN:-}" ]] && auth=(--header "Authorization: Bearer $HF_TOKEN")
  curl -fL --retry 3 --retry-delay 5 "${auth[@]}" -o "$DEST/$out.part" "$url"
  mv "$DEST/$out.part" "$DEST/$out"
}

echo "=== multi-view depth init (UniMatch) -> $DEST"
fetch "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth" \
      "gmflow-scale1-things-e9887eda.pth"

echo "=== monocular depth init (Depth Anything V2) -> $DEST"
for v in "${VARIANTS[@]}"; do
  case "$v" in
    vits) repo="Depth-Anything-V2-Small" ;;
    vitb) repo="Depth-Anything-V2-Base"  ;;
    vitl) repo="Depth-Anything-V2-Large" ;;
    *) echo "unknown variant '$v' (expected vits|vitb|vitl)"; exit 1 ;;
  esac
  fetch "https://huggingface.co/depth-anything/$repo/resolve/main/depth_anything_v2_$v.pth" \
        "depth_anything_v2_$v.pth"
done

echo
# A truncated or HTML error page is the usual failure mode and it surfaces much
# later as an opaque state_dict error, so check loadability now.
python - "$DEST" <<'PY'
import sys, pathlib, torch
dest = pathlib.Path(sys.argv[1])
bad = 0
for p in sorted(dest.glob("*.pth")):
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            sd = sd.get("state_dict", sd.get("model", sd))
        print(f"  ok   {p.name:44s} {len(sd)} tensors, {p.stat().st_size/2**20:.0f} MiB")
    except Exception as exc:
        print(f"  FAIL {p.name:44s} {type(exc).__name__}: {exc}")
        bad += 1
sys.exit(1 if bad else 0)
PY
