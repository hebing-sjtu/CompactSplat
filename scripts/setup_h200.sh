#!/usr/bin/env bash
# Provision a CptSplat environment on an H200 (sm_90) node.
#
#   bash scripts/setup_h200.sh /path/to/venv
#
# Run this once per node unless the venv lives on shared storage that every node
# mounts at the same path. Lightning under srun imports from whatever $VENV the
# job script activates, so the path must resolve identically on both nodes.
set -euo pipefail

VENV="${1:?usage: setup_h200.sh /path/to/venv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# H200 is compute capability 9.0. Without this, the rasteriser build falls back to
# whatever nvcc guesses from the visible device, which fails outright on a
# login/build node that has no GPU attached.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-16}"
# The one xformers use site is guarded; keep it off so nothing tries to load a
# wheel built for a different torch.
export XFORMERS_DISABLED=1

echo "=== repo: $REPO"
echo "=== venv: $VENV"

# --- 1. locate the interpreter that actually has torch --------------------
# Must happen before we activate our own venv, because activating overwrites
# VIRTUAL_ENV. Images commonly keep torch inside their own venv (/opt/venv),
# and venvs do not chain: --system-site-packages exposes the base interpreter's
# site-packages, never a parent venv's. So find where torch really lives and
# deal with both layouts below.
IMAGE_PYTHON="${IMAGE_PYTHON:-}"
if [[ -z "$IMAGE_PYTHON" ]]; then
  for cand in "${VIRTUAL_ENV:-/nonexistent}/bin/python" /opt/venv/bin/python \
              "$(command -v python3 || true)" /usr/bin/python3; do
    [[ -x "$cand" ]] || continue
    if "$cand" -c 'import torch' >/dev/null 2>&1; then
      IMAGE_PYTHON="$cand"
      break
    fi
  done
fi
if [[ -z "$IMAGE_PYTHON" ]]; then
  echo "!! no interpreter with torch found."
  echo "   Point IMAGE_PYTHON at the image's python and re-run, e.g."
  echo "     IMAGE_PYTHON=/opt/venv/bin/python bash scripts/setup_h200.sh $VENV"
  exit 1
fi
TORCH_SITE="$("$IMAGE_PYTHON" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent.parent)')"
echo "=== image python: $IMAGE_PYTHON"
echo "=== image site:   $TORCH_SITE"

# --- 2. venv that can see the image's torch -------------------------------
if [[ ! -d "$VENV" ]]; then
  "$IMAGE_PYTHON" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# If torch is invisible here, the image keeps it in a venv rather than in the
# base install, so --system-site-packages did nothing. Put that directory on
# sys.path explicitly. The venv's own site-packages is still searched first, so
# anything installed below shadows the image's copy, same as the flag's
# semantics.
if ! python -c 'import torch' >/dev/null 2>&1; then
  SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "$TORCH_SITE" > "$SITE/_image_site.pth"
  echo "=== linked image site-packages via $SITE/_image_site.pth"
fi
python -c 'import torch' >/dev/null 2>&1 || {
  echo "!! torch still not importable inside $VENV"
  echo "   delete the venv and re-run: rm -rf '$VENV'"
  exit 1
}

python -m pip install --upgrade pip

# --- 3. confirm the image's torch is what we pinned against ----------------
echo "=== pinning what the image provides ==="
python - "$REPO/constraints-image.txt" <<'PY'
import pathlib, sys
import torch, torchvision

print(f"  torch=={torch.__version__}")
print(f"  torchvision=={torchvision.__version__}")
print(f"  cuda runtime: {torch.version.cuda}")
try:
    print(f"  nccl: {torch.cuda.nccl.version()}")
except Exception as exc:
    print(f"  nccl: unavailable ({exc})")

want = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = line.split("#")[0].strip()
    if "==" in line:
        name, version = line.split("==", 1)
        want[name.strip()] = version.strip()

have = {"torch": torch.__version__, "torchvision": torchvision.__version__}
bad = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
if bad:
    print("\n!! image differs from constraints-image.txt:")
    for name, (expected, actual) in bad.items():
        print(f"     {name}: pinned {expected}, image has {actual}")
    print("   Update constraints-image.txt to match the image, then re-run.")
    sys.exit(1)
PY

# --- 3. python deps --------------------------------------------------------
# -c keeps pip from resolving any transitive torch requirement to a PyPI wheel.
pip install -r "$REPO/requirements-h200.txt" -c "$REPO/constraints-image.txt"

# --- 4. Cython extension for the Gaussian pattern --------------------------
# Builds quadtree_core in place next to the package so `from .quadtree_core import`
# resolves. cythonize regenerates the checked-in .cpp whenever the .pyx is newer.
( cd "$REPO/src/dataset/GS_pattern" && python setup.py build_ext --inplace )

# --- 5. import smoke test --------------------------------------------------
# These all sit on the module-level import path of src/main.py, so anything that
# fails here fails the training job during startup rather than at first use.
cd "$REPO"
python - <<'PY'
import importlib, sys

checks = [
    ("diff_gaussian_rasterization", "CUDA rasteriser"),
    ("skvideo.io",                  "sk-video (misc/image_io.py, module-level)"),
    ("moviepy.editor",              "moviepy 1.x (model_wrapper.py, module-level)"),
    ("lpips",                       "lpips loss"),
    ("e3nn.o3",                     "e3nn (misc/sh_rotation.py)"),
    ("pytorch_lightning",           "lightning"),
]
failed = []
for mod, what in checks:
    try:
        importlib.import_module(mod)
        print(f"  ok    {mod:32s} {what}")
    except Exception as exc:
        print(f"  FAIL  {mod:32s} {what}: {type(exc).__name__}: {exc}")
        failed.append(mod)

# Repo-local imports, which is where the Cython extension and the pattern
# registry get exercised.
try:
    from src.dataset.GS_pattern.quadtree_core import compute_quadtree_cython  # noqa: F401
    print("  ok    quadtree_core                    Cython extension")
except Exception as exc:
    print(f"  FAIL  quadtree_core: {type(exc).__name__}: {exc}")
    failed.append("quadtree_core")

try:
    from src.dataset.GS_pattern import get_GS_pattern  # noqa: F401
    print("  ok    GS_pattern registry")
except Exception as exc:
    print(f"  FAIL  GS_pattern registry: {type(exc).__name__}: {exc}")
    failed.append("GS_pattern")

try:
    import src.model.model_wrapper  # noqa: F401
    print("  ok    src.model.model_wrapper         full training import chain")
except Exception as exc:
    print(f"  FAIL  src.model.model_wrapper: {type(exc).__name__}: {exc}")
    failed.append("model_wrapper")

if failed:
    print(f"\n{len(failed)} import(s) failed: {', '.join(failed)}")
    sys.exit(1)
print("\nall imports clean")
PY

echo
echo "=== GPU visibility ==="
python - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"  cuda devices: {n}")
for i in range(n):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"  [{i}] {torch.cuda.get_device_name(i)} sm_{major}{minor}")
if n == 0:
    print("  (none visible; fine on a login node, but Trainer(devices=device_count())")
    print("   means the training job itself must run where the GPUs are)")
PY

echo
echo "done. activate with: source $VENV/bin/activate"
