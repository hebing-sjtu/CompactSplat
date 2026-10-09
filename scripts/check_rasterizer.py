"""Launch one real rasterisation to prove the CUDA extension matches this GPU.

    python scripts/check_rasterizer.py

`import diff_gaussian_rasterization` succeeds even when the extension was built
for the wrong compute capability -- the mismatch only surfaces as
"no kernel image is available for execution on the device" at the first kernel
launch, which during training means a crash several minutes into the job. This
pushes synthetic Gaussians through src.model.decoder.cuda_splatting.render_cuda,
the exact path the decoder uses, so a clean run here rules the extension out.
"""
import sys

import torch

from src.model.decoder.cuda_splatting import render_cuda

N = 256
H = W = 64


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible; run this where the GPUs are")
        return 1

    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print(f"device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})")
    print(f"torch:  {torch.__version__}, cuda {torch.version.cuda}")

    g = torch.Generator(device="cpu").manual_seed(0)

    # Normalised intrinsics, OpenCV c2w extrinsics: the repo's camera convention.
    extrinsics = torch.eye(4, device=dev)[None]
    intrinsics = torch.tensor(
        [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], device=dev
    )[None]
    near = torch.tensor([0.5], device=dev)
    far = torch.tensor([100.0], device=dev)

    # A slab of Gaussians a few units down +Z, which is into the screen here.
    means = torch.rand((1, N, 3), generator=g).to(dev) * 2 - 1
    means[..., 2] = means[..., 2].abs() + 2.0
    covariances = torch.eye(3, device=dev).expand(1, N, 3, 3) * 1e-3
    # d_sh == 1 so degree 0; paired with use_sh=False this takes the
    # colors_precomp branch, which is what the decoder does for depth/normals.
    sh = torch.rand((1, N, 3, 1), generator=g).to(dev)
    opacities = torch.full((1, N), 0.5, device=dev)
    valid = torch.ones((1, N), dtype=torch.bool, device=dev)
    background = torch.zeros((1, 3), device=dev)

    image = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        (H, W),
        background,
        means,
        covariances,
        sh,
        opacities,
        valid,
        use_sh=False,
    )
    torch.cuda.synchronize()

    print(f"rendered: {tuple(image.shape)} dtype={image.dtype}")
    if image.shape != (1, 3, H, W):
        print(f"FAIL unexpected shape, wanted (1, 3, {H}, {W})")
        return 1
    if not torch.isfinite(image).all():
        print("FAIL output contains nan/inf")
        return 1
    # An all-zero image would mean nothing was splatted, i.e. the camera or the
    # Gaussians never met, so the launch proved nothing.
    covered = (image > 0).any(dim=1).float().mean().item()
    print(f"coverage: {covered:.1%} of pixels non-zero, max={image.max().item():.4f}")
    if covered == 0.0:
        print("FAIL nothing was rasterised; the kernel ran but produced an empty image")
        return 1

    print("\nrasteriser OK for this device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
