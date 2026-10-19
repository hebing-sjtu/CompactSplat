# Reproducibility notes

This document separates the CompactSplat paper protocol from the current implementation snapshot and records known reproduction gaps. Reproduction of the paper's results has not been validated for this snapshot.

## Contribution and scope

CompactSplat contributes texture-aware partitioning (TAP), hierarchical pyramid sampling (HPS), and inference-time control of Gaussian allocation. Its feature/depth backbone and surface-oriented geometric priors build on SurfSplat and DepthSplat. HRRC was introduced by SurfSplat.

The reported setting uses two posed input views with calibrated intrinsics. TAP controls allocation within each view; the current method does not explicitly merge redundant Gaussians across views. Texture variation is an allocation cue and is not a guaranteed detector of geometric boundaries. HRRC describes rendering consistency as output resolution changes; it does not prove watertightness or geometric accuracy.

## Paper versus current implementation

| Item | Paper protocol | Current implementation |
| :--- | :--- | :--- |
| TAP split statistic | Maximum per-channel color range: `max_c(max_p I[p,c] - min_p I[p,c])` | Maximum absolute deviation from the per-channel block mean: `max_c,p(abs(I[p,c] - mean_p I[p,c]))` |
| Maximum root block | 16 pixels | `max_subblock_size: 16` |
| Initial training stage | Dense initialization with threshold 0 | No automatic two-stage scheduler; thresholds must be configured per run |
| Sparsity-aware training | Threshold sampled from `U(0, 0.02)` | Default bounds are 0.02 and 0.1 |
| Main evaluation threshold | Fixed threshold 0.02 | The dataset samples a threshold at train, validation, and test time unless both bounds are equal |
| HPS | Match sampling scale to partition footprint | Area downsampling and bilinear center sampling, with a separate LoD bias for normal/scale fields |
| HRRC | Same Gaussians rendered at ×1, ×2, and ×4 | Implemented in the non-chunked test path; targets are bilinearly upsampled for ×2 and ×4 comparisons |
| Trained CompactSplat checkpoint | Needed to reproduce reported results | Weights are hosted at [`binghe-sjtu/compactsplat`](https://huggingface.co/binghe-sjtu/compactsplat); per-checkpoint reproduction remains to be validated |

The implemented TAP statistic differs from the paper's max–min definition. Threshold values are not interchangeable between the two statistics. Changing the YAML bounds alone therefore does **not** recover the paper method. A paper-matched code revision, checkpoint, and resolved configuration must be paired and validated before publishing reproduction claims.

Implementation references:

- [quadtree_core.pyx](../src/dataset/GS_pattern/quadtree_core.pyx): active split statistic and leaf construction.
- [GS_pattern_quadtree.py](../src/dataset/GS_pattern/GS_pattern_quadtree.py): threshold interpolation and padded output buffers. Historical commented implementations are not the active path.
- [quadtree.yaml](../config/dataset/GS_pattern/quadtree.yaml): current threshold bounds and maximum block size.
- [dataset_re10k.py](../src/dataset/dataset_re10k.py): threshold sampling after crop/augmentation and an 8-pixel border mask.
- [gaussian_adapter.py](../src/model/encoder/common/gaussian_adapter.py): HPS and Gaussian scale construction.
- [model_wrapper.py](../src/model/model_wrapper.py): rendering, HRRC comparisons, metrics, and qualitative export.

## Threshold controls

Hydra exposes both bounds for the active statistic:

```text
# Constant threshold, useful for repeatable evaluation
dataset.GS_pattern.threshold_min=0.02
dataset.GS_pattern.threshold_max=0.02

# Paper-described sparsity-training interval (the statistic still differs)
dataset.GS_pattern.threshold_min=0.0
dataset.GS_pattern.threshold_max=0.02

# Dense-initialization threshold
dataset.GS_pattern.threshold_min=0.0
dataset.GS_pattern.threshold_max=0.0
```

Equal bounds are intentional for fixed-threshold runs. They disable threshold jitter; they are not an invalid setting. With the active strict `statistic > threshold` test, a perfectly constant block may remain unsplit even at threshold zero. The 8-pixel boundary mask also means zero threshold is not an unconditional one-Gaussian-per-pixel switch.

Larger thresholds generally retain larger blocks and fewer primitives. Report the statistic, threshold, image shape, number of views, mask, and maximum block size alongside Gaussian counts. `prune_ratio` is calculated as `1 - total_nodes / (views × height × width)`, so it includes effects of the border mask. Padded storage capacity is not the active Gaussian count.

## Training protocol

The paper describes dense geometric initialization followed by sparsity-aware fine-tuning. The supplement reports ViT-S training for 300K iterations with batch size 16 and ViT-B/ViT-L for 600K iterations with batch size 8 on A100 GPUs. It gives learning rates of `2e-6` for the monocular backbone and `2e-4` for newly initialized layers, with MSE plus LPIPS weighted by 0.05.

The current `+experiment=re10k` configuration defaults to a small backbone, batch size 14, and 300,001 steps. The H200 SLURM launcher uses its own 16-GPU setup and defaults. These development configurations should not be presented as the complete paper recipe. The stage boundary, initialization checkpoint, exact effective batch size, and resolved configuration for each reported variant still need to be attached to the release.

Use `checkpointing.pretrained_model=/path/to/stage1.ckpt` to initialize model weights for a separate fine-tuning run. Use a new `output_dir` and leave `checkpointing.resume=false` for that new run. `checkpointing.resume=true` instead resumes the latest checkpoint under the existing output directory, including training state. The legacy training scripts contain example SurfSplat checkpoint references; they do not supply those files.

## Evaluation details and known gaps

- Use the supplied dataset-specific evaluation index and `num_context_views=2` for the paper setting. Preserve the same scenes and target views when comparing thresholds or models.
- Fix both threshold bounds for evaluation. Random thresholds are otherwise sampled even in test mode, so repeated results may vary.
- Use `test.render_chunk_size=null`. The current chunked branch creates only the standard-resolution output, while later code expects ×2 and ×4 outputs as well.
- The non-chunked path encodes once and renders the resulting Gaussians at all three resolutions. Changing `dataset.image_shape` would change reconstruction inputs and is not the HRRC protocol.
- `psnr`, `ssim`, and `lpips` refer to ×1; suffix `2` refers to ×2 and suffix `3` to ×4. The current code appends each ×4 LPIPS value twice. This leaves its arithmetic mean unchanged when every value is duplicated, but duplicates entries in the per-scene output list.
- Qualitative saving is gated by `test_scene_names` in `model_wrapper.py`. Review that allowlist before expecting all requested images, videos, depths, or PLYs to be exported.
- Encoder timing excludes dataset-side TAP preprocessing. Decoder timing includes the non-chunked ×1/×2/×4 render calls. Do not interpret these logs as separately measured end-to-end single-resolution reconstruction latency.
- The finest-level-only versus multilevel HPS comparison remains an isolated ablation to complete. Existing sparsity sweeps do not by themselves establish that contribution.

## Release record to complete

For a reproducible model release, attach the exact code revision, resolved Hydra config, trained checkpoint checksum, data/evaluation split, GPU/software environment, and result JSON files for each model. Reconcile the active TAP statistic with the paper and verify the complete training/evaluation path with the released checkpoints.

CUDA training, checkpoint loading, and numerical reproduction have not been validated for this release snapshot.
