# Model Zoo

## CompactSplat release status

**CompactSplat checkpoints are not yet released.** This repository contains implementation and experiment scripts, but the example checkpoint paths do not include trained weights. A download, exact training configuration, code revision, and checksum are pending validation and release.

Do not report results from an upstream DepthSplat checkpoint as CompactSplat results. The adapted Gaussian head is different, so full upstream reconstruction weights are not guaranteed to load into this implementation without a deliberate conversion.

| Variant | Monocular backbone | CompactSplat checkpoint |
| :--- | :--- | :--- |
| CompactSplat-S | Depth Anything V2 ViT-S | Pending release |
| CompactSplat-B | Depth Anything V2 ViT-B | Pending release |
| CompactSplat-L | Depth Anything V2 ViT-L | Pending release |

See [README.md](README.md#training) for a current-code smoke test and [reproducibility notes](docs/REPRODUCIBILITY.md) for the paper/current-code differences.

## Initialization weights

The training entry point loads two upstream components before fitting a model:

| Component | Weight | Hydra key |
| :--- | :--- | :--- |
| UniMatch multi-view initialization | `gmflow-scale1-things-e9887eda.pth` | `checkpointing.pretrained_mvdepth` |
| Depth Anything V2 monocular initialization | `depth_anything_v2_vits.pth`, `depth_anything_v2_vitb.pth`, or `depth_anything_v2_vitl.pth` | `checkpointing.pretrained_monodepth` |

```bash
# Select the backbone(s) needed for your experiment.
bash scripts/download_pretrained.sh vits
# bash scripts/download_pretrained.sh vitb vitl
```

The [download script](scripts/download_pretrained.sh) stores files in `pretrained/` by default and checks that they can be loaded. Match the downloaded variant to `model.encoder.monodepth_vit_type`. Follow the providers' terms for [UniMatch](https://github.com/autonomousvision/unimatch) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

## Inherited upstream reference catalog

The tables below preserve the original [DepthSplat model catalog](https://huggingface.co/haofeixu/depthsplat) for provenance and baseline use. **Every model below is a DepthSplat model, not a CompactSplat release.** Sizes, training data, and training settings in these tables are upstream metadata.

Store weights in `pretrained/` or point configuration paths to their actual location. Upstream filenames include a SHA-256 prefix; compare it with `sha256sum filename.pth` (Linux) or `shasum -a 256 filename.pth` (macOS). A checksum check confirms the downloaded file's identity, not compatibility with the modified CompactSplat architecture.

## Upstream DepthSplat Gaussian-splatting checkpoints

- According to the upstream DepthSplat model documentation, these models are trained on RealEstate10K (re10k) and/or DL3DV (dl3dv) datasets at resolutions of 256x256, 256x448, and 448x768. The number of training views ranges from 2 to 10.

- The "&rarr;" symbol indicates that the models are trained in two stages. For example, "re10k &rarr; (re10k+dl3dv)" means the model is firstly trained on the RealEstate10K dataset and then fine-tuned using a combination of the RealEstate10K and DL3DV datasets.


| Model                                                        |       Training Data        |  Training Resolution  | Training Views | Params (M) |                           Download                           |
| ------------------------------------------------------------ | :------------------------: | :-------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth         |           re10k            |        256x256        |       2        |     37     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth) |
| depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth          |           re10k            |        256x256        |       2        |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth) |
| depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth         |           re10k            |        256x256        |       2        |    360     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth) |
| depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth          |           re10k            |        256x448        |       2        |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth) |
| depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth    |     re10k &rarr; dl3dv     |        256x448        |      2-6       |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth) |
| depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth | re10k &rarr; (re10k+dl3dv) | 256x448 &rarr;448x768 |      4-10      |     37     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth) |
| depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth | re10k &rarr; (re10k+dl3dv) | 256x448 &rarr;448x768 |      2-6       |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth) |



## Upstream DepthSplat depth-prediction checkpoints

- The upstream DepthSplat depth models were trained with the following procedure:
  - Initialize the monocular feature with Depth Anything V2 and the multi-view Transformer with UniMatch.
  - Train the full DepthSplat model end-to-end on the mixed RealEstate10K and DL3DV datasets.
  - Fine-tune the pre-trained depth model on the depth datasets with ground truth depth supervision. The depth datasets used for fine-tuning include ScanNet, TartanAir, and VKITTI2.
- The depth models are fine-tuned with random numbers (2-8) of input images, and the training image resolution is 352x640.
- The scale of the predicted depth is aligned with the scale of camera pose's translation.

| Model                                                   |                  Training Data                   |  Training Resolution   | Training Views | Params (M) |                           Download                           |
| ------------------------------------------------------- | :----------------------------------------------: | :--------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| depthsplat-depth-small-352x640-randview2-8-e807bd82.pth | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2-8       |     36     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-small-352x640-randview2-8-e807bd82.pth) |
| depthsplat-depth-base-352x640-randview2-8-65a892c5.pth  | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2-8       |    111     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth) |


