# Datasets

CompactSplat uses posed image sequences. The paper reports RealEstate10K and ACID experiments and evaluates cross-dataset generalization on DL3DV, DTU, and ScanNet. The repository also retains upstream DL3DV training utilities. A conversion script or experiment config is not evidence that a CompactSplat checkpoint has been trained on that dataset.

## Layout and paths

The chunk-based loaders expect an `index.json` mapping scene keys to `.torch` chunk filenames in each split:

```text
datasets/
├── re10k/
│   ├── train/
│   │   ├── 000000.torch
│   │   ├── ...
│   │   └── index.json
│   └── test/
│       ├── 000000.torch
│       ├── ...
│       └── index.json
├── acid/
├── dl3dv/
├── dtu/
└── scannet/
```

Evaluation requires a `test/` split. Training also reads `test/` for validation. Keep datasets outside the checkout if desired and supply a Hydra list pointing to the **dataset root**, for example:

```text
dataset.roots=[/data/datasets/re10k]
```

Alternatively, link an existing parent directory containing `re10k/`, `acid/`, and other datasets:

```bash
ln -s /data/datasets datasets
```

The chunks contain encoded images, camera information, and scene keys. Follow the [camera conventions](README.md#camera-conventions): normalized intrinsics and OpenCV camera-to-world extrinsics. The RE10K loader converts stored world-to-camera poses into camera-to-world matrices.

## RealEstate10K

The paper's standard experiment uses 256 × 256 cropped/resized images from [RealEstate10K](https://google.github.io/realestate10k/index.html), with two context views. Follow the [pixelSplat dataset acquisition instructions](https://github.com/dcharatan/pixelsplat#acquiring-datasets) for processed 360p data. The `+experiment=re10k` configuration defaults to `datasets/re10k` and uses [evaluation_index_re10k.json](assets/evaluation_index_re10k.json) for evaluation.

For source-video preparation, see the [cross_attention_renderer download utilities](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download) and [real_estate_10k_tools](https://github.com/dcharatan/real_estate_10k_tools). Video availability can change, so record which scenes were successfully prepared and compare them with the evaluation index.

For 720p inputs, obtain and convert the higher-resolution videos, then set `dataset.highres=true`. In the RE10K loader, `highres=false` expects 360 × 640 source images and `highres=true` expects 720 × 1280 source images before cropping. This input-data option is separate from HRRC, which changes **render resolution** while keeping the reconstructed Gaussians fixed.

DepthSplat provides a [two-scene 720p test subset](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/re10k_720p_test_subset.zip). It is an **upstream demonstration subset**, not CompactSplat training data or the complete evaluation benchmark.

## ACID

Acquire [ACID](https://infinite-nature.github.io/) chunks through the [pixelSplat dataset instructions](https://github.com/dcharatan/pixelsplat#acquiring-datasets). Prepare `datasets/acid/train/` and `datasets/acid/test/` for ACID training, or only `test/` for evaluation. The current `+experiment=acid` config uses the RE10K-format loader with ACID-specific bounds and root.

Pass the ACID index explicitly when evaluating:

```text
+experiment=acid
dataset/view_sampler=evaluation
dataset.view_sampler.index_path=assets/evaluation_index_acid.json
```

The inherited dataset-specific Hydra override is keyed to the loader name `re10k`, so selecting the ACID experiment alone does not select the ACID evaluation index. Distinguish an ACID-trained checkpoint from a RealEstate10K checkpoint evaluated zero-shot on ACID.

## DL3DV

Use the official [DL3DV dataset](https://github.com/DL3DV-10K/Dataset) and its [DL3DV-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark) evaluation split. The inherited preprocessing workflow uses:

- [convert_dl3dv_test.py](src/scripts/convert_dl3dv_test.py) for test chunks.
- [convert_dl3dv_train.py](src/scripts/convert_dl3dv_train.py) for training chunks from [DL3DV-ALL-480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P), excluding benchmark scenes.
- [generate_dl3dv_index.py](src/scripts/generate_dl3dv_index.py) for `index.json`.

These scripts are **adaptation templates**. Set the required paths and inspect the following layout assumptions; command-line flags alone do not handle every dataset variant:

- Set the converters' `--input_dir` and `--output_dir` arguments for your dataset layout.
- Both converters hard-code `images_8` in scene validation and image loading. Although `--img_subdir` accepts `images_4`, it only selects the expected image shape. For [DL3DV-ALL-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P), change both hard-coded directory references to `images_4` and select the matching flag. Retain `images_8` for 480p data.
- The training converter requires `--test_index /path/to/dl3dv/test/index.json`. Supply your prepared benchmark index so benchmark scenes are excluded from training.
- The index generator requires `--dataset_path /path/to/dl3dv` and processes only `test`; change its split list when indexing a training split.

The utilities use Linux `du -b`. Record preprocessing choices and retain frame identifiers that match the evaluation index. Follow the [DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md) and cite the dataset.

The CompactSplat cross-dataset evaluation entry is `+experiment=dl3dv_test`, using the RE10K-format loader and two views. The separate `+experiment=dl3dv` configuration retains the upstream multi-view training setup and should not be confused with that paper setting.

An [upstream two-scene 960p test subset](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv_960p_test_subset.zip) is available from DepthSplat for demonstration. It is not the full benchmark or a training set.

## DTU

Follow the [MVSplat dataset preparation instructions](https://github.com/donydchen/mvsplat) for [DTU](https://roboimagedata.compute.dtu.dk/). The evaluation config is [config/experiment/dtu.yaml](config/experiment/dtu.yaml), and the legacy evaluation script references [evaluation_index_dtu.json](assets/evaluation_index_dtu.json). The repository also contains `evaluation_index_dtu_nctx2.json`; confirm which protocol belongs to the checkpoint and reported experiment before switching indices.

The current DTU experiment adds `dataset.test_times_per_scene`, but the RE10K-format loader ignores that field and hard-codes one pass per scene in test mode. It does not implement a four-repeat evaluation protocol. Resolve this with the paper's evaluation record before claiming exact reproduction.

## ScanNet

Obtain ScanNet through its [official project](http://www.scan-net.org/) and prepare the required RGB/camera data. The repository includes [convert_scan.py](src/scripts/convert_scan.py), which requires explicit input and output paths, plus [evaluation_index_scannet.json](assets/evaluation_index_scannet.json):

```bash
python -m src.scripts.convert_scan \
  --input_dir /path/to/scannet \
  --output_dir /path/to/scannet-chunks
```

The converter currently processes the `test` split. Inspect its scene layout assumptions before converting your data.

The current `+experiment=scannet` selects the RE10K-format loader with `skip_bad_shape=false`. A separately named ScanNet loader also exists; do not substitute it without validating the expected data format and evaluation protocol. The exact preprocessed CompactSplat evaluation split is pending release documentation.

## Evaluation configuration map

Use these with the matching prepared dataset and a trained CompactSplat checkpoint. All entries below use two context views in the paper setting.

| Dataset | Experiment | Default root | Explicit evaluation index |
| :--- | :--- | :--- | :--- |
| RealEstate10K | `re10k` | `datasets/re10k` | `assets/evaluation_index_re10k.json` |
| ACID | `acid` | `datasets/acid` | `assets/evaluation_index_acid.json` |
| DL3DV cross-dataset | `dl3dv_test` | `datasets/dl3dv` | `assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json` |
| DTU | `dtu` | `datasets/dtu` | `assets/evaluation_index_dtu.json` |
| ScanNet | `scannet` | `datasets/scannet` | `assets/evaluation_index_scannet.json` |

Set `dataset/view_sampler=evaluation`, `dataset.view_sampler.num_context_views=2`, and the corresponding `dataset.view_sampler.index_path`. Use `dataset.test_chunk_interval=1` for the full configured evaluation split. Keep the scene list, frame IDs, camera normalization, crop, and checkpoint fixed for comparisons.

## Additional datasets

Adapt the [conversion utilities](src/scripts) to the same chunk format, provide an evaluation index, and create a dataset/experiment configuration with appropriate image shape and near/far bounds. Verify camera conventions and a small sample before processing the full collection. Dataset access and use remain governed by the original providers' terms.
