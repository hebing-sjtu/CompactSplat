#!/usr/bin/env bash

# or 
# small model
# a single A100 (80GB) for 600K steps, batch size 8 on each gpu
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=16 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=300000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# checkpointing.pretrained_model=pretrained/re10k-256x256-surfsplat-small/checkpoints/epoch_18-step_300000.ckpt \
# output_dir=checkpoints/re10k-256x256-cptsplat-small

# base model
# a single A100 (80GB) for 600K steps, batch size 8 on each gpu
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.resume=true \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# checkpointing.pretrained_model=pretrained/re10k-256x256-surfsplat-base/checkpoints/epoch_72-step_600000.ckpt \
# output_dir=checkpoints/re10k-256x256-cptsplat-base

# large model
# train on 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitl.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# checkpointing.pretrained_model=pretrained/re10k-256x256-surfsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# output_dir=checkpoints/re10k-256x256-cptsplat-large


# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# dataset.image_shape=[256,448] \
# trainer.max_steps=600000 \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=8 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# checkpointing.pretrained_model=pretrained/re10k-256x448-surfsplat-base/checkpoints/epoch_18-step_600000.ckpt \
# output_dir=checkpoints/re10k-256x448-cptsplat-base


python -m src.main +experiment=acid \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=600000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.resume=true \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitl.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
checkpointing.pretrained_model=pretrained/acid-256x256-surfsplat-large/checkpoints/epoch_433-step_600000.ckpt \
output_dir=checkpoints/acid-256x256-cptsplat-large