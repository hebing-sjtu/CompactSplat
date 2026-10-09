# test


# tmux 0
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# checkpointing.load=checkpoints/re10k-256x256-cptsplat-small/checkpoints/epoch_72-step_600000.ckpt \
# output_dir=outputs/re10k \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \

# tmux 1
# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset/view_sampler=evaluation \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.load=checkpoints/re10k-256x256-cptsplat-base/checkpoints/epoch_72-step_600000.ckpt \
# output_dir=outputs/re10k_base \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \



# tmux 2
python -m src.main +experiment=re10k \
mode=test \
wandb.name=re10k \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
dataset/view_sampler=evaluation \
checkpointing.load=checkpoints/re10k-256x256-cptsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# output_dir=outputs/re10k_large \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \



# python -m src.main +experiment=dtu \
# mode=test \
# wandb.name=dtu \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_dtu.json \
# checkpointing.load=checkpoints/re10k-256x256-cptsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=false \
# test.save_gt_image=false \
# test.save_input_images=false \
# test.save_depth=false \
# test.save_depth_concat_img=false \
# test.save_depth_npy=false \
# test.save_gaussian=false \
# output_dir=outputs/dtu_large

# python -m src.main +experiment=scannet \
# mode=test \
# wandb.name=scannet \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_scannet.json \
# checkpointing.load=checkpoints/re10k-256x256-cptsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=false \
# test.save_gt_image=false \
# test.save_input_images=false \
# test.save_depth=false \
# test.save_depth_concat_img=false \
# test.save_depth_npy=false \
# test.save_gaussian=false \
# output_dir=outputs/scannet_large

# python -m src.main +experiment=dl3dv_test \
# mode=test \
# wandb.name=dl3dv \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
# checkpointing.load=checkpoints/re10k-256x256-cptsplat-large/checkpoints/epoch_72-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/dl3dv_large


# python -m src.main +experiment=acid \
# mode=test \
# wandb.name=acid \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=2 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.monodepth_vit_type=vitl \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
# checkpointing.load=checkpoints/acid-256x256-cptsplat-large/checkpoints/epoch_433-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/acid_large


# python -m src.main +experiment=re10k \
# mode=test \
# wandb.name=re10k \
# dataset.image_shape=[256,448] \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=8 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.load=checkpoints/re10k-256x448-ff2dgs-base/checkpoints/epoch_18-step_600000.ckpt \
# test.save_image=true \
# test.save_gt_image=true \
# test.save_input_images=true \
# test.save_depth=true \
# test.save_depth_concat_img=true \
# test.save_depth_npy=true \
# test.save_gaussian=true \
# output_dir=outputs/re10k-256x448
