from dataclasses import dataclass
from typing import Literal
import numpy as np

import torch
from jaxtyping import Float, Bool
from torch import Tensor

from .GS_pattern import GaussianPattern
from .quadtree_core import compute_quadtree_cython
# from .parallel_splits.quadtree import quadtree

@dataclass
class GaussianPatternQuadTreeCfg:
    name: Literal["quadtree"]

    # GS parm
    scl_overlapping: float
    
    # multi-view config
    bias_flag: bool
    uniform_bias: bool
    n_views: int

    # Seg Block config
    block_height: int
    block_width: int
    threshold_min: float
    threshold_max: float
    max_subblock_size: int
    max_aspect_ratio: float


class GaussianPatternQuadTree(GaussianPattern[GaussianPatternQuadTreeCfg]):
    def generate(
        self,
        context: Float[Tensor, "view 3 height width"], 
        masks: Bool[Tensor, "view height width"], 
        threshold_ratio: float,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Generate Gaussian Splatting parameters for each view.

        Args:
            context: Batch of images.
            device: Calculation device.

        Returns:
            A tuple of dictionaries.
            Length of tuple == context.shape[0] (number of views).
            Each dictionary contains GS parameters keys (e.g., 'means', 'scales', 'quats') 
            and their corresponding Tensor values for that specific view.
        """
        V, C, H, W = context.shape
        max_blocks = H * W
        device = context.device
        S_max = self.cfg.max_subblock_size
        threshold = self.cfg.threshold_min + (self.cfg.threshold_max - self.cfg.threshold_min) * threshold_ratio
        
        # 1. Prepare contiguous numpy arrays for Cython input
        # Note: Quadtree requires sequential CPU operations, so we must pull to CPU
        context_np = context.detach().cpu().float().contiguous().numpy()
        masks_np = masks.detach().cpu().byte().contiguous().numpy() # byte = uint8
        
        # 2. Pre-allocate output arrays on CPU (Cython will fill them in-place)
        mu_cpu = np.zeros((V, max_blocks, 2), dtype=np.float32)
        b_len_cpu = np.zeros((V, max_blocks, 2), dtype=np.float32)
        lod_cpu = np.zeros((V, max_blocks, 1), dtype=np.int32)
        valid_mask_cpu = np.zeros((V, max_blocks), dtype=np.uint8)
        
        # 3. Execute zero-overhead C++ computation
        total_nodes = compute_quadtree_cython(
            context_np, masks_np, threshold, S_max,
            mu_cpu, b_len_cpu, lod_cpu, valid_mask_cpu
        )
        
        # 4. Transfer generated data directly to original device (GPU)
        # Using torch.from_numpy avoids data copying in CPU memory
        mu = torch.from_numpy(mu_cpu).to(device)
        b_len = torch.from_numpy(b_len_cpu).to(device)
        lod = torch.from_numpy(lod_cpu).to(device)
        valid_mask = torch.from_numpy(valid_mask_cpu).to(device).bool()
        
        self.prune_ratio_value = 1.0 - (total_nodes / (max_blocks * V))
        self.total_nodes_value = total_nodes
        
        return {
            "mu": mu,
            "b_len": b_len,
            "lod": lod,
            "valid_mask": valid_mask
        }
        
    @property
    def prune_ratio(self) -> float:
        return self.prune_ratio_value
    
    @property
    def total_nodes(self) -> int:
        return self.total_nodes_value    
    
################ VERSION 2 ###################
        # # Assuming context, device, threshold_ratio, and self.cfg.max_subblock_size are defined
        # V, C, H, W = context.shape
        # S_max = self.cfg.max_subblock_size
        # threshold = self.cfg.threshold_min + (self.cfg.threshold_max - self.cfg.threshold_min) * threshold_ratio
        
        # # In the worst-case scenario, every pixel becomes an independent block
        # max_blocks = H * W

        # # Pre-allocate output tensors with uniform shapes to accommodate Lightning's batch collate
        # mu = torch.zeros((V, max_blocks, 2), device=device, dtype=torch.float32)
        # b_len = torch.zeros((V, max_blocks, 2), device=device, dtype=torch.float32)
        # lod = torch.zeros((V, max_blocks, 1), device=device, dtype=torch.int32)
        # valid_mask = torch.zeros((V, max_blocks), device=device, dtype=torch.bool)
        # node_num = 0

        # for v in range(V):
        #     curr_img = context[v]
            
        #     # 1. Core Modification: Build a Quadtree Forest
        #     # Instead of starting with the whole image (0, 0, W, H), we tile the image 
        #     # into a grid of initial blocks, each up to S_max in size.
        #     queue = []
        #     for y_start in range(0, H, S_max):
        #         for x_start in range(0, W, S_max):
        #             # Handle edge cases where image dimensions are not perfectly divisible by S_max
        #             w_chunk = min(S_max, W - x_start)
        #             h_chunk = min(S_max, H - y_start)
                    
        #             # These initial blocks act as the root nodes for independent quadtrees.
        #             # LOD is uniformly defined as 0 at this maximum allowed block size.
        #             queue.append((x_start, y_start, w_chunk, h_chunk, 0))
                    
        #     leaf_idx = 0
            
        #     while queue:
        #         x, y, w, h, current_lod = queue.pop(0)
                
        #         # Extract the image content of the current block
        #         block = curr_img[:, y:y+h, x:x+w]
                
        #         # Calculate color difference within the block (maximum of channel ranges)
        #         c_max = block.amax(dim=(1, 2))
        #         c_min = block.amin(dim=(1, 2))
        #         color_diff = (c_max - c_min).max().item()
                
        #         # 2. Simplified Splitting Condition:
        #         # Since the initial blocks are already capped at max_subblock_size,
        #         # we only need to check the color variance and ensure the block is splittable (w>1, h>1).
        #         # This completely avoids forced, meaningless splits and exploding LOD levels.
        #         if color_diff > threshold and w > 1 and h > 1:
        #             # Calculate split points (compatible with odd-sized dimensions)
        #             w_left = w // 2
        #             w_right = w - w_left
        #             h_top = h // 2
        #             h_bottom = h - h_top
                    
        #             # Generate and enqueue the four child blocks
        #             if w_left > 0 and h_top > 0:
        #                 queue.append((x, y, w_left, h_top, current_lod + 1))
        #             if w_right > 0 and h_top > 0:
        #                 queue.append((x + w_left, y, w_right, h_top, current_lod + 1))
        #             if w_left > 0 and h_bottom > 0:
        #                 queue.append((x, y + h_top, w_left, h_bottom, current_lod + 1))
        #             if w_right > 0 and h_bottom > 0:
        #                 queue.append((x + w_left, y + h_top, w_right, h_bottom, current_lod + 1))
        #         else:
        #             # Record leaf nodes that cannot or do not need to be split further
        #             # Format for grid_sample mapping
        #             center_x = x + w / 2.0
        #             center_y = y + h / 2.0
                    
        #             mu[v, leaf_idx, 0] = (center_x / W) * 2.0 - 1.0  # Center x coordinate
        #             mu[v, leaf_idx, 1] = (center_y / H) * 2.0 - 1.0  # Center y coordinate
        #             b_len[v, leaf_idx, 0] = w         # Block width
        #             b_len[v, leaf_idx, 1] = h         # Block height
        #             lod[v, leaf_idx, 0] = current_lod # Level of Detail (LOD)
        #             valid_mask[v, leaf_idx] = True    # Set valid mask
        #             leaf_idx += 1
            
        #     node_num += leaf_idx     

        # # Calculate overall prune ratio across all views and blocks
        # self.prune_ratio_value = 1.0 - node_num / (max_blocks * V)

        # return {
        #     "mu": mu,
        #     "b_len": b_len,
        #     "lod": lod,
        #     "valid_mask": valid_mask
        # }
    

################ VERSION 1 ###################
        # v, _, h, w = context.shape
        # assert h % self.cfg.block_height == 0, \
        #     f"Image height {h} is not divisible by block_height {self.cfg.block_height}"
        # assert w % self.cfg.block_width == 0, \
        #     f"Image width {w} is not divisible by block_width {self.cfg.block_width}"
        # pixel_num = v * h * w
        # max_gs_per_view = h * w
        # threshold = self.cfg.threshold_min + (self.cfg.threshold_max - self.cfg.threshold_min) * threshold_ratio
        # tdgs_sets ={
        #     'mu':         torch.zeros((v, max_gs_per_view, 2), dtype=torch.float32, device=device),
        #     'b_len':      torch.zeros((v, max_gs_per_view, 2), dtype=torch.float32, device=device),
        #     'lod':        torch.zeros((v, max_gs_per_view),    dtype=torch.float32, device=device),
        #     'label':      torch.zeros((v, max_gs_per_view),    dtype=torch.float32, device=device),
        #     'valid_mask': torch.zeros((v, max_gs_per_view),    dtype=torch.bool,    device=device)
        # }
        # total_gs_num = 0
  
        # for i in range(v):
        #     img = context[i]
        #     h, w = img.shape[1], img.shape[2]
        #     mask = torch.zeros((h, w), dtype=torch.float32)
        #     edge = torch.zeros((h, w), dtype=torch.float32)
        #     coordinates, img_blocks, mask_blocks, edge_blocks = self.convert_img_into_blocks(img, mask, edge, block_size=(self.cfg.block_height, self.cfg.block_width))
        #     mu, b_len, lod, label = quadtree(self.cfg.max_subblock_size, self.cfg.max_aspect_ratio, 
        #                                                     threshold, self.cfg.scl_overlapping, self.cfg.bias_flag, self.cfg.uniform_bias, 
        #                                                     coordinates, img_blocks, mask_blocks, edge_blocks)
            
        #     n_pts = mu.shape[0]
        #     total_gs_num += n_pts
            
        #     if n_pts > 0:

        #         tdgs_sets['mu'][i, :n_pts] = torch.from_numpy(mu).float().contiguous().to(device)
        #         tdgs_sets['b_len'][i, :n_pts] = torch.from_numpy(b_len).float().contiguous().to(device)
        #         tdgs_sets['lod'][i, :n_pts] = torch.from_numpy(lod).float()
        #         tdgs_sets['label'][i, :n_pts] = torch.from_numpy(label).float().contiguous().to(device)
        #         tdgs_sets['valid_mask'][i, :n_pts] = True
            
        # self.prune_ratio_value = 1.0 - total_gs_num / pixel_num
        # return tdgs_sets

