from dataclasses import dataclass
from typing import Literal
import numpy as np

import torch
from jaxtyping import Float
from torch import Tensor

from .GS_pattern import GaussianPattern

@dataclass
class GaussianPatternRectGrowthCfg:
    name: Literal["rect_growth"]

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


class GaussianPatternRectGrowth(GaussianPattern[GaussianPatternRectGrowthCfg]):
    def generate(
        self,
        context: Float[Tensor, "view 3 height width"], 
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
        v, _, h, w = context.shape
        assert h % self.cfg.block_height == 0, \
            f"Image height {h} is not divisible by block_height {self.cfg.block_height}"
        assert w % self.cfg.block_width == 0, \
            f"Image width {w} is not divisible by block_width {self.cfg.block_width}"
        pixel_num = v * h * w
        max_gs_per_view = h * w
        threshold = self.cfg.threshold_min + (self.cfg.threshold_max - self.cfg.threshold_min) * threshold_ratio
        tdgs_sets ={
            'mu':         torch.zeros((v, max_gs_per_view, 2), dtype=torch.float32, device=device),
            'b_len':      torch.zeros((v, max_gs_per_view, 2), dtype=torch.float32, device=device),
            'lod':        torch.zeros((v, max_gs_per_view),    dtype=torch.float32, device=device),
            'label':      torch.zeros((v, max_gs_per_view),    dtype=torch.float32, device=device),
            'valid_mask': torch.zeros((v, max_gs_per_view),    dtype=torch.bool,    device=device)
        }
        total_gs_num = 0

        # parallel_splits is a separate Cython extension; import it here so that the
        # package still loads when only quadtree_core has been built.
        from .parallel_splits.rect_growth import rect_growth
  
        for i in range(v):
            img = context[i]
            h, w = img.shape[1], img.shape[2]
            mask = torch.zeros((h, w), dtype=torch.float32)
            edge = torch.zeros((h, w), dtype=torch.float32)
            coordinates, img_blocks, mask_blocks, edge_blocks = self.convert_img_into_blocks(img, mask, edge, block_size=(self.cfg.block_height, self.cfg.block_width))
            mu, b_len, lod, label = rect_growth(self.cfg.max_subblock_size, self.cfg.max_aspect_ratio, 
                                                            threshold, self.cfg.scl_overlapping, self.cfg.bias_flag, self.cfg.uniform_bias, 
                                                            coordinates, img_blocks, mask_blocks, edge_blocks)
            
            n_pts = mu.shape[0]
            total_gs_num += n_pts
            
            if n_pts > 0:

                tdgs_sets['mu'][i, :n_pts] = torch.from_numpy(mu).float().contiguous().to(device)
                tdgs_sets['b_len'][i, :n_pts] = torch.from_numpy(b_len).float().contiguous().to(device)
                tdgs_sets['lod'][i, :n_pts] = torch.from_numpy(lod).float().contiguous().to(device)
                tdgs_sets['label'][i, :n_pts] = torch.from_numpy(label).float().contiguous().to(device)
                tdgs_sets['valid_mask'][i, :n_pts] = True
            
        self.prune_ratio_value = 1.0 - total_gs_num / pixel_num
        return tdgs_sets
        
        
    @property
    def prune_ratio(self) -> float:
        return self.prune_ratio_value

    def convert_img_into_blocks(self, img, mask, edge, block_size=(64,64)):   
        '''
        input img, mask
        split them into blocks for further parallel processing
        return coordinates, img_blocks, depth_blocks, mask_blocks
        '''
        _, h, w = img.shape
        block_height, block_width = block_size
        num_blocks_h, num_blocks_w = h // block_height, w // block_width
        img_blocks = img.cpu().float().numpy().reshape(3, num_blocks_h, block_height, num_blocks_w, block_width).transpose(1, 3, 0, 2, 4).reshape(-1, 3, block_height, block_width)
        mask_blocks = mask.cpu().float().numpy().reshape(num_blocks_h, block_height, num_blocks_w, block_width).transpose(0, 2, 1, 3).reshape(-1, block_height, block_width)
        edge_blocks = edge.cpu().float().numpy().reshape(num_blocks_h, block_height, num_blocks_w, block_width).transpose(0, 2, 1, 3).reshape(-1, block_height, block_width)

        i_indices = np.arange(num_blocks_h) * block_height
        j_indices = np.arange(num_blocks_w) * block_width
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
        coordinates = np.stack((i_grid, j_grid), axis=-1).reshape(-1, 2).astype(np.float32)

        return coordinates, img_blocks, mask_blocks, edge_blocks
