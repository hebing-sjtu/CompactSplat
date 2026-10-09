# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from libcpp.queue cimport queue

# Define a lightweight C-struct for the queue elements
cdef struct Block:
    int x
    int y
    int w
    int h
    int lod

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_quadtree_cython(
    float[:, :, :, ::1] images,       # [V, C, H, W] Base image features
    unsigned char[:, :, ::1] masks,   # [V, H, W] Mask (0 for invalid edge, 1 for valid)
    float threshold,
    int max_subblock_size,
    float[:, :, ::1] mu,              # Output: [V, max_blocks, 2]
    float[:, :, ::1] b_len,           # Output: [V, max_blocks, 2]
    int[:, :, ::1] lod_out,           # Output: [V, max_blocks, 1]
    unsigned char[:, ::1] valid_mask  # Output: [V, max_blocks]
):
    cdef int V = images.shape[0]
    cdef int C = images.shape[1]
    cdef int H = images.shape[2]
    cdef int W = images.shape[3]
    
    cdef int v, x_start, y_start, w_chunk, h_chunk
    cdef int x, y, w, h, current_lod
    cdef int leaf_idx, total_nodes = 0
    cdef queue[Block] q
    cdef Block blk, child_blk
    
    cdef int i, j, c
    cdef float mean_val, pixel_num_inv, diff, max_diff
    cdef float center_x, center_y
    cdef int w_left, w_right, h_top, h_bottom
    
    # Release the GIL for pure C execution
    with nogil:
        # Simple range(V) is safe because step is 1
        for v in range(V):
            leaf_idx = 0
            
            # 1. Initialize the quadtree forest using pure C while loops
            y_start = 0
            while y_start < H:
                x_start = 0
                while x_start < W:
                    # Calculate chunk width safely
                    if x_start + max_subblock_size <= W:
                        w_chunk = max_subblock_size
                    else:
                        w_chunk = W - x_start
                        
                    # Calculate chunk height safely
                    if y_start + max_subblock_size <= H:
                        h_chunk = max_subblock_size
                    else:
                        h_chunk = H - y_start
                    
                    blk.x = x_start
                    blk.y = y_start
                    blk.w = w_chunk
                    blk.h = h_chunk
                    blk.lod = 0
                    q.push(blk)
                    
                    x_start += max_subblock_size
                y_start += max_subblock_size
                    
            # 2. Process the queue
            while not q.empty():
                blk = q.front()
                q.pop()
                
                x = blk.x
                y = blk.y
                w = blk.w
                h = blk.h
                current_lod = blk.lod
                
                # 3. Calculate the max deviation from the per-channel block mean purely in C
                max_diff = 0.0
                pixel_num_inv = 1.0 / (w * h)
                for c in range(C):
                    mean_val = 0.0
                    for i in range(y, y + h):
                        for j in range(x, x + w):
                            mean_val += images[v, c, i, j]
                    mean_val = mean_val * pixel_num_inv

                    for i in range(y, y + h):
                        for j in range(x, x + w):
                            diff = fabs(mean_val - images[v, c, i, j])
                            if diff > max_diff:
                                max_diff = diff

                    if max_diff > threshold:
                        break

                # 4. Splitting conditions
                if max_diff > threshold and w > 1 and h > 1:
                    w_left = w / 2
                    w_right = w - w_left
                    h_top = h / 2
                    h_bottom = h - h_top
                    
                    if w_left > 0 and h_top > 0:
                        child_blk.x = x
                        child_blk.y = y
                        child_blk.w = w_left
                        child_blk.h = h_top
                        child_blk.lod = current_lod + 1
                        q.push(child_blk)
                    if w_right > 0 and h_top > 0:
                        child_blk.x = x + w_left
                        child_blk.y = y
                        child_blk.w = w_right
                        child_blk.h = h_top
                        child_blk.lod = current_lod + 1
                        q.push(child_blk)
                    if w_left > 0 and h_bottom > 0:
                        child_blk.x = x
                        child_blk.y = y + h_top
                        child_blk.w = w_left
                        child_blk.h = h_bottom
                        child_blk.lod = current_lod + 1
                        q.push(child_blk)
                    if w_right > 0 and h_bottom > 0:
                        child_blk.x = x + w_left
                        child_blk.y = y + h_top
                        child_blk.w = w_right
                        child_blk.h = h_bottom
                        child_blk.lod = current_lod + 1
                        q.push(child_blk)
                else:
                    # 5. Record leaf node and apply MASK filtering
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    
                    # Core Mask Filter: Keep Gaussians in valid mask area
                    if masks[v, <int>center_y, <int>center_x] > 0:
                        mu[v, leaf_idx, 0] = (center_x / W) * 2.0 - 1.0
                        mu[v, leaf_idx, 1] = (center_y / H) * 2.0 - 1.0
                        b_len[v, leaf_idx, 0] = w
                        b_len[v, leaf_idx, 1] = h
                        lod_out[v, leaf_idx, 0] = current_lod
                        valid_mask[v, leaf_idx] = 1
                        leaf_idx += 1
                        
            total_nodes += leaf_idx
            
    return total_nodes