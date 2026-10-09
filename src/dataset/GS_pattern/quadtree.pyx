# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport log2, ceil, fabs

# --- Helper Functions for Array Concatenation ---
# Retain Python-level logic; negligible performance impact here.
def convert_all_and_concatenate(float[:, :, :] mu_c, int[:, :, :] blen_c, int[:, :] lod_c, int[:, :] label_c, int[:] idx, int n, float scale_shrink_ratio, bint bias_flag, bint uniform_bias):
    cdef np.ndarray[np.float32_t, ndim=3] np_mu = np.asarray(mu_c)
    cdef np.ndarray[np.float32_t, ndim=3] np_blen = np.asarray(blen_c, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] np_lod = np.asarray(lod_c)
    cdef np.ndarray[np.int32_t, ndim=2] np_label = np.asarray(label_c)
    
    # Use list comprehension for fast slicing
    selected_mu = [np_mu[i, :idx[i], :] for i in range(n)]
    selected_blen = [np_blen[i, :idx[i], :] for i in range(n)]
    selected_lod = [np_lod[i, :idx[i]] for i in range(n)]
    selected_label = [np_label[i, :idx[i]] for i in range(n)]
    
    mu_res_tmp = np.concatenate(selected_mu, axis=0)
    blen_res = np.concatenate(selected_blen, axis=0)
    lod_res = np.concatenate(selected_lod, axis=0)
    label_res = np.concatenate(selected_label, axis=0)
    
    blen_res = blen_res * scale_shrink_ratio
    
    if bias_flag:
        if uniform_bias:
            biases = np.random.uniform(-0.5, 0.5, size=blen_res.shape)
        else:
            biases = np.random.normal(loc=0.0, scale=0.17, size=blen_res.shape)
        mu_res = mu_res_tmp + biases * blen_res
    else:
        mu_res = mu_res_tmp

    return mu_res, blen_res, lod_res, label_res

# --- High-Performance Nogil Utility Functions ---

cdef inline void calc_img_mean_nogil(const float[:, :, :, :] img, int n, int start_y, int start_x, int length, float* color_out, int channels) noexcept nogil:
    cdef int c, y, x
    cdef float sum_val
    cdef float pixel_num_inv = 1.0 / (length * length)
    
    for c in range(channels):
        sum_val = 0.0
        for y in range(start_y, start_y + length):
            for x in range(start_x, start_x + length):
                sum_val += img[n, c, y, x]
        color_out[c] = sum_val * pixel_num_inv

cdef inline bint check_mask_occupied(const float[:, :, :] mask, int n, int start_y, int start_x, int length) noexcept nogil:
    cdef int y, x
    cdef int end_y = start_y + length
    cdef int end_x = start_x + length
    
    if end_y > mask.shape[1] or end_x > mask.shape[2]:
        return True

    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            if mask[n, y, x] >= 0.99:
                return True
    return False

cdef inline bint check_color_variance(const float[:, :, :, :] img, int n, float* mean_col, int start_y, int start_x, int length, float thresh, int channels) noexcept nogil:
    cdef int c, y, x
    for c in range(channels):
        for y in range(start_y, start_y + length):
            for x in range(start_x, start_x + length):
                if fabs(mean_col[c] - img[n, c, y, x]) > thresh:
                    return True
    return False

cdef inline int calc_edge_label_nogil(const float[:, :, :] edge_map, int n, int start_y, int start_x, int length, float density_thresh) noexcept nogil:
    cdef int y, x, count = 0
    cdef int limit = <int>((length * length) * density_thresh)
    
    for y in range(start_y, start_y + length):
        for x in range(start_x, start_x + length):
            if edge_map[n, y, x] > 0.1:
                count += 1
                if count > limit:
                    return 1
    return 0

cdef inline void mark_mask_done(float[:, :, :] mask, int n, int start_y, int start_x, int length) noexcept nogil:
    cdef int y, x
    for y in range(start_y, start_y + length):
        for x in range(start_x, start_x + length):
            mask[n, y, x] = 1.0

# --- Main Processing Function ---
def quadtree(int max_subblock_size, float unused_aspect_ratio, 
             float threshold, float scale_shrink_ratio, bint bias_flag, bint uniform_bias,
             float[:, :] coordinates,
             const float[:, :, :, :] img_blocks, 
             float[:, :, :] mask_blocks,
             const float[:, :, :] edge_blocks):

    cdef int num_blocks = coordinates.shape[0]
    cdef int channels = img_blocks.shape[1]
    cdef int block_height = mask_blocks.shape[1]
    cdef int block_width = mask_blocks.shape[2]

    if channels > 4: 
        raise ValueError("Image has too many channels (max 4).")

    # --- 1. Pre-allocate Output Memory ---
    cdef int max_possible_elements = block_height * block_width 
    cdef float[:, :, :] mu = np.zeros((num_blocks, max_possible_elements, 2), dtype=np.float32)
    cdef float[:, :, :] col = np.zeros((num_blocks, max_possible_elements, channels), dtype=np.float32)
    cdef int[:, :] label = np.zeros((num_blocks, max_possible_elements), dtype=np.int32)
    cdef int[:, :, :] b_len = np.zeros((num_blocks, max_possible_elements, 2), dtype=np.int32)
    cdef int[:, :] lod_out = np.zeros((num_blocks, max_possible_elements), dtype=np.int32)
    cdef int[:] idxs = np.zeros(num_blocks, dtype=np.int32)

    # --- 2. Local Buffers (No longer thread-specific) ---
    cdef float current_mean_arr[4]
    cdef float* current_mean_ptr = current_mean_arr

    # --- 3. Loop Variable Declaration ---
    cdef int i, y, x, c, curr_len, cur_idx, current_lod

    # Sequential processing with nogil to retain high C-level performance
    with nogil:
        for i in range(num_blocks):
            for y in range(block_height):
                for x in range(block_width):
                    if mask_blocks[i, y, x] >= 0.99:
                        continue
                    
                    curr_len = max_subblock_size
                    while curr_len > 1:
                        if not check_mask_occupied(mask_blocks, i, y, x, curr_len):
                            calc_img_mean_nogil(img_blocks, i, y, x, curr_len, current_mean_ptr, channels)
                            if not check_color_variance(img_blocks, i, current_mean_ptr, y, x, curr_len, threshold, channels):
                                break 
                        
                        curr_len = curr_len >> 1
                    
                    if curr_len == 1:
                        for c in range(channels):
                            current_mean_ptr[c] = img_blocks[i, c, y, x]

                    current_lod = <int>(log2(<double>curr_len))
                    cur_idx = idxs[i] 
                    
                    mu[i, cur_idx, 0] = coordinates[i, 0] + y + (curr_len / 2.0)
                    mu[i, cur_idx, 1] = coordinates[i, 1] + x + (curr_len / 2.0)
                    
                    for c in range(channels):
                        col[i, cur_idx, c] = current_mean_ptr[c]
                    
                    b_len[i, cur_idx, 0] = curr_len
                    b_len[i, cur_idx, 1] = curr_len
                    lod_out[i, cur_idx] = current_lod
                    
                    label[i, cur_idx] = calc_edge_label_nogil(edge_blocks, i, y, x, curr_len, 0.2)
                    
                    mark_mask_done(mask_blocks, i, y, x, curr_len)
                    idxs[i] += 1

    return convert_all_and_concatenate(
        mu, b_len, lod_out, label, idxs, num_blocks, scale_shrink_ratio, bias_flag, uniform_bias
    )

# --- Main Pro Processing Function ---
def quadtree_pro(int max_subblock_size, float unused_aspect_ratio, 
                 float threshold, float scale_shrink_ratio, bint bias_flag, bint uniform_bias,
                 float[:, :] coordinates,
                 const float[:, :, :, :] img_blocks, 
                 float[:, :, :] mask_blocks,
                 const float[:, :, :] edge_blocks):

    cdef int num_blocks = coordinates.shape[0]
    cdef int channels = img_blocks.shape[1]
    cdef int bh = mask_blocks.shape[1]
    cdef int bw = mask_blocks.shape[2]

    if channels > 4: 
        raise ValueError("Image has too many channels (max 4 supported).")

    # Pre-allocate
    cdef int max_elements = bh * bw * 2 
    cdef float[:, :, :] mu = np.zeros((num_blocks, max_elements, 2), dtype=np.float32)
    cdef float[:, :, :] col = np.zeros((num_blocks, max_elements, channels), dtype=np.float32)
    cdef int[:, :] label = np.zeros((num_blocks, max_elements), dtype=np.int32)
    cdef int[:, :, :] b_len = np.zeros((num_blocks, max_elements, 2), dtype=np.int32)
    cdef int[:, :] lod_out = np.zeros((num_blocks, max_elements), dtype=np.int32)
    cdef int[:] idxs = np.zeros(num_blocks, dtype=np.int32)

    # --- Local Buffers ---
    cdef float current_mean_arr[4]
    cdef float* current_mean_ptr = current_mean_arr

    # --- Loop Variable Declaration ---
    cdef int i, y, x, c, curr_len, cur_idx, current_lod
    cdef int corner_len, row_sign, col_sign
    cdef float offset_s

    # Sequential processing with nogil
    with nogil:
        for i in range(num_blocks):
            for y in range(bh):
                for x in range(bw):
                    if mask_blocks[i, y, x] >= 0.99:
                        continue
                    
                    # 1. Standard Quadtree Splitting Logic
                    curr_len = max_subblock_size
                    while curr_len > 1:
                        if not check_mask_occupied(mask_blocks, i, y, x, curr_len):
                            calc_img_mean_nogil(img_blocks, i, y, x, curr_len, current_mean_ptr, channels)
                            if not check_color_variance(img_blocks, i, current_mean_ptr, y, x, curr_len, threshold, channels):
                                break 
                        
                        curr_len = curr_len >> 1
                    
                    if curr_len == 1:
                        for c in range(channels): 
                            current_mean_ptr[c] = img_blocks[i, c, y, x]

                    # 2. Store Main Node
                    cur_idx = idxs[i]
                    current_lod = <int>(log2(<double>curr_len))
                    
                    mu[i, cur_idx, 0] = coordinates[i, 0] + y + (curr_len / 2.0)
                    mu[i, cur_idx, 1] = coordinates[i, 1] + x + (curr_len / 2.0)
                    
                    for c in range(channels):
                        col[i, cur_idx, c] = current_mean_ptr[c]
                    
                    b_len[i, cur_idx, 0] = curr_len
                    b_len[i, cur_idx, 1] = curr_len
                    lod_out[i, cur_idx] = current_lod
                    idxs[i] += 1

                    # 3. Corner Refinement (Pro Feature)
                    if curr_len >= 4:
                        corner_len = curr_len >> 2 
                        offset_s = corner_len / 2.0
                        
                        for row_sign in range(2):
                            for col_sign in range(2):
                                cur_idx = idxs[i]
                                mu[i, cur_idx, 0] = coordinates[i, 0] + y + (row_sign * (curr_len - corner_len)) + offset_s
                                mu[i, cur_idx, 1] = coordinates[i, 1] + x + (col_sign * (curr_len - corner_len)) + offset_s
                                
                                for c in range(channels):
                                    col[i, cur_idx, c] = current_mean_ptr[c]
                                
                                b_len[i, cur_idx, 0] = corner_len
                                b_len[i, cur_idx, 1] = corner_len
                                lod_out[i, cur_idx] = current_lod - 2 
                                idxs[i] += 1
                    
                    mark_mask_done(mask_blocks, i, y, x, curr_len)

    return convert_all_and_concatenate(
        mu, b_len, lod_out, label, idxs, num_blocks, scale_shrink_ratio, bias_flag, uniform_bias
    )