# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport log2, fabs

# --- Improved Concatenation ---

def convert_all_and_concatenate(float[:, :, :] mu_c, int[:, :, :] blen_c, int[:, :] lod_c, int[:, :] label_c, int[:] idx, int n, float scale_shrink_ratio, bint bias_flag, bint uniform_bias):
    """Aggregates all results into flat NumPy arrays."""
    cdef np.ndarray[np.float32_t, ndim=3] np_mu = np.asarray(mu_c)
    cdef np.ndarray[np.float32_t, ndim=3] np_blen = np.asarray(blen_c, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] np_lod = np.asarray(lod_c)
    cdef np.ndarray[np.int32_t, ndim=2] np_label = np.asarray(label_c)
    
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

# --- High-Performance Nogil Helpers ---

cdef inline bint is_line_occupied(const float[:, :, :] mask, int n, int row, int start_col, int end_col) noexcept nogil:
    cdef int j
    if row >= mask.shape[1] or end_col >= mask.shape[2]:
        return True
    for j in range(start_col, end_col + 1):
        if mask[n, row, j] >= 0.99: return True
    return False

cdef inline bint is_column_occupied(const float[:, :, :] mask, int n, int col, int start_row, int end_row) noexcept nogil:
    cdef int i
    if col >= mask.shape[2] or end_row >= mask.shape[1]:
        return True
    for i in range(start_row, end_row + 1):
        if mask[n, i, col] >= 0.99: return True
    return False

cdef inline bint check_color_variance_line(const float[:, :, :, :] img, int n, float* init_col, int row, int start_col, int end_col, float thresh) noexcept nogil:
    cdef int c, j
    for c in range(3):
        for j in range(start_col, end_col + 1):
            if fabs(init_col[c] - img[n, c, row, j]) > thresh: return True
    return False

cdef inline bint check_color_variance_col(const float[:, :, :, :] img, int n, float* init_col, int col, int start_row, int end_row, float thresh) noexcept nogil:
    cdef int c, i
    for c in range(3):
        for i in range(start_row, end_row + 1):
            if fabs(init_col[c] - img[n, c, i, col]) > thresh: return True
    return False

# --- Optimized Rectangular Segmentation ---

def rect_growth(int max_subblock_size, float max_aspect_ratio, 
                float threshold, float scale_shrink_ratio, bint bias_flag, bint uniform_bias,
                float[:, :] coordinates, 
                const float[:, :, :, :] img_blocks, 
                float[:, :, :] mask_blocks,
                const float[:, :, :] edge_blocks):
    
    cdef int num_blocks = coordinates.shape[0]
    cdef int bh = mask_blocks.shape[1]
    cdef int bw = mask_blocks.shape[2]
    cdef int max_el = bh * bw 

    # Pre-allocate output
    cdef float[:, :, :] mu = np.zeros((num_blocks, max_el, 2), dtype=np.float32)
    cdef float[:, :, :] col = np.zeros((num_blocks, max_el, 3), dtype=np.float32)
    cdef int[:, :] label = np.zeros((num_blocks, max_el), dtype=np.int32)
    cdef int[:, :, :] b_len = np.zeros((num_blocks, max_el, 2), dtype=np.int32)
    cdef int[:, :] lod_out = np.zeros((num_blocks, max_el), dtype=np.int32)
    cdef int[:] idxs = np.zeros(num_blocks, dtype=np.int32)

    # --- Replaced TLS Buffers with Simple C Arrays ---
    cdef float c_init_arr[3]
    cdef float c_mean_arr[3]
    cdef float* c_init_ptr = c_init_arr
    cdef float* c_mean_ptr = c_mean_arr

    # Variables
    cdef int i, p_i, p_j, c_i, c_j, cur_h, cur_w, max_h, max_w, c, pix_num, row, col_idx, idx
    cdef bint flag_i, flag_j
    cdef float sum_val
    cdef int short_side, current_lod

    # Sequential processing with nogil for high performance
    with nogil:
        for i in range(num_blocks):
            for p_i in range(bh):
                for p_j in range(bw):
                    if mask_blocks[i, p_i, p_j] >= 0.99: continue

                    for c in range(3): 
                        c_init_ptr[c] = img_blocks[i, c, p_i, p_j]
                    
                    flag_i = False
                    flag_j = False
                    c_i = p_i
                    c_j = p_j
                    max_h = 2
                    max_w = 2

                    while not (flag_i and flag_j):
                        # Expand Height (i direction)
                        if not flag_i:
                            if (c_i + 1 >= bh) or (c_i - p_i + 1 >= max_subblock_size) or (c_i - p_i + 1 > max_h):
                                flag_i = True
                            elif is_line_occupied(mask_blocks, i, c_i + 1, p_j, c_j):
                                flag_i = True
                            elif check_color_variance_line(img_blocks, i, c_init_ptr, c_i + 1, p_j, c_j, threshold):
                                flag_i = True
                            else:
                                c_i = c_i + 1
                                max_w = <int>((c_i - p_i + 1) * max_aspect_ratio)

                        # Expand Width (j direction)
                        if not flag_j:
                            if (c_j + 1 >= bw) or (c_j - p_j + 1 >= max_subblock_size) or (c_j - p_j + 1 > max_w):
                                flag_j = True
                            elif is_column_occupied(mask_blocks, i, c_j + 1, p_i, c_i):
                                flag_j = True
                            elif check_color_variance_col(img_blocks, i, c_init_ptr, c_j + 1, p_i, c_i, threshold):
                                flag_j = True
                            else:
                                c_j = c_j + 1
                                max_h = <int>((c_j - p_j + 1) * max_aspect_ratio)

                    # Finalize block
                    cur_h = c_i - p_i + 1
                    cur_w = c_j - p_j + 1
                    pix_num = cur_h * cur_w
                    
                    for c in range(3):
                        sum_val = 0.0
                        for row in range(p_i, c_i + 1):
                            for col_idx in range(p_j, c_j + 1):
                                sum_val = sum_val + img_blocks[i, c, row, col_idx]
                        c_mean_ptr[c] = sum_val / pix_num

                    if cur_h < cur_w: short_side = cur_h
                    else: short_side = cur_w
                    
                    current_lod = <int>log2(<double>short_side)

                    idx = idxs[i]
                    mu[i, idx, 0] = coordinates[i, 0] + (p_i + c_i + 1.0) / 2.0
                    mu[i, idx, 1] = coordinates[i, 1] + (p_j + c_j + 1.0) / 2.0
                    
                    for c in range(3): col[i, idx, c] = c_mean_ptr[c]
                    
                    b_len[i, idx, 0] = cur_h
                    b_len[i, idx, 1] = cur_w
                    lod_out[i, idx] = current_lod
                    
                    for row in range(p_i, c_i + 1):
                        for col_idx in range(p_j, c_j + 1):
                            mask_blocks[i, row, col_idx] = 1.0
                    
                    idxs[i] += 1

    return convert_all_and_concatenate(
        mu, b_len, lod_out, label, idxs, num_blocks, scale_shrink_ratio, bias_flag, uniform_bias
    )


# --- Pro Rectangular Processing Function ---

def rect_growth_pro(int max_subblock_size, float max_aspect_ratio, 
                    float threshold, float scale_shrink_ratio, bint bias_flag, bint uniform_bias,
                    float[:, :] coordinates, 
                    const float[:, :, :, :] img_blocks, 
                    float[:, :, :] mask_blocks,
                    const float[:, :, :] edge_blocks):
    
    cdef int num_blocks = coordinates.shape[0]
    cdef int bh = mask_blocks.shape[1]
    cdef int bw = mask_blocks.shape[2]
    # Safety: Pro version adds 4 corners per large node, so we double the buffer.
    cdef int max_el = bh * bw * 2 

    cdef float[:, :, :] mu = np.zeros((num_blocks, max_el, 2), dtype=np.float32)
    cdef float[:, :, :] col = np.zeros((num_blocks, max_el, 3), dtype=np.float32)
    cdef int[:, :] label = np.zeros((num_blocks, max_el), dtype=np.int32)
    cdef int[:, :, :] b_len = np.zeros((num_blocks, max_el, 2), dtype=np.int32)
    cdef int[:, :] lod_out = np.zeros((num_blocks, max_el), dtype=np.int32)
    cdef int[:] idxs = np.zeros(num_blocks, dtype=np.int32)

    # --- Replaced TLS Buffers with Simple C Arrays ---
    cdef float c_init_arr[3]
    cdef float c_mean_arr[3]
    cdef float* c_init_ptr = c_init_arr
    cdef float* c_mean_ptr = c_mean_arr

    # Variable Declarations (Must be at top)
    cdef int i, p_i, p_j, c_i, c_j, cur_h, cur_w, c, row, col_idx, cur_idx
    cdef bint flag_i, flag_j
    cdef float sum_val, offset_h, offset_w, c_h, c_w
    cdef int short_side, current_lod
    cdef int r_s, c_s 

    with nogil:
        for i in range(num_blocks):
            for p_i in range(bh):
                for p_j in range(bw):
                    if mask_blocks[i, p_i, p_j] >= 0.99: continue

                    # Step 1: Initialize growth from top-left pixel
                    for c in range(3): 
                        c_init_ptr[c] = img_blocks[i, c, p_i, p_j]
                    
                    flag_i = False
                    flag_j = False
                    c_i = p_i
                    c_j = p_j

                    # Step 2: Adaptive Rectangular Growth
                    while not (flag_i and flag_j):
                        if not flag_i:
                            # Growth criteria for height
                            if (c_i + 1 >= bh) or (c_i - p_i + 1 >= max_subblock_size) or \
                               (c_i - p_i + 1 > (c_j - p_j + 1) * max_aspect_ratio):
                                flag_i = True
                            elif is_line_occupied(mask_blocks, i, c_i + 1, p_j, c_j): 
                                flag_i = True
                            elif check_color_variance_line(img_blocks, i, c_init_ptr, c_i + 1, p_j, c_j, threshold): 
                                flag_i = True
                            else:
                                c_i = c_i + 1

                        if not flag_j:
                            # Growth criteria for width
                            if (c_j + 1 >= bw) or (c_j - p_j + 1 >= max_subblock_size) or \
                               (c_j - p_j + 1 > (c_i - p_i + 1) * max_aspect_ratio):
                                flag_j = True
                            elif is_column_occupied(mask_blocks, i, c_j + 1, p_i, c_i): 
                                flag_j = True
                            elif check_color_variance_col(img_blocks, i, c_init_ptr, c_j + 1, p_i, c_i, threshold): 
                                flag_j = True
                            else:
                                c_j = c_j + 1

                    # Step 3: Calculate Block Statistics
                    cur_h = c_i - p_i + 1
                    cur_w = c_j - p_j + 1
                    
                    for c in range(3):
                        sum_val = 0.0
                        for row in range(p_i, c_i + 1):
                            for col_idx in range(p_j, c_j + 1):
                                sum_val = sum_val + img_blocks[i, c, row, col_idx]
                        c_mean_ptr[c] = sum_val / (cur_h * cur_w)

                    # LoD based on the short side
                    if cur_h < cur_w: short_side = cur_h 
                    else: short_side = cur_w
                    current_lod = <int>log2(<double>short_side)

                    # Step 4: Save Main Rectangular Node
                    cur_idx = idxs[i]
                    mu[i, cur_idx, 0] = coordinates[i, 0] + (p_i + c_i + 1.0) / 2.0
                    mu[i, cur_idx, 1] = coordinates[i, 1] + (p_j + c_j + 1.0) / 2.0
                    
                    for c in range(3): col[i, cur_idx, c] = c_mean_ptr[c]
                    
                    b_len[i, cur_idx, 0] = cur_h
                    b_len[i, cur_idx, 1] = cur_w
                    lod_out[i, cur_idx] = current_lod
                    idxs[i] += 1

                    # Step 5: Pro Feature - Corner Refinement
                    # Add 4 corner nodes if the block is large enough (short side >= 4)
                    if short_side >= 4:
                        c_h = cur_h / 4.0
                        c_w = cur_w / 4.0
                        offset_h = c_h / 2.0
                        offset_w = c_w / 2.0
                        
                        for r_s in range(2): 
                            for c_s in range(2): 
                                cur_idx = idxs[i]
                                
                                mu[i, cur_idx, 0] = coordinates[i, 0] + p_i + (r_s * (cur_h - c_h)) + offset_h
                                mu[i, cur_idx, 1] = coordinates[i, 1] + p_j + (c_s * (cur_w - c_w)) + offset_w
                                
                                for c in range(3): col[i, cur_idx, c] = c_mean_ptr[c]
                                
                                b_len[i, cur_idx, 0] = <int>c_h
                                b_len[i, cur_idx, 1] = <int>c_w
                                lod_out[i, cur_idx] = current_lod - 2 
                                idxs[i] += 1

                    # Step 6: Update Mask
                    for row in range(p_i, c_i + 1):
                        for col_idx in range(p_j, c_j + 1):
                            mask_blocks[i, row, col_idx] = 1.0

    return convert_all_and_concatenate(
        mu, b_len, lod_out, label, idxs, num_blocks, scale_shrink_ratio, bias_flag, uniform_bias
    )