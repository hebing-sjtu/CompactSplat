# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

def convert_int_and_concatenate(int[:, :] c_array, int[:] idx, int n):
    cdef np.ndarray[np.int32_t, ndim=2] np_array = np.asarray(c_array)
    cdef np.ndarray[np.int32_t, ndim=1] np_idx = np.asarray(idx)
    
    # extract first n elements
    selected_arrays = [np_array[i, :np_idx[i]] for i in range(n)]
    result = np.concatenate(selected_arrays, axis=0)
    
    return result


def convert_float_and_concatenate(float[:, :, :] c_array, int[:] idx, int n):
    cdef np.ndarray[np.float32_t, ndim=3] np_array = np.asarray(c_array)
    cdef np.ndarray[np.int32_t, ndim=1] np_idx = np.asarray(idx)
    
    # extract first n elements
    selected_arrays = [np_array[i, :np_idx[i], :] for i in range(n)]
    result = np.concatenate(selected_arrays, axis=0)
    
    return result


def convert_pos_and_concatenate(float[:, :, :] c1_array, int[:, :, :] c2_array, int[:] idx, int n, float scale_shrink_ratio, bint bias_flag, bint uniform_bias):
    cdef np.ndarray[np.float32_t, ndim=3] np_array1 = np.asarray(c1_array)
    cdef np.ndarray[np.float32_t, ndim=3] np_array2 = np.asarray(c2_array, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] np_idx = np.asarray(idx)

    # extract first n elements
    selected_arrays1 = [np_array1[i, :np_idx[i], :] for i in range(n)]
    selected_arrays2 = [np_array2[i, :np_idx[i], :] for i in range(n)]
    
    area = np.concatenate(selected_arrays2, axis=0)
    result2 = area * scale_shrink_ratio
    result1_tmp = np.concatenate(selected_arrays1, axis=0)
    if bias_flag:
        # import random bias based on scale_shrink_ratio 
        if uniform_bias:
            biases = np.random.uniform(-0.5, 0.5, size=area.shape)
        else:
            biases = np.random.normal(loc=0.0, scale=0.17, size=area.shape)

        result1 = result1_tmp + biases * area
    else:
        result1 = result1_tmp

    return result1, result2


cdef void calc_img_mean(float[:, :, :] img_block_slice, float[:] color_mean_mv_slice, int start_i, int end_i, int start_j, int end_j, int pixel_num) nogil:
    cdef int c, i, j
    for c in prange(img_block_slice.shape[0], nogil=True):
        color_mean_mv_slice[c] = 0
        for i in range(start_i, end_i + 1):
            for j in range(start_j, end_j + 1):
                color_mean_mv_slice[c] += img_block_slice[c, i, j]
        color_mean_mv_slice[c] = color_mean_mv_slice[c] / pixel_num


cdef int calc_edge_label(float[:, :] edge_blocks_slice, int start_i, int end_i, int start_j, int end_j, float edge_threshold) nogil:
    cdef int i, j, edge_label_count = 0

    for i in range(start_i, end_i + 1):
        for j in range(start_j, end_j + 1):
            if edge_blocks_slice[i, j] > 0:
                edge_label_count += 1
    if edge_label_count > edge_threshold:
        return 1
    else:
        return 0

@cython.inline
cdef bint check_any_equal_1d(float[:] equal, float value) nogil:
    cdef int i
    for i in range(equal.shape[0]):
        if equal[i] == value:
            return True
    return False

@cython.inline
cdef bint check_any_greater_1d(float[:] col_init, float[:] col_curr, float threshold) nogil:
    cdef int i
    for i in range(col_curr.shape[0]):
        if abs(col_init[i]-col_curr[i]) > threshold:
            return True
    return False

@cython.inline
cdef bint check_any_greater_2d(float[:] col_init, float[:,:] col_curr, float threshold) nogil:
    cdef int i,j
    for i in range(col_curr.shape[0]):
        for j in range(col_curr.shape[1]):
            if abs(col_init[i]-col_curr[i,j]) > threshold:
                return True
    return False


def process_blocks_in_parallel(int max_subblock_size, float max_aspect_ratio, 
                                  float threshold, float scale_shrink_ratio, bint bias_flag, bint uniform_bias,
                                  np.ndarray[np.float32_t, ndim=2] coordinates, 
                                  np.ndarray[np.float32_t, ndim=4] img_blocks, 
                                  np.ndarray[np.float32_t, ndim=3] mask_blocks,
                                  np.ndarray[np.float32_t, ndim=3] edge_blocks,):
    cdef int num_blocks = coordinates.shape[0]
    cdef int block_height = mask_blocks.shape[1]
    cdef int block_width = mask_blocks.shape[2]

    # Pre-allocate space for intermediate variables
    cdef np.ndarray[np.float32_t, ndim=3] OM = 1.0-mask_blocks
    cdef np.ndarray[np.float32_t, ndim=2] color_init = np.zeros((num_blocks, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] color_mean = np.zeros((num_blocks, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] edge_threshold = np.zeros((num_blocks), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] edge_label = np.zeros((num_blocks), dtype=np.int32)

    # Pre-allocate space for output variables
    cdef np.ndarray[np.float32_t, ndim=3] mu = np.zeros((num_blocks, block_height * block_width, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] col = np.zeros((num_blocks, block_height * block_width, 3), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] label = np.zeros((num_blocks, block_height * block_width), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=3] b_len = np.zeros((num_blocks, block_height * block_width, 2), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] idxs = np.zeros((num_blocks), dtype=np.int32)

    # Convert to memoryviews for better performance in parallel processing
    #input variables and local variables
    cdef int i, prev_i, prev_j, curr_i, curr_j, pixel_num, curr_h, curr_w, max_h, max_w
    cdef bint flag_i, flag_j
    cdef float[:, :] coordinates_mv = coordinates
    cdef float[:, :, :, :] img_blocks_mv = img_blocks
    cdef float[:, :, :] edge_blocks_mv = edge_blocks

    # intermediate variable
    cdef float[:, :, :] OM_mv = OM
    cdef float[:, :] color_init_mv = color_init
    cdef float[:, :] color_mean_mv = color_mean
    cdef float[:] edge_threshold_mv = edge_threshold
    cdef int[:] edge_label_mv = edge_label

    # Output variables
    cdef float[:, :, :] mu_mv = mu
    cdef float[:, :, :] col_mv = col
    cdef int[:, :] label_mv = label
    cdef int[:, :, :] b_len_mv = b_len
    cdef int[:] idxs_mv = idxs
    
    # Parallel processing with pre-allocated arrays
    for i in prange(num_blocks, nogil=True):  # Adjust chunksize
        for prev_i in range(block_height):
            for prev_j in range(block_width):
                if OM_mv[i, prev_i, prev_j] == 1:
                    continue
                else:
                    color_init_mv[i, :] = img_blocks_mv[i, :, prev_i, prev_j]
                    flag_i = False
                    flag_j = False
                    curr_i = prev_i+1
                    curr_j = prev_j+1
                    max_h = 2
                    max_w = 2

                    while not flag_i or not flag_j:
                        if not flag_i:
                            curr_i = curr_i+1
                            curr_h = curr_i-prev_i

                            if curr_i >= block_height:
                                curr_i = block_height-1
                                flag_i = True
                            elif check_any_equal_1d(OM_mv[i, curr_i, prev_j:curr_j + 1], 1):
                                curr_i = curr_i-1
                                flag_i = True
                            elif curr_h > max_h:
                                curr_i = curr_i-1
                                flag_i = True
                            else:
                                if check_any_greater_2d(color_init_mv[i, :], img_blocks_mv[i, :, curr_i, prev_j:curr_j + 1], threshold):
                                    curr_i = curr_i-1
                                    flag_i = True
                                elif curr_i == prev_i + max_subblock_size - 1:
                                    flag_i = True
                                    curr_h = curr_h+1
                                else:
                                    curr_h = curr_h+1
                            
                            max_w = int((curr_h-1)*max_aspect_ratio)

                        if not flag_j:
                            curr_j = curr_j+1
                            curr_w = curr_j-prev_j

                            if curr_j >= block_width:
                                curr_j = block_width-1
                                flag_j = True
                            elif check_any_equal_1d(OM_mv[i, prev_i:curr_i + 1, curr_j], 1):
                                curr_j = curr_j-1
                                flag_j = True
                            elif curr_w > max_w:
                                curr_j = curr_j-1
                                flag_j = True
                            else:
                                if check_any_greater_2d(color_init_mv[i, :], img_blocks_mv[i, :, prev_i:curr_i + 1, curr_j], threshold):
                                    curr_j = curr_j-1
                                    flag_j = True
                                elif curr_j == prev_j + max_subblock_size - 1:
                                    flag_j = True
                                    curr_w = curr_w+1
                                else:
                                    curr_w = curr_w+1
                                
                            max_h = int((curr_w-1)* max_aspect_ratio)

                    # calculate center and mean color
                    # alter
                    pixel_num = (curr_i - prev_i + 1)*(curr_j - prev_j + 1)
                    calc_img_mean(img_blocks_mv[i, :, :, :], color_mean_mv[i,:], prev_i, curr_i, prev_j, curr_j, pixel_num)
                    edge_threshold[i] = pixel_num*0.2
                    edge_label_mv[i] = calc_edge_label(edge_blocks_mv[i, :, :], prev_i, curr_i, prev_j, curr_j, edge_threshold[i])

                    OM_mv[i, prev_i:curr_i + 1, prev_j:curr_j + 1] = 1
                    mu_mv[i, idxs_mv[i], 0] = coordinates_mv[i,0] + float((curr_i + prev_i + 1.0) / 2.0)
                    mu_mv[i, idxs_mv[i], 1] = coordinates_mv[i,1] + float((curr_j + prev_j + 1.0) / 2.0)
                    col_mv[i, idxs_mv[i], :] = color_mean_mv[i,:]
                    label_mv[i, idxs_mv[i]] = edge_label_mv[i]
                    b_len_mv[i, idxs_mv[i], 0] = curr_i - prev_i + 1
                    b_len_mv[i, idxs_mv[i], 1] = curr_j - prev_j + 1
                    idxs_mv[i] = idxs_mv[i] + 1

    result_mu, result_b_len = convert_pos_and_concatenate(mu_mv, b_len_mv, idxs_mv, num_blocks, scale_shrink_ratio, bias_flag, uniform_bias)
    result_col = convert_float_and_concatenate(col_mv, idxs_mv, num_blocks)
    result_label = convert_int_and_concatenate(label_mv, idxs_mv, num_blocks)
    return result_mu, result_b_len, result_col, result_label
