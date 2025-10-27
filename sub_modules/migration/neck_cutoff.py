"""
neck_cutoff.py
"""


import numpy as np
import cupy as cp
from numba import cuda
import math

_GPU_AVAILABLE = True


def check_neck_cutoff_gpu_sequential(params, all_channels_data):
    """
    GPU river channel neck cutoff detection algorithm.

    Args:
        params: Parameter configuration dictionary
        all_channels_data: List of channel data

    Returns:
        list: List of cutoff results, None if failed
    """
    import numpy as np
    import time

    if not _GPU_AVAILABLE:
        print("GPU unavailable")
        return None

    start_time = time.time()
    print("Starting GPU neck cutoff detection")

    # Parameter retrieval
    cutoff_factor = params.get('cutoff_factor', 0.6)
    allow_endpoint_migration = params.get('allow_endpoint_migration', True)
    small_segment_threshold = params.get('small_segment_cutoff_threshold', 25)
    print(f"Parameters: cutoff_factor={cutoff_factor}, small_threshold={small_segment_threshold}")

    cutoff_results = []
    total_cutoffs = 0
    small_segment_channels = 0

    for channel_idx, channel_data in enumerate(all_channels_data):
        channel_start_time = time.time()

        centerline_x = np.array(channel_data['centerline_x'])
        centerline_y = np.array(channel_data['centerline_y'])
        width = np.array(channel_data['width'])

        original_start_x, original_start_y = centerline_x[0], centerline_y[0]
        original_end_x, original_end_y = centerline_x[-1], centerline_y[-1]

        n_points = len(centerline_x)
        is_small_segment = n_points <= small_segment_threshold

        if is_small_segment:
            small_segment_channels += 1

        print(f"Channel {channel_idx + 1}: {n_points} points")

        # Calculate average segment length
        avg_segment_length = 0
        for i in range(1, n_points):
            dx = centerline_x[i] - centerline_x[i - 1]
            dy = centerline_y[i] - centerline_y[i - 1]
            avg_segment_length += np.sqrt(dx ** 2 + dy ** 2)
        avg_segment_length /= max(1, (n_points - 1))

        mean_width = np.mean(width)

        # Adaptive distance threshold calculation
        if is_small_segment:
            dis_thresh_factor = params.get('small_cutoff_distance_threshold_factor', 1.5)
            min_dis_thresh = params.get('small_min_cutoff_distance_threshold', 3)
            dis_thresh = max(int(mean_width / max(0.001, avg_segment_length)) * dis_thresh_factor, min_dis_thresh)
            dis_thresh = min(dis_thresh, n_points // 4)
        else:
            dis_thresh_factor = params.get('cutoff_distance_threshold_factor', 2.0)
            min_dis_thresh = params.get('min_cutoff_distance_threshold', 25)
            dis_thresh = max(int(mean_width / max(0.001, avg_segment_length)) * dis_thresh_factor, min_dis_thresh)
            dis_thresh = min(dis_thresh, n_points // 3)

        modified_x = centerline_x.copy()
        modified_y = centerline_y.copy()
        width_array = width.copy()
        cutoff_occurred = False
        cutoff_count = 0

        max_cutoffs = params.get('max_cutoffs_per_channel', 10)
        continue_search = True

        while continue_search and cutoff_count < max_cutoffs:
            continue_search = False
            current_n_points = len(modified_x)

            if current_n_points < 4:
                break

            # Cutoff point detection
            cutoff_point = find_first_cutoff_point_gpu(
                modified_x, modified_y, width_array, cutoff_factor, dis_thresh
            )

            if cutoff_point is not None:
                i, j = cutoff_point
                cutoff_occurred = True
                cutoff_count += 1
                total_cutoffs += 1

                new_x, new_y, new_width = apply_single_cutoff(
                    modified_x, modified_y, width_array, i, j,
                    original_start_x, original_start_y, original_end_x, original_end_y,
                    allow_endpoint_migration
                )

                modified_x = new_x
                modified_y = new_y
                width_array = new_width

                # Update distance threshold
                if len(modified_x) <= small_segment_threshold:
                    dis_thresh = min(dis_thresh, len(modified_x) // 4)

                continue_search = True

        # Extend arrays to original length
        if len(modified_x) < n_points:
            temp_x = np.zeros(n_points)
            temp_y = np.zeros(n_points)

            temp_x[:len(modified_x)] = modified_x
            temp_y[:len(modified_y)] = modified_y

            if len(modified_x) > 0:
                temp_x[len(modified_x):] = modified_x[-1]
                temp_y[len(modified_y):] = modified_y[-1]

            modified_x = temp_x
            modified_y = temp_y

        channel_end_time = time.time()
        print(f"Channel {channel_idx + 1} complete: {cutoff_count} cutoffs, time {channel_end_time - channel_start_time:.3f}s")

        cutoff_results.append({
            'modified_x': modified_x,
            'modified_y': modified_y,
            'cutoff_occurred': cutoff_occurred
        })

    end_time = time.time()
    print(f"Cutoff detection complete, total time {end_time - start_time:.3f}s, total cutoffs {total_cutoffs}")
    return cutoff_results


def apply_single_cutoff(modified_x, modified_y, width_array, i, j,
                        original_start_x, original_start_y, original_end_x, original_end_y,
                        allow_endpoint_migration):
    """
    Execute single cutoff operation.

    Args:
        modified_x, modified_y: Coordinate arrays
        width_array: Width array
        i, j: Cutoff point indices
        original_start_x, original_start_y: Original start point coordinates
        original_end_x, original_end_y: Original end point coordinates
        allow_endpoint_migration: Whether to allow endpoint migration

    Returns:
        tuple: New coordinate and width arrays (new_x, new_y, new_width)
    """
    n_points = len(modified_x)
    new_n_points = n_points - (j - i)

    new_x = np.zeros(new_n_points)
    new_y = np.zeros(new_n_points)
    new_width = np.zeros(new_n_points)

    # Copy first half
    new_x[:i + 1] = modified_x[:i + 1]
    new_y[:i + 1] = modified_y[:i + 1]
    new_width[:i + 1] = width_array[:i + 1]

    # Copy second half
    count = 1
    for k in range(j, n_points):
        if i + count < new_n_points:
            new_x[i + count] = modified_x[k]
            new_y[i + count] = modified_y[k]
            new_width[i + count] = width_array[k]
            count += 1

    # Boundary handling
    if allow_endpoint_migration:
        boundary_zone = min(8, new_n_points // 12)

        # Start point gradient constraint
        for idx in range(min(boundary_zone, len(new_x))):
            factor = float(idx) / float(boundary_zone) if boundary_zone > 0 else 1.0
            factor = factor ** 2
            new_x[idx] = original_start_x + factor * (new_x[idx] - original_start_x)
            new_y[idx] = original_start_y + factor * (new_y[idx] - original_start_y)

        # End point gradient constraint
        for idx in range(max(0, len(new_x) - boundary_zone), len(new_x)):
            dist_from_end = len(new_x) - 1 - idx
            factor = float(dist_from_end) / float(boundary_zone) if boundary_zone > 0 else 1.0
            factor = factor ** 2
            new_x[idx] = original_end_x + factor * (new_x[idx] - original_end_x)
            new_y[idx] = original_end_y + factor * (new_y[idx] - original_end_y)
    else:
        # Fixed endpoints
        if len(new_x) > 0:
            new_x[0] = original_start_x
            new_y[0] = original_start_y
            new_x[-1] = original_end_x
            new_y[-1] = original_end_y

    return new_x, new_y, new_width


def find_first_cutoff_point_gpu(centerline_x, centerline_y, width_array, cutoff_factor, dis_thresh):
    """
    Find first cutoff point using GPU.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        width_array: Width array
        cutoff_factor: Cutoff factor
        dis_thresh: Distance threshold

    Returns:
        tuple: Cutoff point indices (i, j), None if not found
    """
    n_points = len(centerline_x)

    # Select processing mode
    use_small_segment_mode = n_points <= 25
    use_angle_verification = n_points > 10

    d_centerline_x = cp.array(centerline_x)
    d_centerline_y = cp.array(centerline_y)
    d_width_array = cp.array(width_array)
    d_result = cp.array([-1, -1, -1], dtype=cp.int32)

    threads_per_block = 1
    blocks_per_grid = 1

    if use_small_segment_mode:
        min_loop_size = max(3, n_points // 8)
        find_cutoff_kernel_small_segments[blocks_per_grid, threads_per_block](
            d_centerline_x, d_centerline_y, d_width_array,
            cutoff_factor, n_points, d_result, min_loop_size
        )
    else:
        angle_threshold = 45.0
        find_cutoff_kernel[blocks_per_grid, threads_per_block](
            d_centerline_x, d_centerline_y, d_width_array,
            cutoff_factor, dis_thresh, n_points, d_result,
            angle_threshold, use_angle_verification
        )

    result = cp.asnumpy(d_result)

    if result[0] != -1 and result[1] != -1:
        return (int(result[0]), int(result[1]))
    else:
        return None


@cuda.jit
def find_cutoff_kernel_small_segments(centerline_x, centerline_y, width_array,
                                      cutoff_factor, n_points, result,
                                      min_loop_size):
    """
    Small segment specialized cutoff detection CUDA kernel.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        width_array: Width array
        cutoff_factor: Cutoff factor
        n_points: Number of points
        result: Result array
        min_loop_size: Minimum loop size

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)

    if i == 0:
        min_separation = max(1, min_loop_size)

        # Tight search mode
        for start_i in range(0, n_points - min_separation):
            for end_j in range(start_i + min_separation, n_points):

                # Distance calculation
                dx = centerline_x[start_i] - centerline_x[end_j]
                dy = centerline_y[start_i] - centerline_y[end_j]
                cdist_sq = dx * dx + dy * dy

                # Cutoff threshold
                aggressive_thresh = min(width_array[start_i], width_array[end_j]) * cutoff_factor * 0.8
                aggressive_thresh_sq = aggressive_thresh * aggressive_thresh

                if cdist_sq < aggressive_thresh_sq:
                    # Loop validation
                    segment_length = end_j - start_i
                    if segment_length >= min_loop_size:

                        # Calculate path length
                        path_length = 0.0
                        for k in range(start_i, end_j):
                            if k + 1 < end_j:
                                dx_seg = centerline_x[k + 1] - centerline_x[k]
                                dy_seg = centerline_y[k + 1] - centerline_y[k]
                                path_length += math.sqrt(dx_seg ** 2 + dy_seg ** 2)

                        direct_distance = math.sqrt(cdist_sq)

                        # Loop coefficient check
                        if path_length > direct_distance * 2.0:
                            result[0] = start_i
                            result[1] = end_j
                            result[2] = 999
                            return

        # No cutoff found
        result[0] = -1
        result[1] = -1
        result[2] = -1


@cuda.jit
def find_cutoff_kernel(centerline_x, centerline_y, width_array,
                       cutoff_factor, dis_thresh, n_points, result,
                       angle_threshold, enable_angle_check):
    """
    Standard multi-scale cutoff detection CUDA kernel.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        width_array: Width array
        cutoff_factor: Cutoff factor
        dis_thresh: Distance threshold
        n_points: Number of points
        result: Result array
        angle_threshold: Angle threshold
        enable_angle_check: Whether to enable angle checking

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)

    if i == 0:
        # Multi-scale search
        max_scales = 5
        min_dis_thresh = max(1, dis_thresh // 4)

        for scale_idx in range(max_scales):
            # Progressively increase search distance
            current_dis_thresh = min_dis_thresh + (dis_thresh - min_dis_thresh) * scale_idx // (max_scales - 1)
            current_dis_thresh = min(current_dis_thresh, n_points // 3)

            if current_dis_thresh < 1:
                current_dis_thresh = 1

            # Adaptive search range
            if n_points <= 20:
                start_range_begin = 0
                end_range_limit = n_points
                allow_endpoints = True
            elif n_points <= 50:
                start_range_begin = 1
                end_range_limit = n_points - 1
                allow_endpoints = False
            else:
                start_range_begin = max(1, current_dis_thresh // 2)
                end_range_limit = n_points - max(1, current_dis_thresh // 2)
                allow_endpoints = False

            if start_range_begin >= n_points - current_dis_thresh - 1:
                continue

            # Main search loop
            for start_i in range(start_range_begin, min(n_points - current_dis_thresh - 1, end_range_limit)):
                for end_j in range(start_i + current_dis_thresh, min(n_points, end_range_limit)):

                    # Endpoint protection check
                    if not allow_endpoints:
                        if start_i == 0 or end_j == n_points - 1:
                            continue

                    # Distance check
                    dx = centerline_x[start_i] - centerline_x[end_j]
                    dy = centerline_y[start_i] - centerline_y[end_j]
                    cdist_sq = dx * dx + dy * dy

                    # Adaptive cutoff threshold
                    scale_factor = 1.0 + 0.3 * float(scale_idx) / float(max_scales)
                    adaptive_thresh = (width_array[start_i] + width_array[end_j]) * 0.5 * cutoff_factor * scale_factor
                    adaptive_thresh_sq = adaptive_thresh * adaptive_thresh

                    if cdist_sq < adaptive_thresh_sq:
                        # Angle validation
                        is_valid_cutoff = True

                        if enable_angle_check and (start_i > 0 and start_i < n_points - 1 and
                                                   end_j > 0 and end_j < n_points - 1):
                            # Calculate direction at start_i
                            dx1_start = centerline_x[start_i] - centerline_x[start_i - 1]
                            dy1_start = centerline_y[start_i] - centerline_y[start_i - 1]
                            dx2_start = centerline_x[start_i + 1] - centerline_x[start_i]
                            dy2_start = centerline_y[start_i + 1] - centerline_y[start_i]

                            # Calculate direction at end_j
                            dx1_end = centerline_x[end_j] - centerline_x[end_j - 1]
                            dy1_end = centerline_y[end_j] - centerline_y[end_j - 1]
                            dx2_end = centerline_x[end_j + 1] - centerline_x[end_j]
                            dy2_end = centerline_y[end_j + 1] - centerline_y[end_j]

                            # Average direction vectors
                            avg_dir_start_x = (dx1_start + dx2_start) * 0.5
                            avg_dir_start_y = (dy1_start + dy2_start) * 0.5
                            avg_dir_end_x = (dx1_end + dx2_end) * 0.5
                            avg_dir_end_y = (dy1_end + dy2_end) * 0.5

                            # Normalize
                            len_start = math.sqrt(avg_dir_start_x ** 2 + avg_dir_start_y ** 2)
                            len_end = math.sqrt(avg_dir_end_x ** 2 + avg_dir_end_y ** 2)

                            if len_start > 1e-10 and len_end > 1e-10:
                                avg_dir_start_x /= len_start
                                avg_dir_start_y /= len_start
                                avg_dir_end_x /= len_end
                                avg_dir_end_y /= len_end

                                # Calculate direction angle
                                dot_product = avg_dir_start_x * avg_dir_end_x + avg_dir_start_y * avg_dir_end_y

                                # Angle check
                                if dot_product > math.cos(angle_threshold * 3.141592654 / 180.0):
                                    is_valid_cutoff = False

                        if is_valid_cutoff:
                            result[0] = start_i
                            result[1] = end_j
                            result[2] = scale_idx
                            return

        # No cutoff found
        result[0] = -1
        result[1] = -1
        result[2] = -1