"""
intersection_cutoff.py

This module implements GPU-parallel algorithms for river channel self-intersection detection
and cutoff processing, utilizing spatial proximity analysis methods to identify self-intersection
phenomena in river channel centerlines and execute geometric reconstruction.

The algorithm employs efficient spatial indexing and parallel computing techniques to detect
loops and self-intersections in meandering river channels, automatically applying geometric
cutoffs to maintain realistic channel morphology and prevent non-physical channel configurations.

Core Algorithmic Features:
- GPU-Accelerated Detection: Parallel processing of intersection point identification using CUDA kernels
- Spatial Proximity Analysis: Distance-based threshold methods for intersection detection
- Batch Processing: Simultaneous processing of multiple channels for computational efficiency
- Geometric Reconstruction: Automated cutoff application with smooth transitions and interpolation
- Fallback Mechanisms: CPU-based alternatives ensuring robust operation across different hardware configurations
- Memory Optimization: Efficient data structures and memory management for large-scale channel networks

Technical Implementation:
- CUDA kernel functions for parallel distance calculations and intersection identification
- Adaptive thresholding based on grid resolution and channel characteristics
- Segment merging algorithms to handle overlapping intersection regions
- Progressive cutoff application with geometric continuity preservation
- Comprehensive error handling and hardware compatibility checks
"""

import numpy as np
import time
import cupy as cp
from numba import cuda

_GPU_AVAILABLE = True

def apply_batch_intersection_cutoff(params, ncc, nc, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos, centerline_y_data):
    """
    Apply batch processing for self-intersection detection and cutoff across all channels.

    This function orchestrates the complete intersection detection and cutoff workflow,
    processing all channels in the simulation domain simultaneously for computational efficiency.
    It handles data preparation, GPU processing coordination, and result integration.

    Args:
        params (dict): Simulation parameter dictionary containing grid distances and enable flags
        ncc (int): Total number of channel complexes
        nc (np.ndarray): Array of channel counts per complex
        ccx (np.ndarray): Channel complex X-position array
        ccz (np.ndarray): Channel complex Z-position array
        cco (np.ndarray): Channel complex orientation angle array
        cx (np.ndarray): Channel X-position array
        cz (np.ndarray): Channel Z-position array
        ccl (np.ndarray): Channel centerline coordinate array
        ct (np.ndarray): Channel thickness array
        cw (np.ndarray): Channel width array
        crelpos (np.ndarray): Channel relative position array
        centerline_y_data (np.ndarray): Channel centerline Y-coordinate data

    Returns:
        None: Modifies input arrays in-place
    """
    print("Batch self-intersection detection and cutoff processing")

    # Check if intersection detection is enabled
    intersection_cutoff_enabled = params.get('intersection_cutoff_enabled', True)
    if not intersection_cutoff_enabled:
        print("Self-intersection detection disabled")
        return

    # Retrieve grid distance parameters
    xsiz = params.get('xsiz', 24.0)
    ysiz = params.get('ysiz', 24.0)
    grid_distance = min(xsiz, ysiz)

    # Set cutoff threshold to 0.3 times grid distance
    cutoff_factor = 0.3
    print(f"Grid distance: {grid_distance:.2f}, Cutoff threshold: {cutoff_factor * grid_distance:.2f}")

    # Prepare data for all channels
    all_channels_data = []
    channel_mapping = []

    total_channels = 0
    for icc in range(1, ncc + 1):
        for ic in range(1, nc[icc - 1] + 1):
            # Get actual step count for this channel
            channel_key = f"{icc}_{ic}"
            actual_steps = params.get('channel_actual_steps', {}).get(channel_key, ccl.shape[2])

            if actual_steps < 10:  # Skip channels that are too short
                continue

            # Get channel data
            centerline_x = ccl[icc - 1, ic - 1, :actual_steps].copy()

            # Get Y-coordinate data
            if centerline_y_data is not None and centerline_y_data.shape[2] >= actual_steps:
                centerline_y = centerline_y_data[icc - 1, ic - 1, :actual_steps].copy()
            else:
                # Use uniformly distributed Y coordinates
                ymn = params.get('ymn', 0.0)
                step_size = params.get('step_size', ysiz)
                centerline_y = np.array([ymn + i * step_size for i in range(actual_steps)])

            # Get width data
            width_data = cw[icc - 1, ic - 1, :actual_steps].copy()

            # Check data consistency
            if len(centerline_x) != len(centerline_y) or len(centerline_x) != len(width_data):
                print(f"Channel {icc}.{ic} data length inconsistency, skipping")
                continue

            all_channels_data.append({
                'centerline_x': centerline_x,
                'centerline_y': centerline_y,
                'width': width_data
            })

            channel_mapping.append((icc, ic, actual_steps))
            total_channels += 1

    if total_channels == 0:
        print("No channels requiring processing")
        return

    print(f"Processing {total_channels} channels")

    # Use GPU batch processing for all channel self-intersection detection
    try:
        print(f"GPU batch intersection detection initiated")
        cutoff_results = check_intersection_cutoff_gpu_batch(
            all_channels_data, grid_distance, cutoff_factor, channel_mapping
        )

        if cutoff_results is None:
            print("Intersection detection failed, maintaining original channels")
            return

        # Process detection results
        total_cutoff_count = 0
        updated_channels = 0

        for idx, (result, (icc, ic, original_steps)) in enumerate(zip(cutoff_results, channel_mapping)):
            if result['cutoff_occurred']:
                total_cutoff_count += 1

                # Get cutoff coordinates
                new_x = result['modified_x']
                new_y = result['modified_y']

                # Calculate step count
                new_steps = len(new_x)
                if new_steps < original_steps and new_steps >= 5:  # Ensure sufficient points remain after cutoff
                    # Update ccl array
                    if new_steps <= ccl.shape[2]:
                        ccl[icc - 1, ic - 1, :new_steps] = new_x[:new_steps]
                        if new_steps < ccl.shape[2]:
                            ccl[icc - 1, ic - 1, new_steps:] = 0.0

                    # Update Y-coordinate data
                    if centerline_y_data is not None and new_steps <= centerline_y_data.shape[2]:
                        centerline_y_data[icc - 1, ic - 1, :new_steps] = new_y[:new_steps]
                        if new_steps < centerline_y_data.shape[2]:
                            centerline_y_data[icc - 1, ic - 1, new_steps:] = centerline_y_data[
                                icc - 1, ic - 1, new_steps - 1]

                    # Update channel property arrays
                    if new_steps < original_steps:

                        # Resample thickness array
                        original_thickness = ct[icc - 1, ic - 1, :original_steps].copy()
                        new_indices = np.linspace(0, original_steps - 1, new_steps)
                        new_thickness = np.interp(new_indices, np.arange(original_steps), original_thickness)

                        if new_steps <= ct.shape[2]:
                            ct[icc - 1, ic - 1, :new_steps] = new_thickness
                            if new_steps < ct.shape[2]:
                                ct[icc - 1, ic - 1, new_steps:] = new_thickness[-1]

                        # Resample width array
                        original_width = cw[icc - 1, ic - 1, :original_steps].copy()
                        new_width = np.interp(new_indices, np.arange(original_steps), original_width)

                        if new_steps <= cw.shape[2]:
                            cw[icc - 1, ic - 1, :new_steps] = new_width
                            if new_steps < cw.shape[2]:
                                cw[icc - 1, ic - 1, new_steps:] = new_width[-1]

                        # Resample relative position array
                        original_relpos = crelpos[icc - 1, ic - 1, :original_steps].copy()
                        new_relpos = np.interp(new_indices, np.arange(original_steps), original_relpos)

                        if new_steps <= crelpos.shape[2]:
                            crelpos[icc - 1, ic - 1, :new_steps] = new_relpos
                            if new_steps < crelpos.shape[2]:
                                crelpos[icc - 1, ic - 1, new_steps:] = 0.5

                    # Update actual step count record
                    channel_key = f"{icc}_{ic}"
                    if 'channel_actual_steps' not in params:
                        params['channel_actual_steps'] = {}
                    params['channel_actual_steps'][channel_key] = new_steps

                    # Update complete centerline record
                    if 'full_centerlines' in params:
                        final_centerline = np.zeros((new_steps, 2))
                        final_centerline[:, 0] = new_x[:new_steps]
                        final_centerline[:, 1] = new_y[:new_steps]
                        params['full_centerlines'][channel_key] = final_centerline

                    updated_channels += 1
                else:
                    print(f"Channel {icc}.{ic} insufficient points after cutoff, maintaining original length")

        if updated_channels > 0:
            print(f"Batch intersection detection complete, updated channels: {updated_channels}")
        else:
            print("No self-intersections detected")

    except Exception as e:
        print(f"Batch intersection detection failed: {e}")
        import traceback
        traceback.print_exc()
        print("Maintaining original channels")


def check_intersection_cutoff_gpu_batch(all_channels_data, grid_distance, cutoff_factor, channel_mapping):
    """
    Execute batch channel self-intersection detection and cutoff processing using GPU acceleration.

    This function coordinates the GPU-based processing pipeline for detecting and resolving
    self-intersections across multiple channels simultaneously, optimizing computational
    efficiency through parallel processing.

    Args:
        all_channels_data (list): List of all channel data including centerline coordinates and widths
        grid_distance (float): Grid distance parameter for spatial resolution
        cutoff_factor (float): Cutoff factor for calculating threshold distance
        channel_mapping (list): Channel mapping information including complex and channel numbers

    Returns:
        list: Cutoff result list with modified coordinates and cutoff flags for each channel
    """
    start_time = time.time()
    print(f"GPU batch processing {len(all_channels_data)} channels, threshold: {cutoff_factor * grid_distance:.2f}")
    cutoff_results = []

    intersected_channels = 0

    for channel_idx, (channel_data, (icc, ic, original_steps)) in enumerate(zip(all_channels_data, channel_mapping)):
        centerline_x = np.array(channel_data['centerline_x'])
        centerline_y = np.array(channel_data['centerline_y'])
        width = np.array(channel_data['width'])

        n_points = len(centerline_x)

        if n_points < 10:
            cutoff_results.append({
                'modified_x': centerline_x.copy(),
                'modified_y': centerline_y.copy(),
                'cutoff_occurred': False,
                'intersections_found': 0
            })
            continue

        # Find all intersection points
        intersection_pairs = find_all_intersection_points_gpu(
            centerline_x, centerline_y, grid_distance, cutoff_factor
        )

        if intersection_pairs:
            intersected_channels += 1
            print(f"Channel {icc}.{ic} (points: {n_points}) found {len(intersection_pairs)} intersection pairs")

            # Merge overlapping intersection intervals
            cutoff_segments = merge_overlapping_intersections(intersection_pairs)
            print(f"  Merged into {len(cutoff_segments)} cutoff segments")

            # Apply multiple cutoffs
            modified_x, modified_y, modified_width = apply_multiple_intersection_cutoffs(
                centerline_x, centerline_y, width, cutoff_segments
            )

            print(f"  Channel {icc}.{ic} cutoff complete: {original_steps} -> {len(modified_x)} points, removed {len(cutoff_segments)} intersection segments")

            cutoff_occurred = True
            intersections_found = len(cutoff_segments)
        else:
            modified_x = centerline_x.copy()
            modified_y = centerline_y.copy()
            cutoff_occurred = False
            intersections_found = 0

        cutoff_results.append({
            'modified_x': modified_x,
            'modified_y': modified_y,
            'cutoff_occurred': cutoff_occurred,
            'intersections_found': intersections_found
        })

    end_time = time.time()

    print(f"GPU batch detection complete, total time: {end_time - start_time:.3f}s")

    return cutoff_results


def merge_overlapping_intersections(intersection_pairs):
    """
    Merge overlapping intersection intervals to optimize cutoff processing.

    This function consolidates multiple overlapping intersection regions into
    unified segments to prevent redundant processing and ensure geometric consistency.

    Args:
        intersection_pairs (list): List of intersection point pairs, each element is a tuple (point_index1, point_index2)

    Returns:
        list: List of merged cutoff intervals, each element is a tuple (start_index, end_index)
    """
    if not intersection_pairs:
        return []

    # Convert to cutoff intervals
    segments = []
    for i, j in intersection_pairs:
        start = min(i, j)
        end = max(i, j)
        segments.append([start, end])

    # Sort by starting point
    segments.sort(key=lambda x: x[0])

    # Merge overlapping intervals
    merged = []
    for start, end in segments:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def find_all_intersection_points_gpu(centerline_x, centerline_y, grid_distance, cutoff_factor):
    """
    Find all self-intersection point pairs in river channel centerline using GPU acceleration.

    This function implements parallel processing for efficient detection of channel self-intersections
    by computing pairwise distances between centerline points and identifying proximity-based intersections.

    Args:
        centerline_x (np.ndarray): Centerline X-coordinate array
        centerline_y (np.ndarray): Centerline Y-coordinate array
        grid_distance (float): Grid distance parameter for spatial resolution
        cutoff_factor (float): Cutoff factor for calculating threshold distance

    Returns:
        list: List of intersection point pairs, each element is a tuple (point_index1, point_index2)
    """
    if not _GPU_AVAILABLE:
        return find_all_intersection_points_cpu(centerline_x, centerline_y, grid_distance, cutoff_factor)

    try:
        n_points = len(centerline_x)
        max_intersections = min(n_points * 10, 1000)

        # Prepare GPU data
        d_centerline_x = cp.array(centerline_x, dtype=cp.float64)
        d_centerline_y = cp.array(centerline_y, dtype=cp.float64)
        d_intersection_pairs = cp.full((max_intersections, 3), -1, dtype=cp.int32)
        d_intersection_count = cp.array([0], dtype=cp.int32)

        # Launch GPU kernel
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

        find_all_intersection_points_kernel[blocks_per_grid, threads_per_block](
            d_centerline_x, d_centerline_y, grid_distance, cutoff_factor, n_points,
            d_intersection_pairs, max_intersections, d_intersection_count
        )

        # Retrieve results
        intersection_count = int(cp.asnumpy(d_intersection_count)[0])
        intersection_pairs = cp.asnumpy(d_intersection_pairs)

        if intersection_count > 0:
            # Extract valid intersection pairs
            valid_pairs = intersection_pairs[:intersection_count]

            # Sort by distance
            sorted_indices = np.argsort(valid_pairs[:, 2])
            sorted_pairs = valid_pairs[sorted_indices]

            # Return intersection point pairs
            result_pairs = [(int(pair[0]), int(pair[1])) for pair in sorted_pairs]
            return result_pairs
        else:
            return []

    except Exception as e:
        print(f"GPU intersection detection failed, falling back to CPU: {e}")
        return find_all_intersection_points_cpu(centerline_x, centerline_y, grid_distance, cutoff_factor)


def find_all_intersection_points_cpu(centerline_x, centerline_y, grid_distance, cutoff_factor):
    """
    CPU-based implementation of river channel self-intersection point detection algorithm.

    This function provides a robust fallback mechanism for intersection detection when
    GPU resources are unavailable, implementing the same proximity-based detection logic
    with sequential processing.

    Args:
        centerline_x (np.ndarray): Centerline X-coordinate array
        centerline_y (np.ndarray): Centerline Y-coordinate array
        grid_distance (float): Grid distance parameter for spatial resolution
        cutoff_factor (float): Cutoff factor for calculating threshold distance

    Returns:
        list: List of intersection point pairs, each element is a tuple (point_index1, point_index2)
    """
    n_points = len(centerline_x)
    intersection_pairs = []
    threshold_distance = grid_distance * cutoff_factor
    threshold_distance_sq = threshold_distance * threshold_distance

    for i in range(n_points - 3):
        for j in range(i + 3, n_points):
            dx = centerline_x[i] - centerline_x[j]
            dy = centerline_y[i] - centerline_y[j]
            cdist_sq = dx * dx + dy * dy

            if cdist_sq < threshold_distance_sq:
                intersection_pairs.append((i, j, cdist_sq))

    # Sort by distance
    intersection_pairs.sort(key=lambda x: x[2])

    return [(i, j) for i, j, _ in intersection_pairs]


def apply_multiple_intersection_cutoffs(centerline_x, centerline_y, width_array, cutoff_segments):
    """
    Apply multiple cutoff intervals to river channel centerline with geometric continuity preservation.

    This function processes multiple intersection segments sequentially, removing self-intersecting
    portions while maintaining smooth geometric transitions and preserving channel connectivity.

    Args:
        centerline_x (np.ndarray): Centerline X-coordinate array
        centerline_y (np.ndarray): Centerline Y-coordinate array
        width_array (np.ndarray): River channel width array
        cutoff_segments (list): List of cutoff intervals, each element is a tuple (start_index, end_index)

    Returns:
        tuple: (cutoff X-coordinate array, cutoff Y-coordinate array, cutoff width array)
    """
    if not cutoff_segments:
        return centerline_x.copy(), centerline_y.copy(), width_array.copy()

    # Process from back to front to avoid index offset issues
    cutoff_segments = sorted(cutoff_segments, key=lambda x: x[0], reverse=True)

    current_x = centerline_x.copy()
    current_y = centerline_y.copy()
    current_width = width_array.copy()

    total_removed_points = 0

    for start_idx, end_idx in cutoff_segments:
        # Ensure valid indices
        start_idx = max(1, min(start_idx, len(current_x) - 2))
        end_idx = max(start_idx + 1, min(end_idx, len(current_x) - 1))

        if start_idx >= end_idx:
            continue

        # Calculate length after cutoff
        points_to_remove = end_idx - start_idx
        new_length = len(current_x) - points_to_remove

        if new_length < 5:
            print(f"    Cutoff interval [{start_idx}, {end_idx}] would result in too few points, skipping")
            continue

        # Create arrays
        new_x = np.zeros(new_length)
        new_y = np.zeros(new_length)
        new_width = np.zeros(new_length)

        # Copy first part
        new_x[:start_idx] = current_x[:start_idx]
        new_y[:start_idx] = current_y[:start_idx]
        new_width[:start_idx] = current_width[:start_idx]

        # Copy second part
        remaining_length = len(current_x) - end_idx
        if remaining_length > 0:
            new_x[start_idx:start_idx + remaining_length] = current_x[end_idx:]
            new_y[start_idx:start_idx + remaining_length] = current_y[end_idx:]
            new_width[start_idx:start_idx + remaining_length] = current_width[end_idx:]

        # Add smooth transition at connection point
        if start_idx > 0 and start_idx < len(new_x) - 1:
            prev_x, prev_y = new_x[start_idx - 1], new_y[start_idx - 1]
            next_x, next_y = new_x[start_idx], new_y[start_idx]

            dx = next_x - prev_x
            dy = next_y - prev_y
            dist = np.sqrt(dx * dx + dy * dy)

            # If connection distance is too large, add intermediate interpolation point
            if dist > 24 * 2.0 and len(new_x) < 1000:
                mid_x = (prev_x + next_x) / 2
                mid_y = (prev_y + next_y) / 2
                mid_width = (new_width[start_idx - 1] + new_width[start_idx]) / 2

                # Insert intermediate point
                insert_x = np.concatenate([new_x[:start_idx], [mid_x], new_x[start_idx:]])
                insert_y = np.concatenate([new_y[:start_idx], [mid_y], new_y[start_idx:]])
                insert_width = np.concatenate([new_width[:start_idx], [mid_width], new_width[start_idx:]])

                new_x = insert_x
                new_y = insert_y
                new_width = insert_width

        current_x = new_x
        current_y = new_y
        current_width = new_width
        total_removed_points += points_to_remove

        print(f"    Cutoff interval [{start_idx}, {end_idx}]: removed {points_to_remove} points, current points: {len(current_x)}")

    print(f"  Total removed {total_removed_points} self-intersection points")
    return current_x, current_y, current_width


@cuda.jit
def find_all_intersection_points_kernel(centerline_x, centerline_y, grid_distance,
                                        cutoff_factor, n_points,
                                        intersection_pairs, max_intersections, intersection_count):
    """
    GPU kernel function for finding all self-intersection point pairs using parallel processing.

    This CUDA kernel implements efficient parallel distance calculations between all pairs
    of centerline points, identifying intersections based on proximity thresholds and
    storing results in shared output arrays with atomic operations for thread safety.

    Args:
        centerline_x (cp.ndarray): Centerline X-coordinate array on GPU
        centerline_y (cp.ndarray): Centerline Y-coordinate array on GPU
        grid_distance (float): Grid distance parameter for spatial resolution
        cutoff_factor (float): Cutoff factor for threshold calculation
        n_points (int): Total number of points in centerline
        intersection_pairs (cp.ndarray): Output array for intersection point pairs
        max_intersections (int): Maximum number of intersections to store
        intersection_count (cp.ndarray): Atomic counter for intersection count

    Returns:
        None: Results written directly to output arrays
    """
    i = cuda.grid(1)

    if i < n_points - 5:
        local_intersections = 0
        threshold_distance = grid_distance * cutoff_factor
        threshold_distance_sq = threshold_distance * threshold_distance

        # Check distance to all subsequent points
        for j in range(i + 3, n_points):
            dx = centerline_x[i] - centerline_x[j]
            dy = centerline_y[i] - centerline_y[j]
            cdist_sq = dx * dx + dy * dy

            if cdist_sq < threshold_distance_sq:
                # Use atomic operation to safely add to result array
                idx = cuda.atomic.add(intersection_count, 0, 1)
                if idx < max_intersections:
                    intersection_pairs[idx, 0] = i
                    intersection_pairs[idx, 1] = j
                    intersection_pairs[idx, 2] = int(cdist_sq * 1000)
                    local_intersections += 1

        # Limit number of intersections found per thread
        if local_intersections > 50:
            return