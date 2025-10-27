"""
Voxel_removes_module.py
"""

import numpy as np
from numba import cuda
import math

@cuda.jit(device=True)
def calculate_river_id_from_channel_info(icc, ic, mxc):
    """
    Calculate river ID from channel complex and channel number
    Parameters:
        icc: Channel complex number
        ic: Channel number
        mxc: Maximum number of channels
    Returns:
        int: River ID (limited to 9999)
    """
    base_id = (icc - 1) * mxc + ic
    return min(base_id, 9999)

@cuda.jit(device=True)
def extract_channel_info_from_levee_id(levee_id, mxc):
    """
    Extract channel information from levee ID
    Parameters:
        levee_id: Levee ID
        mxc: Maximum number of channels
    Returns:
        tuple: (channel complex number, channel number), returns (-1, -1) on failure
    """
    if 10000 <= levee_id < 20000:
        levee_base_id = levee_id - 10000
        mxc_multiplier = mxc * 2
        icc = levee_base_id // mxc_multiplier
        ic = (levee_base_id % mxc_multiplier) // 2
        if ic == 0:
            ic = mxc
        return icc, ic
    else:
        return -1, -1


@cuda.jit(device=True)
def extract_channel_info_from_splay_id(splay_id, crevasse_color_offset):
    """
    Extract channel information from crevasse splay ID
    Parameters:
        splay_id: Crevasse splay ID
        crevasse_color_offset: Crevasse splay color offset
    Returns:
        tuple: (channel complex number, channel number), returns (-1, -1) on failure
    """
    crevasse_color_range = 10000  # Crevasse splay color range
    if crevasse_color_offset <= splay_id < crevasse_color_offset + crevasse_color_range:
        splay_base_id = splay_id - crevasse_color_offset
        icc = splay_base_id // 100
        ic = (splay_base_id % 100) // 10
        return icc, ic
    else:
        return -1, -1

@cuda.jit(device=True)
def generate_improved_splay_id(crevasse_color_offset, icc, ic, splay_sub_idx):
    """
    Generate crevasse splay ID containing channel information
    Parameters:
        crevasse_color_offset: Crevasse splay color offset
        icc: Channel complex number
        ic: Channel number
        splay_sub_idx: Crevasse splay sub-index
    Returns:
        int: Generated crevasse splay ID
    """
    splay_id = crevasse_color_offset + icc * 100 + ic * 10 + splay_sub_idx
    return splay_id

# =============================================================================
# CUDA Kernel Function Definitions
# =============================================================================

@cuda.jit
def clear_overlying_levees_kernel(channel, nx, ny, nz, mxc, levee_color_offset,
                                 levee_color_range, cleared_count):
    """
    Clear overlying levee voxels above channels (CUDA kernel function)
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        mxc: Maximum number of channels
        levee_color_offset: Levee color offset
        levee_color_range: Levee color range
        cleared_count: Clearing counter array
    Returns:
        None: Directly modifies input array
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    # Scan vertical column from bottom to top
    for iz in range(nz):
        cell_value = channel[ix, iy, iz]

        # Check channel voxels (1-9999)
        if 0 < cell_value < levee_color_offset:
            river_id = cell_value

            # Check all layers above this channel voxel
            for upper_iz in range(iz + 1, nz):
                upper_cell_value = channel[ix, iy, upper_iz]

                # Check levee voxels (10000-19999)
                if levee_color_offset <= upper_cell_value < levee_color_offset + levee_color_range:
                    # Extract channel information from levee ID
                    levee_icc, levee_ic = extract_channel_info_from_levee_id(upper_cell_value, mxc)

                    if levee_icc >= 0 and levee_ic >= 0:
                        # Calculate source river ID of the levee
                        levee_source_river_id = calculate_river_id_from_channel_info(levee_icc, levee_ic, mxc)

                        # Clear levee if current river ID > levee source river ID
                        if river_id > levee_source_river_id:
                            channel[ix, iy, upper_iz] = 0
                            cuda.atomic.add(cleared_count, 0, 1)

                # Stop upward checking when encountering other channel voxels
                elif 0 < upper_cell_value < levee_color_offset:
                    break

@cuda.jit
def clear_overlying_crevasse_kernel(channel, nx, ny, nz, crevasse_color_offset,
                                   crevasse_color_range, cleared_count):
    """
    Clear overlying crevasse splay voxels above channels (CUDA kernel function)
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        crevasse_color_offset: Crevasse splay color offset
        crevasse_color_range: Crevasse splay color range
        cleared_count: Clearing counter array
    Returns:
        None: Directly modifies input array
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    # Scan vertical column from bottom to top
    for iz in range(nz):
        cell_value = channel[ix, iy, iz]

        # Check channel voxels (1-9999)
        if 0 < cell_value < 10000:
            river_id = cell_value

            # Check all layers above this channel voxel
            for upper_iz in range(iz + 1, nz):
                upper_cell_value = channel[ix, iy, upper_iz]

                # Check crevasse splay voxels (20000-29999)
                if crevasse_color_offset <= upper_cell_value < crevasse_color_offset + crevasse_color_range:
                    # Extract channel information from crevasse splay ID
                    splay_icc, splay_ic = extract_channel_info_from_splay_id(upper_cell_value, crevasse_color_offset)

                    if splay_icc >= 0 and splay_ic >= 0:
                        # Calculate source river ID of the crevasse splay
                        splay_source_river_id = calculate_river_id_from_channel_info(splay_icc, splay_ic, 10)

                        # Clear crevasse splay if current river ID > splay source river ID
                        if river_id > splay_source_river_id:
                            channel[ix, iy, upper_iz] = 0
                            cuda.atomic.add(cleared_count, 0, 1)

                # Stop upward checking when encountering other channel voxels
                elif 0 < upper_cell_value < 10000:
                    break

@cuda.jit
def find_max_channel_id_kernel(channel, nx, ny, nz, levee_color_offset, max_id):
    """
    Find maximum channel ID (CUDA kernel function)
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        levee_color_offset: Levee color offset
        max_id: Maximum ID storage array
    Returns:
        None: Result stored in max_id array
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    for iz in range(nz):
        cell_value = channel[ix, iy, iz]
        # Check main channel ID (1-9999)
        if 0 < cell_value < levee_color_offset:
            cuda.atomic.max(max_id, 0, cell_value)

# =============================================================================
# Main Interface Function Definitions
# =============================================================================

def clear_overlying_levees_for_pre_channel_id(channel, nx, ny, nz, mxc,
                                              levee_color_offset=10000, levee_color_range=10000):
    """
    Clear overlying levee voxels above channels
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        mxc: Maximum number of channels
        levee_color_offset: Levee color offset (default 10000)
        levee_color_range: Levee color range (default 10000)
    Returns:
        tuple: (maximum channel ID, number of cleared voxels)
    """
    print("Starting levee voxel clearing")

    # Transfer data to GPU
    d_channel = cuda.to_device(channel)
    d_cleared_count = cuda.to_device(np.array([0], dtype=np.int32))
    d_max_id = cuda.to_device(np.array([0], dtype=np.int32))

    # Configure GPU execution parameters
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(nx / threads_per_block[0])
    blocks_per_grid_y = math.ceil(ny / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Find maximum channel ID
    find_max_channel_id_kernel[blocks_per_grid, threads_per_block](
        d_channel, nx, ny, nz, levee_color_offset, d_max_id)

    max_channel_id = d_max_id.copy_to_host()[0]

    if max_channel_id == 0:
        print("No channel voxels found")
        return 0, 0

    print(f"Maximum channel ID: {max_channel_id}")

    # Execute clearing kernel
    clear_overlying_levees_kernel[blocks_per_grid, threads_per_block](
        d_channel, nx, ny, nz, mxc, levee_color_offset, levee_color_range, d_cleared_count)

    # Copy results back to CPU
    channel[:] = d_channel.copy_to_host()
    cleared_count = d_cleared_count.copy_to_host()[0]

    print(f"Clearing complete, cleared {cleared_count} levee voxels")
    return max_channel_id, cleared_count

def clear_overlying_crevasse_for_pre_channel_id(channel, nx, ny, nz,
                                               crevasse_color_offset=20000, crevasse_color_range=10000):
    """
    Clear overlying crevasse splay voxels above channels
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        crevasse_color_offset: Crevasse splay color offset (default 20000)
        crevasse_color_range: Crevasse splay color range (default 10000)
    Returns:
        tuple: (maximum channel ID, number of cleared voxels)
    """
    print("Starting crevasse splay voxel clearing")

    # Transfer data to GPU
    d_channel = cuda.to_device(channel)
    d_cleared_count = cuda.to_device(np.array([0], dtype=np.int32))
    d_max_id = cuda.to_device(np.array([0], dtype=np.int32))

    # Configure GPU execution parameters
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(nx / threads_per_block[0])
    blocks_per_grid_y = math.ceil(ny / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Find maximum channel ID
    find_max_channel_id_kernel[blocks_per_grid, threads_per_block](
        d_channel, nx, ny, nz, 10000, d_max_id)

    max_channel_id = d_max_id.copy_to_host()[0]

    if max_channel_id == 0:
        print("No channel voxels found")
        return 0, 0

    print(f"Maximum channel ID: {max_channel_id}")

    # Execute clearing kernel
    clear_overlying_crevasse_kernel[blocks_per_grid, threads_per_block](
        d_channel, nx, ny, nz, crevasse_color_offset, crevasse_color_range, d_cleared_count)

    # Copy results back to CPU
    channel[:] = d_channel.copy_to_host()
    cleared_count = d_cleared_count.copy_to_host()[0]

    print(f"Clearing complete, cleared {cleared_count} crevasse splay voxels")
    return max_channel_id, cleared_count


@cuda.jit
def clear_overlying_levees_above_all_splays_kernel(channel, nx, ny, nz,
                                                   crevasse_color_offset, crevasse_color_range,
                                                   levee_color_offset, levee_color_range,
                                                   cleared_count):
    """
    Clear natural levee voxels overlying all crevasse splays (CUDA kernel function)
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        crevasse_color_offset: Crevasse splay color offset
        crevasse_color_range: Crevasse splay color range
        levee_color_offset: Natural levee color offset
        levee_color_range: Natural levee color range
        cleared_count: Clearing counter array
    Returns:
        None: Directly modifies input array
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    # Scan vertical column from bottom to top
    for iz in range(nz):
        cell_value = channel[ix, iy, iz]

        # Check crevasse splay voxels
        if crevasse_color_offset <= cell_value < crevasse_color_offset + crevasse_color_range:
            # Found crevasse splay voxel, check all layers above it
            for upper_iz in range(iz + 1, nz):
                upper_cell_value = channel[ix, iy, upper_iz]

                # Check natural levee voxels (10000-19999)
                if levee_color_offset <= upper_cell_value < levee_color_offset + levee_color_range:
                    # Clear natural levee voxel
                    channel[ix, iy, upper_iz] = 0
                    cuda.atomic.add(cleared_count, 0, 1)

                # Note: Continue searching upward without stopping



def clear_overlying_levees_above_all_splays(channel, nx, ny, nz,
                                            crevasse_color_offset=20000, crevasse_color_range=10000,
                                            levee_color_offset=10000, levee_color_range=10000):
    """
    Clear natural levee voxels overlying all crevasse splays
    Parameters:
        channel: 3D channel array
        nx, ny, nz: Array dimensions
        crevasse_color_offset: Crevasse splay color offset (default 20000)
        crevasse_color_range: Crevasse splay color range (default 10000)
        levee_color_offset: Natural levee color offset (default 10000)
        levee_color_range: Natural levee color range (default 10000)
    Returns:
        int: Number of cleared natural levee voxels
    """
    print("Starting clearing of natural levee voxels above all crevasse splays")

    # Transfer data to GPU
    d_channel = cuda.to_device(channel)
    d_cleared_count = cuda.to_device(np.array([0], dtype=np.int32))

    # Configure GPU execution parameters
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(nx / threads_per_block[0])
    blocks_per_grid_y = math.ceil(ny / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Execute clearing kernel
    clear_overlying_levees_above_all_splays_kernel[blocks_per_grid, threads_per_block](
        d_channel, nx, ny, nz,
        crevasse_color_offset, crevasse_color_range,
        levee_color_offset, levee_color_range,
        d_cleared_count)

    # Copy results back to CPU
    channel[:] = d_channel.copy_to_host()
    cleared_count = d_cleared_count.copy_to_host()[0]

    print(f"Natural levee clearing above crevasse splays complete, cleared {cleared_count} natural levee voxels")
    return cleared_count