"""
render_channel.py
"""
import numpy as np
import time
import cupy as cp
from numba import cuda
import math
_GPU_AVAILABLE = True
import warnings
from numba.cuda.dispatcher import NumbaPerformanceWarning
from .render_facies import render_levees_batch_gpu, render_crevasse_splays_batch_gpu, gpu_distance_to_line_segment
from ..utils.utils import gpu_random, get_value_from_distribution
from .Voxel_removes_module import clear_overlying_levees_for_pre_channel_id, clear_overlying_crevasse_for_pre_channel_id, clear_overlying_levees_above_all_splays


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
# Constants definition
EPSLON = 1.0e-10
DEG2RAD = 0.017453293  # pi/180

class ChannelGpuData:
    """GPU channel data structure"""

    def __init__(self, ncc, nc, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos):
        """
        Initialize GPU channel data structure
        Args:
            ncc: Number of channel complexes
            nc: Array of channel counts for each complex
            ccx, ccz, cco: Complex center coordinates and angles
            cx, cz: Channel center coordinates
            ccl, ct, cw, crelpos: Channel centerline, thickness, width and relative position data
        Returns:
            None: No return value
        """
        # Copy data
        self.ncc = ncc
        self.nc = np.copy(nc)
        self.ccx = np.copy(ccx)
        self.ccz = np.copy(ccz)
        self.cco = np.copy(cco)
        self.cx = np.copy(cx)
        self.cz = np.copy(cz)
        self.ccl = np.copy(ccl)
        self.ct = np.copy(ct)
        self.cw = np.copy(cw)
        self.crelpos = np.copy(crelpos)

        # Generate channel index information
        self.channels = []
        channel_idx = 0
        for icc in range(1, ncc + 1):
            for ic in range(1, nc[icc-1] + 1):
                self.channels.append({
                    'icc': icc,
                    'ic': ic,
                    'idx': channel_idx
                })
                channel_idx += 1

        self.total_channels = channel_idx

    def transfer_to_gpu(self):
        """
        Transfer data to GPU
        Args:
            No arguments
        Returns:
            None: No return value
        """
        # Create GPU array versions
        self.d_nc = cp.asarray(self.nc)
        self.d_ccx = cp.asarray(self.ccx)
        self.d_ccz = cp.asarray(self.ccz)
        self.d_cco = cp.asarray(self.cco)
        self.d_cx = cp.asarray(self.cx)
        self.d_cz = cp.asarray(self.cz)
        self.d_ccl = cp.asarray(self.ccl)
        self.d_ct = cp.asarray(self.ct)
        self.d_cw = cp.asarray(self.cw)
        self.d_crelpos = cp.asarray(self.crelpos)

        # Prepare channel information for CUDA kernels
        channels_data = []
        for ch in self.channels:
            channels_data.append((ch['icc'], ch['ic'], ch['idx']))

        self.d_channels = cp.asarray(channels_data)



def rasterc_with_gpu(ncc, nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
                     ccx, ccz, cco, cct, ccw, cx, cz, nc,
                     channel, por, ccl, ct, cw, crelpos, pcurvea, pmapa,
                     chanon, getpor, params=None, is_final_iteration=False):
    """
    GPU-accelerated channel rasterization
    Args:
        ncc, nc: Number of channel complexes and channel count per complex
        nx, ny, nz: Grid dimensions
        xmn, ymn, zmn: Grid origin coordinates
        xsiz, ysiz, zsiz: Grid spacing
        ccx, ccz, cco, cct, ccw: Complex parameters
        cx, cz: Channel coordinates
        channel, por: Output grids
        ccl, ct, cw, crelpos: Channel geometry parameters
        pcurvea, pmapa: Proportion curves and area maps
        chanon: Channel switch array
        getpor: Whether to compute porosity
        params: Global parameter dictionary
        is_final_iteration: Whether this is the final iteration
    Returns:
        float: Global channel proportion
    """
    start_time = time.time()

    # Channel trend processing
    if params and params.get('enable_river_trends', False):
        z_downstream_trend = params.get('z_downstream_trend', 0.0)
        if z_downstream_trend != 0.0:
            print(f"Applying channel Z downstream trend {z_downstream_trend}")
            if 'centerline_z_offset' not in params:
                ny3 = ccl.shape[2]
                params['centerline_z_offset'] = apply_z_downstream_trends_gpu(
                    params, ccl, cz, params['ysiz'], ny3
                )

    if params is None:
        params = {}

    if 'centerline_z_offset' not in params or params['centerline_z_offset'] is None:
        ny3 = ccl.shape[2]
        max_channels = max(nc) if len(nc) > 0 else 1
        params['centerline_z_offset'] = np.zeros((ncc, max_channels, ny3), dtype=np.float32)
        print("Creating zero-offset centerline_z_offset array")

    enable_migration = params.get('enable_migration', False) if params else False
    print(f"GPU rasterization: Migration algorithm {'enabled' if enable_migration else 'disabled'}")

    print(f"Processing {ncc} complexes, {sum(nc)} channels, grid {nx}x{ny}x{nz}")

    # Initialize output grids
    channel.fill(0)
    if getpor:
        por.fill(-9.0)

    # Build channel index mapping
    global_channel_map = {}
    enabled_channels = []
    global_id = 1

    original_channels = []
    migration_channels = []

    channel_idx = 0
    for icc in range(1, ncc + 1):
        for ic in range(1, nc[icc - 1] + 1):
            if chanon[channel_idx]:
                original_channel_count = min(params.get('mxc', 10), nc[icc - 1])
                if ic <= original_channel_count:
                    original_channels.append((icc, ic, channel_idx))
                else:
                    migration_channels.append((icc, ic, channel_idx))
            channel_idx += 1

    print(f"Original channels {len(original_channels)}, migration channels {len(migration_channels)}")

    # Allocate IDs
    for icc, ic, chan_idx in original_channels:
        global_channel_map[(icc, ic)] = global_id
        enabled_channels.append((icc, ic, chan_idx, global_id))
        global_id += 1

    for icc, ic, chan_idx in migration_channels:
        global_channel_map[(icc, ic)] = global_id
        enabled_channels.append((icc, ic, chan_idx, global_id))
        global_id += 1

    if not enabled_channels:
        print("No enabled channels")
        return 0.0

    # Build channel metadata dictionary
    channel_metadata = {}
    for icc, ic, chan_idx, global_id in enabled_channels:
        channel_metadata[global_id] = {
            'icc': icc,
            'ic': ic,
            'z_top': cz[icc - 1, ic - 1],
            'global_id': global_id,
            'is_migration': (icc, ic, chan_idx) in migration_channels
        }

    params['channel_metadata'] = channel_metadata
    params['enable_migration'] = enable_migration

    # Build channel priority arrays
    channel_z_tops = np.zeros(len(enabled_channels), dtype=np.float32)
    channel_priorities = np.zeros(len(enabled_channels), dtype=np.int32)

    for idx, (icc, ic, chan_idx, global_id) in enumerate(enabled_channels):
        channel_z_tops[idx] = cz[icc - 1, ic - 1]
        channel_priorities[idx] = global_id

    # Channel porosity parameters
    use_channel_specific_porosity = params.get('use_channel_specific_porosity', False)
    d_centerline_z_offset = cuda.to_device(params['centerline_z_offset'].astype(np.float32))

    if use_channel_specific_porosity:
        max_channels = max(nc) if len(nc) > 0 else 1
        por_base = np.full((ncc, max_channels), 0.20)
        por_range = np.full((ncc, max_channels), 0.07)



        fcpor_base = params.get('fcpor_base', [0.15, 0.20, 0.25])
        fcpor_range = params.get('fcpor_range', [0.05, 0.07, 0.10])

        for icc in range(1, ncc + 1):
            for ic in range(1, nc[icc - 1] + 1):
                por_base[icc - 1, ic - 1] = get_value_from_distribution(fcpor_base)
                por_range[icc - 1, ic - 1] = get_value_from_distribution(fcpor_range)
    else:
        max_channels = max(nc) if len(nc) > 0 else 1
        por_base = np.full((ncc, max_channels), 0.20)
        por_range = np.full((ncc, max_channels), 0.07)

    try:
        print("Starting GPU rendering")

        # Stage 1: Precompute distance matrix
        print("Precomputing distance matrix")

        num_channels = len(enabled_channels)
        distance_matrix_size = nx * ny * num_channels * 3 * 4
        distance_matrix_size_mb = distance_matrix_size / (1024 * 1024)
        print(f"Distance matrix memory {distance_matrix_size_mb:.1f} MB")

        distance_matrix = np.zeros((nx, ny, num_channels, 3), dtype=np.float32)
        channel_data = np.array([(icc, ic, chan_idx) for icc, ic, chan_idx, _ in enabled_channels],
                                dtype=np.int32)

        # Transfer data to GPU
        d_channel_data = cuda.to_device(channel_data)
        d_distance_matrix = cuda.to_device(distance_matrix)
        d_ccx = cuda.to_device(ccx.astype(np.float32))
        d_ccz = cuda.to_device(ccz.astype(np.float32))
        d_cco = cuda.to_device(cco.astype(np.float32))
        d_cx = cuda.to_device(cx.astype(np.float32))
        d_cz = cuda.to_device(cz.astype(np.float32))
        d_ccl = cuda.to_device(ccl.astype(np.float32))

        if 'centerline_y_data' in params and params['centerline_y_data'] is not None:
            d_ccy = cuda.to_device(params['centerline_y_data'].astype(np.float32))
        else:
            d_ccy = None

        xmn_f32 = np.float32(xmn)
        ymn_f32 = np.float32(ymn)
        xsiz_f32 = np.float32(xsiz)
        ysiz_f32 = np.float32(ysiz)

        # Precompute distance matrix
        threads_per_block_x = 8
        threads_per_block_y = 8
        threads_per_block_z = 4

        blocks_per_grid_x = (nx + threads_per_block_x - 1) // threads_per_block_x
        blocks_per_grid_y = (ny + threads_per_block_y - 1) // threads_per_block_y
        blocks_per_grid_z = (num_channels + threads_per_block_z - 1) // threads_per_block_z

        threads_per_block = (threads_per_block_x, threads_per_block_y, threads_per_block_z)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        ny3 = ccl.shape[2]

        precompute_distance_matrix_kernel[blocks_per_grid, threads_per_block](
            d_channel_data, nx, ny, xmn_f32, ymn_f32, xsiz_f32, ysiz_f32,
            d_ccx, d_ccz, d_cco, d_cx, d_cz, d_ccl, d_ccy, ny3,
            d_distance_matrix
        )

        cuda.synchronize()

        distance_computation_time = time.time() - start_time
        print(f"Distance matrix precomputation completed, elapsed time {distance_computation_time:.3f}s")

        # Stage 2: Count candidates
        print("Counting candidates")

        total_voxels = nx * ny * nz
        candidate_counts = np.zeros(total_voxels, dtype=np.int32)
        d_candidate_counts = cuda.to_device(candidate_counts)

        # Transfer data to GPU
        d_ct = cuda.to_device(ct.astype(np.float32))
        d_cw = cuda.to_device(cw.astype(np.float32))
        d_crelpos = cuda.to_device(crelpos.astype(np.float32))
        d_channel_z_tops = cuda.to_device(channel_z_tops)
        d_channel_priorities = cuda.to_device(channel_priorities)
        d_por_base = cuda.to_device(por_base.astype(np.float32))
        d_por_range = cuda.to_device(por_range.astype(np.float32))

        zmn_f32 = np.float32(zmn)
        zsiz_f32 = np.float32(zsiz)

        # Configure 2D grid for candidate counting
        threads_per_block_x = 16
        threads_per_block_y = 16
        blocks_per_grid_x = (nx + threads_per_block_x - 1) // threads_per_block_x
        blocks_per_grid_y = (ny + threads_per_block_y - 1) // threads_per_block_y

        threads_per_block = (threads_per_block_x, threads_per_block_y)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        candidate_counting_kernel_with_precomputed_distance[blocks_per_grid, threads_per_block](
            d_channel_data, nx, ny, nz, xmn_f32, ymn_f32, zmn_f32, xsiz_f32, ysiz_f32, zsiz_f32,
            d_ccx, d_ccz, d_cco, d_cx, d_cz, d_ccl, d_ccy, d_ct, d_cw, d_crelpos,
            d_por_base, d_por_range, ny3,
            enable_migration, d_channel_z_tops, d_channel_priorities,
            d_candidate_counts, d_centerline_z_offset, d_distance_matrix
        )

        cuda.synchronize()

        # Transfer candidate counts back to CPU
        d_candidate_counts.copy_to_host(candidate_counts)

        # Statistics results
        total_candidates = np.sum(candidate_counts)
        voxels_with_candidates = np.count_nonzero(candidate_counts)
        avg_candidates_per_voxel = total_candidates / max(voxels_with_candidates, 1)
        max_candidates_in_voxel = np.max(candidate_counts)

        print(f"Candidate statistics: total {total_candidates:,}, active voxels {voxels_with_candidates:,}/{total_voxels:,}")
        print(f"Average per voxel {avg_candidates_per_voxel:.2f}, maximum single voxel {max_candidates_in_voxel}")

        if total_candidates == 0:
            print("No candidates found")
            return 0.0

        # Stage 3: Create compressed sparse format candidate buffer
        print("Creating candidate buffer")

        # Calculate starting offset for each voxel in the candidate buffer
        candidate_offsets = np.zeros(total_voxels + 1, dtype=np.int32)
        candidate_offsets[1:] = np.cumsum(candidate_counts)

        # Create candidate buffer
        candidate_buffer = np.zeros((total_candidates, 4), dtype=np.float32)
        candidate_current_pos = np.zeros(total_voxels, dtype=np.int32)

        # Calculate memory usage
        old_memory_mb = total_voxels * 50 * 4 * 4 / (1024 * 1024)
        new_memory_mb = total_candidates * 4 * 4 / (1024 * 1024)
        memory_savings = old_memory_mb - new_memory_mb

        print(f"Candidate buffer {new_memory_mb:.1f} MB, saved {memory_savings:.1f} MB")

        # Transfer to GPU
        d_candidate_buffer = cuda.to_device(candidate_buffer)
        d_candidate_offsets = cuda.to_device(candidate_offsets)
        d_candidate_current_pos = cuda.to_device(candidate_current_pos)

        # Stage 4: Collect candidates
        print("Candidate collection")

        random_seed = int(time.time() * 1000) % 10000000

        candidate_collection_kernel_dynamic_with_precomputed_distance[blocks_per_grid, threads_per_block](
            d_channel_data, nx, ny, nz, xmn_f32, ymn_f32, zmn_f32, xsiz_f32, ysiz_f32, zsiz_f32,
            d_ccx, d_ccz, d_cco, d_cx, d_cz, d_ccl, d_ccy, d_ct, d_cw, d_crelpos,
            d_por_base, d_por_range, random_seed, ny3,
            enable_migration, d_channel_z_tops, d_channel_priorities,
            d_candidate_buffer, d_candidate_offsets, d_candidate_current_pos,
            d_centerline_z_offset, d_distance_matrix
        )

        cuda.synchronize()

        # Verify candidate collection results
        d_candidate_current_pos.copy_to_host(candidate_current_pos)
        collected_candidates = np.sum(candidate_current_pos)
        print(f"Collected candidates {collected_candidates:,}/{total_candidates:,}")

        # Stage 5: Conflict resolution
        print("Conflict resolution")

        # Create output grid
        d_output_grid = cuda.to_device(channel)
        if getpor:
            d_output_por = cuda.to_device(por)
        else:
            d_output_por = cuda.to_device(np.zeros((1, 1, 1), dtype=np.float32))

        # Configure voxel-parallel CUDA grid
        voxel_threads_per_block = 256
        voxel_blocks_per_grid = (total_voxels + voxel_threads_per_block - 1) // voxel_threads_per_block

        conflict_resolution_kernel_dynamic[voxel_blocks_per_grid, voxel_threads_per_block](
            d_candidate_buffer, d_candidate_offsets, d_candidate_counts,
            d_output_grid, d_output_por, getpor,
            nx, ny, nz, enable_migration
        )

        cuda.synchronize()

        # Transfer results back to CPU
        d_output_grid.copy_to_host(channel)
        if getpor:
            d_output_por.copy_to_host(por)

        print(f"GPU rendering completed, elapsed time {time.time() - start_time:.3f}s")

    except Exception as e:
        print(f"GPU rendering failed: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to CPU rendering")

    # Levee rendering
    if params.get('levee_enabled', False) and 'levees_data' in params:
        levees_data = params['levees_data']

        # Levee data validation
        valid_levees = []
        valid_count = 0

        print(f"Validating {len(levees_data)} levee data entries")

        for i, levee in enumerate(levees_data):
            if 'icc' in levee and 'ic' in levee and 'left_edge' in levee and 'right_edge' in levee and 'height' in levee:
                non_zero_left = np.count_nonzero(levee['left_edge'])
                non_zero_right = np.count_nonzero(levee['right_edge'])
                non_zero_height = np.count_nonzero(levee['height'])

                if non_zero_left > 0 and non_zero_right > 0:
                    # Fix if height is all zeros
                    if non_zero_height == 0:
                        avg_width = np.mean(np.abs(np.array(levee['left_edge']))) + np.mean(
                            np.abs(np.array(levee['right_edge'])))
                        default_height = max(0.5, avg_width * 0.1)
                        levee['height'] = np.ones_like(levee['height']) * default_height

                    valid_levees.append(levee)
                    valid_count += 1

        print(f"Valid levee count {valid_count}/{len(levees_data)}")

        # Render levees
        if valid_count > 0:
            params['levees_data'] = valid_levees
            print(f"Levee rendering: {valid_count} natural levees")

            total_filled = render_levees_batch_gpu(
                valid_levees, 10000,
                nx, ny, nz, cz, cx, xmn, ccx, ct, ymn, cco, zmn, xsiz, ysiz, zsiz,
                channel, por, getpor, nc,
                channel_metadata=params.get('channel_metadata'),
                enable_migration=params.get('enable_migration', False)
            )

            print(f"Levee rendering completed, filled {total_filled} cells")
        else:
            print("No valid levee data")


        # Levee cleanup in migration mode
        enable_migration = params.get('enable_migration', False) if params else False
        if enable_migration:
            mxc = nc.max() if len(nc) > 0 else 1
            max_channel_id, cleared_count = clear_overlying_levees_for_pre_channel_id(
                channel, nx, ny, nz,mxc,
                levee_color_offset=10000,
                levee_color_range=10000
            )

            if cleared_count > 0:
                print(f"Channel migration cleanup: target channel ID {max_channel_id}, cleared levee voxels {cleared_count}")




    # Handle crevasse splays
    if params.get('crevasse_enabled', False) and 'crevasse_data' in params:
        crevasse_data = params['crevasse_data']
        print(f"Rendering {len(crevasse_data)} crevasse splays")

        total_filled = render_crevasse_splays_batch_gpu(
            crevasse_data, 20000,
            nx, ny, nz, cz, cx, xmn, ccx, ct, ymn, cco, zmn, xsiz, ysiz, zsiz,
            channel, por, getpor, nc,
            channel_metadata=params.get('channel_metadata'),
            enable_migration=params.get('enable_migration', False)
        )

        print(f"Crevasse splay rendering completed, filled {total_filled} cells")

        # Crevasse splay cleanup in migration mode
        enable_migration = params.get('enable_migration', False) if params else False
        if enable_migration:
            mxc = nc.max() if len(nc) > 0 else 1
            max_channel_id_crevasse, cleared_crevasse_count = clear_overlying_crevasse_for_pre_channel_id(
                channel, nx, ny, nz,
                crevasse_color_offset=20000,
                crevasse_color_range=10000
            )

            if cleared_crevasse_count > 0:
                print(f"Channel migration crevasse cleanup: cleared crevasse voxels {cleared_crevasse_count}")
            splay_levee_cleared_count = clear_overlying_levees_above_all_splays(
                channel, nx, ny, nz,
                crevasse_color_offset=20000,
                crevasse_color_range=10000,
                levee_color_offset=10000,
                levee_color_range=10000
            )

            if splay_levee_cleared_count > 0:
                print(f"Levee cleanup above crevasse splays: cleared {splay_levee_cleared_count} levee voxels")


    # Calculate proportions
    channel_binary = (channel > 0).astype(np.float64)
    pcurvea[:] = np.mean(channel_binary, axis=(0, 1))
    pmapa[:] = np.mean(channel_binary, axis=2)
    prop = float(np.mean(channel_binary))

    total_time = time.time() - start_time
    print(f"Rasterization completed, total elapsed time {total_time:.3f}s")

    # Rendering result verification
    channel_cells = np.count_nonzero((channel > 0) & (channel < 10000))
    print(f"Main channel cells {channel_cells}, global proportion {prop:.6f}")

    # Verify priority correctness
    if enable_migration and len(migration_channels) > 0:
        migration_ids = set(enabled_channels[i][3] for i in range(len(original_channels), len(enabled_channels)))
        rendered_migration_ids = set()

        for migration_id in migration_ids:
            if np.any(channel == migration_id):
                rendered_migration_ids.add(migration_id)

        coverage_ratio = len(rendered_migration_ids) / len(migration_ids) if migration_ids else 0
        print(f"Migration channel rendering verification: {len(rendered_migration_ids)}/{len(migration_ids)} ({coverage_ratio:.1%})")

    return prop

def apply_z_downstream_trends_gpu(params, ccl, cz, ysiz, ny3):
    """
    GPU version of channel Z downstream trend application
    Args:
        params: Global parameter dictionary
        ccl: Channel centerline array
        cz: Channel Z position array
        ysiz: Y-direction grid spacing
        ny3: Number of centerline points
    Returns:
        ndarray: Z offset array, returns None if trend is 0
    """
    z_downstream_trend = params.get('z_downstream_trend', 0.0)

    if z_downstream_trend == 0.0:
        return None

    print(f"Applying Z downstream trend: {z_downstream_trend}")

    # Create Z offset array
    ncc = ccl.shape[0]
    mxc = ccl.shape[1]
    centerline_z_offset = np.zeros((ncc, mxc, ny3), dtype=np.float32)

    # Calculate total channel length
    total_length = ny3 * ysiz

    # Calculate Z offset for each channel
    for icc in range(ncc):
        for ic in range(mxc):
            for iy in range(ny3):
                # Calculate relative position
                relative_position = iy / max(1, ny3 - 1)

                # Calculate Z offset
                z_offset = -z_downstream_trend * relative_position * total_length

                centerline_z_offset[icc, ic, iy] = z_offset

    return centerline_z_offset

@cuda.jit
def precompute_distance_matrix_kernel(channel_indices, nx, ny, xmn, ymn, xsiz, ysiz,
                                      ccx, ccz, cco, cx, cz, ccl, ccy, ny3,
                                      distance_matrix):
    """
    Precompute distance matrix from grid points to channel centerlines
    Args:
        channel_indices: Channel index array
        nx, ny: Grid X and Y dimensions
        xmn, ymn: Grid origin coordinates
        xsiz, ysiz: Grid spacing
        ccx, ccz, cco: Complex center coordinates and angles
        cx, cz: Channel center coordinates
        ccl: Channel centerline array
        ccy: Channel Y coordinate array
        ny3: Number of centerline points
        distance_matrix: Output distance matrix
    Returns:
        None: GPU kernel has no return value
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    channel_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    if ix >= nx or iy >= ny or channel_idx >= len(channel_indices):
        return

    # Precompute common values
    world_x = xmn + (ix + 0.5) * xsiz
    world_y = ymn + (iy + 0.5) * ysiz

    # Get current channel information
    icc, ic, chan_idx = channel_indices[channel_idx]

    # Precompute channel transformation parameters
    global_angle = cco[icc - 1] * 0.017453293
    x_center = ccx[icc - 1] + cx[icc - 1, ic - 1]

    # Precompute trigonometric values
    cos_angle = math.cos(global_angle)
    sin_angle = math.sin(global_angle)

    # Initialize best values
    min_dist_sq = 1e30
    closest_segment = 0
    closest_t = 0.0

    # Get actual steps
    actual_steps = min(ny3, ccl.shape[2])

    # Loop unrolling, process 8 segments at a time
    for i in range(0, actual_steps - 1, 8):
        for j in range(8):
            segment_idx = i + j
            if segment_idx >= actual_steps - 1:
                break

            # Get segment endpoint coordinates
            local_x1 = ccl[icc - 1, ic - 1, segment_idx]
            local_x2 = ccl[icc - 1, ic - 1, segment_idx + 1]

            # Data validity check
            if abs(local_x1) > 1e6 or abs(local_x2) > 1e6:
                continue

            # Get Y coordinates
            local_y1, local_y2 = (ccy[icc - 1, ic - 1, segment_idx],
                                  ccy[icc - 1, ic - 1, segment_idx + 1]) if ccy is not None else \
                (ymn + segment_idx * ysiz,
                 ymn + (segment_idx + 1) * ysiz)

            # Y coordinate validity check
            if ccy is not None and (abs(local_y1) > 1e6 or abs(local_y2) > 1e6):
                continue

            # Coordinate transformation
            y_offset_1 = local_y1 - ymn
            y_offset_2 = local_y2 - ymn

            global_x1 = x_center + local_x1 * cos_angle - y_offset_1 * sin_angle
            global_y1 = y_offset_1 * cos_angle + local_x1 * sin_angle
            global_x2 = x_center + local_x2 * cos_angle - y_offset_2 * sin_angle
            global_y2 = y_offset_2 * cos_angle + local_x2 * sin_angle

            # Point to line segment distance calculation
            dx = global_x2 - global_x1
            dy = global_y2 - global_y1
            seg_len_sq = dx * dx + dy * dy

            if seg_len_sq < 1e-10:
                # Degenerate case: segment becomes a point
                dist_sq = (world_x - global_x1) * (world_x - global_x1) + \
                          (world_y - global_y1) * (world_y - global_y1)

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_segment = segment_idx
                    closest_t = 0.0
                continue

            # Projection calculation
            dot_product = (world_x - global_x1) * dx + (world_y - global_y1) * dy
            t = dot_product / seg_len_sq

            # Range clamping
            t = max(0.0, min(1.0, t))

            # Closest point calculation
            closest_x = global_x1 + t * dx
            closest_y = global_y1 + t * dy

            # Distance squared calculation
            diff_x = world_x - closest_x
            diff_y = world_y - closest_y
            dist_sq = diff_x * diff_x + diff_y * diff_y

            # Update minimum
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_segment = segment_idx
                closest_t = t

    # Compute results
    min_distance = math.sqrt(min_dist_sq)

    # Store results
    distance_matrix[ix, iy, channel_idx, 0] = min_distance
    distance_matrix[ix, iy, channel_idx, 1] = closest_segment
    distance_matrix[ix, iy, channel_idx, 2] = closest_t

@cuda.jit
def candidate_counting_kernel_with_precomputed_distance(
        channel_indices, nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
        ccx, ccz, cco, cx, cz, ccl, ccy, ct, cw, crelpos,
        por_base, por_range, ny3,
        enable_migration, channel_z_tops, channel_priorities,
        candidate_counts, centerline_z_offset, distance_matrix):
    """
    Candidate counting kernel function
    Args:
        channel_indices: Channel index array
        nx, ny, nz: Grid dimensions
        xmn, ymn, zmn: Grid origin coordinates
        xsiz, ysiz, zsiz: Grid spacing
        ccx, ccz, cco: Complex center coordinates and angles
        cx, cz: Channel center coordinates
        ccl: Channel centerline array
        ccy: Channel Y coordinate array
        ct, cw, crelpos: Channel thickness, width, and relative position
        por_base, por_range: Porosity base and range values
        ny3: Number of centerline points
        enable_migration: Whether migration mode is enabled
        channel_z_tops: Channel top Z values array
        channel_priorities: Channel priority array
        candidate_counts: Candidate count array
        centerline_z_offset: Centerline Z offset array
        distance_matrix: Precomputed distance matrix
    Returns:
        None: GPU kernel has no return value
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    # Use precomputed distance information for each channel
    for channel_idx in range(len(channel_indices)):
        icc, ic, chan_idx = channel_indices[channel_idx]
        current_priority = channel_priorities[channel_idx]

        base_z_top = cz[icc - 1, ic - 1]

        # Get distance information from distance matrix
        min_distance = distance_matrix[ix, iy, channel_idx, 0]
        closest_segment = int(distance_matrix[ix, iy, channel_idx, 1])
        closest_t = distance_matrix[ix, iy, channel_idx, 2]

        # Interpolate channel parameters
        thickness, width, relpos = interpolate_channel_parameters(
            closest_segment, closest_t, ct, cw, crelpos, ny3, icc, ic
        )

        # Parameter validity check
        if thickness <= 0.0 or width <= 0.0:
            continue

        # Get Z offset
        z_offset = 0.0
        if centerline_z_offset is not None and closest_segment < ny3:
            z_offset1 = centerline_z_offset[icc - 1, ic - 1, closest_segment]
            if closest_segment + 1 < ny3:
                z_offset2 = centerline_z_offset[icc - 1, ic - 1, closest_segment + 1]
                z_offset = z_offset1 + closest_t * (z_offset2 - z_offset1)
            else:
                z_offset = z_offset1

        current_z_top = base_z_top + z_offset

        # Use distance field to determine if within channel
        river_thickness_at_point = channel_cross_section_distance_field(
            min_distance, width, thickness, relpos
        )

        if river_thickness_at_point <= 0.0:
            continue

        # Calculate Z direction range
        z_bottom = current_z_top - river_thickness_at_point
        z_top_calc = current_z_top

        # Convert to grid indices
        iz_bottom = int((z_bottom - zmn) / zsiz)
        iz_top = int((z_top_calc - zmn) / zsiz)

        # Boundary check
        iz_bottom = max(0, min(nz - 1, iz_bottom))
        iz_top = max(0, min(nz - 1, iz_top))

        # Count candidates in Z direction
        for iz in range(iz_bottom, iz_top + 1):
            voxel_idx = iz * nx * ny + iy * nx + ix
            cuda.atomic.add(candidate_counts, voxel_idx, 1)

@cuda.jit
def candidate_collection_kernel_dynamic_with_precomputed_distance(
        channel_indices, nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
        ccx, ccz, cco, cx, cz, ccl, ccy, ct, cw, crelpos,
        por_base, por_range, random_seed, ny3,
        enable_migration, channel_z_tops, channel_priorities,
        candidate_buffer, candidate_offsets, candidate_current_pos,
        centerline_z_offset, distance_matrix):
    """
    Candidate collection kernel function
    Args:
        channel_indices: Channel index array
        nx, ny, nz: Grid dimensions
        xmn, ymn, zmn: Grid origin coordinates
        xsiz, ysiz, zsiz: Grid spacing
        ccx, ccz, cco: Complex center coordinates and angles
        cx, cz: Channel center coordinates
        ccl: Channel centerline array
        ccy: Channel Y coordinate array
        ct, cw, crelpos: Channel thickness, width, and relative position
        por_base, por_range: Porosity base and range values
        random_seed: Random seed
        ny3: Number of centerline points
        enable_migration: Whether migration mode is enabled
        channel_z_tops: Channel top Z values array
        channel_priorities: Channel priority array
        candidate_buffer: Candidate buffer
        candidate_offsets: Candidate offset array
        candidate_current_pos: Candidate current position array
        centerline_z_offset: Centerline Z offset array
        distance_matrix: Precomputed distance matrix
    Returns:
        None: GPU kernel has no return value
    """
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    for channel_idx in range(len(channel_indices)):
        icc, ic, chan_idx = channel_indices[channel_idx]
        current_priority = channel_priorities[channel_idx]

        base_z_top = cz[icc - 1, ic - 1]

        channel_por_base = por_base[icc - 1, ic - 1]
        channel_por_range = por_range[icc - 1, ic - 1]

        is_migration = 1 if enable_migration else 0

        # Get distance information from distance matrix
        min_distance = distance_matrix[ix, iy, channel_idx, 0]
        closest_segment = int(distance_matrix[ix, iy, channel_idx, 1])
        closest_t = distance_matrix[ix, iy, channel_idx, 2]

        # Interpolate channel parameters
        thickness, width, relpos = interpolate_channel_parameters(
            closest_segment, closest_t, ct, cw, crelpos, ny3, icc, ic
        )

        if thickness <= 0.0 or width <= 0.0:
            continue

        # Get Z offset
        z_offset = 0.0
        if centerline_z_offset is not None and closest_segment < ny3:
            z_offset1 = centerline_z_offset[icc - 1, ic - 1, closest_segment]
            if closest_segment + 1 < ny3:
                z_offset2 = centerline_z_offset[icc - 1, ic - 1, closest_segment + 1]
                z_offset = z_offset1 + closest_t * (z_offset2 - z_offset1)
            else:
                z_offset = z_offset1

        current_z_top = base_z_top + z_offset

        # Use distance field to determine if within channel
        river_thickness_at_point = channel_cross_section_distance_field(
            min_distance, width, thickness, relpos
        )

        if river_thickness_at_point <= 0.0:
            continue

        # Calculate Z direction range
        z_bottom = current_z_top - river_thickness_at_point
        z_top_calc = current_z_top

        iz_bottom = int((z_bottom - zmn) / zsiz)
        iz_top = int((z_top_calc - zmn) / zsiz)

        iz_bottom = max(0, min(nz - 1, iz_bottom))
        iz_top = max(0, min(nz - 1, iz_top))

        # Fill Z direction
        for iz in range(iz_bottom, iz_top + 1):
            voxel_idx = iz * nx * ny + iy * nx + ix

            # Calculate vertical relative position
            z_world = zmn + (iz + 0.5) * zsiz
            z_thickness = z_top_calc - z_bottom

            if z_thickness > 0.01:
                z_rel = (z_world - z_bottom) / z_thickness
                z_rel = max(0.0, min(1.0, z_rel))
            else:
                z_rel = 0.5

            # Calculate normalized distance to channel centerline
            half_width = width / 2.0
            if half_width > 0.0:
                normalized_distance = min_distance / half_width
                normalized_distance = min(normalized_distance, 1.0)
                lateral_position_normalized = normalized_distance
            else:
                lateral_position_normalized = 0.0

            # Calculate porosity
            por_val = calculate_porosity_improved(
                z_rel,
                lateral_position_normalized,
                0.0,
                random_seed,
                ix * ny + iy,
                ix, iy, iz,
                channel_por_base,
                channel_por_range
            )

            # Add candidate to compressed buffer
            start_offset = candidate_offsets[voxel_idx]
            candidate_pos = cuda.atomic.add(candidate_current_pos, voxel_idx, 1)
            buffer_idx = start_offset + candidate_pos

            # Store candidate data
            candidate_buffer[buffer_idx, 0] = current_priority
            candidate_buffer[buffer_idx, 1] = current_z_top
            candidate_buffer[buffer_idx, 2] = por_val
            candidate_buffer[buffer_idx, 3] = is_migration

@cuda.jit(device=True)
def calculate_porosity_improved(rel_z_position, rel_lateral_position, curvature,random_seed,thread_id,ix, iy, iz,por_base,por_range):
    """
    Distance field-based porosity calculation
    Args:
        rel_z_position: Relative Z position
        rel_lateral_position: Relative lateral position
        curvature: Curvature
        random_seed: Random seed
        thread_id: Thread ID
        ix, iy, iz: Voxel coordinates
        por_base: Porosity base value
        por_range: Porosity range
    Returns:
        float: Calculated porosity value
    """
    rel_z_position = max(0.0, min(1.0, rel_z_position))

    por_min = por_base - por_range
    por_max = por_base + por_range

    # Distance-based porosity distribution
    distance_to_center = abs(rel_lateral_position)
    distance_factor = 1.0 - distance_to_center

    # Smooth transition
    smooth_distance_factor = 0.5 + 0.5 * math.cos((1.0 - distance_factor) * math.pi)

    # Distance-based base porosity
    distance_porosity = por_min + (por_max - por_min) * smooth_distance_factor

    # Vertical position influence
    vertical_factor = 0.8 + 0.4 * rel_z_position

    # Random variation
    rand_val = gpu_random(random_seed + thread_id, iz)
    random_variation = (rand_val - 0.5) * por_range * 0.2

    # Final porosity calculation
    final_porosity = distance_porosity * vertical_factor + random_variation

    return final_porosity



@cuda.jit
def conflict_resolution_kernel_dynamic(candidate_buffer, candidate_offsets,
                                       candidate_counts, output_grid, output_por, getpor,
                                       nx, ny, nz, enable_migration):
    """
    Candidate conflict resolution kernel function
    Args:
        candidate_buffer: Candidate buffer
        candidate_offsets: Candidate offset array
        candidate_counts: Candidate count array
        output_grid: Output grid
        output_por: Output porosity grid
        getpor: Whether to compute porosity
        nx, ny, nz: Grid dimensions
        enable_migration: Whether migration mode is enabled
    Returns:
        None: GPU kernel has no return value
    """
    thread_id = cuda.grid(1)

    if thread_id >= nx * ny * nz:
        return

    iz = thread_id // (nx * ny)
    remainder = thread_id % (nx * ny)
    iy = remainder // nx
    ix = remainder % nx

    voxel_idx = thread_id
    num_candidates = candidate_counts[voxel_idx]
    start_offset = candidate_offsets[voxel_idx]

    if num_candidates <= 0:
        return

    # Find optimal candidate
    best_candidate_idx = -1
    best_priority = -1.0
    best_z = -999999.0

    for cand_idx in range(num_candidates):
        buffer_idx = start_offset + cand_idx

        channel_id = candidate_buffer[buffer_idx, 0]
        z_value = candidate_buffer[buffer_idx, 1]
        should_select = False

        if enable_migration:
            # Migration mode: select candidate with highest ID
            if channel_id > best_priority:
                should_select = True
                best_priority = channel_id
        else:
            # Non-migration mode: select candidate with highest Z value
            if z_value > best_z + 0.001:
                should_select = True
                best_z = z_value

        if should_select:
            best_candidate_idx = cand_idx

    # Write optimal candidate
    if best_candidate_idx >= 0:
        buffer_idx = start_offset + best_candidate_idx

        channel_id = int(candidate_buffer[buffer_idx, 0])
        porosity = candidate_buffer[buffer_idx, 2]

        output_grid[ix, iy, iz] = channel_id

        if getpor:
            output_por[ix, iy, iz] = porosity

@cuda.jit(device=True)
def channel_cross_section_distance_field(distance_from_centerline, river_width,
                                           river_thickness, relative_position=0.5):
    """
    Distance field-based channel cross-section thickness calculation
    Args:
        distance_from_centerline: Distance from centerline
        river_width: River width
        river_thickness: River thickness
        relative_position: Relative position
    Returns:
        float: Channel thickness at this point
    """
    half_width = river_width / 2.0

    if distance_from_centerline > half_width:
        return 0.0

    # Calculate lateral relative position
    lateral_position = distance_from_centerline / half_width
    lateral_position = max(0.0, min(1.0, lateral_position))

    # Parabolic cross-section
    depth_factor = 1.0 - lateral_position * lateral_position

    # Consider channel asymmetry
    if relative_position != 0.5:
        asymmetry_factor = 1.0 + (relative_position - 0.5) * 0.3
        depth_factor *= asymmetry_factor

    depth_factor = max(depth_factor, 0.1)

    thickness = river_thickness * depth_factor
    return max(0.0, thickness)

@cuda.jit(device=True)
def interpolate_channel_parameters(segment_idx, t, ct, cw, crelpos, ny3, icc, ic):
    """
    Interpolate channel parameters along centerline
    Args:
        segment_idx: Segment index
        t: Interpolation parameter
        ct: Channel thickness array
        cw: Channel width array
        crelpos: Relative position array
        ny3: Number of centerline points
        icc, ic: Complex and channel indices
    Returns:
        tuple: Interpolated thickness, width, and relative position
    """
    if segment_idx >= ny3 - 1:
        segment_idx = ny3 - 2

    if segment_idx < 0:
        segment_idx = 0

    # Linear interpolation of channel parameters
    thickness1 = ct[icc - 1, ic - 1, segment_idx]
    thickness2 = ct[icc - 1, ic - 1, min(segment_idx + 1, ny3 - 1)]

    width1 = cw[icc - 1, ic - 1, segment_idx]
    width2 = cw[icc - 1, ic - 1, min(segment_idx + 1, ny3 - 1)]

    relpos1 = crelpos[icc - 1, ic - 1, segment_idx]
    relpos2 = crelpos[icc - 1, ic - 1, min(segment_idx + 1, ny3 - 1)]

    # Linear interpolation
    thickness = thickness1 + t * (thickness2 - thickness1)
    width = width1 + t * (width2 - width1)
    relpos = relpos1 + t * (relpos2 - relpos1)

    return thickness, width, relpos