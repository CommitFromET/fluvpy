
"""
render_facies.py

"""
import numpy as np
import time
import math

try:
    import cupy as cp
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available")
from ..utils.utils import gpu_random


def render_levees_batch_gpu(levees_data, color_offset, nx, ny, nz, cz, cx, xmn, ccx, ct, ymn,
                            cco, zmn, xsiz, ysiz, zsiz, channel, por, getpor=True, nc=None,
                            channel_metadata=None, enable_migration=False):
    """
    Natural levee batch GPU rendering
    Parameters:
        levees_data: Natural levee data list
        color_offset: Color offset
        nx, ny, nz: Grid dimensions
        cz, cx: Channel elevation and position arrays
        xmn, ymn, zmn: Grid origin coordinates
        ccx, ct, cco: Channel complex parameters
        xsiz, ysiz, zsiz: Grid cell dimensions
        channel: Channel grid array
        por: Porosity array
        getpor: Whether to calculate porosity
        nc: Channel count array
        channel_metadata: Channel metadata
        enable_migration: Whether to enable migration mode
    Returns:
        int: Number of filled cells
    """
    try:
        print(f"Starting levee rendering: {len(levees_data)} levees")
        print(f"Rendering mode: {'Migration mode' if enable_migration else 'Non-migration mode'}")

        if not GPU_AVAILABLE:
            print("GPU unavailable")
            return 0

        # GPU resource check and configuration
        try:
            device = cuda.get_current_device()
            max_threads_per_block = device.MAX_THREADS_PER_BLOCK

            if max_threads_per_block >= 1024:
                threads_per_block_x = 2
                threads_per_block_y = 16
                threads_per_block_z = 16
            elif max_threads_per_block >= 512:
                threads_per_block_x = 2
                threads_per_block_y = 8
                threads_per_block_z = 16
            else:
                threads_per_block_x = 1
                threads_per_block_y = 8
                threads_per_block_z = 8

        except Exception as e:
            print(f"GPU resource detection failed: {e}")
            threads_per_block_x = 1
            threads_per_block_y = 8
            threads_per_block_z = 8

        # Data preprocessing
        all_centerlines_coords = []
        all_centerlines_params = []
        levee_metadata = []

        # Prepare channel metadata array
        max_channel_id = max(channel_metadata.keys()) if channel_metadata else 9999
        channel_z_tops_array = np.zeros(max_channel_id + 1, dtype=np.float32)

        if channel_metadata:
            for channel_id, metadata in channel_metadata.items():
                if channel_id <= max_channel_id:
                    channel_z_tops_array[channel_id] = metadata['z_top']

        # Process levee data
        for levee_idx, levee in enumerate(levees_data):
            try:
                if not isinstance(levee, dict) or 'global_coords' not in levee:
                    continue

                required_coords = ['x', 'y', 'z', 'inner_edge', 'outer_edge']
                if not all(coord in levee['global_coords'] for coord in required_coords):
                    continue

                icc = levee['icc']
                ic = levee['ic']
                global_coords = levee['global_coords']
                ny3 = len(global_coords['x'])

                if ny3 == 0:
                    continue

                # Get channel data
                if 'true_river_width' in levee and 'true_river_thickness' in levee:
                    true_widths = np.array(levee['true_river_width'])
                    true_z_coords = np.array(levee.get('true_river_z_top', [cz[icc - 1, ic - 1]] * ny3))
                else:
                    true_widths = np.full(ny3, 10.0)
                    true_z_coords = np.full(ny3, cz[icc - 1, ic - 1])

                # Get channel centerline coordinates
                if ('river_coords' in levee and 'x' in levee['river_coords']):
                    river_center_x = np.array(levee['river_coords']['x'])
                    river_center_y = np.array(levee['river_coords']['y'])
                else:
                    # Recalculate
                    x_center_global = ccx[icc - 1] + cx[icc - 1, ic - 1]
                    river_center_x = np.full(ny3, x_center_global)
                    river_center_y = np.array([ymn + i * ysiz for i in range(ny3)])

                # Data validation
                valid_indices = [i for i in range(ny3)
                                 if (not np.isnan(river_center_x[i]) and
                                     not np.isnan(river_center_y[i]) and
                                     true_widths[i] > 0.01 and
                                     levee['height'][i] > 0.01)]

                if len(valid_indices) < max(3, ny3 * 0.3):
                    continue

                # Build centerline data
                valid_points = len(valid_indices)
                centerline_coords = np.zeros((valid_points, 2), dtype=np.float32)
                centerline_params = np.zeros((valid_points, 4), dtype=np.float32)

                for idx, i in enumerate(valid_indices):
                    centerline_coords[idx, 0] = river_center_x[i]
                    centerline_coords[idx, 1] = river_center_y[i]
                    centerline_params[idx, 0] = true_z_coords[i]
                    centerline_params[idx, 1] = true_widths[i]
                    centerline_params[idx, 2] = levee['height'][i]
                    centerline_params[idx, 3] = 1.0

                all_centerlines_coords.append(centerline_coords)
                all_centerlines_params.append(centerline_params)

                # Levee ID encoding
                side_offset = 0 if levee['side'] == 'left' else 1
                mxc = nc.max() if len(nc) > 0 else 1
                levee_id_with_channel_info = 10000 + icc * (mxc * 2) + ic * 2 + side_offset

                # Get source channel ID
                source_channel_id = 0
                source_channel_z_top = cz[icc - 1, ic - 1]

                if channel_metadata:
                    for channel_id, metadata in channel_metadata.items():
                        if metadata['icc'] == icc and metadata['ic'] == ic:
                            source_channel_id = channel_id
                            source_channel_z_top = metadata['z_top']
                            break

                # Metadata
                levee_metadata.append({
                    'levee_idx': levee_idx,
                    'levee_id': levee_id_with_channel_info,
                    'icc': icc,
                    'ic': ic,
                    'side': levee['side'],
                    'valid_points': valid_points,
                    'river_half_width': np.mean(true_widths) / 2.0,
                    'levee_width': np.mean(levee['width']),
                    'levee_height': np.mean(levee['height']),
                    'levee_depth': np.mean(levee.get('depth', levee['height'] * 0.4)),
                    'levee_total_thickness': np.mean(levee.get('total_thickness', levee['height'] * 1.4)),
                    'source_channel_id': source_channel_id,
                    'source_channel_z_top': source_channel_z_top,
                })

            except Exception as e:
                print(f"Error processing levee {levee_idx}: {e}")
                continue

        if not all_centerlines_coords:
            print("No valid levee data")
            return 0

        num_levees = len(all_centerlines_coords)
        print(f"Preparing to process {num_levees} valid levees")

        # Stage 1: Precompute distance matrix
        print("Precomputing levee distance matrix")

        max_points_per_levee = max(coords.shape[0] for coords in all_centerlines_coords)

        # Create padded matrices
        padded_centerlines = np.zeros((num_levees, max_points_per_levee, 2), dtype=np.float32)
        levee_point_counts = np.zeros(num_levees, dtype=np.int32)

        for i, coords in enumerate(all_centerlines_coords):
            points = coords.shape[0]
            padded_centerlines[i, :points, :] = coords
            levee_point_counts[i] = points

        # Distance matrix: [levee_idx, ix, iy, 3]
        distance_matrix = np.zeros((num_levees, nx, ny, 3), dtype=np.float32)

        # Transfer to GPU
        d_padded_centerlines = cuda.to_device(padded_centerlines)
        d_levee_point_counts = cuda.to_device(levee_point_counts)
        d_distance_matrix = cuda.to_device(distance_matrix)

        # Parameters
        xmn_f32 = np.float32(xmn)
        ymn_f32 = np.float32(ymn)
        xsiz_f32 = np.float32(xsiz)
        ysiz_f32 = np.float32(ysiz)

        # Grid configuration
        blocks_per_grid_x = (num_levees + threads_per_block_x - 1) // threads_per_block_x
        blocks_per_grid_y = (nx + threads_per_block_y - 1) // threads_per_block_y
        blocks_per_grid_z = (ny + threads_per_block_z - 1) // threads_per_block_z

        threads_per_block = (threads_per_block_x, threads_per_block_y, threads_per_block_z)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        # Distance matrix precomputation
        batch_precompute_levee_distance_matrix_kernel[blocks_per_grid, threads_per_block](
            d_padded_centerlines, d_levee_point_counts, num_levees, max_points_per_levee,
            nx, ny, xmn_f32, ymn_f32, xsiz_f32, ysiz_f32, d_distance_matrix
        )
        cuda.synchronize()

        # Stage 2: Count candidates
        print("Counting levee candidates")

        total_voxels = nx * ny * nz
        candidate_counts = np.zeros(total_voxels, dtype=np.int32)
        d_candidate_counts = cuda.to_device(candidate_counts)

        # Prepare rendering parameters
        padded_params = np.zeros((num_levees, max_points_per_levee, 4), dtype=np.float32)
        levee_ids = np.zeros(num_levees, dtype=np.int32)
        base_widths = np.zeros(num_levees, dtype=np.float32)
        base_heights = np.zeros(num_levees, dtype=np.float32)
        base_depths = np.zeros(num_levees, dtype=np.float32)
        source_channel_ids = np.zeros(num_levees, dtype=np.int32)
        source_channel_z_tops = np.zeros(num_levees, dtype=np.float32)

        for i, (params_data, metadata) in enumerate(zip(all_centerlines_params, levee_metadata)):
            points = params_data.shape[0]
            padded_params[i, :points, :] = params_data
            levee_ids[i] = metadata['levee_id']
            base_widths[i] = metadata['levee_width']
            base_heights[i] = metadata['levee_height']
            base_depths[i] = metadata['levee_depth']
            source_channel_ids[i] = metadata['source_channel_id']
            source_channel_z_tops[i] = metadata['source_channel_z_top']

        # Transfer rendering data
        d_padded_params = cuda.to_device(padded_params)
        d_levee_ids = cuda.to_device(levee_ids)
        d_base_widths = cuda.to_device(base_widths)
        d_base_heights = cuda.to_device(base_heights)
        d_base_depths = cuda.to_device(base_depths)
        d_source_channel_ids = cuda.to_device(source_channel_ids)
        d_source_channel_z_tops = cuda.to_device(source_channel_z_tops)
        d_current_channel_grid = cuda.to_device(channel.copy())
        d_channel_z_tops_array = cuda.to_device(channel_z_tops_array)

        zmn_f32 = np.float32(zmn)
        zsiz_f32 = np.float32(zsiz)

        # Create centerline_z_offset array
        dummy_centerline_z_offset = np.zeros((num_levees, max_points_per_levee, 1), dtype=np.float32)
        d_dummy_centerline_z_offset = cuda.to_device(dummy_centerline_z_offset)

        # Configure 2D grid
        threads_per_block_2d_x = 16
        threads_per_block_2d_y = 16
        blocks_per_grid_2d_x = (nx + threads_per_block_2d_x - 1) // threads_per_block_2d_x
        blocks_per_grid_2d_y = (ny + threads_per_block_2d_y - 1) // threads_per_block_2d_y

        threads_per_block_2d = (threads_per_block_2d_x, threads_per_block_2d_y)
        blocks_per_grid_2d = (blocks_per_grid_2d_x, blocks_per_grid_2d_y)

        # Candidate counting
        levee_candidate_counting_kernel_with_precomputed_distance[blocks_per_grid_2d, threads_per_block_2d](
            d_padded_params, d_levee_point_counts, d_levee_ids, d_base_widths,
            d_base_heights, d_base_depths, d_source_channel_ids, d_source_channel_z_tops,
            num_levees, max_points_per_levee,
            nx, ny, nz, xmn_f32, ymn_f32, zmn_f32, xsiz_f32, ysiz_f32, zsiz_f32,
            d_current_channel_grid, enable_migration, d_channel_z_tops_array,
            d_candidate_counts, d_dummy_centerline_z_offset, d_distance_matrix
        )
        cuda.synchronize()

        # Transfer candidate counts
        d_candidate_counts.copy_to_host(candidate_counts)

        # Statistics
        total_candidates = np.sum(candidate_counts)
        voxels_with_candidates = np.count_nonzero(candidate_counts)

        print(f"Candidate statistics: {total_candidates:,} total, {voxels_with_candidates:,} valid voxels")

        if total_candidates == 0:
            print("No levee candidates found")
            return 0

        # Stage 3: Create candidate buffer
        print("Creating levee candidate buffer")

        # CSR format offsets
        candidate_offsets = np.zeros(total_voxels + 1, dtype=np.int32)
        candidate_offsets[1:] = np.cumsum(candidate_counts)

        # Candidate buffer
        candidate_buffer = np.zeros((total_candidates, 4), dtype=np.float32)
        candidate_current_pos = np.zeros(total_voxels, dtype=np.int32)

        # Memory usage
        new_memory_mb = total_candidates * 4 * 4 / (1024 * 1024)
        print(f"Candidate buffer: {new_memory_mb:.1f} MB")

        # Transfer to GPU
        d_candidate_buffer = cuda.to_device(candidate_buffer)
        d_candidate_offsets = cuda.to_device(candidate_offsets)
        d_candidate_current_pos = cuda.to_device(candidate_current_pos)

        # Stage 4: Collect candidates
        print("Levee candidate collection")

        random_seed = int(time.time() * 1000) % 10000000

        levee_candidate_collection_kernel_dynamic_with_precomputed_distance[blocks_per_grid_2d, threads_per_block_2d](
            d_padded_params, d_levee_point_counts, d_levee_ids, d_base_widths,
            d_base_heights, d_base_depths, d_source_channel_ids, d_source_channel_z_tops,
            num_levees, max_points_per_levee,
            nx, ny, nz, xmn_f32, ymn_f32, zmn_f32, xsiz_f32, ysiz_f32, zsiz_f32,
            d_current_channel_grid, enable_migration, d_channel_z_tops_array,
            d_candidate_buffer, d_candidate_offsets, d_candidate_current_pos,
            d_dummy_centerline_z_offset, d_distance_matrix, random_seed
        )
        cuda.synchronize()

        # Verify collection results
        d_candidate_current_pos.copy_to_host(candidate_current_pos)
        collected_candidates = np.sum(candidate_current_pos)
        print(f"Collected candidates: {collected_candidates:,}/{total_candidates:,}")

        # Stage 5: Conflict resolution
        print("Levee conflict resolution")

        # Create output grids
        d_output_grid = cuda.to_device(channel.copy())
        if getpor:
            d_output_por = cuda.to_device(por.copy())
        else:
            d_output_por = cuda.to_device(np.zeros((1, 1, 1), dtype=np.float32))

        # Configure voxel parallel CUDA grid
        voxel_threads_per_block = 256
        voxel_blocks_per_grid = (total_voxels + voxel_threads_per_block - 1) // voxel_threads_per_block

        levee_conflict_resolution_kernel_dynamic[voxel_blocks_per_grid, voxel_threads_per_block](
            d_candidate_buffer, d_candidate_offsets, d_candidate_counts,
            d_output_grid, d_output_por, getpor,
            nx, ny, nz, enable_migration
        )
        cuda.synchronize()

        # Transfer results
        d_output_grid.copy_to_host(channel)
        if getpor:
            d_output_por.copy_to_host(por)

        # Count filled cells
        total_filled_cells = np.count_nonzero((channel >= 10000) & (channel < 20000))

        print(f"Levee rendering complete, filled cells: {total_filled_cells}")

        return total_filled_cells

    except Exception as e:
        print(f"Levee rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


@cuda.jit
def batch_precompute_levee_distance_matrix_kernel(padded_centerlines, levee_point_counts,
                                                  num_levees, max_points_per_levee,
                                                  nx, ny, xmn, ymn, xsiz, ysiz,
                                                  batch_distance_matrix):
    """
    Batch precomputation of levee distance matrix CUDA kernel
    Parameters:
        padded_centerlines: Padded centerline coordinate array
        levee_point_counts: Levee point count array
        num_levees: Number of levees
        max_points_per_levee: Maximum points per levee
        nx, ny: Grid X, Y dimensions
        xmn, ymn: Grid origin coordinates
        xsiz, ysiz: Grid cell dimensions
        batch_distance_matrix: Output distance matrix
    Returns:
        void: No return value, results stored in batch_distance_matrix
    """
    # Get thread indices
    levee_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ix = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    iy = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    # Boundary check
    if levee_idx >= num_levees or ix >= nx or iy >= ny:
        return

    # Calculate world coordinates of grid point
    world_x = xmn + (ix + 0.5) * xsiz
    world_y = ymn + (iy + 0.5) * ysiz

    # Get number of points for current levee
    num_centerline_points = levee_point_counts[levee_idx]

    if num_centerline_points <= 1:
        return

    # Find minimum distance to centerline
    min_dist_sq = 1e30
    closest_segment = 0
    closest_t = 0.0

    # Iterate through all centerline segments of current levee
    for i in range(num_centerline_points - 1):
        x1 = padded_centerlines[levee_idx, i, 0]
        y1 = padded_centerlines[levee_idx, i, 1]
        x2 = padded_centerlines[levee_idx, i + 1, 0]
        y2 = padded_centerlines[levee_idx, i + 1, 1]

        # Data validity check
        if abs(x1) > 1e6 or abs(y1) > 1e6 or abs(x2) > 1e6 or abs(y2) > 1e6:
            continue

        # Calculate distance to current line segment
        dist_sq, t = gpu_distance_to_line_segment(world_x, world_y, x1, y1, x2, y2)

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_segment = i
            closest_t = t

    # Store calculation results
    min_distance = math.sqrt(min_dist_sq)
    batch_distance_matrix[levee_idx, ix, iy, 0] = min_distance
    batch_distance_matrix[levee_idx, ix, iy, 1] = closest_segment
    batch_distance_matrix[levee_idx, ix, iy, 2] = closest_t


@cuda.jit(device=True)
def gpu_distance_to_line_segment(px, py, x1, y1, x2, y2):
    """
    GPU device function: Calculate minimum distance from point to line segment
    Parameters:
        px, py: Point coordinates
        x1, y1: Line segment start point coordinates
        x2, y2: Line segment end point coordinates
    Returns:
        tuple: (squared distance, closest point parameter t)
    """

    dx = x2 - x1
    dy = y2 - y1

    # Squared length of line segment
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-10:
        # Degenerate to point
        dist_sq = (px - x1) * (px - x1) + (py - y1) * (py - y1)
        return dist_sq, 0.0

    # Calculate projection parameter t
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq

    # Clamp t to [0,1] range
    t = max(0.0, min(1.0, t))

    # Calculate closest point
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Calculate squared distance
    dist_sq = (px - closest_x) * (px - closest_x) + (py - closest_y) * (py - closest_y)

    return dist_sq, t


@cuda.jit
def levee_candidate_counting_kernel_with_precomputed_distance(
        padded_params, levee_point_counts, levee_ids, base_widths,
        base_heights, base_depths, source_channel_ids, source_channel_z_tops,
        num_levees, max_points_per_levee,
        nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
        current_channel_grid, enable_migration, channel_z_tops_array,
        candidate_counts, centerline_z_offset, distance_matrix):
    """
    Levee candidate counting kernel
    Parameters:
        padded_params: Padded parameter array
        levee_point_counts: Levee point count array
        levee_ids: Levee ID array
        base_widths, base_heights, base_depths: Levee base dimension parameters
        source_channel_ids, source_channel_z_tops: Source channel information
        num_levees, max_points_per_levee: Number of levees and maximum points
        nx, ny, nz: Grid dimensions
        xmn, ymn, zmn: Grid origin coordinates
        xsiz, ysiz, zsiz: Grid cell dimensions
        current_channel_grid: Current channel grid
        enable_migration: Whether migration mode is enabled
        channel_z_tops_array: Channel top elevation array
        candidate_counts: Candidate count array
        centerline_z_offset: Centerline Z offset
        distance_matrix: Precomputed distance matrix
    Returns:
        void: No return value, results stored in candidate_counts
    """
    # Get current grid point indices
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    # Use precomputed distance information for each levee
    for levee_idx in range(num_levees):
        # Get number of points for current levee
        num_centerline_points = levee_point_counts[levee_idx]

        if num_centerline_points <= 1:
            continue

        # Get distance information from precomputed distance matrix
        min_distance = distance_matrix[levee_idx, ix, iy, 0]
        closest_segment_idx = int(distance_matrix[levee_idx, ix, iy, 1])
        closest_t = distance_matrix[levee_idx, ix, iy, 2]

        # Boundary check
        if closest_segment_idx < 0 or closest_segment_idx >= num_centerline_points - 1:
            continue

        # Interpolate channel parameters on closest segment
        i = closest_segment_idx
        t = closest_t

        # Interpolate channel centerline parameters
        interp_z = gpu_linear_interpolate(padded_params[levee_idx, i, 0],
                                          padded_params[levee_idx, i + 1, 0], t)
        interp_river_width = gpu_linear_interpolate(padded_params[levee_idx, i, 1],
                                                    padded_params[levee_idx, i + 1, 1], t)
        interp_levee_height = gpu_linear_interpolate(padded_params[levee_idx, i, 2],
                                                     padded_params[levee_idx, i + 1, 2], t)

        # Calculate channel half width
        river_half_width = interp_river_width / 2.0

        # Get levee depth parameters
        interp_levee_depth = base_depths[levee_idx]

        # Channel top surface Z coordinate
        river_top_z = interp_z

        # Get current levee's source channel information
        source_channel_id = source_channel_ids[levee_idx]
        source_channel_z_top = source_channel_z_tops[levee_idx]

        # Calculate levee geometric range
        levee_z_bottom = river_top_z - interp_levee_depth
        levee_z_top = river_top_z + interp_levee_height

        # Calculate influence range
        levee_outer_distance = river_half_width + base_widths[levee_idx]

        # Check if within levee influence range
        if min_distance > levee_outer_distance:
            continue

        # Lower half: linear depth decay
        relative_distance = min_distance / levee_outer_distance
        current_depth = interp_levee_depth * (1.0 - relative_distance)
        min_depth = zsiz * 0.5
        current_depth = max(current_depth, min_depth)

        # Lower half Z direction range
        z_bottom_lower = river_top_z - current_depth
        z_top_lower = river_top_z

        # Convert to grid indices
        iz_bottom_lower = max(0, min(nz - 1, int((z_bottom_lower - zmn) / zsiz)))
        iz_top_lower = max(0, min(nz - 1, int((z_top_lower - zmn) / zsiz)))

        # Ensure at least one layer is filled
        if iz_bottom_lower >= iz_top_lower:
            iz_bottom_lower = iz_top_lower - 1
            iz_bottom_lower = max(0, iz_bottom_lower)

        # Upper half: use distance field cross-section function
        levee_upper_thickness = gpu_levee_cross_section_distance_field(
            min_distance, river_half_width, base_widths[levee_idx],
            interp_levee_height, 1.0
        )

        # Upper half Z direction range
        if levee_upper_thickness > 0.0:
            safety_gap = 0.5
            if min_distance >= river_half_width + safety_gap:
                z_top_upper = river_top_z + interp_levee_height
                if levee_upper_thickness < 1.0:
                    actual_height = interp_levee_height * levee_upper_thickness
                    z_top_upper = river_top_z + actual_height

                iz_river_top = max(0, min(nz - 1, int((river_top_z - zmn) / zsiz)))
                iz_top_upper = max(0, min(nz - 1, int((z_top_upper - zmn) / zsiz)))
            else:
                iz_river_top = iz_top_lower
                iz_top_upper = iz_top_lower
        else:
            iz_river_top = iz_top_lower
            iz_top_upper = iz_top_lower

        # Count lower half candidates
        for iz in range(iz_bottom_lower, iz_top_lower + 1):
            current_val = current_channel_grid[ix, iy, iz]

            # Candidate priority judgment
            can_be_candidate = levee_priority_check_for_candidate(
                current_val, source_channel_id, source_channel_z_top,
                enable_migration, channel_z_tops_array
            )

            if can_be_candidate:
                voxel_idx = iz * nx * ny + iy * nx + ix
                cuda.atomic.add(candidate_counts, voxel_idx, 1)

        # Count upper half candidates
        for iz in range(iz_river_top + 1, iz_top_upper + 1):
            current_val = current_channel_grid[ix, iy, iz]

            # Candidate priority judgment
            can_be_candidate = levee_priority_check_for_candidate(
                current_val, source_channel_id, source_channel_z_top,
                enable_migration, channel_z_tops_array
            )

            if can_be_candidate:
                voxel_idx = iz * nx * ny + iy * nx + ix
                cuda.atomic.add(candidate_counts, voxel_idx, 1)


@cuda.jit
def levee_candidate_collection_kernel_dynamic_with_precomputed_distance(
        padded_params, levee_point_counts, levee_ids, base_widths,
        base_heights, base_depths, source_channel_ids, source_channel_z_tops,
        num_levees, max_points_per_levee,
        nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
        current_channel_grid, enable_migration, channel_z_tops_array,
        candidate_buffer, candidate_offsets, candidate_current_pos,
        centerline_z_offset, distance_matrix, random_seed):
    """
    Levee candidate collection kernel
    Parameters:
        padded_params: Padded parameter array
        levee_point_counts: Levee point count array
        levee_ids: Levee ID array
        base_widths, base_heights, base_depths: Levee base dimension parameters
        source_channel_ids, source_channel_z_tops: Source channel information
        num_levees, max_points_per_levee: Number of levees and maximum points
        nx, ny, nz: Grid dimensions
        xmn, ymn, zmn: Grid origin coordinates
        xsiz, ysiz, zsiz: Grid cell dimensions
        current_channel_grid: Current channel grid
        enable_migration: Whether migration mode is enabled
        channel_z_tops_array: Channel top elevation array
        candidate_buffer: Candidate buffer
        candidate_offsets: Candidate offsets
        candidate_current_pos: Candidate current positions
        centerline_z_offset: Centerline Z offset
        distance_matrix: Precomputed distance matrix
        random_seed: Random seed
    Returns:
        void: No return value, results stored in candidate_buffer
    """
    # Get current grid point indices
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if ix >= nx or iy >= ny:
        return

    for levee_idx in range(num_levees):
        # Get number of points for current levee
        num_centerline_points = levee_point_counts[levee_idx]

        if num_centerline_points <= 1:
            continue

        # Get distance information from precomputed distance matrix
        min_distance = distance_matrix[levee_idx, ix, iy, 0]
        closest_segment_idx = int(distance_matrix[levee_idx, ix, iy, 1])
        closest_t = distance_matrix[levee_idx, ix, iy, 2]

        if closest_segment_idx < 0 or closest_segment_idx >= num_centerline_points - 1:
            continue

        # Interpolation parameters
        i = closest_segment_idx
        t = closest_t

        interp_z = gpu_linear_interpolate(padded_params[levee_idx, i, 0],
                                          padded_params[levee_idx, i + 1, 0], t)
        interp_river_width = gpu_linear_interpolate(padded_params[levee_idx, i, 1],
                                                    padded_params[levee_idx, i + 1, 1], t)
        interp_levee_height = gpu_linear_interpolate(padded_params[levee_idx, i, 2],
                                                     padded_params[levee_idx, i + 1, 2], t)

        river_half_width = interp_river_width / 2.0
        interp_levee_depth = base_depths[levee_idx]
        river_top_z = interp_z

        # Get levee ID and source channel information
        current_levee_id = levee_ids[levee_idx]
        source_channel_id = source_channel_ids[levee_idx]
        source_channel_z_top = source_channel_z_tops[levee_idx]

        # Calculate levee geometric range
        levee_z_bottom = river_top_z - interp_levee_depth
        total_levee_thickness = interp_levee_depth + interp_levee_height

        # Calculate influence range
        levee_outer_distance = river_half_width + base_widths[levee_idx]

        if min_distance > levee_outer_distance:
            continue

        # Calculate distance-related parameters
        relative_distance = min_distance / levee_outer_distance
        relative_distance = max(0.0, min(1.0, relative_distance))

        # Lower half calculation
        current_depth = interp_levee_depth * (1.0 - relative_distance)
        min_depth = zsiz * 0.5
        current_depth = max(current_depth, min_depth)

        z_bottom_lower = river_top_z - current_depth
        z_top_lower = river_top_z

        iz_bottom_lower = max(0, min(nz - 1, int((z_bottom_lower - zmn) / zsiz)))
        iz_top_lower = max(0, min(nz - 1, int((z_top_lower - zmn) / zsiz)))

        if iz_bottom_lower >= iz_top_lower:
            iz_bottom_lower = iz_top_lower - 1
            iz_bottom_lower = max(0, iz_bottom_lower)

        # Upper half calculation
        levee_upper_thickness = gpu_levee_cross_section_distance_field(
            min_distance, river_half_width, base_widths[levee_idx],
            interp_levee_height, 1.0
        )

        if levee_upper_thickness > 0.0:
            safety_gap = 0.5
            if min_distance >= river_half_width + safety_gap:
                z_top_upper = river_top_z + interp_levee_height
                if levee_upper_thickness < 1.0:
                    actual_height = interp_levee_height * levee_upper_thickness
                    z_top_upper = river_top_z + actual_height

                iz_river_top = max(0, min(nz - 1, int((river_top_z - zmn) / zsiz)))
                iz_top_upper = max(0, min(nz - 1, int((z_top_upper - zmn) / zsiz)))
            else:
                iz_river_top = iz_top_lower
                iz_top_upper = iz_top_lower
        else:
            iz_river_top = iz_top_lower
            iz_top_upper = iz_top_lower

        # Calculate channel top surface position
        river_top_position = interp_levee_depth / total_levee_thickness if total_levee_thickness > 0 else 0.5

        # Collect lower half candidates
        for iz in range(iz_bottom_lower, iz_top_lower + 1):
            current_val = current_channel_grid[ix, iy, iz]

            can_be_candidate = levee_priority_check_for_candidate(
                current_val, source_channel_id, source_channel_z_top,
                enable_migration, channel_z_tops_array
            )

            if can_be_candidate:
                voxel_idx = iz * nx * ny + iy * nx + ix

                # Add candidate to compressed buffer
                start_offset = candidate_offsets[voxel_idx]
                candidate_pos = cuda.atomic.add(candidate_current_pos, voxel_idx, 1)
                buffer_idx = start_offset + candidate_pos

                # Calculate porosity
                z_loc = zmn + (iz + 0.5) * zsiz
                por_val = gpu_unified_levee_porosity_calculation(
                    z_loc, levee_z_bottom, total_levee_thickness, river_top_position,
                    relative_distance, random_seed, ix, iy, iz
                )

                # Store candidate data
                candidate_buffer[buffer_idx, 0] = current_levee_id
                candidate_buffer[buffer_idx, 1] = source_channel_z_top
                candidate_buffer[buffer_idx, 2] = por_val
                candidate_buffer[buffer_idx, 3] = 1 if enable_migration else 0

        # Collect upper half candidates
        for iz in range(iz_river_top + 1, iz_top_upper + 1):
            current_val = current_channel_grid[ix, iy, iz]

            can_be_candidate = levee_priority_check_for_candidate(
                current_val, source_channel_id, source_channel_z_top,
                enable_migration, channel_z_tops_array
            )

            if can_be_candidate:
                voxel_idx = iz * nx * ny + iy * nx + ix

                start_offset = candidate_offsets[voxel_idx]
                candidate_pos = cuda.atomic.add(candidate_current_pos, voxel_idx, 1)
                buffer_idx = start_offset + candidate_pos

                z_loc = zmn + (iz + 0.5) * zsiz
                por_val = gpu_unified_levee_porosity_calculation(
                    z_loc, levee_z_bottom, total_levee_thickness, river_top_position,
                    relative_distance, random_seed, ix, iy, iz
                )

                candidate_buffer[buffer_idx, 0] = current_levee_id
                candidate_buffer[buffer_idx, 1] = source_channel_z_top
                candidate_buffer[buffer_idx, 2] = por_val
                candidate_buffer[buffer_idx, 3] = 1 if enable_migration else 0


@cuda.jit(device=True)
def gpu_unified_levee_porosity_calculation(z_loc, levee_z_bottom, total_levee_thickness,
                                           river_top_position, relative_distance,
                                           random_seed, ix, iy, iz):
    """
    Unified levee porosity calculation function
    Parameters:
        z_loc: Z location coordinate
        levee_z_bottom: Levee bottom Z coordinate
        total_levee_thickness: Total levee thickness
        river_top_position: River top relative position
        relative_distance: Relative distance
        random_seed: Random seed
        ix, iy, iz: Grid indices
    Returns:
        float: Calculated porosity value
    """
    # Normalized position relative to entire levee thickness
    if total_levee_thickness > 0:
        z_rel_global = (z_loc - levee_z_bottom) / total_levee_thickness
    else:
        z_rel_global = 0.5
    z_rel_global = max(0.0, min(1.0, z_rel_global))

    # Base porosity model
    if z_rel_global <= river_top_position:
        # Lower half: from bottom to channel top surface
        local_progress = z_rel_global / max(river_top_position, 0.01)
        por_base = 0.14 + 0.04 * local_progress  # 0.14 → 0.18
    else:
        # Upper half: from channel top surface
        local_progress = (z_rel_global - river_top_position) / max(1.0 - river_top_position, 0.01)
        por_base = 0.18 - 0.03 * local_progress  # 0.18 → 0.15

    # Effect factors
    distance_effect = 0.015 * relative_distance
    depth_effect = 0.01 * (z_rel_global - 0.5)

    # Random variation
    cell_seed = random_seed + ix * 1000 + iy * 100 + iz
    rand_val = gpu_random(cell_seed, iz)
    random_variation = (rand_val - 0.5) * 0.02

    # Combine all effects
    por_val = por_base + distance_effect + depth_effect + random_variation

    # Boundary constraints
    por_val = max(0.12, min(0.28, por_val))

    return por_val


@cuda.jit(device=True)
def levee_priority_check_for_candidate(current_val, source_channel_id, source_channel_z_top,
                                       enable_migration, channel_z_tops_array):
    """
    Levee candidate priority check function
    Parameters:
        current_val: Current voxel value
        source_channel_id: Source channel ID
        source_channel_z_top: Source channel top elevation
        enable_migration: Whether migration mode is enabled
        channel_z_tops_array: Channel top elevation array
    Returns:
        bool: True if can be candidate, False if cannot be candidate
    """
    # If voxel is empty, can be candidate
    if current_val == 0:
        return True

    # Determine current voxel type
    if 1 <= current_val < 10000:
        # Current voxel is channel
        existing_channel_id = current_val

        # If same source channel, cannot be candidate
        if existing_channel_id == source_channel_id:
            return False

        # Different channel, judge based on mode
        if enable_migration:
            # Migration mode: compare channel IDs
            if existing_channel_id > source_channel_id:
                return False
            else:
                return True
        else:
            # Non-migration mode: compare Z values
            existing_channel_z_top = channel_z_tops_array[existing_channel_id]
            if existing_channel_z_top > source_channel_z_top:
                return False
            else:
                return True

    elif 10000 <= current_val < 20000:
        # Current voxel is other levee
        return True

    elif current_val >= 20000:
        # Current voxel is crevasse splay
        return False

    return False


@cuda.jit
def levee_conflict_resolution_kernel_dynamic(candidate_buffer, candidate_offsets,
                                             candidate_counts, output_grid, output_por, getpor,
                                             nx, ny, nz, enable_migration):
    """
    Levee conflict resolution kernel
    Parameters:
        candidate_buffer: Candidate buffer
        candidate_offsets: Candidate offsets
        candidate_counts: Candidate counts
        output_grid: Output grid
        output_por: Output porosity
        getpor: Whether to get porosity
        nx, ny, nz: Grid dimensions
        enable_migration: Whether migration mode is enabled
    Returns:
        void: No return value, results stored in output_grid and output_por
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

    # Find best candidate
    best_candidate_idx = -1
    best_priority = -1.0
    best_z = -999999.0

    for cand_idx in range(num_candidates):
        buffer_idx = start_offset + cand_idx

        levee_id = candidate_buffer[buffer_idx, 0]
        source_z_top = candidate_buffer[buffer_idx, 1]

        should_select = False

        if enable_migration:
            # Migration mode: select candidate with highest channel ID
            if levee_id > best_priority:
                should_select = True
                best_priority = levee_id
        else:
            # Non-migration mode: select candidate with highest Z value
            if source_z_top > best_z + 0.001:
                should_select = True
                best_z = source_z_top

        if should_select:
            best_candidate_idx = cand_idx

    # Write best candidate
    if best_candidate_idx >= 0:
        buffer_idx = start_offset + best_candidate_idx

        levee_id = int(candidate_buffer[buffer_idx, 0])
        porosity = candidate_buffer[buffer_idx, 2]

        output_grid[ix, iy, iz] = levee_id

        if getpor:
            output_por[ix, iy, iz] = porosity


@cuda.jit(device=True)
def gpu_levee_cross_section_distance_field(distance_from_centerline, river_half_width,
                                           levee_width, levee_height, levee_asymmetry=1.0):
    """
    Levee cross-section distance field function
    Parameters:
        distance_from_centerline: Distance from centerline
        river_half_width: River half width
        levee_width: Levee width
        levee_height: Levee height
        levee_asymmetry: Levee asymmetry factor, default 1.0
    Returns:
        float: Calculated levee thickness value
    """
    # Safety gap
    safety_gap = 0.5
    levee_inner_distance = river_half_width + safety_gap

    # Levee outer edge distance
    levee_outer_distance = levee_inner_distance + levee_width

    # Check if within levee range
    if distance_from_centerline < levee_inner_distance or distance_from_centerline > levee_outer_distance:
        return 0.0

    # Relative position
    relative_pos = (distance_from_centerline - levee_inner_distance) / levee_width
    relative_pos = max(0.0, min(1.0, relative_pos))

    # Width factor
    min_width = 1.0
    max_width = 15.0
    width_factor = min(1.0, max(0.0, (levee_width - min_width) / (max_width - min_width)))

    # Peak position
    peak_position_min = 0.20
    peak_position_max = 0.35
    peak_position = peak_position_min + (peak_position_max - peak_position_min) * width_factor

    # Apply asymmetry factor
    peak_position = peak_position * levee_asymmetry
    peak_position = max(0.15, min(0.45, peak_position))

    # Base thickness calculation
    zsiz = 0.4
    min_thickness_ratio = zsiz / levee_height
    min_thickness_ratio = max(0.05, min(0.3, min_thickness_ratio))

    # Thickness distribution calculation
    if relative_pos <= peak_position:
        # From inner edge to peak
        if peak_position > 0.0:
            progress = relative_pos / peak_position

            growth_power_min = 1.5
            growth_power_max = 2.0
            growth_power = growth_power_min + (growth_power_max - growth_power_min) * width_factor

            thickness_factor = min_thickness_ratio + (1.0 - min_thickness_ratio) * (progress ** growth_power)
        else:
            thickness_factor = 1.0

    else:
        # Exponential decay from peak to outer edge
        decay_start = peak_position
        decay_end = 1.0
        decay_length = decay_end - decay_start

        if decay_length > 0.0:
            decay_progress = (relative_pos - decay_start) / decay_length

            # Exponential decay
            decay_intensity_min = 2.5
            decay_intensity_max = 4.0
            decay_intensity = decay_intensity_min + (decay_intensity_max - decay_intensity_min) * width_factor

            exp_factor = math.exp(-decay_intensity * decay_progress)

            exp_min = math.exp(-decay_intensity)
            normalized_exp_factor = (exp_factor - exp_min) / (1.0 - exp_min)
            normalized_exp_factor = max(0.0, min(1.0, normalized_exp_factor))

            thickness_factor = min_thickness_ratio + (1.0 - min_thickness_ratio) * normalized_exp_factor

            # Steepening treatment
            if decay_progress > 0.6:
                steep_progress = (decay_progress - 0.6) / 0.4
                steep_factor = steep_progress ** 3.0

                additional_reduction = (thickness_factor - min_thickness_ratio) * steep_factor
                thickness_factor = thickness_factor - additional_reduction

            thickness_factor = max(min_thickness_ratio, min(1.0, thickness_factor))

        else:
            thickness_factor = min_thickness_ratio

    # Validate thickness factor
    thickness_factor = max(min_thickness_ratio, min(1.0, thickness_factor))

    # Topographic undulation
    if 0.05 < relative_pos < 0.95 and thickness_factor > (min_thickness_ratio + 0.02):
        ripple_intensity = 0.008 * (1.0 + width_factor * 0.3)
        ripple_frequency = 12.0 + width_factor * 6.0

        distance_to_peak = abs(relative_pos - peak_position)
        peak_influence = max(0.3, 1.0 - distance_to_peak * 1.5)

        height_above_min = (thickness_factor - min_thickness_ratio) / (1.0 - min_thickness_ratio)

        ripple_intensity = ripple_intensity * peak_influence * height_above_min
        ripple_effect = ripple_intensity * math.sin(relative_pos * ripple_frequency)

        thickness_factor = thickness_factor + ripple_effect
        thickness_factor = max(min_thickness_ratio, min(1.0, thickness_factor))

    # Edge treatment
    edge_zone = 0.02

    if relative_pos <= edge_zone:
        edge_blend = relative_pos / edge_zone
        thickness_factor = min_thickness_ratio + (thickness_factor - min_thickness_ratio) * edge_blend

    elif relative_pos >= (1.0 - edge_zone):
        edge_blend = (1.0 - relative_pos) / edge_zone

        target_thickness = min_thickness_ratio
        thickness_factor = target_thickness + (thickness_factor - target_thickness) * edge_blend

        if relative_pos >= 0.98:
            thickness_factor = min_thickness_ratio

    # Final check
    thickness_factor = max(min_thickness_ratio, min(1.0, thickness_factor))

    # Calculate final thickness
    thickness = levee_height * thickness_factor

    return max(zsiz, thickness)


def render_crevasse_splays_batch_gpu(crevasse_data, color_offset, nx, ny, nz, cz, cx, xmn, ccx, ct, ymn, cco, zmn,
                                     xsiz, ysiz, zsiz, channel, por, getpor=True, nc=None,
                                     channel_metadata=None, enable_migration=False):
    """Crevasse splay batch GPU rendering"""
    try:
        print(f"Starting crevasse splay rendering: {len(crevasse_data)} crevasse splays")
        print(f"Rendering mode: {'Migration mode' if enable_migration else 'Non-migration mode'}")

        # Levee clearing control
        enable_splay_levee_clearing = not enable_migration
        print(f"Crevasse splay-levee clearing: {'Disabled (migration mode)' if enable_migration else 'Enabled (non-migration mode)'}")

        if not GPU_AVAILABLE:
            print("GPU unavailable")
            return 0

        # Prepare channel metadata array
        max_channel_id = max(channel_metadata.keys()) if channel_metadata else 9999
        channel_z_tops_array = np.zeros(max_channel_id + 1, dtype=np.float32)

        if channel_metadata:
            for channel_id, metadata in channel_metadata.items():
                if channel_id <= max_channel_id:
                    channel_z_tops_array[channel_id] = metadata['z_top']

        # Data preprocessing
        all_centerlines_coords = []
        all_centerlines_params = []
        splay_metadata = []

        for splay_idx, splay in enumerate(crevasse_data):
            try:
                if not isinstance(splay, dict) or 'centerline' not in splay:
                    continue

                centerline = splay['centerline']
                if len(centerline) < 2:
                    continue

                # Get channel information
                icc = splay.get('icc', 1)
                ic = splay.get('ic', 1)

                # Get source channel ID
                source_channel_id = 0
                source_channel_z_top = cz[icc - 1, ic - 1] if icc > 0 and ic > 0 else 50.0

                if channel_metadata:
                    for channel_id, metadata in channel_metadata.items():
                        if metadata['icc'] == icc and metadata['ic'] == ic:
                            source_channel_id = channel_id
                            source_channel_z_top = metadata['z_top']
                            break

                # Key modification: calculate Z value offset
                z_offset = 0.0  # Default no offset
                if 'river_node_z_top' in splay and 'river_z_top' in splay:
                    # Calculate difference between actual river Z value at crevasse splay location and original river Z value
                    actual_river_z = splay['river_node_z_top']  # Actual river Z value at node
                    original_river_z = splay['river_z_top']  # Original river Z value
                    z_offset = actual_river_z - original_river_z

                    if splay_idx < 3:
                        print(
                            f"  Crevasse splay {splay_idx}: Z offset={z_offset:.3f} (from {original_river_z:.3f} to {actual_river_z:.3f})")
                elif 'z_top' in splay and 'river_z_top' in splay:
                    # Reverse calculate offset from crevasse splay z_top
                    splay_z_top = splay['z_top']
                    original_river_z = splay['river_z_top']
                    # Assume crevasse splay height is 60% of original thickness
                    estimated_height = splay.get('thickness', 2.0) * 0.6
                    actual_river_z = splay_z_top - estimated_height
                    z_offset = actual_river_z - original_river_z

                    if splay_idx < 3:
                        print(f"  Crevasse splay {splay_idx}: Estimated Z offset={z_offset:.3f}")

                # Calculate cumulative distances
                cumulative_distances = np.zeros(len(centerline))
                for i in range(1, len(centerline)):
                    dist = np.linalg.norm(centerline[i] - centerline[i - 1])
                    cumulative_distances[i] = cumulative_distances[i - 1] + dist

                # Normalize to target length
                total_length = splay.get('total_length', 100.0)
                if cumulative_distances[-1] > 0:
                    cumulative_distances = cumulative_distances * total_length / cumulative_distances[-1]

                # Build centerline coordinate data
                centerline_coords = centerline.astype(np.float32)

                # Build parameter data
                centerline_params = np.zeros((len(centerline), 4), dtype=np.float32)
                for i in range(len(centerline)):
                    centerline_params[i, 0] = cumulative_distances[i]
                    centerline_params[i, 1] = total_length
                    centerline_params[i, 2] = splay.get('cone_length', total_length * 0.6)
                    centerline_params[i, 3] = splay.get('max_width', 30.0)

                all_centerlines_coords.append(centerline_coords)
                all_centerlines_params.append(centerline_params)

                # Crevasse splay ID
                splay_sub_idx = splay_idx % 10
                improved_splay_id = generate_improved_splay_id(color_offset, icc, ic, splay_sub_idx)

                # Metadata
                splay_metadata.append({
                    'splay_idx': splay_idx,
                    'splay_id': improved_splay_id,
                    'valid_points': len(centerline),
                    'total_length': total_length,
                    'cone_length': splay.get('cone_length', total_length * 0.6),
                    'max_width': splay.get('max_width', 30.0),
                    'start_width': splay.get('start_width', 5.0),
                    'height_ratio': splay.get('height_ratio', 0.25),
                    'depth_ratio': splay.get('depth_ratio', 0.15),
                    'river_z_top': source_channel_z_top,  # Keep original river Z value for shape calculation
                    'z_offset': z_offset,  # New: Z value offset
                    'icc': icc,
                    'ic': ic,
                    'source_channel_id': source_channel_id,
                    'source_channel_z_top': source_channel_z_top
                })

            except Exception as e:
                print(f"Error processing crevasse splay {splay_idx}: {e}")
                continue

        if not all_centerlines_coords:
            print("No valid crevasse splay data")
            return 0

        num_splays = len(all_centerlines_coords)
        print(f"Preparing to process {num_splays} valid crevasse splays")

        # GPU rendering configuration
        try:
            device = cuda.get_current_device()
            max_threads_per_block = device.MAX_THREADS_PER_BLOCK

            if max_threads_per_block >= 1024:
                threads_per_block_x = 2
                threads_per_block_y = 16
                threads_per_block_z = 16
            elif max_threads_per_block >= 512:
                threads_per_block_x = 2
                threads_per_block_y = 8
                threads_per_block_z = 16
            else:
                threads_per_block_x = 1
                threads_per_block_y = 8
                threads_per_block_z = 8
        except:
            threads_per_block_x = 1
            threads_per_block_y = 8
            threads_per_block_z = 8

        try:
            print("Starting crevasse splay rendering")

            # Stage 1: Precompute distance matrix
            print("Precomputing crevasse splay distance matrix")

            max_points_per_splay = max(coords.shape[0] for coords in all_centerlines_coords)

            # Distance matrix
            distance_matrix = np.zeros((num_splays, nx, ny, 3), dtype=np.float32)
            padded_centerlines = np.zeros((num_splays, max_points_per_splay, 2), dtype=np.float32)
            splay_point_counts = np.zeros(num_splays, dtype=np.int32)

            for i, coords in enumerate(all_centerlines_coords):
                points = coords.shape[0]
                padded_centerlines[i, :points, :] = coords
                splay_point_counts[i] = points

            # Transfer to GPU
            d_padded_centerlines = cuda.to_device(padded_centerlines)
            d_splay_point_counts = cuda.to_device(splay_point_counts)
            d_distance_matrix = cuda.to_device(distance_matrix)

            # Parameters
            xmn_f32 = np.float32(xmn)
            ymn_f32 = np.float32(ymn)
            xsiz_f32 = np.float32(xsiz)
            ysiz_f32 = np.float32(ysiz)

            # Grid configuration
            blocks_per_grid_x = (num_splays + threads_per_block_x - 1) // threads_per_block_x
            blocks_per_grid_y = (nx + threads_per_block_y - 1) // threads_per_block_y
            blocks_per_grid_z = (ny + threads_per_block_z - 1) // threads_per_block_z

            threads_per_block = (threads_per_block_x, threads_per_block_y, threads_per_block_z)
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

            # Distance matrix precomputation
            batch_precompute_splay_distance_matrix_kernel[blocks_per_grid, threads_per_block](
                d_padded_centerlines, d_splay_point_counts, num_splays, max_points_per_splay,
                nx, ny, xmn_f32, ymn_f32, xsiz_f32, ysiz_f32, d_distance_matrix
            )
            cuda.synchronize()

            # Stage 2: Prepare rendering parameters
            print("Preparing rendering parameters")

            padded_params = np.zeros((num_splays, max_points_per_splay, 4), dtype=np.float32)
            splay_ids = np.zeros(num_splays, dtype=np.int32)
            start_widths = np.zeros(num_splays, dtype=np.float32)
            height_ratios = np.zeros(num_splays, dtype=np.float32)
            depth_ratios = np.zeros(num_splays, dtype=np.float32)
            river_z_tops = np.zeros(num_splays, dtype=np.float32)
            z_offsets = np.zeros(num_splays, dtype=np.float32)  # New: Z offset array

            # Source channel information arrays
            source_channel_ids = np.zeros(num_splays, dtype=np.int32)
            source_channel_z_tops = np.zeros(num_splays, dtype=np.float32)

            # icc and ic arrays
            splay_icc_values = np.zeros(num_splays, dtype=np.int32)
            splay_ic_values = np.zeros(num_splays, dtype=np.int32)

            for i, (params_data, metadata) in enumerate(zip(all_centerlines_params, splay_metadata)):
                points = params_data.shape[0]
                padded_params[i, :points, :] = params_data
                splay_ids[i] = metadata['splay_id']
                start_widths[i] = metadata['start_width']
                height_ratios[i] = metadata['height_ratio']
                depth_ratios[i] = metadata['depth_ratio']
                river_z_tops[i] = metadata['river_z_top']  # Keep original river Z value
                z_offsets[i] = metadata['z_offset']  # New: Z offset
                source_channel_ids[i] = metadata['source_channel_id']
                source_channel_z_tops[i] = metadata['source_channel_z_top']
                splay_icc_values[i] = metadata['icc']
                splay_ic_values[i] = metadata['ic']

            # Transfer rendering data
            d_padded_params = cuda.to_device(padded_params)
            d_splay_ids = cuda.to_device(splay_ids)
            d_start_widths = cuda.to_device(start_widths)
            d_height_ratios = cuda.to_device(height_ratios)
            d_depth_ratios = cuda.to_device(depth_ratios)
            d_river_z_tops = cuda.to_device(river_z_tops)
            d_z_offsets = cuda.to_device(z_offsets)  # New: transfer Z offsets
            d_channel_grid = cuda.to_device(channel.copy())

            # Transfer data
            d_source_channel_ids = cuda.to_device(source_channel_ids)
            d_source_channel_z_tops = cuda.to_device(source_channel_z_tops)
            d_channel_z_tops_array = cuda.to_device(channel_z_tops_array)
            d_splay_icc_values = cuda.to_device(splay_icc_values)
            d_splay_ic_values = cuda.to_device(splay_ic_values)

            if getpor:
                d_por = cuda.to_device(por)
            else:
                d_por = cuda.to_device(np.zeros((1, 1, 1), dtype=np.float32))

            # Stage 3: Render all crevasse splays
            print("Rendering all crevasse splays")

            zmn_f32 = np.float32(zmn)
            zsiz_f32 = np.float32(zsiz)
            random_seed = int(time.time() * 1000) % 10000000

            mxc = nc.max() if nc is not None and len(nc) > 0 else 1

            # Crevasse splay rendering - using kernel with Z offset
            batch_fortran_splay_distance_field_kernel_with_z_offset[blocks_per_grid, threads_per_block](
                d_padded_params, d_splay_point_counts, d_splay_ids, d_start_widths,
                d_height_ratios, d_depth_ratios, d_river_z_tops, d_z_offsets,  # New z_offsets parameter
                d_source_channel_ids, d_source_channel_z_tops,
                num_splays, max_points_per_splay, d_splay_icc_values, d_splay_ic_values,
                nx, ny, nz, xmn_f32, ymn_f32, zmn_f32, xsiz_f32, ysiz_f32, zsiz_f32,
                d_channel_grid, d_por, getpor, random_seed, d_distance_matrix,
                enable_migration, d_channel_z_tops_array,
                enable_splay_levee_clearing, mxc)
            cuda.synchronize()

            # Transfer results
            d_channel_grid.copy_to_host(channel)
            if getpor:
                d_por.copy_to_host(por)

            # Count filled cells
            total_filled_cells = np.count_nonzero((channel >= color_offset) &
                                                  (channel < color_offset + 10000))

            print(f"Crevasse splay rendering complete, filled cells: {total_filled_cells}")

        except Exception as render_error:
            print(f"Crevasse splay rendering failed: {render_error}")
            import traceback
            traceback.print_exc()
            return 0

        print(f"Crevasse splay rendering complete")
        print(f"Successfully processed {num_splays} crevasse splays")
        print(f"Total filled cells: {total_filled_cells}")
        print(f"Levee clearing mechanism: {'Disabled (migration mode)' if enable_migration else 'Executed (non-migration mode)'}")

        return total_filled_cells

    except Exception as e:
        print(f"Crevasse splay rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


@cuda.jit
def batch_fortran_splay_distance_field_kernel_with_z_offset(
        padded_params, splay_point_counts, splay_ids,
        start_widths, height_ratios, depth_ratios, river_z_tops, z_offsets,
        source_channel_ids, source_channel_z_tops,
        num_splays, max_points_per_splay, splay_icc_values, splay_ic_values,
        nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
        output_grid, output_por, getpor, random_seed,
        batch_distance_matrix, enable_migration, channel_z_tops_array,
        enable_splay_levee_clearing, mxc):
    """
    Crevasse splay rendering kernel
    """
    # Get current grid point indices
    splay_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ix = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    iy = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    if splay_idx >= num_splays or ix >= nx or iy >= ny:
        return

    # Get number of points for current crevasse splay
    num_centerline_points = splay_point_counts[splay_idx]

    if num_centerline_points <= 1:
        return

    # Get distance information from precomputed distance matrix
    min_distance = batch_distance_matrix[splay_idx, ix, iy, 0]
    closest_segment_idx = int(batch_distance_matrix[splay_idx, ix, iy, 1])
    closest_t = batch_distance_matrix[splay_idx, ix, iy, 2]

    # Boundary check
    if closest_segment_idx < 0 or closest_segment_idx >= num_centerline_points - 1:
        return

    # Interpolate parameters on closest line segment
    i = closest_segment_idx
    t = closest_t

    # Interpolate crevasse splay parameters
    s_position = gpu_linear_interpolate(padded_params[splay_idx, i, 0],
                                        padded_params[splay_idx, i + 1, 0], t)
    total_length = gpu_linear_interpolate(padded_params[splay_idx, i, 1],
                                          padded_params[splay_idx, i + 1, 1], t)
    cone_length = gpu_linear_interpolate(padded_params[splay_idx, i, 2],
                                         padded_params[splay_idx, i + 1, 2], t)
    max_width = gpu_linear_interpolate(padded_params[splay_idx, i, 3],
                                       padded_params[splay_idx, i + 1, 3], t)

    # Get crevasse splay parameters
    start_width = start_widths[splay_idx]
    height_ratio = height_ratios[splay_idx]
    depth_ratio = depth_ratios[splay_idx]
    river_z_top = river_z_tops[splay_idx]  # Original river Z value for shape calculation
    z_offset = z_offsets[splay_idx]  # Z offset
    current_splay_id = splay_ids[splay_idx]

    # Get source channel information
    source_channel_id = source_channel_ids[splay_idx]
    source_channel_z_top = source_channel_z_tops[splay_idx]

    current_icc = splay_icc_values[splay_idx]
    current_ic = splay_ic_values[splay_idx]

    # Key modification: keep shape calculation unchanged, only translate Z values at the end
    # Use original Fortran algorithm to calculate crevasse splay shape (based on original river Z value)
    splay_top_original, splay_bottom_original = gpu_fortran_crevasse_distance_field(
        min_distance, s_position, total_length, cone_length,
        max_width, start_width, height_ratio, depth_ratio, river_z_top
    )

    # Apply Z offset, translate entire crevasse splay
    splay_top = splay_top_original + z_offset
    splay_bottom = splay_bottom_original + z_offset

    # Check thickness
    if splay_top <= splay_bottom:
        return

    thickness = splay_top - splay_bottom
    if thickness < 0.05 or thickness > 60.0:
        return

    # Calculate Z direction grid indices
    iz_bottom_raw = (splay_bottom - zmn) / zsiz
    iz_top_raw = (splay_top - zmn) / zsiz

    # Convert to integer indices
    iz_bottom = int(math.floor(iz_bottom_raw))
    iz_top = int(math.ceil(iz_top_raw))

    # Boundary check
    iz_bottom = max(0, min(nz - 1, iz_bottom))
    iz_top = max(0, min(nz - 1, iz_top))

    # Ensure at least one layer to fill
    if iz_bottom > iz_top:
        iz_bottom = iz_top

    # Limit maximum fill layers
    max_layers = int(thickness / zsiz) + 5
    max_layers = min(max_layers, 150)

    if (iz_top - iz_bottom + 1) > max_layers:
        iz_top = iz_bottom + max_layers - 1
        iz_top = min(iz_top, nz - 1)

    # Crevasse splay voxel filling
    filled_count = 0
    max_fill_count = 100

    for iz in range(iz_bottom, iz_top + 1):
        if filled_count >= max_fill_count:
            break

        z_loc = zmn + (iz + 0.5) * zsiz

        # Z direction boundary check
        if z_loc < splay_bottom - zsiz * 0.5 or z_loc > splay_top + zsiz * 0.5:
            continue

        current_val = output_grid[ix, iy, iz]

        # Key modification: pass mxc parameter to priority judgment function
        can_fill = smart_priority_check_for_splay(
            current_val, source_channel_id, source_channel_z_top,
            enable_migration, channel_z_tops_array, mxc
        )

        if can_fill:
            output_grid[ix, iy, iz] = current_splay_id
            filled_count += 1

            # Only clear overlying levee voxels of same channel in non-migration mode
            if enable_splay_levee_clearing:
                clear_overlying_levee_voxels_same_channel(
                    output_grid, ix, iy, iz, nz, current_icc, current_ic, mxc
                )

            # Calculate porosity
            if getpor:
                # Relative position calculation
                z_relative = (z_loc - splay_bottom) / max(thickness, 0.1)
                z_relative = max(0.0, min(1.0, z_relative))

                distance_relative = min_distance / max(max_width, 0.1)
                distance_relative = max(0.0, min(1.0, distance_relative))

                # Fortran compatible porosity model
                base_porosity = 0.20
                depth_effect = -0.03 * z_relative
                distance_effect = 0.02 * distance_relative

                # Random variation
                cell_seed = random_seed + ix * 1000 + iy * 100 + iz
                rand_val = gpu_random(cell_seed, iz)
                random_variation = (rand_val - 0.5) * 0.025

                por_val = base_porosity + depth_effect + distance_effect + random_variation
                por_val = max(0.10, min(0.30, por_val))

                output_por[ix, iy, iz] = por_val


@cuda.jit
def batch_precompute_splay_distance_matrix_kernel(padded_centerlines, splay_point_counts,
                                                  num_splays, max_points_per_splay,
                                                  nx, ny, xmn, ymn, xsiz, ysiz,
                                                  batch_distance_matrix):
    """
    Batch precomputation of crevasse splay distance matrix CUDA kernel
    Parameters:
        padded_centerlines: Padded centerline coordinate array
        splay_point_counts: Crevasse splay point count array
        num_splays: Number of crevasse splays
        max_points_per_splay: Maximum points per crevasse splay
        nx, ny: Grid X, Y dimensions
        xmn, ymn: Grid origin coordinates
        xsiz, ysiz: Grid cell dimensions
        batch_distance_matrix: Output distance matrix
    Returns:
        void: No return value, results stored in batch_distance_matrix
    """
    # Get thread indices
    splay_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ix = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    iy = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    # Boundary check
    if splay_idx >= num_splays or ix >= nx or iy >= ny:
        return

    # Calculate world coordinates of grid point
    world_x = xmn + (ix + 0.5) * xsiz
    world_y = ymn + (iy + 0.5) * ysiz

    # Get number of points for current crevasse splay
    num_centerline_points = splay_point_counts[splay_idx]

    if num_centerline_points <= 1:
        return

    # Find minimum distance to centerline
    min_dist_sq = 1e30
    closest_segment = 0
    closest_t = 0.0

    # Iterate through all centerline segments of current crevasse splay
    for i in range(num_centerline_points - 1):
        x1 = padded_centerlines[splay_idx, i, 0]
        y1 = padded_centerlines[splay_idx, i, 1]
        x2 = padded_centerlines[splay_idx, i + 1, 0]
        y2 = padded_centerlines[splay_idx, i + 1, 1]

        # Data validity check
        if abs(x1) > 1e6 or abs(y1) > 1e6 or abs(x2) > 1e6 or abs(y2) > 1e6:
            continue

        # Calculate distance to current line segment
        dist_sq, t = gpu_distance_to_line_segment(world_x, world_y, x1, y1, x2, y2)

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_segment = i
            closest_t = t

    # Store calculation results
    min_distance = math.sqrt(min_dist_sq)
    batch_distance_matrix[splay_idx, ix, iy, 0] = min_distance
    batch_distance_matrix[splay_idx, ix, iy, 1] = closest_segment
    batch_distance_matrix[splay_idx, ix, iy, 2] = closest_t


@cuda.jit(device=True)
def gpu_fortran_crevasse_distance_field(distance_from_centerline, s_position, total_length, cone_length,
                                        max_width, start_width, height_ratio, depth_ratio, river_z_top):
    """
    GPU device function based on Fortran calc_lobe algorithm
    Parameters:
        distance_from_centerline: Distance from centerline
        s_position: Position along centerline
        total_length: Total length
        cone_length: Cone segment length
        max_width: Maximum width
        start_width: Starting width
        height_ratio: Height ratio
        depth_ratio: Depth ratio
        river_z_top: River top elevation
    Returns:
        tuple: (crevasse splay top elevation, crevasse splay bottom elevation)
    """
    # Width function calculation
    width_function = 0.0

    if 0 <= s_position <= cone_length:
        # Cone segment
        if cone_length > 0:
            progress = s_position / cone_length
            width_function = max_width - (max_width - start_width) * (1.0 - progress) * (1.0 - progress)
    elif cone_length < s_position <= total_length:
        # Fan segment
        if (total_length - cone_length) > 0:
            relative_pos = (s_position - cone_length) / (total_length - cone_length)
            sqrt_term = 1.0 - relative_pos * relative_pos
            if sqrt_term > 0:
                width_function = max_width * math.sqrt(sqrt_term)

    # Check if within width range
    if distance_from_centerline > width_function or width_function <= 0:
        return 0.0, 0.0

    # Extend width function maximum value limit
    max_reasonable_width = max(start_width * 5.0, 200.0)
    width_function = min(width_function, max_reasonable_width)

    # Height factor calculation
    relative_width = distance_from_centerline / width_function if width_function > 0 else 0
    height_factor = max(0.0, 1.0 - relative_width * relative_width)

    # Thickness calculation
    estimated_river_thickness = 5.0
    if height_ratio > 0 and depth_ratio > 0:
        normal_height_ratio = 0.25
        thickness_scale_factor = (height_ratio + depth_ratio) / (normal_height_ratio * 2)
        estimated_river_thickness = estimated_river_thickness * thickness_scale_factor

    base_thickness = estimated_river_thickness * (height_ratio + depth_ratio)

    # Distance decay
    distance_decay = max(0.1, 1.0 - distance_from_centerline / max(width_function, 1.0))

    # Flow direction decay
    flow_decay = 1.0
    if s_position > cone_length:
        flow_progress = (s_position - cone_length) / max(total_length - cone_length, 1.0)
        flow_decay = max(0.2, 1.0 - flow_progress * 0.8)

    # Calculate effective thickness
    effective_thickness = base_thickness * height_factor * distance_decay * flow_decay

    # Thickness range limits
    min_thickness = 0.1
    max_thickness = 50.0
    effective_thickness = max(min_thickness, min(effective_thickness, max_thickness))

    # Calculate top and bottom
    height_above = effective_thickness * 0.6
    depth_below = effective_thickness * 0.4

    splay_top = river_z_top + height_above
    splay_bottom = river_z_top - depth_below

    # Boundary check
    max_height_above_river = 30.0
    max_depth_below_river = 20.0

    splay_top = min(splay_top, river_z_top + max_height_above_river)
    splay_bottom = max(splay_bottom, river_z_top - max_depth_below_river)

    # Ensure top is greater than bottom
    if splay_top <= splay_bottom:
        splay_top = splay_bottom + min_thickness

    return splay_top, splay_bottom


@cuda.jit(device=True)
def smart_priority_check_for_splay(current_val, source_channel_id, source_channel_z_top,
                                   enable_migration, channel_z_tops_array, mxc):
    """
    Crevasse splay priority check function (improved version)
    Parameters:
        current_val: Current voxel value
        source_channel_id: Crevasse splay source channel ID
        source_channel_z_top: Crevasse splay source channel top elevation
        enable_migration: Whether migration mode is enabled
        channel_z_tops_array: Channel top elevation array
        mxc: Maximum channel count
    Returns:
        bool: True if crevasse splay can be rendered, False if cannot be rendered
    """
    # If voxel is empty, render directly
    if current_val == 0:
        return True

    # Determine current voxel type
    if 1 <= current_val < 10000:
        # Current voxel is channel
        existing_channel_id = current_val

        # If same source channel, do not overwrite
        if existing_channel_id == source_channel_id:
            return False

        # Different channel, judge based on mode
        if enable_migration:
            # Migration mode: compare channel IDs
            if existing_channel_id > source_channel_id:
                return False
            else:
                return True
        else:
            # Non-migration mode: compare Z values
            existing_channel_z_top = channel_z_tops_array[existing_channel_id]
            if existing_channel_z_top > source_channel_z_top:
                return False
            else:
                return True

    elif 10000 <= current_val < 20000:
        # Current voxel is levee
        if enable_migration:
            # Migration mode: need to compare source channel IDs of crevasse splay and levee
            levee_icc, levee_ic = extract_channel_info_from_levee_id_for_splay(current_val, mxc)

            if levee_icc >= 0 and levee_ic >= 0:
                # Calculate levee's source channel ID
                levee_source_channel_id = calculate_river_id_from_channel_info_for_splay(
                    levee_icc, levee_ic, mxc
                )

                # Compare crevasse splay source channel ID with levee source channel ID
                if source_channel_id > levee_source_channel_id:
                    # Crevasse splay source channel ID is larger, can overwrite levee
                    return True
                else:
                    # Levee source channel ID is larger or equal, crevasse splay cannot overwrite
                    return False
            else:
                # Cannot extract levee channel information, default allow overwrite
                return True
        else:
            # Non-migration mode: crevasse splay has higher priority than levee (keep original logic)
            return True

    elif current_val >= 20000:
        # Current voxel is other crevasse splay, allow overwrite (competition between crevasse splays)
        return True

    return False


@cuda.jit(device=True)
def clear_overlying_levee_voxels_same_channel(output_grid, ix, iy, iz_start, nz, target_icc, target_ic, mxc):
    """
    Clear natural levee voxels of the same channel above crevasse splay
    Parameters:
        output_grid: Output grid
        ix, iy, iz_start: Grid indices and starting Z layer
        nz: Grid size in Z direction
        target_icc, target_ic: Target channel complex and channel indices
        mxc: Maximum channel count
    Returns:
        void: No return value, directly modifies output_grid
    """
    # Check all Z layers above the crevasse splay
    for iz_above in range(iz_start + 1, nz):
        current_val = output_grid[ix, iy, iz_above]

        # Check if it is a natural levee voxel
        if 10000 <= current_val < 20000:
            # Extract channel information from natural levee ID
            levee_base_id = current_val - 10000
            mxc_multiplier = mxc * 2
            levee_icc = levee_base_id // mxc_multiplier - 1
            levee_ic = (levee_base_id % mxc_multiplier) // 2 + 1

            # Only clear natural levee voxels of the same channel
            if levee_icc == target_icc and levee_ic == target_ic:
                output_grid[ix, iy, iz_above] = 0
        elif current_val >= 20000:
            # If encountering other crevasse splay, stop clearing
            break
        elif 1 <= current_val < 10000:
            # If encountering channel voxel, stop clearing
            break

@cuda.jit(device=True)
def gpu_linear_interpolate(v1, v2, t):
    """
    GPU device function: Linear interpolation
    Parameters:
        v1: Starting value
        v2: Ending value
        t: Interpolation parameter (between 0 and 1)
    Returns:
        float: Interpolation result
    """
    return v1 + t * (v2 - v1)


def generate_improved_splay_id(crevasse_color_offset, icc, ic, splay_sub_idx):
    """Generate crevasse splay ID containing channel information"""
    splay_id = crevasse_color_offset + icc * 100 + ic * 10 + splay_sub_idx
    return splay_id

# Add these two device functions to render_facies.py (place after existing device functions)

@cuda.jit(device=True)
def calculate_river_id_from_channel_info_for_splay(icc, ic, mxc):
    """
    Calculate river ID from channel complex and channel number (crevasse splay version)
    Parameters:
        icc: Channel complex number
        ic: Channel number
        mxc: Maximum channel count
    Returns:
        int: River ID (limited to within 9999)
    """
    base_id = (icc - 1) * mxc + ic
    return min(base_id, 9999)

@cuda.jit(device=True)
def extract_channel_info_from_levee_id_for_splay(levee_id, mxc):
    """
    Extract channel information from levee ID (crevasse splay version)
    Parameters:
        levee_id: Levee ID
        mxc: Maximum channel count
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