"""
facies_generator.py
"""
import numpy as np
from ..engine import constants as const
from ..utils.utils import getval, get_value_from_distribution, is_distribution_dict
import time

try:
    import cupy as cp
    from numba import cuda

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries unavailable, using CPU rendering")


def generate_levees(params, icc, ic, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos):
    """
    Generate natural levees on both sides of channels with support for
    top/bottom cross-sections and thickness smoothing.

    Args:
        params: Global parameters dictionary
        icc, ic: Channel complex and channel indices
        ccx, ccz, cco: Complex center coordinates and orientation
        cx, cz: Channel center coordinates
        ccl, ct, cw, crelpos: Channel centerline, thickness, width and relative position data

    Returns:
        levees: List of levee data containing geometric and property information for left/right levees
    """
    levee_seed = const.get_levee_seed(icc, ic)
    current_state = np.random.get_state()
    np.random.seed(levee_seed)

    start_time = time.time()

    if not params.get('levee_enabled', False):
        np.random.set_state(current_state)
        return []

    print(f"Generating levees {icc}.{ic}")

    if not isinstance(ccl, np.ndarray) or ccl.ndim < 3:
        print(f"Error: Invalid ccl parameter")
        np.random.set_state(current_state)
        return []

    ny3 = ccl.shape[2]
    levees = []

    # Get basic parameters
    global_angle = cco[icc - 1] * np.pi / 180.0
    x_center_global = ccx[icc - 1] + cx[icc - 1, ic - 1]
    z_top = cz[icc - 1, ic - 1]
    ysiz = params['ysiz']
    ymn = params.get('ymn', 0.0)
    verbose = params.get('debug_parameter_resolution', False)

    # Convert channel centerline to global coordinates
    cos_ang = np.cos(global_angle)
    sin_ang = np.sin(global_angle)

    river_points = np.zeros((ny3, 2))

    # Get Y coordinate data
    if 'centerline_y_data' in params and params['centerline_y_data'] is not None:
        centerline_y_data = params['centerline_y_data']
        if (icc - 1 < centerline_y_data.shape[0] and
                ic - 1 < centerline_y_data.shape[1]):
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                if i < centerline_y_data.shape[2]:
                    local_y = centerline_y_data[icc - 1, ic - 1, i] - ymn
                else:
                    local_y = i * ysiz

                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
        else:
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                local_y = i * ysiz
                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
    else:
        if 'full_centerlines' in params:
            channel_key = f"{icc}_{ic}"
            if channel_key in params['full_centerlines']:
                full_centerline = params['full_centerlines'][channel_key]

                min_points = min(ny3, len(full_centerline))
                for i in range(min_points):
                    river_points[i, 0] = full_centerline[i, 0]
                    river_points[i, 1] = full_centerline[i, 1]

                if min_points < ny3:
                    for i in range(min_points, ny3):
                        river_points[i] = river_points[min_points - 1]
            else:
                for i in range(ny3):
                    local_x = ccl[icc - 1, ic - 1, i]
                    local_y = i * ysiz
                    river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                    river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
        else:
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                local_y = i * ysiz
                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang

    # Calculate normal vectors
    normal_vectors = calculate_simple_normal_vectors(river_points)

    # Preprocess channel geometry data
    processed_thickness = np.zeros(ny3)
    processed_width = np.zeros(ny3)

    # Process thickness data
    for i in range(ny3):
        if i < len(ct[icc - 1, ic - 1]):
            thickness_val = ct[icc - 1, ic - 1, i]
            processed_thickness[i] = thickness_val if not np.isnan(thickness_val) and thickness_val > 0.05 else 2.0
        else:
            processed_thickness[i] = 2.0

    # Process width data
    for i in range(ny3):
        if i < len(cw[icc - 1, ic - 1]):
            width_val = cw[icc - 1, ic - 1, i]
            processed_width[i] = width_val if not np.isnan(width_val) and width_val > 0.1 else 10.0
        else:
            processed_width[i] = 10.0

    # Store channel geometry data
    true_river_widths = processed_width.copy()
    true_river_thickness = processed_thickness.copy()

    # Separate channel Z coordinate data - consider Z value trend control
    true_river_z_top = np.zeros(ny3)
    true_river_z_bottom = np.zeros(ny3)

    # Get channel Z value trend offset data
    enable_river_trends = params.get('enable_river_trends', False)
    z_downstream_trend = params.get('z_downstream_trend', 0.0)

    if enable_river_trends and abs(z_downstream_trend) > 0.0001:
        # Use channel Z offset data
        centerline_z_offset = params.get('centerline_z_offset', None)
        if (centerline_z_offset is not None and
            icc - 1 < centerline_z_offset.shape[0] and
            ic - 1 < centerline_z_offset.shape[1]):

            for i in range(ny3):
                if i < centerline_z_offset.shape[2]:
                    z_offset = centerline_z_offset[icc - 1, ic - 1, i]
                    true_river_z_top[i] = z_top + z_offset
                else:
                    true_river_z_top[i] = z_top
                true_river_z_bottom[i] = true_river_z_top[i] - true_river_thickness[i]

            if verbose:
                print(f"    Levees applying Z trend control: start Z={true_river_z_top[0]:.2f}, end Z={true_river_z_top[-1]:.2f}")
        else:
            # Fallback to fixed Z value
            for i in range(ny3):
                true_river_z_top[i] = z_top
                true_river_z_bottom[i] = z_top - true_river_thickness[i]
            if verbose:
                print(f"    Warning: Cannot get channel Z offset data, levees using fixed Z value")
    else:
        # Trend control not enabled, use fixed Z value
        for i in range(ny3):
            true_river_z_top[i] = z_top
            true_river_z_bottom[i] = z_top - true_river_thickness[i]

    # Get levee parameters
    try:
        levee_width_ratio_param = params.get('levee_width_ratio', [0.5, 0.7, 0.9])
        levee_height_ratio_param = params.get('levee_height_ratio', [0.8, 1.0, 1.2])
        levee_depth_ratio_param = params.get('levee_depth_ratio', [0.3, 0.4, 0.5])

        levee_width_ratio = get_value_from_distribution(levee_width_ratio_param) if is_distribution_dict(
            levee_width_ratio_param) else getval(levee_width_ratio_param)
        levee_height_ratio = get_value_from_distribution(levee_height_ratio_param) if is_distribution_dict(
            levee_height_ratio_param) else getval(levee_height_ratio_param)
        levee_depth_ratio = get_value_from_distribution(levee_depth_ratio_param) if is_distribution_dict(
            levee_depth_ratio_param) else getval(levee_depth_ratio_param)

        # Ensure parameters within reasonable range
        levee_width_ratio = max(0.1, min(50.0, levee_width_ratio))
        levee_height_ratio = max(0.1, min(3.0, levee_height_ratio))
        levee_depth_ratio = max(0.1, min(20.0, levee_depth_ratio))

    except Exception as e:
        print(f"Levee parameter error: {e}, using default values")
        levee_width_ratio = 0.7
        levee_height_ratio = 1.0
        levee_depth_ratio = 0.4

    # Geometric parameter calculation
    levee_widths = true_river_widths * levee_width_ratio
    levee_heights = true_river_thickness * levee_height_ratio
    levee_depths = true_river_thickness * levee_depth_ratio

    # Ensure minimum values
    levee_widths = np.maximum(levee_widths, 0.5)
    levee_heights = np.maximum(levee_heights, 0.1)
    levee_depths = np.maximum(levee_depths, 0.1)

    # Levee thickness smoothing
    smoothing_enabled = params.get('levee_thickness_smoothing_enabled', True)
    smoothing_iterations = params.get('levee_thickness_smoothing_iterations', 3)
    smoothing_strength = params.get('levee_thickness_smoothing_strength', 0.4)

    if smoothing_enabled and ny3 > 3:
        print("Applying thickness smoothing")
        levee_heights, levee_depths, levee_widths = smooth_levee_thickness_profiles(
            levee_heights, levee_depths, levee_widths,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength
        )

    # Distance calculation
    levee_distances = calculate_simple_levee_distance(river_points, true_river_widths, None)

    # Generate levees for left and right sides
    for side_name, side_factor in [('left', -1.0), ('right', 1.0)]:

        # Calculate levee boundary point positions
        inner_edge_points = np.zeros((ny3, 2))
        outer_edge_points = np.zeros((ny3, 2))
        center_points = np.zeros((ny3, 2))

        for i in range(ny3):
            # Inner edge position
            inner_offset = side_factor * levee_distances[i]
            inner_edge_points[i] = river_points[i] + inner_offset * normal_vectors[i]

            # Outer edge position
            outer_offset = side_factor * (levee_distances[i] + levee_widths[i])
            outer_edge_points[i] = river_points[i] + outer_offset * normal_vectors[i]

            # Center point position
            center_offset = side_factor * (levee_distances[i] + levee_widths[i] / 2.0)
            center_points[i] = river_points[i] + center_offset * normal_vectors[i]

        # Smoothing
        for points_array in [inner_edge_points, outer_edge_points, center_points]:
            for coord_idx in range(2):
                for _ in range(2):
                    temp_coords = points_array[:, coord_idx].copy()
                    for i in range(1, ny3 - 1):
                        window_start = max(0, i - 2)
                        window_end = min(ny3, i + 3)
                        points_array[i, coord_idx] = np.mean(temp_coords[window_start:window_end])

        # Z coordinate calculation
        levee_z_coords = np.zeros(ny3)
        levee_base_z = np.zeros(ny3)
        levee_total_thickness = np.zeros(ny3)

        for i in range(ny3):
            levee_base_z[i] = true_river_z_top[i] - levee_depths[i]
            levee_z_coords[i] = true_river_z_top[i] + levee_heights[i]
            levee_total_thickness[i] = levee_depths[i] + levee_heights[i]

            # Reasonableness check
            min_total_thickness = 0.2
            if levee_total_thickness[i] < min_total_thickness:
                adjustment_factor = min_total_thickness / levee_total_thickness[i]
                levee_depths[i] *= adjustment_factor
                levee_heights[i] *= adjustment_factor
                levee_base_z[i] = true_river_z_top[i] - levee_depths[i]
                levee_z_coords[i] = true_river_z_top[i] + levee_heights[i]
                levee_total_thickness[i] = levee_depths[i] + levee_heights[i]


        # Create levee data structure
        levee = {
            'side': side_name,
            'icc': icc,
            'ic': ic,
            'angle': global_angle,

            # Global coordinate data
            'global_coords': {
                'x': center_points[:, 0],
                'y': center_points[:, 1],
                'z': levee_z_coords,
                'inner_edge': {
                    'x': inner_edge_points[:, 0],
                    'y': inner_edge_points[:, 1],
                    'z': levee_base_z
                },
                'outer_edge': {
                    'x': outer_edge_points[:, 0],
                    'y': outer_edge_points[:, 1],
                    'z': levee_base_z
                }
            },

            # Levee geometric parameters
            'width': levee_widths,
            'height': levee_heights,
            'depth': levee_depths,
            'total_thickness': levee_total_thickness,
            'base_height': np.full(ny3, 0.1),

            # Channel geometry data
            'true_river_width': true_river_widths,
            'true_river_thickness': true_river_thickness,
            'true_river_z_top': true_river_z_top,
            'true_river_z_bottom': true_river_z_bottom,

            # Quality metrics
            'quality_metrics': {
                'width_ratio_applied': levee_width_ratio,
                'height_ratio_applied': levee_height_ratio,
                'depth_ratio_applied': levee_depth_ratio,
                'thickness_smoothing_applied': smoothing_enabled,
                'smoothing_iterations': smoothing_iterations if smoothing_enabled else 0,
                'smoothing_strength': smoothing_strength if smoothing_enabled else 0,
                'geological_continuity': True,
                'fortran_compatible': True,
                'cross_river_top': True,
                'position_fixed': True,
                'thickness_independent': True,
                'parameter_control_correct': True,
                'parameter_separation_complete': True,
                'z_coordinate_separation_complete': True,
                'z_trend_control_applied': enable_river_trends and abs(z_downstream_trend) > 0.0001,
                'z_downstream_trend_value': z_downstream_trend
            },

            # Channel data
            'river_coords': {
                'x': river_points[:, 0].copy(),
                'y': river_points[:, 1].copy()
            },

            # Compatibility data
            'centerline': np.zeros(ny3),
            'left_edge': np.zeros(ny3),
            'right_edge': np.zeros(ny3),
            'x': center_points[:, 0].copy(),
            'y': center_points[:, 1].copy(),
            'z': levee_z_coords.copy()
        }

        # Calculate compatibility data
        for i in range(ny3):
            # Convert center point to local coordinates
            dx = center_points[i, 0] - x_center_global
            dy = center_points[i, 1] - ymn
            levee['centerline'][i] = dx * cos_ang + dy * sin_ang

            # Calculate left and right edges
            inner_dx = inner_edge_points[i, 0] - x_center_global
            inner_dy = inner_edge_points[i, 1] - ymn
            inner_local_x = inner_dx * cos_ang + inner_dy * sin_ang

            outer_dx = outer_edge_points[i, 0] - x_center_global
            outer_dy = outer_edge_points[i, 1] - ymn
            outer_local_x = outer_dx * cos_ang + outer_dy * sin_ang

            levee['left_edge'][i] = min(inner_local_x, outer_local_x)
            levee['right_edge'][i] = max(inner_local_x, outer_local_x)

        levees.append(levee)

    # Validation and reporting
    total_time = time.time() - start_time
    print(f"Levee generation complete: {len(levees)} levees in {total_time:.1f}s")

    # Save to params
    if len(levees) > 0:
        for levee in levees:
            levee_data = {
                'levee_id': f"{icc}_{ic}_{levee['side']}",
                'channel_id': f"{icc}_{ic}",
                'side': levee['side'],
                'icc': icc,
                'ic': ic,
                'complex_x': ccx[icc - 1],
                'complex_z': ccz[icc - 1],
                'channel_x': cx[icc - 1, ic - 1],
                'channel_z': cz[icc - 1, ic - 1],
                'global_angle': cco[icc - 1],
                'centerline_points': [],
                'inner_edge_points': [],
                'outer_edge_points': [],
                'width_profile': levee['width'].copy(),
                'height_profile': levee['height'].copy(),
                'depth_profile': levee['depth'].copy(),
                'total_thickness_profile': levee['total_thickness'].copy(),
                'base_height_profile': levee['base_height'].copy(),
                'true_river_width_profile': levee['true_river_width'].copy(),
                'true_river_thickness_profile': levee['true_river_thickness'].copy(),
                'true_river_z_top_profile': levee['true_river_z_top'].copy(),
                'true_river_z_bottom_profile': levee['true_river_z_bottom'].copy(),
                'quality_metrics': levee['quality_metrics'].copy()
            }

            # Fill point data
            for i in range(ny3):
                centerline_point = {
                    'point_index': i,
                    'global_x': levee['global_coords']['x'][i],
                    'global_y': levee['global_coords']['y'][i],
                    'global_z': levee['global_coords']['z'][i],
                    'width': levee['width'][i],
                    'height': levee['height'][i],
                    'depth': levee['depth'][i],
                    'total_thickness': levee['total_thickness'][i],
                    'base_height': levee['base_height'][i],
                    'true_river_width': levee['true_river_width'][i],
                    'true_river_thickness': levee['true_river_thickness'][i],
                    'true_river_z_top': levee['true_river_z_top'][i],
                    'true_river_z_bottom': levee['true_river_z_bottom'][i]
                }
                levee_data['centerline_points'].append(centerline_point)

                inner_point = {
                    'point_index': i,
                    'global_x': levee['global_coords']['inner_edge']['x'][i],
                    'global_y': levee['global_coords']['inner_edge']['y'][i],
                    'global_z': levee['global_coords']['inner_edge']['z'][i],
                }
                levee_data['inner_edge_points'].append(inner_point)

                outer_point = {
                    'point_index': i,
                    'global_x': levee['global_coords']['outer_edge']['x'][i],
                    'global_y': levee['global_coords']['outer_edge']['y'][i],
                    'global_z': levee['global_coords']['outer_edge']['z'][i],
                }
                levee_data['outer_edge_points'].append(outer_point)

            if 'levees_data' not in params:
                params['levees_data'] = []
            params['levees_data'].append(levee_data)

    np.random.set_state(current_state)

    return levees

def smooth_levee_thickness_profiles(levee_heights, levee_depths, levee_widths,
                                    smoothing_iterations=3, smoothing_strength=0.5):
    """
    Smooth levee thickness-related parameters to eliminate abrupt changes.

    Args:
        levee_heights: Levee height array
        levee_depths: Levee depth array
        levee_widths: Levee width array
        smoothing_iterations: Number of smoothing iterations
        smoothing_strength: Smoothing strength (0-1 range)

    Returns:
        Tuple of smoothed (levee_heights, levee_depths, levee_widths)
    """
    ny3 = len(levee_heights)

    smooth_heights = levee_heights.copy()
    smooth_depths = levee_depths.copy()
    smooth_widths = levee_widths.copy()

    for iteration in range(smoothing_iterations):
        # Smooth heights
        temp_heights = smooth_heights.copy()
        for i in range(1, ny3 - 1):
            window_start = max(0, i - 1)
            window_end = min(ny3, i + 2)

            local_mean = np.mean(temp_heights[window_start:window_end])
            smooth_heights[i] = (1 - smoothing_strength) * temp_heights[i] + smoothing_strength * local_mean

        # Smooth depths
        temp_depths = smooth_depths.copy()
        for i in range(1, ny3 - 1):
            window_start = max(0, i - 1)
            window_end = min(ny3, i + 2)

            local_mean = np.mean(temp_depths[window_start:window_end])
            smooth_depths[i] = (1 - smoothing_strength) * temp_depths[i] + smoothing_strength * local_mean

        # Lightly smooth widths
        temp_widths = smooth_widths.copy()
        for i in range(1, ny3 - 1):
            window_start = max(0, i - 1)
            window_end = min(ny3, i + 2)

            local_mean = np.mean(temp_widths[window_start:window_end])
            smooth_widths[i] = (1 - smoothing_strength * 0.5) * temp_widths[i] + smoothing_strength * 0.5 * local_mean

    # Boundary treatment - use linear interpolation to smooth endpoints
    if ny3 > 2:
        smooth_heights[0] = 0.7 * smooth_heights[0] + 0.3 * smooth_heights[1]
        smooth_depths[0] = 0.7 * smooth_depths[0] + 0.3 * smooth_depths[1]
        smooth_widths[0] = 0.7 * smooth_widths[0] + 0.3 * smooth_widths[1]

        smooth_heights[-1] = 0.7 * smooth_heights[-1] + 0.3 * smooth_heights[-2]
        smooth_depths[-1] = 0.7 * smooth_depths[-1] + 0.3 * smooth_depths[-2]
        smooth_widths[-1] = 0.7 * smooth_widths[-1] + 0.3 * smooth_widths[-2]

    # Ensure minimum value constraints
    smooth_heights = np.maximum(smooth_heights, 0.1)
    smooth_depths = np.maximum(smooth_depths, 0.1)
    smooth_widths = np.maximum(smooth_widths, 0.5)

    return smooth_heights, smooth_depths, smooth_widths

def calculate_simple_levee_distance(river_points, channel_width, levee_width):
    """
    Calculate distance from levee inner edge to channel centerline.
    Position is fixed and unaffected by levee width.

    Args:
        river_points: Channel centerline coordinate points array
        channel_width: Channel width array
        levee_width: Levee width array

    Returns:
        distances: Levee inner edge distance array
    """
    n = len(river_points)
    distances = np.zeros(n)

    safety_gap = 2.0  # Fixed safety gap

    for i in range(n):
        river_half_width = channel_width[i] / 2.0
        distances[i] = river_half_width + safety_gap

        # Ensure minimum distance
        min_distance = river_half_width + 1.0
        distances[i] = max(distances[i], min_distance)

    return distances

def calculate_simple_normal_vectors(river_points):
    """
    Calculate normal vectors at each point along channel centerline
    using fixed window for tangent calculation.

    Args:
        river_points: Channel centerline coordinate points array (n×2)

    Returns:
        normal_vectors: Normal vector array (n×2)
    """
    n = len(river_points)
    normal_vectors = np.zeros((n, 2))

    for i in range(n):
        window_size = 3

        # Calculate tangent vector
        if i == 0:
            end_idx = min(window_size, n - 1)
            dx = river_points[end_idx, 0] - river_points[0, 0]
            dy = river_points[end_idx, 1] - river_points[0, 1]
        elif i == n - 1:
            start_idx = max(0, n - 1 - window_size)
            dx = river_points[-1, 0] - river_points[start_idx, 0]
            dy = river_points[-1, 1] - river_points[start_idx, 1]
        else:
            start_idx = max(0, i - window_size)
            end_idx = min(n - 1, i + window_size)
            dx = river_points[end_idx, 0] - river_points[start_idx, 0]
            dy = river_points[end_idx, 1] - river_points[start_idx, 1]

        tangent_length = np.sqrt(dx ** 2 + dy ** 2)
        if tangent_length > 1e-8:
            tx, ty = dx / tangent_length, dy / tangent_length
        else:
            tx, ty = 1.0, 0.0

        # Normal vector (tangent rotated 90° counterclockwise)
        normal_vectors[i] = [-ty, tx]

    return normal_vectors


def generate_crevasse_splays(params, icc, ic, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos):
    """
    Generate crevasse splays at convex bank edges based on channel curvature
    with support for sinuosity and thickness ratio control.

    Args:
        params: Global parameters dictionary
        icc, ic: Channel complex and channel indices
        ccx, ccz, cco: Complex center coordinates and orientation
        cx, cz: Channel center coordinates
        ccl, ct, cw, crelpos: Channel centerline, thickness, width and relative position data

    Returns:
        crevasse_splays: List of crevasse splay data
    """
    crevasse_seed = const.get_crevasse_seed(icc, ic)
    current_state = np.random.get_state()
    np.random.seed(crevasse_seed)

    start_time = time.time()

    if not params.get('crevasse_enabled', False):
        np.random.set_state(current_state)
        return []

    print(f"Generating crevasse splays {icc}.{ic}")

    if not isinstance(ccl, np.ndarray) or ccl.ndim < 3:
        print(f"Error: Invalid ccl parameter")
        np.random.set_state(current_state)
        return []

    ny3 = ccl.shape[2]
    crevasse_splays = []

    # Get basic parameters
    global_angle = cco[icc - 1] * np.pi / 180.0
    x_center_global = ccx[icc - 1] + cx[icc - 1, ic - 1]
    z_top = cz[icc - 1, ic - 1]  # Channel start Z value (reference only)
    ysiz = params['ysiz']
    ymn = params.get('ymn', 0.0)
    verbose = params.get('debug_parameter_resolution', False)

    # Get channel Z value trend control parameters
    enable_river_trends = params.get('enable_river_trends', False)
    z_downstream_trend = params.get('z_downstream_trend', 0.0)
    centerline_z_offset = params.get('centerline_z_offset', None)

    if verbose:
        print(f"    Z trend control status: enable_trends={enable_river_trends}, z_trend={z_downstream_trend}")
        if enable_river_trends and centerline_z_offset is not None:
            print(f"    Channel Z offset array shape: {centerline_z_offset.shape}")

    # Get crevasse splay parameters
    try:
        crevasse_per_channel_param = params.get('crevasse_per_channel', [1, 2, 4])
        crevasse_angle_param = params.get('crevasse_angle', [30, 60, 90])
        crevasse_length_ratio_param = params.get('crevasse_length_ratio', [0.3, 0.5, 0.7])
        crevasse_width_ratio_param = params.get('crevasse_width_ratio', [0.3, 0.5, 0.7])

        # Get sinuosity and thickness ratio parameters
        crevasse_sinuosity_param = None
        crevasse_height_ratio_param = None

        region_configs = params.get('region_configs', {})

        # Determine current channel region
        partition_axis = params.get('partition_axis', 'x')
        num_regions = params.get('num_regions', 3)

        if partition_axis == 'x':
            nx = params.get('nx', 250)
            region_width = nx // num_regions
            current_region = min(num_regions - 1, max(0, int((icc - 1) // max(1, region_width))))
        else:
            ny = params.get('ny', 250)
            region_height = ny // num_regions
            current_region = min(num_regions - 1, max(0, int((ic - 1) // max(1, region_height))))

        # Check region configuration
        if region_configs and current_region in region_configs:
            region_config = region_configs[current_region]
            if 'crevasse_sinuosity' in region_config:
                crevasse_sinuosity_param = region_config['crevasse_sinuosity']
            if 'crevasse_height_ratio' in region_config:
                crevasse_height_ratio_param = region_config['crevasse_height_ratio']

        # Use global configuration as default
        if crevasse_sinuosity_param is None:
            crevasse_sinuosity_param = params.get('crevasse_sinuosity', 1.2)
        if crevasse_height_ratio_param is None:
            crevasse_height_ratio_param = params.get('crevasse_height_ratio', 0.5)

        crevasse_per_channel = get_value_from_distribution(crevasse_per_channel_param) if is_distribution_dict(
            crevasse_per_channel_param) else getval(crevasse_per_channel_param)

        crevasse_per_channel = max(1, min(10, int(crevasse_per_channel)))

        # Get sinuosity value
        if isinstance(crevasse_sinuosity_param, (int, float)):
            crevasse_sinuosity = float(crevasse_sinuosity_param)
        elif isinstance(crevasse_sinuosity_param, list):
            crevasse_sinuosity = getval(crevasse_sinuosity_param)
        elif isinstance(crevasse_sinuosity_param, dict):
            crevasse_sinuosity = get_value_from_distribution(crevasse_sinuosity_param)
        else:
            crevasse_sinuosity = 1.2

        crevasse_sinuosity = max(1.0, min(3.0, crevasse_sinuosity))

        # Get thickness ratio value
        if isinstance(crevasse_height_ratio_param, (int, float)):
            crevasse_height_ratio = float(crevasse_height_ratio_param)
        elif isinstance(crevasse_height_ratio_param, list):
            crevasse_height_ratio = getval(crevasse_height_ratio_param)
        elif isinstance(crevasse_height_ratio_param, dict):
            crevasse_height_ratio = get_value_from_distribution(crevasse_height_ratio_param)
        else:
            crevasse_height_ratio = 0.5

        crevasse_height_ratio = max(0.02, min(5.0, crevasse_height_ratio))

    except Exception as e:
        print(f"Crevasse splay parameter error: {e}, using default values")
        crevasse_per_channel = 2
        crevasse_angle_param = [30, 60, 90]
        crevasse_length_ratio_param = [0.3, 0.5, 0.7]
        crevasse_width_ratio_param = [0.3, 0.5, 0.7]
        crevasse_sinuosity = 1.2
        crevasse_height_ratio = 0.5

    # Calculate channel geometry data
    cos_ang = np.cos(global_angle)
    sin_ang = np.sin(global_angle)

    river_points = np.zeros((ny3, 2))
    river_left_bank = np.zeros((ny3, 2))
    river_right_bank = np.zeros((ny3, 2))
    river_widths = np.zeros(ny3)
    river_thicknesses = np.zeros(ny3)

    # Get channel centerline coordinates
    if 'centerline_y_data' in params and params['centerline_y_data'] is not None:
        centerline_y_data = params['centerline_y_data']
        if (icc - 1 < centerline_y_data.shape[0] and
                ic - 1 < centerline_y_data.shape[1]):
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                if i < centerline_y_data.shape[2]:
                    local_y = centerline_y_data[icc - 1, ic - 1, i] - ymn
                else:
                    local_y = i * ysiz

                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
        else:
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                local_y = i * ysiz
                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
    else:
        if 'full_centerlines' in params:
            channel_key = f"{icc}_{ic}"
            if channel_key in params['full_centerlines']:
                full_centerline = params['full_centerlines'][channel_key]

                min_points = min(ny3, len(full_centerline))
                for i in range(min_points):
                    river_points[i, 0] = full_centerline[i, 0]
                    river_points[i, 1] = full_centerline[i, 1]

                if min_points < ny3:
                    for i in range(min_points, ny3):
                        river_points[i] = river_points[min_points - 1]
            else:
                for i in range(ny3):
                    local_x = ccl[icc - 1, ic - 1, i]
                    local_y = i * ysiz
                    river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                    river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang
        else:
            for i in range(ny3):
                local_x = ccl[icc - 1, ic - 1, i]
                local_y = i * ysiz
                river_points[i, 0] = x_center_global + local_x * cos_ang - local_y * sin_ang
                river_points[i, 1] = ymn + local_x * sin_ang + local_y * cos_ang

    # Get channel width and thickness
    for i in range(ny3):
        if i < len(cw[icc - 1, ic - 1]):
            width_val = cw[icc - 1, ic - 1, i]
            river_widths[i] = width_val if not np.isnan(width_val) and width_val > 0.1 else 10.0
        else:
            river_widths[i] = 10.0

        if i < len(ct[icc - 1, ic - 1]):
            thickness_val = ct[icc - 1, ic - 1, i]
            river_thicknesses[i] = thickness_val if not np.isnan(thickness_val) and thickness_val > 0.05 else 2.0
        else:
            river_thicknesses[i] = 2.0

    # Calculate channel curvature and bank edges
    curvatures = np.zeros(ny3)
    normal_vectors = np.zeros((ny3, 2))

    # Calculate tangent and normal vectors for each point
    for i in range(ny3):
        if i == 0:
            tangent = river_points[1] - river_points[0]
        elif i == ny3 - 1:
            tangent = river_points[i] - river_points[i - 1]
        else:
            window_size = min(2, i, ny3 - 1 - i)
            start_idx = max(0, i - window_size)
            end_idx = min(ny3 - 1, i + window_size)
            tangent = river_points[end_idx] - river_points[start_idx]

        tangent_length = np.linalg.norm(tangent)
        if tangent_length > 1e-6:
            tangent = tangent / tangent_length
        else:
            tangent = np.array([1.0, 0.0])

        # Calculate normal vector
        normal_vectors[i] = np.array([-tangent[1], tangent[0]])

    # Calculate channel left and right bank edge points
    for i in range(ny3):
        half_width = river_widths[i] / 2.0
        river_right_bank[i] = river_points[i] + normal_vectors[i] * half_width
        river_left_bank[i] = river_points[i] - normal_vectors[i] * half_width

    # Calculate curvature
    for i in range(1, ny3 - 1):
        p1 = river_points[i - 1]
        p2 = river_points[i]
        p3 = river_points[i + 1]

        v1 = p2 - p1
        v2 = p3 - p2

        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 > 1e-6 and len2 > 1e-6:
            curvatures[i] = cross_product / (len1 * len2 * (len1 + len2) / 2)

    # Light smoothing of curvature
    for _ in range(1):
        temp_curvatures = curvatures.copy()
        for i in range(1, ny3 - 1):
            curvatures[i] = 0.6 * temp_curvatures[i] + 0.2 * temp_curvatures[i - 1] + 0.2 * temp_curvatures[i + 1]

    # Convex bank identification
    convex_bank_indices = []
    convex_bank_data = []

    min_curvature_threshold = 0.0002

    for i in range(8, ny3 - 8):
        curvature_val = curvatures[i]
        abs_curvature = abs(curvature_val)

        if abs_curvature > min_curvature_threshold:
            if curvature_val > 0:
                # Positive curvature: left bank is convex
                convex_side = 'left'
                start_point = river_left_bank[i].copy()
                outward_direction = -normal_vectors[i]
                curvature_description = "right bend, left bank convex"
            else:
                # Negative curvature: right bank is convex
                convex_side = 'right'
                start_point = river_right_bank[i].copy()
                outward_direction = normal_vectors[i]
                curvature_description = "left bend, right bank convex"

            convex_bank_indices.append(i)
            convex_bank_data.append({
                'index': i,
                'side': convex_side,
                'start_point': start_point,
                'outward_direction': outward_direction,
                'curvature': curvature_val,
                'abs_curvature': abs_curvature,
                'weight': np.sqrt(abs_curvature) + 0.1,
                'description': curvature_description
            })

    if len(convex_bank_data) == 0:
        print(f"Warning: No sufficiently curved convex banks")
        np.random.set_state(current_state)
        return []

    print(f"Identified {len(convex_bank_data)} convex bank locations")

    weights = np.array([data['weight'] for data in convex_bank_data])

    # Generate crevasse splays at convex bank edges
    total_splays_generated = 0

    for splay_idx in range(crevasse_per_channel):
        if np.sum(weights) <= 0:
            continue

        prob_weights = weights / np.sum(weights)
        selected_idx = np.random.choice(len(convex_bank_data), p=prob_weights)
        selected_data = convex_bank_data[selected_idx]

        node_idx = selected_data['index']
        convex_side = selected_data['side']
        start_point = selected_data['start_point']
        outward_direction = selected_data['outward_direction']
        curvature_at_node = selected_data['curvature']
        abs_curvature = selected_data['abs_curvature']
        description = selected_data['description']

        base_splay_angle_rad = np.arctan2(outward_direction[1], outward_direction[0])

        angle_range = get_value_from_distribution(crevasse_angle_param) if is_distribution_dict(
            crevasse_angle_param) else getval(crevasse_angle_param)
        angle_range = max(15, min(75, angle_range))

        num_centerlines = np.random.choice([3, 4, 5, 6], p=[0.1, 0.2, 0.5, 0.2])

        print(f"Crevasse splay {splay_idx + 1}: node{node_idx}, {description}")

        for centerline_idx in range(num_centerlines):
            # Calculate angle offset
            if num_centerlines == 1:
                angle_offset = 0
            else:
                angle_step = angle_range / (num_centerlines - 1)
                angle_offset = -angle_range / 2 + centerline_idx * angle_step

            random_perturbation = np.random.uniform(-8, 8)
            final_angle_offset = angle_offset + random_perturbation

            splay_angle_rad = base_splay_angle_rad + np.radians(final_angle_offset)
            splay_angle_deg = np.degrees(splay_angle_rad)

            # Calculate crevasse splay geometric parameters
            base_width = river_widths[node_idx]
            base_thickness = river_thicknesses[node_idx]

            curvature_factor = min(1.4, 1.0 + abs_curvature * 400)

            length_ratio = get_value_from_distribution(crevasse_length_ratio_param) if is_distribution_dict(
                crevasse_length_ratio_param) else getval(crevasse_length_ratio_param)
            width_ratio = get_value_from_distribution(crevasse_width_ratio_param) if is_distribution_dict(
                crevasse_width_ratio_param) else getval(crevasse_width_ratio_param)

            # Calculate length
            model_x_size = params.get('nx', 250) * params.get('xsiz', 24)
            model_y_size = params.get('ny', 250) * params.get('ysiz', 24)
            max_reasonable_length = min(model_x_size, model_y_size) * 0.18

            raw_total_length = base_width * 4 * length_ratio * curvature_factor
            total_length = min(raw_total_length, max_reasonable_length)
            cone_length = total_length * 0.6

            # Calculate width
            max_width = base_width * 0.8 * width_ratio * curvature_factor
            start_width = base_width * 0.15

            base_height_ratio = 0.12 * curvature_factor
            base_depth_ratio = 0.06 * curvature_factor

            height_ratio = base_height_ratio * crevasse_height_ratio
            depth_ratio = base_depth_ratio * crevasse_height_ratio

            crevasse_thickness = base_thickness * crevasse_height_ratio

            # ========== Corrected Z coordinate calculation ==========
            # Get actual Z value of channel node corresponding to crevasse splay generation position (considering trend control)
            base_z_for_crevasse = z_top  # Default to channel start Z value

            if enable_river_trends and abs(z_downstream_trend) > 0.0001 and centerline_z_offset is not None:
                # Check array bounds
                if (icc - 1 < centerline_z_offset.shape[0] and
                        ic - 1 < centerline_z_offset.shape[1] and
                        node_idx < centerline_z_offset.shape[2]):

                    # Get Z offset for this channel node
                    z_offset_at_node = centerline_z_offset[icc - 1, ic - 1, node_idx]
                    base_z_for_crevasse = z_top + z_offset_at_node

                    if verbose and splay_idx == 0 and centerline_idx == 0:
                        print(f"    Crevasse splay Z value correction: node{node_idx}")
                        print(f"      Channel start Z value: {z_top:.3f}")
                        print(f"      Node Z offset: {z_offset_at_node:.3f}")
                        print(f"      Node actual Z value: {base_z_for_crevasse:.3f}")
                        print(f"      Z trend parameter: {z_downstream_trend}")
                else:
                    if verbose and splay_idx == 0 and centerline_idx == 0:
                        print(f"    Warning: Cannot get channel Z offset data for node{node_idx}")
                        print(f"      Array shape: {centerline_z_offset.shape}")
                        print(f"      Access indices: [{icc - 1}, {ic - 1}, {node_idx}]")
                        print(f"      Using channel start Z value: {base_z_for_crevasse:.3f}")
            else:
                if verbose and splay_idx == 0 and centerline_idx == 0:
                    if not enable_river_trends:
                        print(f"    Channel trend control not enabled, using channel start Z value: {base_z_for_crevasse:.3f}")
                    elif abs(z_downstream_trend) <= 0.0001:
                        print(f"    Z trend value too small ({z_downstream_trend}), using channel start Z value: {base_z_for_crevasse:.3f}")
                    else:
                        print(f"    No Z offset data, using channel start Z value: {base_z_for_crevasse:.3f}")

            # Calculate crevasse splay top and bottom elevations based on corrected start point Z value
            if crevasse_height_ratio <= 1.0:
                crevasse_z_top = base_z_for_crevasse
                crevasse_z_bottom = base_z_for_crevasse - crevasse_thickness
            else:
                # Super-thick crevasse splay: extend upward
                crevasse_z_top = base_z_for_crevasse + (crevasse_thickness - base_thickness) * 0.3
                crevasse_z_bottom = crevasse_z_top - crevasse_thickness

            # Ensure minimum thickness
            min_thickness = 0.05
            if crevasse_thickness < min_thickness:
                crevasse_thickness = min_thickness
                crevasse_z_bottom = crevasse_z_top - crevasse_thickness

            if verbose and splay_idx == 0 and centerline_idx == 0:
                print(f"    Crevasse splay final Z values:")
                print(f"      Thickness ratio: {crevasse_height_ratio:.3f}")
                print(f"      Crevasse splay thickness: {crevasse_thickness:.3f}")
                print(f"      Crevasse splay top Z: {crevasse_z_top:.3f}")
                print(f"      Crevasse splay bottom Z: {crevasse_z_bottom:.3f}")
            # ========== End of Z coordinate calculation correction ==========

            # Generate crevasse splay centerline
            splay_centerline = generate_crevasse_centerline_from_bank(
                start_point, splay_angle_deg, total_length, params, crevasse_sinuosity, crevasse_height_ratio
            )

            if len(splay_centerline) < 3:
                continue

            # Create crevasse splay data structure
            unique_splay_id = total_splays_generated
            crevasse_splay = {
                'icc': icc,
                'ic': ic,
                'splay_index': splay_idx,
                'centerline_index': centerline_idx,
                'unique_splay_id': unique_splay_id,
                'node_index': node_idx,
                'side': convex_side,

                # Geometric parameters
                'start_point': start_point.copy(),
                'start_angle': splay_angle_deg,
                'curvature': curvature_at_node,
                'abs_curvature': abs_curvature,
                'curvature_factor': curvature_factor,
                'curvature_description': description,
                'sinuosity': crevasse_sinuosity,

                # Thickness parameters
                'height_ratio_param': crevasse_height_ratio,
                'thickness': crevasse_thickness,
                'z_top': crevasse_z_top,
                'z_bottom': crevasse_z_bottom,
                'base_river_thickness': base_thickness,
                'is_super_thick': crevasse_height_ratio > 1.0,

                # Size parameters
                'total_length': total_length,
                'cone_length': cone_length,
                'max_width': max_width,
                'start_width': start_width,
                'height_ratio': height_ratio,
                'depth_ratio': depth_ratio,

                # Centerline data
                'centerline': splay_centerline,
                'num_points': len(splay_centerline),

                # Channel reference data
                'river_z_top': z_top,  # Channel start Z value (reference)
                'river_node_z_top': base_z_for_crevasse,  # Actual Z value of channel corresponding to crevasse splay position
                'river_width': base_width,
                'river_thickness': base_thickness,

                # Quality metrics
                'quality_metrics': {
                    'convex_bank_only': True,
                    'starts_from_bank_edge': True,
                    'outward_direction': True,
                    'curvature_corrected': True,
                    'sinuosity_controlled': True,
                    'thickness_controlled': True,
                    'thickness_ratio_applied': crevasse_height_ratio,
                    'thickness_limit_fixed': True,
                    'supports_super_thick': True,
                    'fortran_compatible': True,
                    'distance_field_ready': True,
                    'gpu_renderable': True,
                    'z_trend_control_applied': enable_river_trends and abs(z_downstream_trend) > 0.0001,
                    'z_downstream_trend_value': z_downstream_trend,
                    'original_river_z': z_top,
                    'adjusted_river_z': base_z_for_crevasse,
                    'z_trend_correctly_applied': enable_river_trends and abs(z_downstream_trend) > 0.0001,
                    'uses_node_specific_z': True
                }
            }

            crevasse_splays.append(crevasse_splay)
            total_splays_generated += 1

        weights[selected_idx] *= 0.1

    total_time = time.time() - start_time
    print(f"Crevasse splay generation complete: {len(crevasse_splays)} splays in {total_time:.1f}s")

    np.random.set_state(current_state)

    return crevasse_splays






def generate_crevasse_centerline_from_bank(start_point, initial_angle, target_length, params, sinuosity=1.2,
                                           thickness_ratio=0.5):
    """
    Generate crevasse splay centerline from channel bank edge with support
    for sinuosity control and thickness influence.

    Args:
        start_point: Starting point coordinates [x, y]
        initial_angle: Initial angle in degrees
        target_length: Target length in meters
        params: Parameters dictionary
        sinuosity: Sinuosity parameter (1.0=straight, >1.0=curved)
        thickness_ratio: Thickness ratio parameter (0.02-5.0, affects path stability)

    Returns:
        centerline_points: Crevasse splay centerline coordinate points array
    """
    centerline_points = [start_point.copy()]
    current_point = start_point.copy()
    current_angle = initial_angle * np.pi / 180.0
    total_distance = 0.0

    base_step_size = params.get('ysiz', 50) * 0.5
    step_size = max(base_step_size, target_length / 30)

    # Boundary checks
    xmin = params.get('xmn', 0)
    xmax = xmin + params.get('nx', 100) * params.get('xsiz', 50)
    ymin = params.get('ymn', 0)
    ymax = ymin + params.get('ny', 100) * params.get('ysiz', 50)

    buffer_ratio = 0.08
    x_buffer = (xmax - xmin) * buffer_ratio
    y_buffer = (ymax - ymin) * buffer_ratio

    extended_xmin = xmin - x_buffer
    extended_xmax = xmax + x_buffer
    extended_ymin = ymin - y_buffer
    extended_ymax = ymax + y_buffer

    # Calculate curvature control parameters based on sinuosity parameter
    sinuosity_factor = max(0.0, sinuosity - 1.0)
    base_angle_drift = 15.0
    angle_drift_strength = base_angle_drift * sinuosity_factor
    angle_drift_strength = min(angle_drift_strength, 35.0)

    bend_frequency_base = 0.1
    bend_frequency = bend_frequency_base * (1.0 + sinuosity_factor * 0.8)
    bend_frequency = min(bend_frequency, 0.4)

    directional_recovery_strength = max(0.1, 0.8 - sinuosity_factor * 0.3)

    max_angle_change_base = 8.0
    max_angle_change = max_angle_change_base * (1.0 + sinuosity_factor * 0.6)
    max_angle_change = min(max_angle_change, 25.0)

    thickness_ratio = max(0.02, min(5.0, thickness_ratio))

    if thickness_ratio <= 1.0:
        thickness_stability_factor = 0.3 + 0.7 * thickness_ratio
    else:
        excess_ratio = thickness_ratio - 1.0
        thickness_stability_factor = 1.0 + 0.4 * np.log(1.0 + excess_ratio)
        thickness_stability_factor = min(thickness_stability_factor, 2.0)

    # Thickness influence on curvature parameters
    if thickness_ratio <= 1.0:
        thickness_adjusted_drift = angle_drift_strength * (2.2 - thickness_stability_factor)
        thickness_adjusted_drift = max(1.0, min(thickness_adjusted_drift, angle_drift_strength * 1.8))
        thickness_adjusted_frequency = bend_frequency * thickness_stability_factor
        thickness_adjusted_recovery = directional_recovery_strength * thickness_stability_factor
        thickness_adjusted_step = step_size * (0.7 + 0.6 * thickness_ratio)

    else:
        super_thick_factor = min(0.3, 0.8 / thickness_ratio)
        thickness_adjusted_drift = angle_drift_strength * super_thick_factor
        thickness_adjusted_drift = max(0.5, thickness_adjusted_drift)

        thickness_adjusted_frequency = bend_frequency * 0.4

        thickness_adjusted_recovery = directional_recovery_strength * thickness_stability_factor

        thickness_adjusted_step = step_size * (1.2 + 0.3 * np.log(thickness_ratio))
        thickness_adjusted_step = min(thickness_adjusted_step, step_size * 2.5)

    # Main generation loop
    consecutive_boundary_hits = 0
    max_consecutive_hits = 4
    target_angle = current_angle

    bend_phase = 0.0
    bend_amplitude = thickness_adjusted_drift

    max_points = min(40, int(target_length / thickness_adjusted_step) + 8)

    for i in range(1, max_points):

        # Basic random walk
        if thickness_ratio <= 1.0:
            random_intensity = thickness_adjusted_drift * np.random.uniform(0.7, 1.3)
            basic_random_drift = np.random.uniform(-random_intensity, random_intensity)
        else:
            random_intensity = thickness_adjusted_drift * np.random.uniform(0.8, 1.1)
            basic_random_drift = np.random.uniform(-random_intensity, random_intensity)

        # Periodic bending component
        bend_phase += thickness_adjusted_frequency * np.random.uniform(0.9, 1.1)

        if thickness_ratio <= 1.0:
            if thickness_ratio > 0.6:
                periodic_bend = bend_amplitude * 0.5 * np.sin(bend_phase) * np.random.uniform(0.8, 1.2)
            else:
                periodic_bend = bend_amplitude * 0.3 * np.sin(bend_phase) * np.random.uniform(0.5, 1.5)
        else:
            periodic_bend = bend_amplitude * 0.2 * np.sin(bend_phase) * np.random.uniform(0.9, 1.1)

        # Cumulative bending trend
        if i > 5:
            if thickness_ratio <= 1.0:
                if thickness_ratio > 0.5:
                    cumulative_drift = np.random.uniform(-thickness_adjusted_drift * 0.2,
                                                         thickness_adjusted_drift * 0.2)
                else:
                    cumulative_drift = np.random.uniform(-thickness_adjusted_drift * 0.4,
                                                         thickness_adjusted_drift * 0.4)
            else:
                cumulative_drift = np.random.uniform(-thickness_adjusted_drift * 0.1, thickness_adjusted_drift * 0.1)
        else:
            cumulative_drift = 0.0

        # Thickness-related special effects
        if thickness_ratio < 0.3:
            instability_drift = np.random.uniform(-5, 5) if np.random.random() < 0.3 else 0
        elif thickness_ratio > 2.0:
            stability_boost = np.random.uniform(-1, 1) if np.random.random() < 0.1 else 0
            instability_drift = stability_boost
        else:
            instability_drift = 0

        total_random_drift = basic_random_drift + periodic_bend + cumulative_drift + instability_drift

        # Directional recovery force
        angle_diff = target_angle - current_angle
        if abs(angle_diff) > np.pi:
            angle_diff = angle_diff - 2 * np.pi * np.sign(angle_diff)

        directional_correction = angle_diff * thickness_adjusted_recovery

        # Calculate new angle
        angle_change_raw = np.radians(total_random_drift) + directional_correction * 0.05

        if thickness_ratio <= 1.0:
            thickness_adjusted_max_change = max_angle_change * (0.7 + 0.6 * thickness_ratio)
        else:
            thickness_adjusted_max_change = max_angle_change * 0.5

        max_change_rad = np.radians(thickness_adjusted_max_change)
        if abs(angle_change_raw) > max_change_rad:
            angle_change_raw = np.sign(angle_change_raw) * max_change_rad

        new_angle = current_angle + angle_change_raw

        # New position calculation
        next_x = current_point[0] + thickness_adjusted_step * np.cos(new_angle)
        next_y = current_point[1] + thickness_adjusted_step * np.sin(new_angle)
        new_point = np.array([next_x, next_y])

        # Boundary check
        is_out_of_bounds = (next_x < extended_xmin or next_x > extended_xmax or
                            next_y < extended_ymin or next_y > extended_ymax)

        if is_out_of_bounds:
            consecutive_boundary_hits += 1
            if consecutive_boundary_hits >= max_consecutive_hits:
                break
            continue
        else:
            consecutive_boundary_hits = 0

        centerline_points.append(new_point)

        # Update distance
        distance_increment = np.linalg.norm(new_point - current_point)
        total_distance += distance_increment

        if total_distance >= target_length:
            break

        current_point = new_point
        current_angle = new_angle

        # Dynamically adjust curvature parameters
        if i % 5 == 0:
            adjustment_factor = 0.05 if thickness_ratio > 1.0 else 0.1
            bend_amplitude *= np.random.uniform(1.0 - adjustment_factor, 1.0 + adjustment_factor)
            thickness_adjusted_frequency *= np.random.uniform(0.98, 1.02)

    return np.array(centerline_points)