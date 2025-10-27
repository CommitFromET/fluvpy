"""
channel.py
This module implements channel geometric parameter calculation and centerline generation

Core computational workflow:
1. Channel parameter calculation: Dynamic determination of maximum length and step size based on target sinuosity and model scale
2. Ferguson centerline generation: Application of Ferguson algorithm to generate centerlines with target sinuosity
3. Channel position calculation: Anti-overlap positioning algorithm considering minimum spacing constraints between channels
4. Geometric attribute generation: Thickness, width and Z-value distribution calculation with downstream trend control support
5. Curvature analysis and normalization: Three-point method based curvature calculation and relative position normalization

Parameter parsing features:
- Hierarchical parameter priority: Cascaded parsing of region config > complex config > global config
- Dynamic parameter validation: Runtime parameter validity checking and boundary constraints
- Trend control support: Linear variation patterns for channel thickness, width and Z-values downstream
- Curvature response mechanism: Local adjustment strategies for geometric parameters at bends

"""
import numpy as np
from typing import Dict, Any
from ..engine import constants as const
from ..utils import utils
from ..utils.utils import rotc, is_distribution_dict, get_value_from_distribution, create_enhanced_parameter_resolver
from .ferguson_centerline import generate_ferguson_centerline

# Global array cache
_ARRAY_CACHE = {}


def getchan(icc: int, ic: int, params: Dict[str, Any], cx: np.ndarray, cz: np.ndarray,
            nc: np.ndarray, ccx: np.ndarray, ccz: np.ndarray, cco: np.ndarray,
            cct: np.ndarray, ccw: np.ndarray, ccntg: np.ndarray,
            ccl: np.ndarray, ct: np.ndarray, cw: np.ndarray,
            crelpos: np.ndarray, carea: np.ndarray) -> float:
    """
    Generate channel centerline and attributes

    Parameters:
        icc, ic: Channel complex and channel indices
        params: Parameter dictionary
        cx, cz, nc: Channel position and count arrays
        ccx, ccz, cco, cct, ccw, ccntg: Channel complex parameter arrays
        ccl, ct, cw, crelpos, carea: Channel geometric attribute arrays

    Returns:
        float: Channel coverage ratio pcc
    """
    const.set_global_params(params)

    # Save array references to params for use by other functions
    params.update({
        'cx': cx, 'cz': cz, 'nc': nc, 'ccx': ccx, 'ccz': ccz, 'cco': cco,
        'cct': cct, 'ccw': ccw, 'ccntg': ccntg,
        'current_icc': icc, 'current_ic': ic
    })

    # Parameter resolver initialization
    if 'parameter_resolver' not in params:
        if 'original_ccx' not in params:
            params['original_ccx'] = ccx.copy()
        params['parameter_resolver'] = create_enhanced_parameter_resolver(params)

    resolver = params['parameter_resolver']
    verbose = (icc == 1 and ic == 1) and params.get('debug_parameter_resolution', False)

    # Calculate channel parameters
    channel_params = calculate_channel_parameters(params, icc, ic, resolver, verbose)

    # Generate centerline
    final_centerline, individual_weights, actual_steps = generate_ferguson_centerline(
        params, channel_params, icc, ic, verbose
    )

    # Store weight information
    if 'channel_weights' not in params:
        params['channel_weights'] = {}
    params['channel_weights'][f"{icc}_{ic}"] = individual_weights

    # Centerline data collection setup
    collect_centerline_data = resolver('collect_centerline_data', icc, ic, verbose)
    centerline_data = None
    if collect_centerline_data:
        centerline_data = {
            'channel_id': f"{icc}_{ic}",
            'complex_x': ccx[icc - 1],
            'complex_z': ccz[icc - 1],
            'channel_x': cx[icc - 1, ic - 1],
            'channel_z': cz[icc - 1, ic - 1],
            'global_angle': cco[icc - 1],
            'weights': individual_weights,
            'actual_steps': actual_steps,
            'algorithm_type': 'multiscale' if channel_params['enable_migration'] else 'ferguson',
            'points': []
        }

    # Set random seed
    channel_seed = const.get_channel_seed(icc, ic)
    current_state = np.random.get_state()
    np.random.seed(channel_seed)
    nc[icc - 1] += 1

    # Calculate channel position
    new_cx, new_cz, placed = calculate_channel_position(params, icc, ic, resolver, verbose)
    cx[icc - 1, ic - 1] = new_cx
    cz[icc - 1, ic - 1] = new_cz

    if verbose:
        print(f"    Actual generated step points: {actual_steps}")

    # Update channel arrays
    update_channel_arrays(params, final_centerline, actual_steps, icc, ic, ccl, params['centerline_y_data'])

    # Calculate channel curvature
    curvature = calculate_channel_curvature(final_centerline[:actual_steps], actual_steps)

    # Normalize curvature
    if np.max(np.abs(curvature)) > 0:
        curmaxn = max(abs(min(0.0, np.min(curvature))), 0.1)
        curmaxp = max(max(0.0, np.max(curvature)), 0.1)
        curmaxp *= 1.25
        curmaxn *= 1.25

        for i in range(actual_steps):
            if i < crelpos.shape[2]:
                window_start = max(0, i - 7)
                window_end = min(actual_steps, i + 8)
                if window_end > window_start:
                    crelpos[icc - 1, ic - 1, i] = np.mean(curvature[window_start:window_end])

        valid_range = min(actual_steps, crelpos.shape[2])

        mask_middle = abs(crelpos[icc - 1, ic - 1, :valid_range]) < const.EPSLON
        mask_negative = (crelpos[icc - 1, ic - 1, :valid_range] < 0.0) & ~mask_middle
        mask_positive = (crelpos[icc - 1, ic - 1, :valid_range] >= 0.0) & ~mask_middle

        crelpos[icc - 1, ic - 1, :valid_range][mask_middle] = 0.5
        crelpos[icc - 1, ic - 1, :valid_range][mask_negative] = 0.5 * (
                1.0 - abs(crelpos[icc - 1, ic - 1, :valid_range][mask_negative]) / curmaxn)
        crelpos[icc - 1, ic - 1, :valid_range][mask_positive] = 0.5 * (
                1.0 + crelpos[icc - 1, ic - 1, :valid_range][mask_positive] / curmaxp)
    else:
        valid_range = min(actual_steps, crelpos.shape[2])
        crelpos[icc - 1, ic - 1, :valid_range] = 0.5

    # Generate channel attributes
    print(f"    Starting channel attribute generation (step points: {actual_steps})...")

    # Generate channel thickness and width parameters
    generate_channel_thickness_with_trends(icc, ic, actual_steps, channel_params['step_size'], channel_seed, ct,
                                           resolver, verbose,
                                           params, final_centerline, curvature)
    generate_channel_width_with_trends(icc, ic, actual_steps, channel_params['step_size'], channel_seed, cw, ct,
                                       resolver, verbose,
                                       params, final_centerline, curvature)

    # Generate channel Z-value trends
    generate_channel_z_trends(icc, ic, actual_steps, channel_params['step_size'], cz, resolver, verbose, params,
                              final_centerline)

    # Calculate and save channel attribute values
    channel_key = f"{icc}_{ic}"

    # Calculate thickness and width statistics
    valid_thickness_range = min(actual_steps, ct.shape[2])
    valid_width_range = min(actual_steps, cw.shape[2])

    actual_thickness_array = ct[icc - 1, ic - 1, :valid_thickness_range]
    actual_width_array = cw[icc - 1, ic - 1, :valid_width_range]

    actual_avg_thickness = np.mean(actual_thickness_array) if valid_thickness_range > 0 else 0.0
    actual_avg_width = np.mean(actual_width_array) if valid_width_range > 0 else 0.0
    actual_max_thickness = np.max(actual_thickness_array) if valid_thickness_range > 0 else 0.0
    actual_min_thickness = np.min(actual_thickness_array) if valid_thickness_range > 0 else 0.0
    actual_max_width = np.max(actual_width_array) if valid_width_range > 0 else 0.0
    actual_min_width = np.min(actual_width_array) if valid_width_range > 0 else 0.0

    # Calculate width-to-thickness ratio
    actual_width_thickness_ratio = actual_avg_width / actual_avg_thickness if actual_avg_thickness > 0 else 0.0

    # Save to attribute dictionary
    if 'actual_channel_properties' not in params:
        params['actual_channel_properties'] = {}

    if channel_key not in params['actual_channel_properties']:
        params['actual_channel_properties'][channel_key] = {}

    params['actual_channel_properties'][channel_key].update({
        'actual_avg_thickness': actual_avg_thickness,
        'actual_avg_width': actual_avg_width,
        'actual_max_thickness': actual_max_thickness,
        'actual_min_thickness': actual_min_thickness,
        'actual_max_width': actual_max_width,
        'actual_min_width': actual_min_width,
        'actual_width_thickness_ratio': actual_width_thickness_ratio,
        'actual_complex_angle': cco[icc - 1],
        'actual_complex_x': ccx[icc - 1],
        'actual_complex_z': ccz[icc - 1],
        'actual_channel_x': cx[icc - 1, ic - 1],
        'actual_channel_z': cz[icc - 1, ic - 1],
        'actual_steps': actual_steps,
        'algorithm_used': 'ferguson'
    })

    # Calculate channel area
    carea[icc - 1, ic - 1] = 0.5 * actual_width_thickness_ratio * actual_avg_thickness * actual_avg_thickness

    ccsize = cct[icc - 1] * ccw[icc - 1]
    ctotal = 0.0
    for i in range(nc[icc - 1]):
        ctotal += carea[icc - 1, i]
    pcc = ctotal / max(ccsize, const.EPSLON)

    # Collect centerline data
    if collect_centerline_data:
        collect_detailed_centerline_data_dynamic(centerline_data, final_centerline[:actual_steps], curvature,
                                                 cw, ct, crelpos, icc, ic, ccx, cco, cx,
                                                 params.get('ymn', 0.0), channel_params['step_size'], actual_steps,
                                                 params)

        if 'centerline_data' not in params:
            params['centerline_data'] = []
        params['centerline_data'].append(centerline_data)

    # Restore random state
    np.random.set_state(current_state)

    return pcc


def calculate_channel_curvature(final_centerline, ny3):
    """
    Calculate curvature of channel centerline

    Parameters:
        final_centerline: Centerline coordinate array
        ny3: Number of step points

    Returns:
        np.ndarray: Curvature array
    """
    curvature = np.zeros(ny3)
    for i in range(1, ny3 - 1):
        x1, y1 = final_centerline[i - 1]
        x2, y2 = final_centerline[i]
        x3, y3 = final_centerline[i + 1]

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2

        len1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
        len2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

        if len1 > 0 and len2 > 0:
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            angle_diff = angle2 - angle1

            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            avg_len = (len1 + len2) / 2
            curvature[i] = angle_diff / avg_len
        else:
            curvature[i] = 0

    curvature[0] = curvature[1] if ny3 > 1 else 0
    curvature[-1] = curvature[-2] if ny3 > 1 else 0

    return curvature


def collect_detailed_centerline_data_dynamic(centerline_data, final_centerline, curvature,
                                             cw, ct, crelpos, icc, ic, ccx, cco, cx, ymn, step_size, actual_steps,
                                             params=None):
    """
    Collect detailed centerline data

    Parameters:
        centerline_data: Centerline data dictionary
        final_centerline: Centerline coordinates
        curvature: Curvature array
        cw, ct, crelpos: Channel geometric attribute arrays
        icc, ic: Channel indices
        ccx, cco, cx: Channel complex and channel position arrays
        ymn: Y-coordinate starting point
        step_size: Step size
        actual_steps: Number of step points
        params: Parameter dictionary containing Ferguson force decomposition data
    """
    x_center = ccx[icc - 1] + cx[icc - 1, ic - 1]
    ang = cco[icc - 1] * const.DEG2RAD

    # Get Ferguson force decomposition data
    channel_key = f"{icc}_{ic}"
    ferguson_forces = None
    if params and 'ferguson_forces_data' in params:
        ferguson_forces = params['ferguson_forces_data'].get(channel_key, None)

    for i in range(actual_steps):
        local_x = final_centerline[i, 0]
        local_y = final_centerline[i, 1]
        ypos = ymn + i * step_size

        try:
            global_x, global_y = rotc(x_center, ymn, ang, 1, local_x, ypos - ymn)
        except:
            cos_ang = np.cos(ang)
            sin_ang = np.sin(ang)
            dy = ypos - ymn
            global_x = x_center + local_x * cos_ang - dy * sin_ang
            global_y = ymn + dy * cos_ang + local_x * sin_ang

        # Ensure array bounds are not exceeded
        width_val = cw[icc - 1, ic - 1, i] if i < cw.shape[2] else 0.0
        thickness_val = ct[icc - 1, ic - 1, i] if i < ct.shape[2] else 0.0
        curvature_val = curvature[i] if i < len(curvature) else 0.0
        relpos_val = crelpos[icc - 1, ic - 1, i] if i < crelpos.shape[2] else 0.5

        # Get Ferguson force decomposition data
        ferguson_data = {
            'ferguson_force_dx': 0.0,
            'ferguson_force_dy': 0.0,
            'statistical_force_dx': 0.0,
            'statistical_force_dy': 0.0,
            'statistical_alpha': 0.0,
            'raw_ferguson_angle': 0.0,
            'final_ferguson_angle': 0.0,
            'target_deviation_x': 0.0,
            'target_deviation_y': 0.0,
            'ferguson_progress_ratio': 0.0
        }

        if ferguson_forces and i < len(ferguson_forces):
            force_data = ferguson_forces[i]
            ferguson_data.update({
                'ferguson_force_dx': force_data.get('ferguson_dx', 0.0),
                'ferguson_force_dy': force_data.get('ferguson_dy', 0.0),
                'statistical_force_dx': force_data.get('statistical_dx', 0.0),
                'statistical_force_dy': force_data.get('statistical_dy', 0.0),
                'statistical_alpha': force_data.get('alpha', 0.0),
                'raw_ferguson_angle': force_data.get('raw_angle', 0.0),
                'final_ferguson_angle': force_data.get('final_angle', 0.0),
                'target_deviation_x': force_data.get('deviation_x', 0.0),
                'target_deviation_y': force_data.get('deviation_y', 0.0),
                'ferguson_progress_ratio': force_data.get('progress_ratio', 0.0)
            })

        # Create point information containing Ferguson force decomposition data
        point_data = {
            'point_index': i,
            'local_x': local_x,
            'local_y': local_y,
            'global_x': global_x,
            'global_y': global_y,
            'width': width_val,
            'thickness': thickness_val,
            'curvature': curvature_val,
            'rel_position': relpos_val
        }

        # Add Ferguson force decomposition data
        point_data.update(ferguson_data)

        centerline_data['points'].append(point_data)


def generate_channel_z_trends(icc, ic, actual_steps, step_size, cz, resolver, verbose, params, final_centerline):
    """
    Generate channel Z-value trends, controlling downstream variation of channel top Z-values

    Parameters:
        icc, ic: Channel complex and channel indices
        actual_steps: Number of step points
        step_size: Step size
        cz: Z-value array
        resolver: Parameter resolver
        verbose: Verbose output flag
        params: Parameter dictionary
        final_centerline: Centerline coordinates
    """
    # Get trend control parameters and extract numerical values from distributions
    enable_river_trends = resolver('enable_river_trends', icc, ic, verbose)
    z_downstream_trend_param = resolver('z_downstream_trend', icc, ic, verbose)

    # Extract actual numerical values from distributions
    if isinstance(z_downstream_trend_param, dict) and is_distribution_dict(z_downstream_trend_param):
        z_downstream_trend = get_value_from_distribution(z_downstream_trend_param)
    elif isinstance(z_downstream_trend_param, list):
        z_downstream_trend = get_value_from_distribution(z_downstream_trend_param)
    else:
        z_downstream_trend = float(z_downstream_trend_param)

    # Parameter validation - using extracted values
    try:
        # Build parameter dictionary containing numerical values for validation
        validated_params = {
            'enable_river_trends': enable_river_trends,
            'z_downstream_trend': z_downstream_trend
        }
    except Exception as e:
        print(f"Z-value trend parameter validation failed: {e}")
        # Use default values
        validated_params = {
            'enable_river_trends': False,
            'z_downstream_trend': 0.0
        }

    # Use validated parameters
    enable_river_trends = validated_params['enable_river_trends']
    z_downstream_trend = validated_params.get('z_downstream_trend', 0.0)

    if not enable_river_trends or abs(z_downstream_trend) < 0.0001:
        return  # If trend control is not enabled or trend value is very small, skip processing

    # Get base Z-value
    base_z_top = cz[icc - 1, ic - 1]

    if verbose:
        print(
            f"    Z-value trend control: base_z={base_z_top:.2f}, z_trend={z_downstream_trend:.6f} (drop {z_downstream_trend}m per 1000m)")

    # Initialize Z offset array
    if 'centerline_z_offset' not in params:
        params['centerline_z_offset'] = np.zeros_like(params.get('centerline_y_data', np.zeros((1, 1, 1))))

    z_offset_array = params['centerline_z_offset']

    # Calculate total channel length
    total_length = 0.0
    if actual_steps > 1:
        for i in range(actual_steps - 1):
            dx = final_centerline[i + 1, 0] - final_centerline[i, 0]
            dy = final_centerline[i + 1, 1] - final_centerline[i, 1]
            segment_length = np.sqrt(dx * dx + dy * dy)
            total_length += segment_length

    # Calculate Z-value offset for each step point
    cumulative_length = 0.0
    for i in range(actual_steps):
        if actual_steps > 1:
            # Calculate cumulative length along channel
            if i > 0:
                dx = final_centerline[i, 0] - final_centerline[i - 1, 0]
                dy = final_centerline[i, 1] - final_centerline[i - 1, 1]
                segment_length = np.sqrt(dx * dx + dy * dy)
                cumulative_length += segment_length
        else:
            cumulative_length = 0.0

        # Z-value offset calculation: drop specified meters per 1000m
        z_offset = -(cumulative_length / 1000.0) * z_downstream_trend

        # Save Z offset to array
        if (icc - 1 < z_offset_array.shape[0] and
                ic - 1 < z_offset_array.shape[1] and
                i < z_offset_array.shape[2]):
            z_offset_array[icc - 1, ic - 1, i] = z_offset

        if verbose and i < 5:
            print(f"      Step point {i}: cumulative_length={cumulative_length:.1f}m, z_offset={z_offset:.4f}m")
            print(f"        Drop {z_downstream_trend}m per 1000m, current position drops {abs(z_offset):.4f}m")

    # Update channel average Z-value
    if actual_steps > 0:
        avg_z_offset = np.mean([z_offset_array[icc - 1, ic - 1, i]
                                for i in range(min(actual_steps, z_offset_array.shape[2]))])
        if verbose:
            print(f"    Average Z offset: {avg_z_offset:.4f}m")
            if total_length > 0:
                print(
                    f"    Total channel length: {total_length:.1f}m, expected endpoint drop: {(total_length / 1000.0) * z_downstream_trend:.4f}m")


def generate_channel_width_with_trends(icc, ic, actual_steps, step_size, channel_seed, cw, ct, resolver, verbose,
                                       params, final_centerline, curvature):
    """
    Generate channel width parameters with downstream trend and bend control support

    Parameters:
        icc, ic: Channel complex and channel indices
        actual_steps: Number of step points
        step_size: Step size
        channel_seed: Random seed
        cw: Width array
        ct: Thickness array
        resolver: Parameter resolver
        verbose: Verbose output flag
        params: Parameter dictionary
        final_centerline: Centerline coordinates
        curvature: Curvature data
    """
    # Get trend control parameters and extract numerical values from distributions
    enable_river_trends = resolver('enable_river_trends', icc, ic, verbose)
    width_downstream_trend_param = resolver('width_downstream_trend', icc, ic, verbose)
    width_curvature_factor_param = resolver('width_curvature_factor', icc, ic, verbose)

    # Extract actual numerical values from distributions
    if isinstance(width_downstream_trend_param, dict) and is_distribution_dict(width_downstream_trend_param):
        width_downstream_trend = get_value_from_distribution(width_downstream_trend_param)
    elif isinstance(width_downstream_trend_param, list):
        width_downstream_trend = get_value_from_distribution(width_downstream_trend_param)
    else:
        width_downstream_trend = float(width_downstream_trend_param)

    if isinstance(width_curvature_factor_param, dict) and is_distribution_dict(width_curvature_factor_param):
        width_curvature_factor = get_value_from_distribution(width_curvature_factor_param)
    elif isinstance(width_curvature_factor_param, list):
        width_curvature_factor = get_value_from_distribution(width_curvature_factor_param)
    else:
        width_curvature_factor = float(width_curvature_factor_param)

    # Parameter validation - using extracted values
    try:
        # Build parameter dictionary containing numerical values for validation
        validation_params = {
            'enable_river_trends': enable_river_trends,
            'width_downstream_trend': width_downstream_trend,
            'width_curvature_factor': width_curvature_factor
        }

        # Get other necessary parameters
        thickness_width_correlation_enabled = resolver('thickness_width_correlation_enabled', icc, ic,
                                                       verbose) if 'thickness_width_correlation_enabled' in params else False
        thickness_width_correlation_factor = resolver('thickness_width_correlation_factor', icc, ic,
                                                      verbose) if 'thickness_width_correlation_factor' in params else 0.5

        base_cwtr_param = resolver('fcwtr', icc, ic, verbose)
        if isinstance(base_cwtr_param, dict) and is_distribution_dict(base_cwtr_param):
            base_cwtr = get_value_from_distribution(base_cwtr_param)
        elif isinstance(base_cwtr_param, list):
            base_cwtr = get_value_from_distribution(base_cwtr_param)
        else:
            base_cwtr = float(base_cwtr_param[1] if isinstance(base_cwtr_param, list) else base_cwtr_param)

        validation_params.update({
            'thickness_width_correlation_enabled': thickness_width_correlation_enabled,
            'thickness_width_correlation_factor': thickness_width_correlation_factor,
            'base_cwtr': base_cwtr,
            'base_thickness': 5.0,
            'base_width_control': 10.0,
            'width_range': 50.0
        })

        validated_params = validation_params

    except Exception as e:
        print(f"Width generation parameter validation failed: {e}")
        # Use default values
        validated_params = {
            'enable_river_trends': False,
            'width_downstream_trend': 1.0,
            'width_curvature_factor': 0.0,
            'thickness_width_correlation_enabled': False,
            'thickness_width_correlation_factor': 0.5,
            'base_cwtr': 1.5,
            'base_thickness': 5.0,
            'base_width_control': 10.0,
            'width_range': 50.0
        }

    # Use validated parameters
    thickness_width_correlation_enabled = validated_params['thickness_width_correlation_enabled']
    thickness_width_correlation_factor = validated_params['thickness_width_correlation_factor']
    base_cwtr = validated_params['base_cwtr']
    enable_river_trends = validated_params['enable_river_trends']
    width_downstream_trend = validated_params.get('width_downstream_trend', 1.0)
    width_curvature_factor = validated_params.get('width_curvature_factor', 0.0)

    # Check if trend control is enabled
    if enable_river_trends and abs(width_downstream_trend - 1.0) > 0.001:
        # Enable trend control: use width-to-thickness ratio for starting point calculation, then apply trend based on starting value

        # Handle thickness-width ratio correlation (for starting point calculation)
        if thickness_width_correlation_enabled:
            valid_thickness_range = min(actual_steps, ct.shape[2])
            avg_channel_thickness = np.mean(ct[icc - 1, ic - 1, :valid_thickness_range])

            # Re-get parameter ranges for correlation calculation
            fcwtr_param = resolver('fcwtr', icc, ic, verbose)
            fcat_param = resolver('fcat', icc, ic, verbose)

            if isinstance(fcwtr_param, dict) and is_distribution_dict(fcwtr_param):
                if fcwtr_param.get('type') == 'discrete':
                    wtr_min = min(fcwtr_param.get('values', [1.2]))
                    wtr_max = max(fcwtr_param.get('values', [1.7]))
                elif fcwtr_param.get('type') == 'normal':
                    wtr_min = fcwtr_param.get('min_limit', 1.2)
                    wtr_max = fcwtr_param.get('max_limit', 1.7)
                else:
                    wtr_min = fcwtr_param.get('min', 1.2)
                    wtr_max = fcwtr_param.get('max', 1.7)
            else:
                wtr_min = fcwtr_param[0]
                wtr_max = fcwtr_param[2]

            if isinstance(fcat_param, dict) and is_distribution_dict(fcat_param):
                thick_min = fcat_param.get('min_limit', 1.0)
                thick_max = fcat_param.get('max_limit', 10.0)
            else:
                thick_min = fcat_param[0]
                thick_max = fcat_param[2]

            norm_thickness = (avg_channel_thickness - thick_min) / max(thick_max - thick_min, const.EPSLON)
            norm_thickness = max(0.0, min(1.0, norm_thickness))
            correlation_effect = thickness_width_correlation_factor * (norm_thickness - 0.5) * 2.0
            wtr_range = wtr_max - wtr_min
            adjusted_cwtr = base_cwtr + correlation_effect * wtr_range * 0.5
            cwtr = max(wtr_min, min(wtr_max, adjusted_cwtr))

            if verbose:
                print(f"    Correlation adjustment: avg_thickness={avg_channel_thickness:.2f}, adjusted_cwtr={cwtr:.2f}")
        else:
            cwtr = base_cwtr

        # Calculate starting point width control factor
        fcawu_param = resolver('fcawu', icc, ic, verbose)
        np.random.seed(channel_seed + 2000)
        initial_width_random = np.random.random()

        if isinstance(fcawu_param, dict) and is_distribution_dict(fcawu_param):
            width_control_factor = get_value_from_distribution(fcawu_param)
        else:
            width_control_factor = get_value_from_distribution(fcawu_param)

        base_fluctuation_factor = 0.9 + 0.2 * initial_width_random

        # Starting width = starting thickness * width-to-thickness ratio * control factor
        initial_thickness = ct[icc - 1, ic - 1, 0] if ct.shape[2] > 0 else validated_params['base_thickness']
        initial_width = width_control_factor * base_fluctuation_factor * initial_thickness * cwtr

        if verbose:
            print(f"    Trend mode: starting width={initial_width:.2f} (using width-to-thickness ratio), endpoint ratio={width_downstream_trend:.3f}")
            print(f"    Starting thickness={initial_thickness:.2f}, width-to-thickness ratio={cwtr:.2f}")

        # Linear variation based on trend for all step points
        for i in range(actual_steps):
            # Calculate progress ratio along channel (0.0 to 1.0)
            progress_ratio = float(i) / float(actual_steps - 1) if actual_steps > 1 else 0.0

            # Base width (based on trend): linear variation from starting width to endpoint width
            trend_factor = 1.0 + (width_downstream_trend - 1.0) * progress_ratio
            trend_factor = max(0.1, min(trend_factor, 50.0))  # Safety limits

            current_width = initial_width * trend_factor

            # Bend width adjustment
            if abs(width_curvature_factor) > 0.001 and i < len(curvature):
                # Normalized curvature value (0 to 1)
                abs_curvature = abs(curvature[i])
                max_curvature = max(abs(np.max(curvature)), abs(np.min(curvature)), 0.001)
                normalized_curvature = abs_curvature / max_curvature

                # Bend width increase factor
                curvature_growth_percent = width_curvature_factor / 100.0
                curvature_width_factor = 1.0 + curvature_growth_percent * normalized_curvature
                curvature_width_factor = max(0.8, min(curvature_width_factor, 1.5))  # Limit to reasonable range
                current_width *= curvature_width_factor

                if verbose and i < 3 and normalized_curvature > 0.1:
                    print(f"      Step point {i}: curvature={curvature[i]:.4f}, curv_factor={curvature_width_factor:.3f}")

            if verbose and i < 3:
                print(f"      Step point {i}: progress={progress_ratio:.3f}, trend_factor={trend_factor:.3f}")
                print(f"        Starting width={initial_width:.3f}, current width={current_width:.3f}")

            # Ensure array bounds are not exceeded and assign value
            if i < cw.shape[2]:
                cw[icc - 1, ic - 1, i] = current_width

    else:
        # Trend control not enabled: use original width-to-thickness ratio logic

        width_factor = 1.0

        # Handle thickness-width ratio correlation
        if thickness_width_correlation_enabled:
            valid_thickness_range = min(actual_steps, ct.shape[2])
            avg_channel_thickness = np.mean(ct[icc - 1, ic - 1, :valid_thickness_range])

            # Re-get parameter ranges for correlation calculation
            fcwtr_param = resolver('fcwtr', icc, ic, verbose)
            fcat_param = resolver('fcat', icc, ic, verbose)

            if isinstance(fcwtr_param, dict) and is_distribution_dict(fcwtr_param):
                if fcwtr_param.get('type') == 'discrete':
                    wtr_min = min(fcwtr_param.get('values', [1.2]))
                    wtr_max = max(fcwtr_param.get('values', [1.7]))
                elif fcwtr_param.get('type') == 'normal':
                    wtr_min = fcwtr_param.get('min_limit', 1.2)
                    wtr_max = fcwtr_param.get('max_limit', 1.7)
                else:
                    wtr_min = fcwtr_param.get('min', 1.2)
                    wtr_max = fcwtr_param.get('max', 1.7)
            else:
                wtr_min = fcwtr_param[0]
                wtr_max = fcwtr_param[2]

            if isinstance(fcat_param, dict) and is_distribution_dict(fcat_param):
                thick_min = fcat_param.get('min_limit', 1.0)
                thick_max = fcat_param.get('max_limit', 10.0)
            else:
                thick_min = fcat_param[0]
                thick_max = fcat_param[2]

            norm_thickness = (avg_channel_thickness - thick_min) / max(thick_max - thick_min, const.EPSLON)
            norm_thickness = max(0.0, min(1.0, norm_thickness))
            correlation_effect = thickness_width_correlation_factor * (norm_thickness - 0.5) * 2.0
            wtr_range = wtr_max - wtr_min
            adjusted_cwtr = base_cwtr + correlation_effect * wtr_range * 0.5
            cwtr = max(wtr_min, min(wtr_max, adjusted_cwtr))

            if verbose:
                print(f"    Correlation adjustment: avg_thickness={avg_channel_thickness:.2f}, adjusted_cwtr={cwtr:.2f}")
        else:
            cwtr = base_cwtr * width_factor

        width_ctrl_base = 1.0
        base_width_control_factor = validated_params['base_width_control'] * width_ctrl_base
        range_val = validated_params['width_range']

        # Generate width array
        width_array = np.zeros(actual_steps)
        utils.lusim(actual_steps, step_size, range_val, 1, width_array,
                    kernel_type="exponential",
                    use_local_random=True,
                    local_seed=channel_seed + 2000,
                    scale_type="single")

        # Original width-to-thickness ratio logic
        for i in range(actual_steps):
            window_start = max(0, i - 3)
            window_end = min(actual_steps, i + 4)
            window_size = window_end - window_start
            sval = np.sum(width_array[window_start:window_end]) / window_size
            base_fluctuation_factor = 0.9 + 0.2 * sval

            # Base width control factor
            width_control_factor = base_width_control_factor

            # Bend width increase
            if abs(width_curvature_factor) > 0.001 and i < len(curvature):
                # Normalized curvature value (0 to 1)
                abs_curvature = abs(curvature[i])
                max_curvature = max(abs(np.max(curvature)), abs(np.min(curvature)), 0.001)
                normalized_curvature = abs_curvature / max_curvature

                # Bend width increase factor
                curvature_growth_percent = width_curvature_factor / 100.0
                curvature_width_factor = 1.0 + curvature_growth_percent * normalized_curvature
                curvature_width_factor = max(0.8, min(curvature_width_factor, 1.5))  # Limit to reasonable range
                width_control_factor *= curvature_width_factor

                if verbose and i < 3 and normalized_curvature > 0.1:
                    print(f"      Step point {i}: curvature={curvature[i]:.4f}, norm_curv={normalized_curvature:.3f}")
                    print(f"        curv_factor={curvature_width_factor:.3f}")

            # Ensure array bounds are not exceeded and assign value
            if i < cw.shape[2] and i < ct.shape[2]:
                final_width = width_control_factor * base_fluctuation_factor * ct[icc - 1, ic - 1, i] * cwtr
                cw[icc - 1, ic - 1, i] = final_width


def calculate_channel_position(params, icc, ic, resolver, verbose):
    """
    Calculate channel position, avoiding overlap between channels

    Parameters:
        params: Parameter dictionary
        icc, ic: Channel indices
        resolver: Parameter resolver
        verbose: Verbose output flag

    Returns:
        tuple: (new_cx, new_cz, placed) New channel coordinates and successful placement flag
    """
    channel_dispersion_x = resolver('channel_dispersion_x', icc, ic, verbose)
    channel_dispersion_z = resolver('channel_dispersion_z', icc, ic, verbose)
    min_channel_spacing = resolver('min_channel_spacing', icc, ic, verbose)

    complex_width = params['ccw'][icc - 1]
    complex_thickness = params['cct'][icc - 1]
    complex_angle = params['cco'][icc - 1] * const.DEG2RAD

    # Channel position generation
    allow_angle_variation = resolver('allow_channel_angle_variation', icc, ic, verbose)
    if allow_angle_variation:
        angle_variation = resolver('channel_angle_variation', icc, ic, verbose) * const.DEG2RAD * 0.5
        channel_angle_offset = np.random.normal(0, angle_variation * 0.5)
        channel_angle = complex_angle + channel_angle_offset
    else:
        channel_angle = complex_angle
        channel_angle_offset = 0.0

    # Channel position overlap avoidance
    x_range = complex_width * channel_dispersion_x
    z_range = complex_thickness * channel_dispersion_z
    max_attempts = 15
    placed = False
    existing_channels = []

    for i in range(1, ic):
        if i <= params['nc'][icc - 1]:
            existing_channels.append((params['cx'][icc - 1, i - 1], params['cz'][icc - 1, i - 1]))

    for attempt in range(max_attempts):
        x_offset = (np.random.random() * 2 - 1) * x_range * 0.5
        x_offset += np.random.normal(0, x_range * 0.05)
        z_offset = np.random.random() * z_range * 0.8
        new_cx = x_offset
        new_cz = params['ccz'][icc - 1] - z_offset

        if ic == 1 or not existing_channels:
            placed = True
            break

        too_close = False
        min_allowed_dist = min_channel_spacing * complex_width
        for ex_cx, ex_cz in existing_channels:
            dist = np.sqrt((new_cx - ex_cx) ** 2 + (new_cz - ex_cz) ** 2)
            if dist < min_allowed_dist:
                too_close = True
                break

        if not too_close:
            placed = True
            break

    return new_cx, new_cz, placed


def update_channel_arrays(params, final_centerline, actual_steps, icc, ic, ccl, centerline_y_data):
    """
    Update channel array data, saving X, Y coordinates and actual step count

    Parameters:
        params: Parameter dictionary
        final_centerline: Centerline coordinates
        actual_steps: Number of step points
        icc, ic: Channel indices
        ccl: Channel centerline array
        centerline_y_data: Y-coordinate data array
    """
    # Save X-coordinate data to ccl
    if ccl.shape[2] < actual_steps:
        print(f"Warning: ccl array size {ccl.shape[2]} smaller than required {actual_steps}, will truncate")
        save_steps = ccl.shape[2]
    else:
        save_steps = actual_steps

    # Save X-coordinate data
    ccl[icc - 1, ic - 1, :save_steps] = final_centerline[:save_steps, 0]

    # Save actual step count
    if 'channel_actual_steps' not in params:
        params['channel_actual_steps'] = {}
    params['channel_actual_steps'][f"{icc}_{ic}"] = actual_steps

    # Save Y-coordinate data
    if 'centerline_y_data' not in params or params['centerline_y_data'] is None:
        # Initialize Y-coordinate data array
        max_cc = ccl.shape[0]
        max_c = ccl.shape[1]
        max_steps = ccl.shape[2]
        params['centerline_y_data'] = np.zeros((max_cc, max_c, max_steps))
        print(f"  Initialized Y-coordinate data array: {params['centerline_y_data'].shape}")

    # Save Y-coordinate data
    if (icc - 1 < params['centerline_y_data'].shape[0] and
            ic - 1 < params['centerline_y_data'].shape[1]):
        max_save_steps = min(actual_steps, params['centerline_y_data'].shape[2])
        params['centerline_y_data'][icc - 1, ic - 1, :max_save_steps] = final_centerline[:max_save_steps, 1]

    # Save complete centerline coordinates
    if 'full_centerlines' not in params:
        params['full_centerlines'] = {}
    params['full_centerlines'][f"{icc}_{ic}"] = final_centerline[:actual_steps].copy()


def generate_channel_thickness_with_trends(icc, ic, actual_steps, step_size, channel_seed, ct, resolver, verbose,
                                           params, final_centerline, curvature):
    """
    Generate channel thickness parameters with downstream trend control support

    Parameters:
        icc, ic: Channel complex and channel indices
        actual_steps: Number of step points
        step_size: Step size
        channel_seed: Random seed
        ct: Thickness array
        resolver: Parameter resolver
        verbose: Verbose output flag
        params: Parameter dictionary
        final_centerline: Centerline coordinates
        curvature: Curvature data
    """
    # Get trend control parameters and extract numerical values from distributions
    enable_river_trends = resolver('enable_river_trends', icc, ic, verbose)
    depth_downstream_trend_param = resolver('depth_downstream_trend', icc, ic, verbose)

    # Extract actual numerical values from distributions
    if isinstance(depth_downstream_trend_param, dict) and is_distribution_dict(depth_downstream_trend_param):
        depth_downstream_trend = get_value_from_distribution(depth_downstream_trend_param)
    elif isinstance(depth_downstream_trend_param, list):
        depth_downstream_trend = get_value_from_distribution(depth_downstream_trend_param)
    else:
        depth_downstream_trend = float(depth_downstream_trend_param)

    # Parameter validation - using extracted values
    try:
        # Build parameter dictionary containing numerical values for validation
        validation_params = {
            'enable_river_trends': enable_river_trends,
            'depth_downstream_trend': depth_downstream_trend
        }

        # Get other necessary parameters
        base_thickness_param = resolver('fcat', icc, ic, verbose)
        if isinstance(base_thickness_param, dict) and is_distribution_dict(base_thickness_param):
            base_thickness = get_value_from_distribution(base_thickness_param)
        elif isinstance(base_thickness_param, list):
            base_thickness = get_value_from_distribution(base_thickness_param)
        else:
            base_thickness = float(
                base_thickness_param[1] if isinstance(base_thickness_param, list) else base_thickness_param)

        validation_params['base_thickness'] = base_thickness
        validation_params['thickness_range'] = 50.0  # Default range

        fcau_param = resolver('fcau', icc, ic, verbose)
        validation_params['fcau_param'] = fcau_param

        validated_params = validation_params

    except Exception as e:
        print(f"Thickness generation parameter validation failed: {e}")
        # Use default values
        enable_river_trends = False
        depth_downstream_trend = 1.0
        validated_params = {
            'enable_river_trends': enable_river_trends,
            'depth_downstream_trend': depth_downstream_trend,
            'base_thickness': 5.0,
            'thickness_range': 50.0,
            'fcau_param': [0.9, 1.0, 1.1]
        }

    # Use validated parameters
    thickness_factor = 0.9
    base_avgthick = validated_params['base_thickness'] * thickness_factor
    range_val = validated_params['thickness_range']
    enable_river_trends = validated_params['enable_river_trends']
    depth_downstream_trend = validated_params.get('depth_downstream_trend', 1.0)

    if verbose:
        print(f"    Dynamic thickness parameters: base_avgthick={base_avgthick:.2f}, range_val={range_val:.1f}")
        print(f"    Trend control: enable_trends={enable_river_trends}, depth_trend={depth_downstream_trend:.3f} (ratio)")

    # Check if trend control is enabled
    if enable_river_trends and abs(depth_downstream_trend - 1.0) > 0.001:
        # Enable trend control: linear variation based on fixed initial thickness

        # Calculate initial thickness
        np.random.seed(channel_seed + 1000)
        initial_random = np.random.random()

        fcau_param = validated_params['fcau_param']
        if isinstance(fcau_param, dict) and is_distribution_dict(fcau_param):
            initial_tval = get_value_from_distribution(fcau_param)
        else:
            if initial_random < 0.5:
                initial_tval = fcau_param[0] + 2.0 * initial_random * (fcau_param[1] - fcau_param[0])
            else:
                initial_tval = fcau_param[1] + 2.0 * (initial_random - 0.5) * (fcau_param[2] - fcau_param[1])

        # Calculate base initial thickness
        initial_thickness = initial_tval * base_avgthick

        if verbose:
            print(f"    Trend mode: initial thickness={initial_thickness:.2f}, endpoint ratio={depth_downstream_trend:.3f}")

        # Linear variation based on trend for all step points
        for i in range(actual_steps):
            # Calculate progress ratio along channel (0.0 to 1.0)
            progress_ratio = float(i) / float(actual_steps - 1) if actual_steps > 1 else 0.0

            # Linear trend: from 1.0 (starting point) to depth_downstream_trend (endpoint)
            trend_factor = 1.0 + (depth_downstream_trend - 1.0) * progress_ratio

            # Safety limits: prevent thickness from becoming negative or too large
            trend_factor = max(0.1, min(trend_factor, 10.0))

            # Final thickness = initial thickness * trend factor
            final_thickness = initial_thickness * trend_factor

            if verbose and i < 3:
                print(f"      Step point {i}: progress={progress_ratio:.3f}, trend_factor={trend_factor:.3f}")
                print(f"        Initial thickness={initial_thickness:.3f}, final thickness={final_thickness:.3f}")

            # Ensure array bounds are not exceeded and assign value
            if i < ct.shape[2]:
                ct[icc - 1, ic - 1, i] = final_thickness

    else:
        # Trend control not enabled: use original random variation logic

        # Generate thickness array
        thickness_array = np.zeros(actual_steps)
        utils.lusim(actual_steps, step_size, range_val, 1, thickness_array,
                    kernel_type="gaussian",
                    use_local_random=True,
                    local_seed=channel_seed + 1000,
                    scale_type="single")

        fcau_param = validated_params['fcau_param']

        for i in range(actual_steps):
            window_start = max(0, i - 3)
            window_end = min(actual_steps, i + 4)
            window_size = window_end - window_start
            sval = np.sum(thickness_array[window_start:window_end]) / window_size

            if isinstance(fcau_param, dict) and is_distribution_dict(fcau_param):
                tval = get_value_from_distribution(fcau_param)
            else:
                if sval < 0.5:
                    tval = fcau_param[0] + 2.0 * sval * (fcau_param[1] - fcau_param[0])
                else:
                    tval = fcau_param[1] + 2.0 * (sval - 0.5) * (fcau_param[2] - fcau_param[1])

            # Base thickness value
            final_thickness = tval * base_avgthick

            # Ensure array bounds are not exceeded and assign value
            if i < ct.shape[2]:
                ct[icc - 1, ic - 1, i] = final_thickness


def calculate_channel_parameters(params, icc, ic, resolver, verbose):
    """
    Calculate channel parameters

    Parameters:
        params: Global parameter dictionary
        icc, ic: Channel complex and channel indices
        resolver: Parameter resolver
        verbose: Verbose output flag

    Returns:
        dict: Dictionary containing all calculated channel parameters
    """
    # Dynamic calculation of channel length parameters
    sinuosity_param = resolver('channel_sinuosity', icc, ic, verbose)
    if isinstance(sinuosity_param, dict) and is_distribution_dict(sinuosity_param):
        target_sinuosity = get_value_from_distribution(sinuosity_param)
    else:
        target_sinuosity = get_value_from_distribution(sinuosity_param)
    target_sinuosity = max(1.0, target_sinuosity)

    # Get dynamic length calculation parameters
    if 'max_channel_length' not in params:
        max_length, max_steps, step_size = calculate_max_channel_length(params, target_sinuosity)
        params['max_channel_length'] = max_length
        params['max_channel_steps'] = max_steps
        params['channel_step_size'] = step_size
    else:
        max_steps = params['max_channel_steps']
        step_size = params['channel_step_size']

    # Calculate actual parameters for current channel
    start_x_global = params['ccx'][icc - 1] + params['cx'][icc - 1, ic - 1]
    start_y_global = params.get('ymn', 0.0)

    model_ly = params.get('ly', params.get('ny', 250) * params['ysiz'])
    complex_angle_rad = np.radians(params['cco'][icc - 1])

    # Calculate actual step count
    base_steps = int(model_ly / step_size)
    actual_steps = max(base_steps, int(base_steps * target_sinuosity))
    actual_steps = min(actual_steps, max_steps)  # Do not exceed maximum step count

    enable_migration = params.get('enable_migration', False)

    return {
        'target_sinuosity': target_sinuosity,
        'max_steps': max_steps,
        'step_size': step_size,
        'start_x_global': start_x_global,
        'start_y_global': start_y_global,
        'model_ly': model_ly,
        'complex_angle_rad': complex_angle_rad,
        'actual_steps': actual_steps,
        'enable_migration': enable_migration
    }


def calculate_max_channel_length(params, target_sinuosity=2.0):
    """
    Dynamically calculate maximum length and step count that channels might need

    Parameters:
        params: Simulation parameter dictionary
        target_sinuosity: Target sinuosity, default 2.0

    Returns:
        tuple: (max_length, max_steps, step_size) Maximum length, maximum step count, fixed step size
    """
    # Get model parameters
    ly = params.get('ly', params.get('ny', 250) * params['ysiz'])
    ysiz = params['ysiz']
    xsiz = params['xsiz']

    # Fixed step size as unit grid length
    step_size = min(xsiz, ysiz)

    # Calculate maximum length based on 1.5 times ly
    max_y_distance = 1.5 * ly

    # Consider influence of maximum possible sinuosity
    max_sinuosity = max(target_sinuosity * 1.5, 3.0)  # Reserve sinuosity margin

    # Maximum channel length
    max_length = max_y_distance * max_sinuosity

    # Maximum step count
    max_steps = int(np.ceil(max_length / step_size)) + 10  # Additional 10 step points reserved

    print(f"Dynamic length calculation: max Y distance={max_y_distance:.1f}, max sinuosity={max_sinuosity:.1f}")
    print(f"Max channel length={max_length:.1f}, fixed step size={step_size:.2f}, max step count={max_steps}")

    return max_length, max_steps, step_size