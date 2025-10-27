"""
utils.py
"""
from numba import njit
from typing import List
from ..engine import constants as const
import numpy as np
try:
    import cupy as cp
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries unavailable, using CPU rendering")


@cuda.jit(device=True)
def gpu_random(seed, idx):
    """
    Simple pseudo-random number generator on GPU device
    Parameters:
        seed: Random seed
        idx: Index value
    Returns:
        float: Random number between 0 and 1
    """
    # Linear congruential generator
    a = 1664525
    c = 1013904223
    m = 2**32

    # Mix thread ID and seed
    val = (seed + idx * 1973) % m
    val = (a * val + c) % m

    return val / float(m)


def calculate_z_position(icc, ncc, params):
    """
    Calculate Z position coordinate for channel complex based on multiple distribution modes
    Parameters:
        icc: Channel complex index
        ncc: Total number of channel complexes
        params: Parameter dictionary
    Returns:
        float: Calculated Z position coordinate
    """

    mode = params.get('z_distribution_mode', 'uniform')
    z_min = params.get('z_min_depth', 0.01 * params['nz'] * params['zsiz'])
    z_max = params.get('z_max_depth', 1.0 * params['nz'] * params['zsiz'])
    z_var = params.get('z_variation', 0.1)
    avg_thick = params['avgthick']

    norm_idx = np.random.uniform(0, 1)

    if mode == 'uniform':
        base_z = z_min + norm_idx * (z_max - z_min)
        z_pos = base_z + np.random.uniform(-1, 1) * avg_thick * z_var
    elif mode == 'normal':
        mean_z = (z_min + z_max) / 2
        std_dev = (z_max - z_min) / 6
        z_pos = np.random.normal(mean_z, std_dev)

    elif mode == 'stratified':
        num_strata = params.get('z_num_strata', 5)
        strata_idx = (icc - 1) % num_strata
        strata_thickness = (z_max - z_min) / num_strata
        base_z = z_min + (strata_idx + 0.5) * strata_thickness
        z_pos = base_z + np.random.uniform(-0.4, 0.4) * strata_thickness

    elif mode == 'custom':
        weights = params.get('z_custom_weights', None)
        if weights is None or len(weights) < 2:
            z_pos = z_min + norm_idx * (z_max - z_min)
        else:
            weights = np.array(weights) / sum(weights)
            cum_weights = np.cumsum(weights)
            for i, w in enumerate(cum_weights):
                if norm_idx <= w:
                    segment_start = 0 if i == 0 else cum_weights[i - 1]
                    segment_size = weights[i]
                    segment_pos = (norm_idx - segment_start) / segment_size if segment_size > 0 else 0
                    segment_z_min = z_min + (z_max - z_min) * (i / len(weights))
                    segment_z_max = z_min + (z_max - z_min) * ((i + 1) / len(weights))
                    z_pos = segment_z_min + segment_pos * (segment_z_max - segment_z_min)
                    break
            else:
                z_pos = z_max

            z_pos += np.random.uniform(-1, 1) * avg_thick * z_var

    elif mode == 'cyclic':
        cycles = params.get('z_cycles', 2.0)
        amplitude = (z_max - z_min) / 2
        midpoint = (z_max + z_min) / 2
        z_pos = midpoint + amplitude * np.sin(norm_idx * 2 * np.pi * cycles)
        z_pos += np.random.uniform(-1, 1) * avg_thick * z_var

    elif mode == 'exponential':
        exponent = params.get('z_exponent', 2.0)
        z_pos = z_min + (z_max - z_min) * (1 - norm_idx ** exponent)
        z_pos += np.random.uniform(-1, 1) * avg_thick * z_var

    else:
        z_pos = z_min + norm_idx * (z_max - z_min)

    # Get minimum thickness value
    if is_distribution_dict(params['fcct']):
        if params['fcct'].get('type') == 'discrete':
            min_thickness = min(params['fcct'].get('values', [5.0]))
        elif params['fcct'].get('type') == 'normal':
            min_thickness = params['fcct'].get('min_limit',
                           params['fcct'].get('mean', 5.0) - 2 * params['fcct'].get('std', 1.0))
        else:
            min_thickness = params['fcct'].get('min_limit', 2.2)
    else:
        min_thickness = params['fcct'][0] if len(params['fcct']) > 0 else 2.2

    z_pos = max(min_thickness, min(avg_thick * 1.5, z_pos))

    return z_pos

def getval(fdist: List[float]) -> float:
    """
    Sample value from triangular distribution, handling cases of identical values
    Parameters:
        fdist: Distribution parameter list
    Returns:
        float: Sampled random value
    """
    try:
        if len(set(fdist)) == 1:
            return fdist[0]
        else:
            return get_value_from_distribution(fdist)
    except Exception as e:
        print(f"Distribution sampling error: {e}")
        if len(fdist) >= 2:
            return fdist[1]
        else:
            return fdist[0]


def is_distribution_dict(param):
    """
    Check if parameter is a distribution dictionary
    Parameters:
        param: Parameter to check
    Returns:
        bool: True if distribution dictionary, False otherwise
    """
    return isinstance(param, dict) and 'type' in param

def adapt_distribution_param(param):
    """
    Adapt distribution parameter, compatible with list and dictionary formats
    Parameters:
        param: Original distribution parameter
    Returns:
        dict: Standardized distribution parameter dictionary
    """
    if isinstance(param, list) and len(param) >= 3:
        return {
            'type': 'triangular',
            'min': param[0],
            'mode': param[1],
            'max': param[2]
        }
    elif is_distribution_dict(param):
        return param
    else:
        print(f"Invalid distribution parameter {param}, using default triangular distribution")
        return {'type': 'triangular', 'min': 0, 'mode': 0.5, 'max': 1}

def get_value_from_distribution(dist_param):
    """
    Sample random value from various types of distributions
    Parameters:
        dist_param: Distribution parameter
    Returns:
        float: Sampled random value
    """
    dist_params = adapt_distribution_param(dist_param)

    current_state = np.random.get_state()
    np.random.set_state(const.random_state)

    dist_type = dist_params.get('type', 'triangular')
    value = 0.0

    try:
        if dist_type == 'triangular':
            min_val = dist_params.get('min', 0.0)
            mode_val = dist_params.get('mode', 0.5)
            max_val = dist_params.get('max', 1.0)

            if abs(min_val - max_val) < 1e-6:
                value = min_val
            else:
                if abs(min_val - mode_val) < 1e-6:
                    mode_val += 1e-6
                if abs(max_val - mode_val) < 1e-6:
                    mode_val -= 1e-6
                value = np.random.triangular(min_val, mode_val, max_val)

        elif dist_type == 'uniform':
            min_val = dist_params.get('min', 0.0)
            max_val = dist_params.get('max', 1.0)
            value = np.random.uniform(min_val, max_val)

        elif dist_type == 'normal':
            mean = dist_params.get('mean', 0.0)
            std = dist_params.get('std', 1.0)
            value = np.random.normal(mean, std)

        elif dist_type == 'lognormal':
            mean = dist_params.get('mean', 0.0)
            std = dist_params.get('std', 1.0)
            value = np.random.lognormal(mean, std)

        elif dist_type == 'discrete':
            values = dist_params.get('values', [0.0, 1.0])
            probabilities = dist_params.get('probabilities', None)
            if probabilities is None:
                probabilities = [1.0 / len(values)] * len(values)
            value = np.random.choice(values, p=probabilities)

        elif dist_type == 'custom':
            ranges = dist_params.get('ranges', [(0, 1)])
            weights = dist_params.get('weights', [1.0])

            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]

            selected_idx = np.random.choice(len(ranges), p=norm_weights)
            min_val, max_val = ranges[selected_idx]

            value = np.random.uniform(min_val, max_val)

        else:
            print(f"Unknown distribution type '{dist_type}', using triangular distribution")
            min_val = dist_params.get('min', 0.0)
            mode_val = dist_params.get('mode', 0.5)
            max_val = dist_params.get('max', 1.0)
            value = np.random.triangular(min_val, mode_val, max_val)

    except Exception as e:
        print(f"Distribution sampling error: {e}")
        value = dist_params.get('default', 0.0)

    # Apply constraints
    min_limit = dist_params.get('min_limit', float('-inf'))
    max_limit = dist_params.get('max_limit', float('inf'))
    value = max(min_limit, min(max_limit, value))

    const.random_state = np.random.get_state()
    np.random.set_state(current_state)

    return value

@njit
def rotc(xorig, yorig, angle, ireverse, xx, yy):
    """
    Coordinate rotation function supporting forward and reverse rotation transformations
    Parameters:
        xorig, yorig: Rotation center coordinates
        angle: Rotation angle
        ireverse: Whether to perform reverse rotation
        xx, yy: Coordinates to transform
    Returns:
        tuple: Transformed coordinates (x_new, y_new)
    """
    if ireverse == 1:
        # From rotated coordinates to original coordinates
        yy_new = yorig + yy * np.cos(angle) + xx * np.sin(angle)
        xx_new = xorig - yy * np.sin(angle) + xx * np.cos(angle)
        return xx_new, yy_new
    else:
        # From original coordinates to rotated coordinates
        ry = (yy - yorig) * np.cos(angle) - (xx - xorig) * np.sin(angle)
        rx = (yy - yorig) * np.sin(angle) + (xx - xorig) * np.cos(angle)
        return rx, ry


def create_enhanced_parameter_resolver(params):
    """
    Create parameter resolver supporting intelligent selection of regional and global parameters
    Parameters:
        params: Global parameter dictionary
    Returns:
        function: Parameter resolver function
    """
    DEFAULT_PARAMETERS = {
        'fcat': [3.0, 5.0, 8.0],
        'fcau': [0.9, 1.0, 1.1],
        'fcuw': [100.0, 120.0, 130.0],
        'fcwtr': [8, 12, 16],
        'fcawu': [13, 15, 19],
        'fcwuw': [1, 1.2, 1.3],
        'fcad': [50.0, 120.0, 200.0],
        'fcaw': [100.0, 1200.0, 2500.0],
        'fcadzl': [650, 680.0, 690.0],
        'fcawzl': [100.0, 200.0, 500.0],
        'channel_dispersion_x': 0.85,
        'channel_dispersion_z': 0.75,
        'min_channel_spacing': 0.30,
        'allow_channel_angle_variation': True,
        'channel_angle_variation': 30.0,
        'collect_centerline_data': True,
        'thickness_width_correlation_enabled': False,
        'thickness_width_correlation_factor': 0.5,
        'channel_sinuosity': [1.1, 1.3, 1.8],
        'enable_multiscale_centerline': True,
    }

    def resolve_parameter(param_name, icc=None, ic=None, verbose=False):
        resolved_value = None
        resolution_source = "Unknown"

        use_partitioning = params.get('use_partitioning', False)
        region_controller = params.get('region_controller', None)

        # Priority 1: Get parameter from regional configuration
        if region_controller and icc is not None:
            try:
                # Get region where current complex is located
                region = None
                if region_controller.axis == 'x':
                    original_ccx = params.get('original_ccx', None)
                    if original_ccx is not None and len(original_ccx) >= icc:
                        pos = original_ccx[icc - 1]
                        region = region_controller.get_region_for_position(pos)
                    else:
                        region_id = (icc - 1) % region_controller.num_regions
                        region = region_controller.get_region_by_id(region_id)
                else:
                    region_id = (icc - 1) % region_controller.num_regions
                    region = region_controller.get_region_by_id(region_id)

                # Get parameter from regional configuration
                if region and param_name in region:
                    resolved_value = region[param_name]
                    resolution_source = f"Regional configuration(ID:{region['id']})"
                    if verbose:
                        print(f"Getting {param_name} from regional configuration: {resolved_value}")

                # If not in regional configuration, try to get from region_configs
                if resolved_value is None and 'region_configs' in params:
                    region_id = region['id'] if region else (icc - 1) % region_controller.num_regions
                    region_config = params['region_configs'].get(region_id, {})
                    if param_name in region_config:
                        resolved_value = region_config[param_name]
                        resolution_source = f"Regional original configuration(ID:{region_id})"
                        if verbose:
                            print(f"Getting {param_name} from regional original configuration: {resolved_value}")

            except Exception as e:
                if verbose:
                    print(f"Regional parameter resolution failed ({param_name}): {e}")

        # Priority 2: Get from global parameters
        if resolved_value is None and param_name in params:
            resolved_value = params[param_name]
            resolution_source = "Global configuration"
            if verbose:
                print(f"Getting {param_name} from global configuration: {resolved_value}")

        # Priority 3: Get from default parameters
        if resolved_value is None and param_name in DEFAULT_PARAMETERS:
            resolved_value = DEFAULT_PARAMETERS[param_name]
            resolution_source = "Default configuration"
            if verbose:
                print(f"Getting {param_name} from default configuration: {resolved_value}")

        if resolved_value is None:
            raise KeyError(f"Parameter '{param_name}' not found")

        if verbose:
            print(f"Parameter resolution: {param_name} = {resolved_value} (Source: {resolution_source})")

        return resolved_value

    return resolve_parameter

def validate_channel_params(icc, ic, actual_steps, step_size, channel_seed, resolver, verbose, params):
    """
    Channel parameter validation and parsing function ensuring parameter validity
    Parameters:
        icc, ic: Complex and channel indices
        actual_steps: Actual number of steps
        step_size: Step size
        channel_seed: Channel seed
        resolver: Parameter resolver
        verbose: Whether to output verbose information
        params: Parameter dictionary
    Returns:
        dict: Validated parameter dictionary
    """
    validated_params = {}

    # Basic parameter validation
    if actual_steps <= 0:
        raise ValueError(f"actual_steps must be greater than 0, current value: {actual_steps}")
    if step_size <= 0:
        raise ValueError(f"step_size must be greater than 0, current value: {step_size}")

    # Get trend control related parameters
    validated_params['enable_river_trends'] = params.get('enable_river_trends', False)

    # Thickness-related parameter validation and parsing
    try:
        fcat_param = resolver('fcat', icc, ic, verbose)
        if isinstance(fcat_param, dict) and is_distribution_dict(fcat_param):
            validated_params['base_thickness'] = get_value_from_distribution(fcat_param)
        else:
            validated_params['base_thickness'] = get_value_from_distribution(fcat_param)
    except Exception as e:
        raise ValueError(f"Thickness parameter fcat parsing failed: {e}")

    # Width-related parameter validation and parsing
    try:
        fcwtr_param = resolver('fcwtr', icc, ic, verbose)
        if isinstance(fcwtr_param, dict):
            validated_params['base_cwtr'] = get_value_from_distribution(fcwtr_param)
        else:
            validated_params['base_cwtr'] = get_value_from_distribution(fcwtr_param)
    except Exception as e:
        raise ValueError(f"Width parameter fcwtr parsing failed: {e}")

    # Range parameter validation
    try:
        fcuw_param = resolver('fcuw', icc, ic, verbose)
        validated_params['thickness_range'] = get_value_from_distribution(fcuw_param) if isinstance(fcuw_param,
                                                                                                    dict) else get_value_from_distribution(
            fcuw_param)

        fcwuw_param = resolver('fcwuw', icc, ic, verbose)
        validated_params['width_range'] = get_value_from_distribution(fcwuw_param) if isinstance(fcwuw_param,
                                                                                                 dict) else get_value_from_distribution(
            fcwuw_param)
    except Exception as e:
        raise ValueError(f"Range parameter parsing failed: {e}")

    # Factor parameter validation
    try:
        fcau_param = resolver('fcau', icc, ic, verbose)
        validated_params['fcau_param'] = fcau_param

        fcawu_param = resolver('fcawu', icc, ic, verbose)
        validated_params['base_width_control'] = get_value_from_distribution(fcawu_param) if isinstance(fcawu_param,
                                                                                                        dict) else get_value_from_distribution(
            fcawu_param)
    except Exception as e:
        raise ValueError(f"Factor parameter parsing failed: {e}")

    # Correlation parameter validation
    validated_params['thickness_width_correlation_enabled'] = resolver('thickness_width_correlation_enabled', icc, ic,
                                                                       verbose)
    validated_params['thickness_width_correlation_factor'] = resolver('thickness_width_correlation_factor', icc, ic,
                                                                      verbose)

    # Trend control parameter validation
    if validated_params['enable_river_trends']:
        validated_params['depth_downstream_trend'] = params.get('depth_downstream_trend', 1.0)
        validated_params['width_downstream_trend'] = params.get('width_downstream_trend', 1.0)
        validated_params['z_downstream_trend'] = params.get('z_downstream_trend', 0.0)
        validated_params['width_curvature_factor'] = params.get('width_curvature_factor', 0.0)
        validated_params['asymmetry_curvature_factor'] = params.get('asymmetry_curvature_factor', 0.0)

        if validated_params['depth_downstream_trend'] < 0.1 or validated_params['depth_downstream_trend'] > 10.0:
            raise ValueError(f"Depth trend parameter exceeds reasonable range[0.1, 10.0]: {validated_params['depth_downstream_trend']}")
        if validated_params['width_downstream_trend'] < 0.1 or validated_params['width_downstream_trend'] > 50.0:
            raise ValueError(f"Width trend parameter exceeds reasonable range[0.1, 50.0]: {validated_params['width_downstream_trend']}")

    # Array boundary validation
    required_arrays = ['ct', 'cw', 'cz']
    for array_name in required_arrays:
        if array_name in params:
            array = params[array_name]
            if array.shape[0] <= icc - 1 or array.shape[1] <= ic - 1:
                raise ValueError(f"Array {array_name} boundary insufficient: shape={array.shape}, need access [{icc - 1}, {ic - 1}]")

    if verbose:
        print(f"Parameter validation complete - Channel {icc}.{ic}")
        print(f"Base thickness: {validated_params['base_thickness']:.2f}")
        print(f"Base width ratio: {validated_params['base_cwtr']:.2f}")
        print(f"Trend control: {validated_params['enable_river_trends']}")

    return validated_params

def calculate_z_position_with_polling(icc, total_complexes, params, region_controller, region_id):
    """
    Calculate Z position using polling + probability mechanism, use default method if vertical partitioning allocation fails
    Parameters:
        icc: Channel complex index
        total_complexes: Total number of complexes
        params: Parameter dictionary
        region_controller: Region controller
        region_id: Region ID
    Returns:
        float: Calculated Z position
    """
    z_region_idx = region_controller.get_z_region_for_complex(region_id, icc, total_complexes)

    if z_region_idx is None:
        print(f"Complex {icc}: Not assigned to vertical partition, using default Z position calculation")
        return calculate_z_position(icc, total_complexes, params)

    # Get region and vertical partition parameters
    region = region_controller.regions[region_id]
    z_params = region.get('z_distribution_params', {})
    z_min = z_params.get('z_min_depth', params.get('z_min_depth', 0.01 * params['nz'] * params['zsiz']))
    z_max = z_params.get('z_max_depth', params.get('z_max_depth', 1.0 * params['nz'] * params['zsiz']))
    z_weights = z_params.get('z_custom_weights', [1])
    num_z_regions = len(z_weights)

    # Calculate depth range of vertical partition
    z_range = z_max - z_min
    z_region_size = z_range / num_z_regions

    z_region_start = z_min + z_region_idx * z_region_size
    z_region_end = z_min + (z_region_idx + 1) * z_region_size

    z_position = z_region_start + np.random.random() * (z_region_end - z_region_start)

    print(f"Complex {icc}: Assigned to region {region_id} vertical partition {z_region_idx}, Z position = {z_position:.2f}")

    return z_position


def lusim(ny3, ysiz, range_val, nsim, array, kernel_type="exponential",
                   kernel_params=None, use_local_random=False, local_seed=None,
                   scale_type="single"):
    """
    Generate one-dimensional random field using LU decomposition method for random variation in channel thickness and width
    Parameters:
        ny3: Number of grid points in Y direction
        ysiz: Grid spacing in Y direction
        range_val: Correlation range
        nsim: Number of simulations
        array: Output array
        kernel_type: Kernel function type
        kernel_params: Kernel function parameters
        use_local_random: Whether to use local random
        local_seed: Local random seed
        scale_type: Scale type
    Returns:
        None: Results stored in array
    """
    try:
        import cupy as cp
        use_gpu = True
    except ImportError:
        use_gpu = False

    if kernel_params is None:
        kernel_params = {}

    # Save and set random state
    current_state = np.random.get_state()
    if use_local_random and local_seed is not None:
        np.random.seed(local_seed)
    else:
        np.random.set_state(const.random_state)

    # Calculate correlation range parameter
    iyr = int(range_val / ysiz) + 1
    iyr = max(iyr, 5)

    def create_simple_kernel(kernel_type, size, params):
        """Create simplified convolution kernel"""
        if use_gpu:
            if kernel_type == "exponential":
                decay_rate = params.get('decay_rate', 0.2)
                k = cp.arange(size) - (size - 1) / 2
                kernel = cp.exp(-decay_rate * cp.abs(k))
                return kernel / cp.sum(kernel)
            elif kernel_type == "gaussian":
                sigma = params.get('sigma', range_val / 3)
                k = cp.arange(size) - (size - 1) / 2
                kernel = cp.exp(-0.5 * (k / sigma) ** 2)
                return kernel / cp.sum(kernel)
            else:
                decay_rate = 0.2
                k = cp.arange(size) - (size - 1) / 2
                kernel = cp.exp(-decay_rate * cp.abs(k))
                return kernel / cp.sum(kernel)
        else:
            if kernel_type == "exponential":
                decay_rate = params.get('decay_rate', 0.2)
                k = np.arange(size) - (size - 1) / 2
                kernel = np.exp(-decay_rate * np.abs(k))
                return kernel / np.sum(kernel)
            elif kernel_type == "gaussian":
                sigma = params.get('sigma', range_val / 3)
                k = np.arange(size) - (size - 1) / 2
                kernel = np.exp(-0.5 * (k / sigma) ** 2)
                return kernel / np.sum(kernel)
            else:
                decay_rate = 0.2
                k = np.arange(size) - (size - 1) / 2
                kernel = np.exp(-decay_rate * np.abs(k))
                return kernel / np.sum(kernel)

    if use_gpu:
        # GPU implementation
        r_cpu = np.random.random(ny3) - 0.5
        r = cp.asarray(r_cpu)

        # First smoothing
        window_size = min(2 * iyr + 1, ny3)
        window = create_simple_kernel(kernel_type, window_size, kernel_params)
        array_gpu = cp.convolve(r, window, mode='same')

        # Standardization
        xm = cp.mean(array_gpu)
        xs = cp.sqrt(cp.mean(array_gpu ** 2) - xm ** 2)
        fac = 0.7 / cp.maximum(0.000001, xs)
        array_gpu = (array_gpu - xm) * fac

        # Second smoothing
        second_window_size = min(11, ny3)
        second_window = create_simple_kernel(kernel_type, second_window_size, kernel_params)
        array_gpu = cp.convolve(array_gpu, second_window, mode='same')

        # Return results
        result = cp.asnumpy(array_gpu[:ny3])
        array[:] = result[:len(array)]
    else:
        # CPU implementation
        r = np.random.random(ny3) - 0.5

        # First smoothing
        window_size = min(2 * iyr + 1, ny3)
        window = create_simple_kernel(kernel_type, window_size, kernel_params)
        temp_array = np.convolve(r, window, mode='same')[:ny3]

        # Standardization
        xm = np.mean(temp_array)
        xs = np.sqrt(np.mean(temp_array ** 2) - xm ** 2)
        fac = 0.7 / max(0.000001, xs)
        temp_array = (temp_array - xm) * fac

        # Second smoothing
        second_window_size = min(11, ny3)
        second_window = create_simple_kernel(kernel_type, second_window_size, kernel_params)
        temp_array = np.convolve(temp_array, second_window, mode='same')[:ny3]

        array[:] = temp_array[:len(array)]

    # Restore random state
    if not use_local_random:
        const.random_state = np.random.get_state()
    np.random.set_state(current_state)

    return


def collect_centerlines(ncc, nc, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos, params, tributaries_data=None):
    """
    Collect and convert all channel centerline data to global coordinate system using complete 2D rotation matrix
    Parameters:
        ncc: Number of channel complexes
        nc: Number of channels per complex array
        ccx, ccz, cco: Complex center coordinates and angle
        cx, cz: Channel center coordinates
        ccl: Channel centerline array
        ct, cw, crelpos: Channel thickness, width and relative position
        params: Parameter dictionary
        tributaries_data: Tributary data
    Returns:
        dict: Dictionary containing main channel and tributary centerline data
    """
    centerlines = {
        'main': [],
        'tributaries': []
    }

    ny3 = params['ny3'] if 'ny3' in params else int(params['ny'] * 1.2)
    ymn = params['ymn']
    ysiz = params['ysiz']

    # Collect main channel centerlines
    for icc in range(1, ncc + 1):
        for ic in range(1, nc[icc - 1] + 1):
            # Centerline coordinates
            cl_data = ccl[icc - 1, ic - 1, :]

            x_coords = []
            y_coords = []
            z_coords = []

            # Complex angle
            angle = cco[icc - 1] * (np.pi / 180.0)

            # Center point position
            x_center = ccx[icc - 1] + cx[icc - 1, ic - 1]
            z_top = cz[icc - 1, ic - 1]

            for iy in range(ny3):
                # Channel centerline position - local coordinates
                cl_pos = cl_data[iy]
                local_y = iy * ysiz

                # Calculate global coordinates using 2D rotation matrix
                x_pos = x_center + cl_pos * np.cos(angle) - local_y * np.sin(angle)
                y_pos = ymn + cl_pos * np.sin(angle) + local_y * np.cos(angle)
                z_pos = z_top - ct[icc - 1, ic - 1, iy] / 2

                x_coords.append(x_pos)
                y_coords.append(y_pos)
                z_coords.append(z_pos)

            # Add to centerline list
            centerlines['main'].append({
                'icc': icc,
                'ic': ic,
                'x': x_coords,
                'y': y_coords,
                'z': z_coords
            })
    return centerlines