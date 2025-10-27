"""
engine.py
"""
import time
import random
import numpy as np
from typing import Dict, Any
from . import constants as const
from .process_executor import get_executor
from ..channel import channel_complex
from ..utils.utils import get_value_from_distribution, is_distribution_dict,collect_centerlines
from ..channel.channel import calculate_max_channel_length, getchan
from ..render.render_channel import rasterc_with_gpu
from ..channel.channel_complex import create_fallback_complex, RegionController
from ..channel.intersection_cutoff import apply_batch_intersection_cutoff
from ..migration.gpu_migration import migrate_river_channel_2d_gpu, check_gpu_availability
from ..migration.neck_cutoff import check_neck_cutoff_gpu_sequential
from ..facies.facies_generator import generate_levees, generate_crevasse_splays






def fluvpy(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main river channel simulation function, manages single or multiple parallel channel simulation realizations.

    Args:
        params: Simulation parameter dictionary containing grid dimensions, channel complex count, realization count, etc.

    Returns:
        Dict[str, Any]: Dictionary containing all realization results with keys in 'realization_n' format
    """
    print("\nStarting dynamic step river channel simulation...")

    # Initialize results dictionary
    results = {}
    nsim = params['nsim']
    # Single realization
    if nsim == 1:
        result = fluvpy_single(params)
        results['realization_1'] = result
        return results
    # Multiple realizations using parallel processing
    print(f"\nExecuting {nsim} parallel dynamic step simulation realizations...")

    executor = get_executor()

    # Prepare parameters for each realization
    sim_params = []
    for i in range(1, nsim + 1):
        p = params.copy()
        p['seed'] = params.get('seed', 123456) + i - 1
        p['parallel_inside'] = True if nsim <= 2 else False
        sim_params.append((p, i))

    # Run realizations using process mode
    results_list = executor.execute(
        lambda params_tuple: fluvpy_single(*params_tuple),
        sim_params,
        mode="process"
    )

    # Collect results
    for i, result in enumerate(results_list, 1):
        results[f'realization_{i}'] = result

    print(f"\nCompleted {nsim} parallel dynamic step simulation realizations")

    # Aggregate dynamic step count statistics
    all_step_counts = []
    for key, result in results.items():
        if 'dynamic_step_info' in result and 'channel_actual_steps' in result['dynamic_step_info']:
            step_counts = list(result['dynamic_step_info']['channel_actual_steps'].values())
            all_step_counts.extend(step_counts)

    if all_step_counts:
        print(f"\nAll realizations dynamic step count aggregate statistics")
        print(f"Total channels: {len(all_step_counts)}")
        print(f"Step count range: {min(all_step_counts)} - {max(all_step_counts)}")
        print(f"Average step count: {np.mean(all_step_counts):.1f}")
        print(f"Median step count: {np.median(all_step_counts):.1f}")
        print(f"Standard deviation: {np.std(all_step_counts):.1f}")

    return results


def fluvpy_single(params: Dict[str, Any], isim: int = 1) -> Dict[str, Any]:
    """
    Execute single river channel simulation realization with dynamic step count and GPU acceleration support.

    Args:
        params: Simulation parameter dictionary containing grid configuration, channel parameters, migration settings, etc.
        isim: Simulation realization number, defaults to 1

    Returns:
        Dict[str, Any]: Complete simulation results including channel grid, porosity, permeability, centerline data, etc.
    """
    start_time = time.time()
    print(f"\n\nExecuting dynamic step realization number {isim}")

    # Set random seed
    seed = params.get('seed', 123456)
    random.seed(seed)
    np.random.seed(seed)
    const.set_global_seed(seed)

    # Dynamically calculate maximum channel length
    max_possible_sinuosity = 3.0
    if 'channel_sinuosity' in params:
        sinuosity_param = params['channel_sinuosity']
        if isinstance(sinuosity_param, dict):
            if sinuosity_param.get('type') == 'discrete':
                max_possible_sinuosity = max(sinuosity_param.get('values', [3.0]))
            elif sinuosity_param.get('type') == 'normal':
                max_possible_sinuosity = sinuosity_param.get('max_limit', 3.0)
            else:
                max_possible_sinuosity = sinuosity_param.get('max', 3.0)
        else:
            max_possible_sinuosity = max(sinuosity_param)

    max_length, max_steps, step_size = calculate_max_channel_length(params, max_possible_sinuosity)
    print(f"Dynamic array initialization: max steps={max_steps}, step size={step_size:.2f}")

    # Extract basic parameters
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']
    xmn = params['xmn']
    ymn = params['ymn']
    zmn = params['zmn']
    xsiz = params['xsiz']
    ysiz = params['ysiz']
    zsiz = params['zsiz']
    mxcc = params['mxcc']
    mxc = params['mxc']
    ipor = params['ipor']
    # Get optimization and migration parameters
    enable_migration = params.get('enable_migration', False)
    migration_steps = params.get('migration_steps', 5) if enable_migration else 0

    # Calculate actual channel capacity (considering migration)
    actual_mxc = mxc
    if enable_migration:
        actual_mxc = mxc + migration_steps
        print(f"Channel migration enabled: extended channel capacity to {actual_mxc}")

    # Initialize arrays using dynamic length
    ncc = 0
    nc = np.zeros(mxcc, dtype=int)

    # Channel complex arrays
    ccx = np.zeros(mxcc)
    ccz = np.zeros(mxcc)
    cco = np.zeros(mxcc)
    cct = np.zeros(mxcc)
    ccw = np.zeros(mxcc)
    ccntg = np.zeros(mxcc)

    # Channel arrays (using actual capacity)
    cx = np.zeros((mxcc, actual_mxc))
    cz = np.zeros((mxcc, actual_mxc))
    chanon = np.ones(mxcc * actual_mxc, dtype=bool)

    # Initialize channel geometry arrays using dynamically calculated maximum steps
    ccl = np.zeros((mxcc, actual_mxc, max_steps))
    ct = np.zeros((mxcc, actual_mxc, max_steps))
    cw = np.zeros((mxcc, actual_mxc, max_steps))
    crelpos = np.zeros((mxcc, actual_mxc, max_steps))
    carea = np.zeros((mxcc, actual_mxc))

    # Dynamic Y-coordinate and Z-offset arrays
    centerline_y_data = np.zeros((mxcc, actual_mxc, max_steps))
    centerline_z_offset = np.zeros((mxcc, actual_mxc, max_steps))

    # Save to parameters
    params['centerline_y_data'] = centerline_y_data
    params['centerline_z_offset'] = centerline_z_offset
    params['actual_mxc'] = actual_mxc
    params['max_steps'] = max_steps
    params['step_size'] = step_size

    # Set default Y coordinates for initial channels
    for icc in range(1, mxcc + 1):
        for ic in range(1, actual_mxc + 1):
            for iy in range(max_steps):
                centerline_y_data[icc - 1, ic - 1, iy] = ymn + iy * step_size

    print(f"Dynamic array initialization complete: channel geometry array size = ({mxcc}, {actual_mxc}, {max_steps})")

    # Initialize grid arrays
    channel_grid = np.zeros((nx, ny, nz), dtype=int)
    por = np.zeros((nx, ny, nz))
    per = np.zeros((nx, ny, nz))
    pcurvea = np.zeros(nz)
    pmapa = np.zeros((nx, ny))

    # Channel complex generation
    print("\n  Batch generation of initial channel complex distribution")

    # Check partition control
    if 'region_controller' not in params:
        axis = params.get('partition_axis', 'x')
        num_regions = params.get('num_regions', 3)
        print(f"Partition control enabled: divided into {num_regions} regions along {axis} axis")
        params['region_controller'] = RegionController(params, axis, num_regions)
    # Generate channel complexes

    batch_size = min(5, mxcc)
    ncc = 0
    actual_complex_idx = 0

    getcc_func = channel_complex.getcc_with_density_control

    # First batch complex generation
    while ncc < batch_size and actual_complex_idx < mxcc:
        success = getcc_func(ncc + 1, params, ccx, ccz, cco, cct, ccw, ccntg)
        if success:
            ncc += 1
        actual_complex_idx += 1

    # Render - using GPU version
    print(f"     Initial {ncc} complexes")

    # Continue adding complexes until target ratio is reached
    while ncc < mxcc and actual_complex_idx < mxcc * 2:
        old_ncc = ncc
        next_batch_size = min(batch_size, mxcc - ncc)
        batch_added = 0

        while batch_added < next_batch_size and actual_complex_idx < mxcc * 2:
            success = getcc_func(ncc + 1, params, ccx, ccz, cco, cct, ccw, ccntg)
            if success:
                ncc += 1

                batch_added += 1
            actual_complex_idx += 1

        if ncc > old_ncc:
            print(f"     Added {ncc - old_ncc} complexes")
    if ncc == 0:
        print("\n  Warning: Partition control generated no complexes, activating safety net mechanism")
        success = create_fallback_complex(params, ccx, ccz, cco, cct, ccw, ccntg)
        if success:
            ncc = 1
            print(f"     Safety net mechanism successfully generated 1 complex, current complex count: {ncc}")
        else:
            print("     Safety net mechanism failed, will continue attempting with empty model")

    print(f"\n  Complex generation complete, final complex count: {ncc}/{mxcc}")


    executor = get_executor()

    ichan = 0
    chanon = np.ones(mxcc * actual_mxc, dtype=bool)


    def fill_complex_batch_dynamic(complex_range):
        """
        Dynamic step count supported complex batch filling.

        Args:
            complex_range: Complex index range

        Returns:
            list: List of tuples containing (complex number, channel count)
        """
        local_results = []
        for complex_idx in complex_range:
            icc = complex_idx + 1
            if icc > ncc:
                continue

            # Calculate tolerance
            if isinstance(params['fcat'], dict) and is_distribution_dict(params['fcat']):
                thickness_value = 0.25 * get_value_from_distribution(params['fcat'])
            else:
                thickness_value = 0.25 * params['fcat'][1]

            if isinstance(params['fcwtr'], dict) and is_distribution_dict(params['fcwtr']):
                width_ratio = get_value_from_distribution(params['fcwtr'])
            else:
                width_ratio = params['fcwtr'][1]

            ptol = ccntg[icc - 1] - thickness_value * thickness_value * width_ratio / max(
                cct[icc - 1] * ccw[icc - 1], const.EPSLON)

            local_nc = 0
            local_channels = []

            # Generate at least one channel per complex
            for ic in range(1, max(2, mxc + 1)):
                try:
                    pcc = getchan(icc, ic, params, cx, cz, nc, ccx, ccz, cco, cct, ccw, ccntg,
                                  ccl, ct, cw, crelpos, carea)

                    # Update counter
                    with executor._task_lock:
                        nonlocal ichan
                        ichan += 1
                        chanon[ichan - 1] = True

                    local_nc += 1
                    local_channels.append(ic)

                    # Get actual step count for this channel
                    channel_key = f"{icc}_{ic}"
                    actual_steps = params.get('channel_actual_steps', {}).get(channel_key, max_steps)

                    if icc <= 5:
                        print(f"Debug: Complex {icc}, channel {ic} generated, steps={actual_steps}, pcc={pcc:.4f}")

                    if pcc >= ptol and local_nc > 0:
                        break

                except Exception as e:
                    print(f"Error generating channel {ic} for complex {icc}: {e}")
                    import traceback
                    traceback.print_exc()

                    # Create fallback channel
                    if local_nc == 0 and ic == 1:
                        try:
                            print(f"Generating fallback channel for complex {icc}")
                            cx[icc - 1, 0] = 0
                            cz[icc - 1, 0] = ccz[icc - 1] - 0.1 * cct[icc - 1]

                            default_steps = min(100, max_steps)

                            for i in range(default_steps):
                                if i < ccl.shape[2]:
                                    ccl[icc - 1, 0, i] = 0.0
                                if i < ct.shape[2]:
                                    ct[icc - 1, 0, i] = 0.5 * cct[icc - 1]
                                if i < cw.shape[2]:
                                    cw[icc - 1, 0, i] = 0.5 * ccw[icc - 1]
                                if i < crelpos.shape[2]:
                                    crelpos[icc - 1, 0, i] = 0.5
                                if i < centerline_y_data.shape[2]:
                                    centerline_y_data[icc - 1, 0, i] = ymn + i * step_size

                            params.setdefault('channel_actual_steps', {})[f"{icc}_1"] = default_steps

                            with executor._task_lock:
                                ichan += 1
                                chanon[ichan - 1] = True

                            local_nc += 1
                            local_channels.append(1)
                            print(f"Created fallback channel for complex {icc} (steps: {default_steps})")
                        except Exception as e2:
                            print(f"Fallback channel creation failed: {e2}")

            # Update global count
            with executor._task_lock:
                nc[icc - 1] = local_nc

            local_results.append((icc, local_nc))

        return local_results

    # Create complex batches and process in parallel
    complex_ranges = []
    batch_size = max(1, min(5, ncc // 4))

    for i in range(0, ncc, batch_size):
        complex_ranges.append(range(i, min(i + batch_size, ncc)))

    # Use thread mode for parallel filling
    results = executor.execute(
        fill_complex_batch_dynamic,
        complex_ranges,
        mode="thread"
    )

    # Process results
    for batch_results in results:
        for icc, count in batch_results:
            if count > 0:
                step_counts = []
                for ic in range(1, count + 1):
                    channel_key = f"{icc}_{ic}"
                    actual_steps = params.get('channel_actual_steps', {}).get(channel_key, max_steps)
                    step_counts.append(actual_steps)

                avg_steps = np.mean(step_counts) if step_counts else 0

    # Check channel generation results
    total_channels = sum(nc)
    print(f"\nGenerated total channels: {total_channels}")

    if total_channels == 0:
        print("\nWarning: No channels generated! Adding fallback channels...")
        for icc in range(1, ncc + 1):
            nc[icc - 1] = 1
            default_steps = min(100, max_steps)

            cx[icc - 1, 0] = 0
            cz[icc - 1, 0] = ccz[icc - 1] - 0.1 * cct[icc - 1]

            for iy in range(default_steps):
                if iy < ccl.shape[2]:
                    ccl[icc - 1, 0, iy] = 0.0
                if iy < ct.shape[2]:
                    ct[icc - 1, 0, iy] = 0.5 * cct[icc - 1]
                if iy < cw.shape[2]:
                    cw[icc - 1, 0, iy] = 0.5 * ccw[icc - 1]
                if iy < crelpos.shape[2]:
                    crelpos[icc - 1, 0, iy] = 0.5
                if iy < centerline_y_data.shape[2]:
                    centerline_y_data[icc - 1, 0, iy] = ymn + iy * step_size

            params.setdefault('channel_actual_steps', {})[f"{icc}_1"] = default_steps

            # Activate channel
            chan_index = 0
            for prev_icc in range(1, icc):
                chan_index += nc[prev_icc - 1]
            if chan_index < len(chanon):
                chanon[chan_index] = True

        print(f"Added {ncc} fallback channels")

    active_channels = np.sum(chanon[:sum(nc)])
    print(f"Active channels: {active_channels} / {sum(nc)}")

    # Batch self-intersection detection and cutoff
    apply_batch_intersection_cutoff(params, ncc, nc, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos, centerline_y_data)

    # Channel migration processing
    migration_start_time = time.time()
    if enable_migration:

        print("\n  Executing channel migration simulation (dynamic step mode)")
        migration_steps = params.get('migration_steps', 5)
        use_gpu_migration = params.get('use_gpu_migration', True)

        if use_gpu_migration:
            print("  Attempting to use GPU-accelerated channel migration")
            try:

                gpu_available = check_gpu_availability()
                if gpu_available:
                    print("  GPU migration algorithm available")
                    migrate_func = migrate_river_channel_2d_gpu
                    cutoff_func = check_neck_cutoff_gpu_sequential
                else:
                    print("  GPU unavailable, skipping migration processing")
                    migrate_func = None
                    cutoff_func = None
            except ImportError:
                print("  GPU migration module unavailable, skipping migration processing")
                migrate_func = None
                cutoff_func = None
        else:
            print("  Channel migration disabled")
            migrate_func = None
            cutoff_func = None

        # Migration data storage
        migration_data = {}
        migrated_channels_info = []

        # Prepare initial data for all channels
        all_channels_initial_data = []
        for icc in range(1, ncc + 1):
            for ic in range(1, nc[icc - 1] + 1):
                channel_key = f"{icc}_{ic}"
                actual_steps = params.get('channel_actual_steps', {}).get(channel_key, max_steps)

                current_centerline = ccl[icc - 1, ic - 1, :actual_steps].copy()
                current_centerline_y = centerline_y_data[icc - 1, ic - 1, :actual_steps].copy()
                current_thickness = ct[icc - 1, ic - 1, :actual_steps].copy()
                current_width = cw[icc - 1, ic - 1, :actual_steps].copy()
                current_relpos = crelpos[icc - 1, ic - 1, :actual_steps].copy()
                initial_z_top = cz[icc - 1, ic - 1]

                migration_data[channel_key] = [{
                    'step': 0,
                    'centerline': current_centerline.copy(),
                    'centerline_y': current_centerline_y.copy(),
                    'thickness': current_thickness.copy(),
                    'width': current_width.copy(),
                    'relpos': current_relpos.copy(),
                    'z_top': initial_z_top,
                    'actual_steps': actual_steps
                }]

                all_channels_initial_data.append({
                    'icc': icc,
                    'ic': ic,
                    'key': channel_key,
                    'centerline_x': current_centerline,
                    'centerline_y': current_centerline_y,
                    'thickness': current_thickness,
                    'width': current_width,
                    'relpos': current_relpos,
                    'initial_z_top': initial_z_top,
                    'actual_steps': actual_steps
                })

        # Execute multi-step migration
        migration_z_increment = params.get('migration_z_increment', 0.0)

        for step in range(1, migration_steps + 1):
            print(f"      Migration step {step}/{migration_steps} (dynamic step mode)")

            if use_gpu_migration and migrate_func == migrate_river_channel_2d_gpu:
                # GPU parallel processing
                all_channels_current_data = []
                for channel_key, channel_data in migration_data.items():
                    previous_data = channel_data[-1]
                    all_channels_current_data.append({
                        'centerline_x': previous_data['centerline'],
                        'centerline_y': previous_data['centerline_y'],
                        'thickness': previous_data['thickness'],
                        'width': previous_data['width'],
                        'relpos': previous_data['relpos'],
                        'actual_steps': previous_data['actual_steps']
                    })

                migrated_channels_data = migrate_func(params, all_channels_current_data)

                if params.get('enable_cutoff', True):
                    cutoff_results = cutoff_func(params, migrated_channels_data)
                    for i, result in enumerate(cutoff_results):
                        if result['cutoff_occurred']:
                            migrated_channels_data[i]['centerline_x'] = result['modified_x']
                            migrated_channels_data[i]['centerline_y'] = result['modified_y']

                # Update migration data
                for i, channel in enumerate(all_channels_initial_data):
                    icc = channel['icc']
                    ic = channel['ic']
                    channel_key = channel['key']
                    initial_z_top = channel['initial_z_top']
                    actual_steps = channel['actual_steps']

                    current_z_top = initial_z_top + step * migration_z_increment

                    new_centerline = migrated_channels_data[i]['centerline_x'][:actual_steps]
                    new_centerline_y = migrated_channels_data[i]['centerline_y'][:actual_steps]
                    new_thickness = migrated_channels_data[i]['thickness'][:actual_steps]
                    new_width = migrated_channels_data[i]['width'][:actual_steps]
                    new_relpos = migrated_channels_data[i]['relpos'][:actual_steps]

                    migration_data[channel_key].append({
                        'step': step,
                        'centerline': new_centerline.copy(),
                        'centerline_y': new_centerline_y.copy(),
                        'thickness': new_thickness.copy(),
                        'width': new_width.copy(),
                        'relpos': new_relpos.copy(),
                        'z_top': current_z_top,
                        'actual_steps': actual_steps
                    })

                    migrated_channels_info.append({
                        'icc': icc,
                        'ic': ic + step,
                        'step': step,
                        'centerline': new_centerline.copy(),
                        'centerline_y': new_centerline_y.copy(),
                        'width': new_width.copy(),
                        'thickness': new_thickness.copy(),
                        'x_center': ccx[icc - 1] + cx[icc - 1, ic - 1],
                        'z_top': current_z_top,
                        'angle': cco[icc - 1] * const.DEG2RAD,
                        'actual_steps': actual_steps
                    })

        # Add migrated channels to model
        add_start_time = time.time()
        if migrated_channels_info:
            migration_channels_added = 0

            complex_migrations = {}
            for channel in migrated_channels_info:
                icc = channel['icc']
                if icc not in complex_migrations:
                    complex_migrations[icc] = []
                complex_migrations[icc].append(channel)

            for icc, channels in complex_migrations.items():
                current_channel_count = nc[icc - 1]
                channels_to_add = channels[:actual_mxc - current_channel_count]

                if len(channels_to_add) < len(channels):
                    print(f"  Warning: Complex {icc} cannot add all migrated channels, reached capacity limit")

                for i, channel in enumerate(channels_to_add):
                    new_ic = current_channel_count + i + 1

                    try:
                        centerline_data = channel['centerline']
                        centerline_y_data_local = channel['centerline_y']
                        thickness_data = channel['thickness']
                        width_data = channel['width']
                        z_top_value = channel['z_top']
                        actual_steps = channel['actual_steps']

                        if len(centerline_data) == 0:
                            continue

                        max_length = min(actual_steps, max_steps, len(centerline_data))

                        if max_length <= ccl.shape[2]:
                            ccl[icc - 1, new_ic - 1, :max_length] = centerline_data[:max_length]
                        if max_length <= ct.shape[2]:
                            ct[icc - 1, new_ic - 1, :max_length] = thickness_data[:max_length]
                        if max_length <= cw.shape[2]:
                            cw[icc - 1, new_ic - 1, :max_length] = width_data[:max_length]

                        if len(centerline_y_data_local) > 0 and max_length <= centerline_y_data.shape[2]:
                            centerline_y_data[icc - 1, new_ic - 1, :max_length] = centerline_y_data_local[:max_length]

                        offset = channel['step'] * 0.05 * ccw[icc - 1]
                        cx[icc - 1, new_ic - 1] = cx[icc - 1, 0] + offset
                        cz[icc - 1, new_ic - 1] = z_top_value

                        for j in range(max_length):
                            if j < crelpos.shape[2]:
                                if j > 0 and j < max_length - 1:
                                    dx = centerline_data[j + 1] - centerline_data[j - 1]
                                    crelpos[icc - 1, new_ic - 1, j] = 0.5 + np.sign(dx) * 0.1
                                else:
                                    crelpos[icc - 1, new_ic - 1, j] = 0.5

                        if max_length > 0:
                            avg_width = np.mean(width_data[:max_length])
                            avg_thickness = np.mean(thickness_data[:max_length])
                            carea[icc - 1, new_ic - 1] = 0.5 * avg_width * avg_thickness

                        params.setdefault('channel_actual_steps', {})[f"{icc}_{new_ic}"] = max_length

                        # Generate detailed centerline data for migrated channels
                        migrated_channel_key = f"{icc}_{new_ic}"
                        centerline_data_entry = {
                            'channel_id': migrated_channel_key,
                            'complex_x': ccx[icc - 1],
                            'complex_z': ccz[icc - 1],
                            'channel_x': cx[icc - 1, new_ic - 1],
                            'channel_z': cz[icc - 1, new_ic - 1],
                            'global_angle': cco[icc - 1],
                            'weights': {'algorithm_type': 'migration'},
                            'actual_steps': max_length,
                            'algorithm_type': 'migration',
                            'points': []
                        }

                        # Create detailed data for each point
                        for point_idx in range(max_length):
                            point_data = {
                                'point_index': point_idx,
                                'local_x': centerline_data[point_idx],
                                'local_y': centerline_y_data_local[point_idx] if point_idx < len(
                                    centerline_y_data_local) else ymn + point_idx * step_size,
                                'width': width_data[point_idx] if point_idx < len(width_data) else avg_width,
                                'thickness': thickness_data[point_idx] if point_idx < len(
                                    thickness_data) else avg_thickness,
                                'curvature': 0.0,
                                'rel_position': crelpos[icc - 1, new_ic - 1, point_idx] if point_idx < crelpos.shape[
                                    2] else 0.5,
                                # Ferguson force decomposition data (set to 0 for migrated channels)
                                'ferguson_force_dx': 0.0,
                                'ferguson_force_dy': 0.0,
                                'statistical_force_dx': 0.0,
                                'statistical_force_dy': 0.0,
                                'statistical_alpha': 0.0,
                                'raw_ferguson_angle': 0.0,
                                'final_ferguson_angle': 0.0,
                                'target_deviation_x': 0.0,
                                'target_deviation_y': 0.0,
                                'ferguson_progress_ratio': float(point_idx) / max(max_length - 1, 1)
                            }
                            centerline_data_entry['points'].append(point_data)

                        # Add to centerline_data list
                        if 'centerline_data' not in params:
                            params['centerline_data'] = []
                        params['centerline_data'].append(centerline_data_entry)

                        # Create actual properties for migrated channels
                        original_channel_key = f"{icc}_1"
                        original_properties = params.get('actual_channel_properties', {}).get(original_channel_key, {})

                        migrated_properties = {
                            'actual_sinuosity': original_properties.get('actual_sinuosity', 1.0),
                            'target_sinuosity': original_properties.get('target_sinuosity', 1.0),
                            'actual_complex_angle': cco[icc - 1],
                            'actual_avg_thickness': avg_thickness,
                            'actual_avg_width': avg_width,
                            'actual_width_thickness_ratio': avg_width / max(avg_thickness, 0.001),
                            'actual_max_thickness': np.max(
                                thickness_data[:max_length]) if max_length > 0 else avg_thickness,
                            'actual_min_thickness': np.min(
                                thickness_data[:max_length]) if max_length > 0 else avg_thickness,
                            'actual_max_width': np.max(width_data[:max_length]) if max_length > 0 else avg_width,
                            'actual_min_width': np.min(width_data[:max_length]) if max_length > 0 else avg_width,
                            'algorithm_used': 'migration',
                            'actual_complex_x': ccx[icc - 1],
                            'actual_complex_z': ccz[icc - 1],
                            'actual_channel_x': cx[icc - 1, new_ic - 1],
                            'actual_channel_z': cz[icc - 1, new_ic - 1],
                            'actual_steps': max_length,
                            'generated_steps': max_length
                        }

                        # Save actual properties of migrated channels
                        if 'actual_channel_properties' not in params:
                            params['actual_channel_properties'] = {}
                        params['actual_channel_properties'][migrated_channel_key] = migrated_properties

                        chan_index = 0
                        for prev_icc in range(1, icc):
                            chan_index += nc[prev_icc - 1]
                        chan_index += new_ic - 1

                        if chan_index < len(chanon):
                            chanon[chan_index] = True
                        else:
                            new_chanon = np.ones(chan_index + 10, dtype=bool)
                            new_chanon[:len(chanon)] = chanon
                            chanon = new_chanon
                            chanon[chan_index] = True

                        migration_channels_added += 1
                        nc[icc - 1] += 1

                    except Exception as e:
                        print(f"  Failed to add migrated channel: {str(e)}")

            print(f"  Successfully added {migration_channels_added} migrated channels", {}, )
            add_end_time = time.time()
            print(f"     Addition time: {add_end_time - add_start_time:.3f}s")

            if 'channel_actual_steps' in params:
                step_counts = list(params['channel_actual_steps'].values())
                if step_counts:
                    print(
                        f"  Post-migration step count statistics: min={min(step_counts)}, max={max(step_counts)}, avg={np.mean(step_counts):.1f}")

            params['migration_data'] = migration_data
            params['migrated_channels_info'] = migrated_channels_info
    migration_end_time = time.time()
    print(f"   Channel migration section complete, time: {migration_end_time - migration_start_time:.3f}s")

    # Generate main channel information
    main_channels_info = []
    missing_properties_count = 0

    for icc in range(1, ncc + 1):
        for ic in range(1, nc[icc - 1] + 1):
            channel_key = f"{icc}_{ic}"
            actual_steps = params.get('channel_actual_steps', {}).get(channel_key, max_steps)

            # Get all values from actual calculated properties
            actual_properties = params.get('actual_channel_properties', {}).get(channel_key, {})

            if not actual_properties:
                print(f"Warning: Channel {channel_key} missing actual calculated properties, skipping this channel")
                missing_properties_count += 1
                continue

            # Get actual calculated sinuosity value
            actual_sinuosity = actual_properties.get('actual_sinuosity')
            if actual_sinuosity is None:
                print(f"Warning: Channel {channel_key} missing actual sinuosity value, skipping this channel")
                missing_properties_count += 1
                continue

            # Get actual calculated complex angle
            actual_fcco = actual_properties.get('actual_complex_angle')
            if actual_fcco is None:
                print(f"Warning: Channel {channel_key} missing actual complex angle, skipping this channel")
                missing_properties_count += 1
                continue

            # Get actual geometric properties
            actual_avg_thickness = actual_properties.get('actual_avg_thickness', 0.0)
            actual_avg_width = actual_properties.get('actual_avg_width', 0.0)
            actual_width_thickness_ratio = actual_properties.get('actual_width_thickness_ratio', 0.0)

            # Get actual position properties
            actual_complex_x = actual_properties.get('actual_complex_x', ccx[icc - 1])
            actual_complex_z = actual_properties.get('actual_complex_z', ccz[icc - 1])
            actual_channel_x = actual_properties.get('actual_channel_x', cx[icc - 1, ic - 1])
            actual_channel_z = actual_properties.get('actual_channel_z', cz[icc - 1, ic - 1])

            # Calculate actual x_center
            actual_x_center = actual_complex_x + actual_channel_x

            # Build main channel information
            channel_info = {
                'icc': icc,
                'ic': ic,
                'centerline': ccl[icc - 1, ic - 1, :actual_steps].copy(),
                'width': cw[icc - 1, ic - 1, :actual_steps].copy(),
                'thickness': ct[icc - 1, ic - 1, :actual_steps].copy(),
                'x_center': actual_x_center,
                'z_top': actual_channel_z,
                'angle': actual_fcco * const.DEG2RAD,
                'centerline_y': centerline_y_data[icc - 1, ic - 1, :actual_steps].copy(),
                'channel_sinuosity': actual_sinuosity,
                'fcco': actual_fcco,
                'actual_steps': actual_steps,

                # Add more actual calculated properties
                'actual_avg_thickness': actual_avg_thickness,
                'actual_avg_width': actual_avg_width,
                'actual_width_thickness_ratio': actual_width_thickness_ratio,
                'actual_max_thickness': actual_properties.get('actual_max_thickness', actual_avg_thickness),
                'actual_min_thickness': actual_properties.get('actual_min_thickness', actual_avg_thickness),
                'actual_max_width': actual_properties.get('actual_max_width', actual_avg_width),
                'actual_min_width': actual_properties.get('actual_min_width', actual_avg_width),
                'algorithm_used': actual_properties.get('algorithm_used', 'unknown'),
                'target_sinuosity': actual_properties.get('target_sinuosity', actual_sinuosity),

                # Actual complex properties
                'actual_complex_x': actual_complex_x,
                'actual_complex_z': actual_complex_z,
                'actual_channel_x': actual_channel_x,
                'actual_channel_z': actual_channel_z,
            }

            main_channels_info.append(channel_info)

            # Detailed output of first channel property verification
            if icc == 1 and ic == 1:
                print(f"  Channel {channel_key} actual property verification:")
                print(f"    Actual sinuosity: {actual_sinuosity:.4f}")
                print(f"    Target sinuosity: {actual_properties.get('target_sinuosity', 'N/A')}")
                print(f"    Actual complex angle: {actual_fcco:.1f}°")
                print(f"    Actual average thickness: {actual_avg_thickness:.2f}")
                print(f"    Actual average width: {actual_avg_width:.2f}")
                print(f"    Actual width-thickness ratio: {actual_width_thickness_ratio:.3f}")
                print(f"    Algorithm type: {actual_properties.get('algorithm_used', 'unknown')}")

    if missing_properties_count > 0:
        print(f"Warning: {missing_properties_count} channels missing actual calculated properties, skipped")
        print("Recommend checking if property saving during channel generation is correct")

    # Save to params and results
    params['main_channels_info'] = main_channels_info

    # Generate levees and crevasse splays
    facies_types = params.get('facies_types', ['channel'])

    # Generate levees
    levees = []
    if 'levee' in facies_types and params.get('levee_enabled', False):
        print("\n  Generating levees (dynamic step mode)")

        for icc in range(1, ncc + 1):
            for ic in range(1, nc[icc - 1] + 1):
                channel_levees = generate_levees(params, icc, ic, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos)
                levees.extend(channel_levees)
        print(f"     Total generated {len(levees)} levees")
        params['levees_data'] = levees

    # Generate crevasse splays
    crevasse_splays = []
    if 'crevasse' in facies_types and params.get('crevasse_enabled', False):
        print("\n  Generating crevasse splays (dynamic step mode)")

        for icc in range(1, ncc + 1):
            for ic in range(1, nc[icc - 1] + 1):
                channel_splays = generate_crevasse_splays(params, icc, ic, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos)
                crevasse_splays.extend(channel_splays)
        print(f"     Total generated {len(crevasse_splays)} crevasse splays")
        params['crevasse_data'] = crevasse_splays

    # Final rendering
    print("\n  Final rendering using GPU (dynamic step mode)")

    prop = rasterc_with_gpu(ncc, nx, ny, nz, xmn, ymn, zmn, xsiz, ysiz, zsiz,
                            ccx, ccz, cco, cct, ccw, cx, cz, nc,
                            channel_grid, por, ccl, ct, cw, crelpos, pcurvea, pmapa,
                            chanon, True, params, True)


    # Ensure prop has valid value
    if prop is None:
        channel_binary = (channel_grid > 0).astype(np.float64)
        prop = float(np.mean(channel_binary))
        print(f"Warning: Direct calculation of global proportion: {prop:.4f}")

    end_time = time.time()
    print(f"\n  Dynamic step simulation {isim} completed in {end_time - start_time:.2f} seconds")

    # Collect centerline data

    centerlines = collect_centerlines(
        ncc, nc, ccx, ccz, cco, cx, cz, ccl, ct, cw, crelpos,
        params, params.get('tributary_data', {})
    )

    # Build results dictionary
    result = {
        'channel': channel_grid.copy(),
        'porosity': por.copy(),
        'permeability': per.copy(),
        'vertical_proportion': pcurvea.copy(),
        'areal_proportion': pmapa.copy(),
        'global_proportion': prop,
        'centerlines': centerlines,
        'facies_info': {
            'levees': levees,
            'crevasse_splays': crevasse_splays
        },
        'main_channels_info': main_channels_info,
        'dynamic_step_info': {
            'max_steps': max_steps,
            'step_size': step_size,
            'channel_actual_steps': params.get('channel_actual_steps', {}),
            'step_statistics': {
                'total_channels': len(params.get('channel_actual_steps', {})),
                'min_steps': min(params.get('channel_actual_steps', {}).values()) if params.get(
                    'channel_actual_steps') else 0,
                'max_steps': max(params.get('channel_actual_steps', {}).values()) if params.get(
                    'channel_actual_steps') else 0,
                'avg_steps': np.mean(list(params.get('channel_actual_steps', {}).values())) if params.get(
                    'channel_actual_steps') else 0
            }
        }
    }

    # Add migration results if migration is enabled
    if params.get('enable_migration', False):
        result['migration_info'] = {
            'migration_data': params.get('migration_data', {}),
            'migrated_channels_info': params.get('migrated_channels_info', [])
        }

    # Add detailed centerline data
    if 'centerline_data' in params:
        result['centerline_data'] = params['centerline_data']
        print(f"Collected detailed centerline data for {len(params['centerline_data'])} channels")

    # Print final dynamic step count statistics
    if 'channel_actual_steps' in params and params['channel_actual_steps']:
        step_counts = list(params['channel_actual_steps'].values())
        print(f"\nFinal dynamic step count statistics")
        print(f"Total channels: {len(step_counts)}")
        print(f"Average step count: {np.mean(step_counts):.1f}")
        print(f"Maximum pre-allocated steps: {max_steps}")

    return result