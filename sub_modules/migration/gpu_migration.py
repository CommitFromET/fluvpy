"""
gpu_migration.py
"""
import time
import numpy as np
import math
import os
from numba import cuda
import cupy as cp
from .vegetation_patches import create_vegetation_generator
from ..export.fluvpy_export import export_vegetation_distribution, export_vegetation_evolution_summary, export_vegetation_comparison

# Degrees to radians conversion constant
DEG2RAD = 3.141592654 / 180.0
_GPU_AVAILABLE = True

def check_gpu_availability():
    """
    Check if GPU is available for river channel migration calculations.

    Args:
        None

    Returns:
        bool: Whether GPU is available
    """
    return _GPU_AVAILABLE


def migrate_river_channel_2d_gpu(params, all_channels_data):
    """
    GPU-accelerated main river channel migration function implementing
    curvature-driven, vegetation-influenced, and integral effect migration mechanisms.

    Args:
        params: Parameter configuration dictionary
        all_channels_data: List of channel data

    Returns:
        list: List of migrated channel data
    """
    if not _GPU_AVAILABLE:
        print("GPU unavailable, cannot execute migration")
        return None

    start_time = time.time()

    # Global migration step management
    if 'global_migration_step_count' not in params:
        params['global_migration_step_count'] = 0

    params['global_migration_step_count'] += 1
    current_migration_step = params['global_migration_step_count']

    print(f"Migration step: {current_migration_step}")

    # Vegetation export related parameters
    export_vegetation_enabled = params.get('export_vegetation_enabled', False)
    vegetation_export_dir = params.get('vegetation_export_dir', 'vegetation_distributions')
    export_vegetation_evolution = params.get('export_vegetation_evolution', False)
    export_vegetation_summary = params.get('export_vegetation_summary', True)

    # Vegetation evolution history recording
    vegetation_maps_history = []
    vegetation_update_count = 0

    if export_vegetation_enabled:
        print(f"Vegetation export enabled, directory: {vegetation_export_dir}")

    # Vegetation influence mode determination
    vegetation_factor = params.get('vegetation_factor', 0)
    vegetation_enabled_flag = params.get('vegetation_enabled', False)
    vegetation_influence_strength = params.get('vegetation_influence_strength', 5.0)
    vegetation_value_range = params.get('vegetation_value_range', (0.1, 0.9))

    uniform_vegetation = vegetation_factor > 0 and not vegetation_enabled_flag
    heterogeneous_vegetation = vegetation_enabled_flag and vegetation_factor > 0
    any_vegetation = uniform_vegetation or heterogeneous_vegetation

    vegetation_generator = None

    if any_vegetation:
        if uniform_vegetation:
            print(f"Vegetation mode uniform, strength {vegetation_influence_strength:.1f}")

        elif heterogeneous_vegetation:
            print(f"Vegetation mode spatially heterogeneous, strength {vegetation_influence_strength:.1f}, range {vegetation_value_range}")

            # Vegetation generator initialization and update logic
            if 'vegetation_generator' in params:
                vegetation_generator = params['vegetation_generator']
                print(f"Using existing vegetation generator")
            else:
                print(f"Initializing vegetation system")
                try:

                    vegetation_generator = create_vegetation_generator(params)

                    params['vegetation_generator'] = vegetation_generator

                    # Export initial vegetation distribution
                    if export_vegetation_enabled:
                        try:

                            export_path = export_vegetation_distribution(
                                vegetation_generator=vegetation_generator,
                                params=params,
                                update_step=0,
                                output_dir=vegetation_export_dir,
                                prefix='vegetation'
                            )
                            if export_path:
                                print(f"Initial vegetation distribution exported: {os.path.basename(export_path)}")

                            if export_vegetation_evolution:
                                initial_map = vegetation_generator.get_vegetation_map()
                                if initial_map is not None:
                                    vegetation_maps_history.append((0, initial_map.copy()))
                                    print(f"Vegetation evolution history recorded: step 0")

                        except Exception as e:
                            print(f"Initial vegetation distribution export failed: {e}")

                    # Display statistics
                    veg_stats = vegetation_generator.get_vegetation_statistics()
                    if veg_stats:
                        print(f"Vegetation coefficient distribution: [{veg_stats['min']:.3f}, {veg_stats['max']:.3f}], mean: {veg_stats['mean']:.3f}")

                except Exception as e:
                    print(f"Vegetation system initialization failed: {e}")
                    heterogeneous_vegetation = False

            # Vegetation update logic
            if vegetation_generator is not None:
                vegetation_update_interval = params.get('vegetation_update_interval', 50)
                if current_migration_step % vegetation_update_interval == 0:
                    print(f"Vegetation distribution update: step {current_migration_step}")

                    vegetation_update_count += 1

                    # Export updated vegetation distribution
                    if export_vegetation_enabled:
                        try:

                            export_path = export_vegetation_distribution(
                                vegetation_generator=vegetation_generator,
                                params=params,
                                update_step=vegetation_update_count,
                                output_dir=vegetation_export_dir,
                                prefix='vegetation'
                            )
                            if export_path:
                                print(f"Updated vegetation distribution exported: {os.path.basename(export_path)}")

                            # Add to evolution history record
                            if export_vegetation_evolution:
                                updated_map = vegetation_generator.get_vegetation_map()
                                if updated_map is not None:
                                    vegetation_maps_history.append((vegetation_update_count, updated_map.copy()))
                                    print(f"Vegetation evolution history recorded: step {vegetation_update_count}")

                        except Exception as e:
                            print(f"Updated vegetation distribution export failed: {e}")
                else:
                    steps_until_next_update = vegetation_update_interval - (
                            current_migration_step % vegetation_update_interval)
                    print(f"Vegetation distribution unchanged, {steps_until_next_update} steps until next update")
    else:
        print(f"Vegetation influence disabled")

    # Get global fixed property ranges
    global_min_thickness = params.get('global_min_thickness')
    global_max_thickness = params.get('global_max_thickness')
    global_min_width = params.get('global_min_width')
    global_max_width = params.get('global_max_width')

    if (global_min_thickness is None or global_max_thickness is None or
            global_min_width is None or global_max_width is None):
        global_min_thickness = float('inf')
        global_max_thickness = float('-inf')
        global_min_width = float('inf')
        global_max_width = float('-inf')

        for channel_data in all_channels_data:
            thickness = np.array(channel_data['thickness'])
            width = np.array(channel_data['width'])

            global_min_thickness = min(global_min_thickness, np.min(thickness))
            global_max_thickness = max(global_max_thickness, np.max(thickness))
            global_min_width = min(global_min_width, np.min(width))
            global_max_width = max(global_max_width, np.max(width))

        params['global_min_thickness'] = global_min_thickness
        params['global_max_thickness'] = global_max_thickness
        params['global_min_width'] = global_min_width
        params['global_max_width'] = global_max_width

    # Get migration parameters
    migration_rate = params.get('migration_rate', 1.0)
    migration_factor = params.get('migration_factor', 1.0)
    allow_endpoint_migration = params.get('allow_endpoint_migration', True)

    # Physical parameters
    us0 = params.get('migration_us0', 1.0)
    curvature_factor = params.get('curvature_factor', 1.0)
    migration_time_step = params.get('migration_time_step', 1.0)
    original_vegetation_factor = params.get('vegetation_factor', 1.0)
    courant_factor = params.get('courant_factor', 0.5)
    migration_limiter = params.get('migration_limiter', 0.0)
    boundary_damping_zone = params.get('boundary_damping_zone', 0.1)

    # Integral effect parameters
    enable_integral_effects = params.get('enable_integral_effects', True)
    integral_length_factor = params.get('integral_length_factor', 3.0)
    integral_decay_factor = params.get('integral_decay_factor', 0.8)
    integral_weight = params.get('integral_weight', 0.7)
    local_weight = params.get('local_weight', 0.3)

    # Smoothing parameters
    smoothing_iterations = params.get('smoothing_iterations', 3)
    smoothing_window_size = params.get('smoothing_window_size', 8)
    boundary_factor_param = params.get('boundary_factor_param', 0.5)

    # Post-smoothing parameters
    post_smoothing_iterations = params.get('post_smoothing_iterations', smoothing_iterations)
    post_smoothing_window_size = params.get('post_smoothing_window_size', smoothing_window_size)
    post_boundary_factor_param = params.get('post_boundary_factor_param', boundary_factor_param)

    # Density adjustment parameters
    enable_density_adjustment = params.get('enable_density_adjustment', True)
    density_threshold_factor = params.get('density_threshold_factor', 2.0)
    max_total_points = params.get('max_total_points', 2000)

    # Other parameters
    lateral_factor_base = params.get('lateral_factor_base', 2.0)
    downstream_factor_base = params.get('downstream_factor_base', 0.15)
    endpoint_damping_factor = params.get('endpoint_damping_factor', 0.8)

    print(f"Physical parameters: us0={us0}, curvature_factor={curvature_factor}, vegetation_factor={original_vegetation_factor}")

    if enable_integral_effects:
        print(f"Integral effects: length_factor={integral_length_factor}, decay={integral_decay_factor}, weight={integral_weight}")

    migrated_channels_data = []

    for channel_idx, channel_data in enumerate(all_channels_data):
        channel_start_time = time.time()

        # Get channel data
        centerline_x = channel_data['centerline_x']
        centerline_y = channel_data['centerline_y']
        thickness = channel_data['thickness']
        width = channel_data['width']
        relpos = channel_data['relpos']

        n_points = len(centerline_x)
        average_width = np.mean(width)

        print(f"Channel {channel_idx + 1}: {n_points} points, average width: {average_width:.3f}")
        print(f"  Pre-smoothing...")

        # Transfer original data to GPU
        d_original_x = cp.array(centerline_x, dtype=cp.float64)
        d_original_y = cp.array(centerline_y, dtype=cp.float64)

        # Create smoothed data arrays
        d_smoothed_x = d_original_x.copy()
        d_smoothed_y = d_original_y.copy()

        # GPU grid setup
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

        # Execute pre-smoothing
        for smooth_iter in range(smoothing_iterations):
            d_temp_smooth_x = cp.zeros_like(d_smoothed_x)
            d_temp_smooth_y = cp.zeros_like(d_smoothed_y)

            smooth_centerline_kernel_guss[blocks_per_grid, threads_per_block](
                d_smoothed_x, d_smoothed_y, d_temp_smooth_x, d_temp_smooth_y,
                smoothing_window_size, d_original_x, d_original_y,
                allow_endpoint_migration, 2, boundary_factor_param
            )

            d_smoothed_x = d_temp_smooth_x.copy()
            d_smoothed_y = d_temp_smooth_y.copy()

        # Curvature calculation
        print(f"  Calculating curvature...")
        d_curvature = cp.zeros(n_points, dtype=cp.float64)
        d_radius = cp.zeros(n_points, dtype=cp.float64)

        calculate_curvature[blocks_per_grid, threads_per_block](
            d_smoothed_x, d_smoothed_y, d_curvature, d_radius
        )

        # Use calculated curvature
        d_curvature_final = d_curvature
        curvature_cpu = cp.asnumpy(d_curvature_final)

        # Vegetation influence calculation
        print(f"  Calculating vegetation influence...")
        if vegetation_factor == 0:
            # No vegetation influence
            vegetation_factors = np.zeros(n_points, dtype=np.float64)
            print(f"    Vegetation influence disabled")
        else:
            # Set influence based on mode
            vegetation_factors = np.ones(n_points, dtype=np.float64)

            if uniform_vegetation:
                # Global uniform influence mode
                vegetation_factors.fill(vegetation_influence_strength)
                print(f"    Vegetation influence uniform mode, strength={vegetation_influence_strength:.1f}")


            elif heterogeneous_vegetation and vegetation_generator is not None:

                # Spatial heterogeneous influence mode
                vegetation_influences = []

                for j in range(n_points):
                    veg_influence = vegetation_generator.get_vegetation_at_point(centerline_x[j], centerline_y[j])
                    vegetation_factors[j] = veg_influence
                    vegetation_influences.append(veg_influence)

                vegetation_influences = np.array(vegetation_influences)

                print(f"    Vegetation influence spatial heterogeneous mode, range=[{vegetation_influences.min():.2f}, {vegetation_influences.max():.2f}]")
                print(f"    Average strength: {vegetation_influences.mean():.3f} (target: {vegetation_influence_strength:.3f})")

            else:
                # Fallback: when configuration is incorrect, disable influence
                vegetation_factors = np.zeros(n_points, dtype=np.float64)
                print(f"    Vegetation influence configuration error, disabled")

        # Display vegetation factor statistics
        print(f"    Vegetation factors: range=[{vegetation_factors.min():.3f}, {vegetation_factors.max():.3f}], mean={vegetation_factors.mean():.3f}")

        # Transfer other data to GPU
        d_thickness = cp.array(thickness, dtype=cp.float64)
        d_width = cp.array(width, dtype=cp.float64)
        d_relpos = cp.array(relpos, dtype=cp.float64)
        d_vegetation_factors = cp.array(vegetation_factors, dtype=cp.float64)

        # Bank velocity calculation
        print(f"  Calculating bank velocity...")
        d_bank_velocity = cp.zeros(n_points, dtype=cp.float64)

        calculate_bank_velocity[blocks_per_grid, threads_per_block](
            d_smoothed_x, d_smoothed_y, d_width, d_thickness,
            d_curvature_final,
            us0, curvature_factor,
            original_vegetation_factor, d_bank_velocity
        )

        # Transfer data back to CPU for integral effect calculation
        smoothed_x_cpu = cp.asnumpy(d_smoothed_x)
        smoothed_y_cpu = cp.asnumpy(d_smoothed_y)
        bank_velocity_cpu = cp.asnumpy(d_bank_velocity)
        width_cpu = np.array(width)

        # Upstream integral effect calculation
        if enable_integral_effects:
            print(f"  Calculating integral effects...")

            integral_start_time = time.time()

            (integrated_curvature, integrated_direction_x,
             integrated_direction_y, integrated_velocity) = calculate_upstream_integral_effects_cpu(
                smoothed_x_cpu, smoothed_y_cpu, curvature_cpu,
                bank_velocity_cpu, width_cpu,
                integral_length_factor, integral_decay_factor
            )

            integral_end_time = time.time()
            print(f"    Integral effects complete, time: {integral_end_time - integral_start_time:.3f}s")

            # Transfer integral effects to GPU
            d_integrated_curvature = cp.array(integrated_curvature, dtype=cp.float64)
            d_integrated_direction_x = cp.array(integrated_direction_x, dtype=cp.float64)
            d_integrated_direction_y = cp.array(integrated_direction_y, dtype=cp.float64)
            d_integrated_velocity = cp.array(integrated_velocity, dtype=cp.float64)

        else:
            print(f"  Skipping integral effects")
            # Use local values as "integral" values
            d_integrated_curvature = d_curvature_final.copy()
            d_integrated_direction_x = cp.zeros(n_points, dtype=cp.float64)
            d_integrated_direction_y = cp.zeros(n_points, dtype=cp.float64)
            d_integrated_velocity = d_bank_velocity.copy()

        # GPU parallel migration calculation
        print(f"  GPU migration calculation...")
        migrate_start_time = time.time()

        d_migrated_x = cp.zeros(n_points, dtype=cp.float64)
        d_migrated_y = cp.zeros(n_points, dtype=cp.float64)

        migrate_river_points_with_integral_effects[blocks_per_grid, threads_per_block](
            d_smoothed_x, d_smoothed_y, d_thickness, d_width, d_relpos,
            d_curvature_final, d_bank_velocity,
            d_vegetation_factors,
            # Integral effect inputs
            d_integrated_curvature, d_integrated_direction_x,
            d_integrated_direction_y, d_integrated_velocity,
            # Other parameters
            migration_rate, migration_factor, migration_limiter,
            allow_endpoint_migration, lateral_factor_base, downstream_factor_base,
            endpoint_damping_factor, courant_factor, boundary_damping_zone,
            migration_time_step, d_migrated_x, d_migrated_y,
            d_original_x, d_original_y, average_width,
            integral_weight, local_weight
        )

        migrate_end_time = time.time()
        print(f"    GPU migration complete, time: {migrate_end_time - migrate_start_time:.3f}s")

        # Transfer results back to CPU for density adjustment
        migrated_centerline_x = cp.asnumpy(d_migrated_x)
        migrated_centerline_y = cp.asnumpy(d_migrated_y)

        current_thickness = thickness.copy()
        current_width = width.copy()
        current_relpos = relpos.copy()

        # Point density adjustment
        adjustment_start_time = time.time()

        if enable_density_adjustment:
            print(f"  Density adjustment...")
            (migrated_centerline_x, migrated_centerline_y,
             current_thickness, current_width, current_relpos) = adjust_point_density_gpu(
                migrated_centerline_x, migrated_centerline_y,
                current_thickness, current_width, current_relpos,
                density_threshold_factor, max_total_points,
                global_min_thickness, global_min_width,
                global_max_thickness, global_max_width
            )

        adjustment_end_time = time.time()
        print(f"    Density adjustment complete, time: {adjustment_end_time - adjustment_start_time:.3f}s")

        # Post-smoothing processing
        post_smoothing_start_time = time.time()
        # Transfer migrated data back to GPU for post-smoothing
        updated_n_points = len(migrated_centerline_x)
        blocks_per_grid_updated = (updated_n_points + threads_per_block - 1) // threads_per_block

        d_migrated_x_for_smooth = cp.array(migrated_centerline_x, dtype=cp.float64)
        d_migrated_y_for_smooth = cp.array(migrated_centerline_y, dtype=cp.float64)

        # Save original migrated positions (for endpoint constraints)
        d_migrated_original_x = d_migrated_x_for_smooth.copy()
        d_migrated_original_y = d_migrated_y_for_smooth.copy()

        # Execute post-smoothing
        for smooth_iter in range(post_smoothing_iterations):
            d_temp_smooth_x = cp.zeros_like(d_migrated_x_for_smooth)
            d_temp_smooth_y = cp.zeros_like(d_migrated_y_for_smooth)

            smooth_centerline_kernel_guss[blocks_per_grid_updated, threads_per_block](
                d_migrated_x_for_smooth, d_migrated_y_for_smooth,
                d_temp_smooth_x, d_temp_smooth_y,
                post_smoothing_window_size,
                d_migrated_original_x, d_migrated_original_y,
                allow_endpoint_migration, 2, post_boundary_factor_param
            )

            d_migrated_x_for_smooth = d_temp_smooth_x.copy()
            d_migrated_y_for_smooth = d_temp_smooth_y.copy()

        # Transfer smoothed data back to CPU
        final_smoothed_x = cp.asnumpy(d_migrated_x_for_smooth)
        final_smoothed_y = cp.asnumpy(d_migrated_y_for_smooth)
        post_smoothing_end_time = time.time()
        print(f"  Post-smoothing complete, time: {post_smoothing_end_time - post_smoothing_start_time:.3f}s")

        # Calculate migration distance statistics
        original_n_points = len(centerline_x)
        migration_distances = []
        for j in range(min(original_n_points, len(final_smoothed_x))):
            dx = final_smoothed_x[j] - centerline_x[j]
            dy = final_smoothed_y[j] - centerline_y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            migration_distances.append(dist)

        migration_distances = np.array(migration_distances)
        print(f"  Migration statistics: average distance={migration_distances.mean():.3f}, maximum distance={migration_distances.max():.3f}")

        # Geometric parameter adjustment
        migrated_thickness = current_thickness.copy()
        migrated_width = current_width.copy()

        geometry_adjustment_enabled = params.get('geometry_adjustment_enabled', True)
        if geometry_adjustment_enabled and len(migration_distances) > 0:
            adjustment_threshold = params.get('geometry_adjustment_threshold', 0.05)
            adjustment_factor = params.get('geometry_adjustment_factor', 0.02)
            min_width_factor = params.get('min_width_factor', 0.8)
            min_thickness_factor = params.get('min_thickness_factor', 0.9)

            print(f"  Geometry adjustment...")

            adjustment_count = 0
            for j in range(min(len(migration_distances), len(migrated_width))):
                if j < len(migration_distances):
                    migrate_dist = migration_distances[j]
                    if j < len(width) and migrate_dist > width[j] * adjustment_threshold:
                        factor = 1.0 - adjustment_factor * (migrate_dist / width[j])

                        adjusted_width = migrated_width[j] * max(min_width_factor, factor)
                        adjusted_thickness = migrated_thickness[j] * max(min_thickness_factor, factor)

                        migrated_width[j] = np.clip(adjusted_width, global_min_width, global_max_width)
                        migrated_thickness[j] = np.clip(adjusted_thickness, global_min_thickness, global_max_thickness)
                        adjustment_count += 1

            print(f"    Geometry adjustment: {adjustment_count} points")

        channel_end_time = time.time()
        print(f"  Channel {channel_idx + 1} complete: {len(final_smoothed_x)} points, {channel_end_time - channel_start_time:.3f}s")

        # Save results
        migrated_channels_data.append({
            'centerline_x': final_smoothed_x,
            'centerline_y': final_smoothed_y,
            'thickness': migrated_thickness,
            'width': migrated_width,
            'relpos': current_relpos.copy()
        })

    # Vegetation evolution summary export
    if heterogeneous_vegetation and vegetation_generator is not None:
        final_vegetation_map = vegetation_generator.get_vegetation_map()
        params['final_vegetation_map'] = final_vegetation_map
        print(f"Vegetation data saved")

        if export_vegetation_enabled:
            try:


                if export_vegetation_summary:
                    export_vegetation_evolution_summary(
                        vegetation_generator=vegetation_generator,
                        params=params,
                        output_dir=vegetation_export_dir
                    )
                    print(f"Vegetation evolution summary exported")

                if export_vegetation_evolution and len(vegetation_maps_history) > 1:
                    export_vegetation_comparison(
                        vegetation_maps_history=vegetation_maps_history,
                        params=params,
                        output_dir=vegetation_export_dir
                    )
                    print(f"Vegetation evolution comparison exported")

            except Exception as e:
                print(f"Vegetation evolution data export failed: {e}")

    end_time = time.time()
    print(f"River channel migration complete, total time: {end_time - start_time:.3f}s")

    return migrated_channels_data


@cuda.jit
def calculate_bank_velocity(centerline_x, centerline_y, channel_width, channel_thickness,
                                       curvature_array,
                                       us0, curvature_factor,
                                       vegetation_factor,
                                       out_bank_velocity):
    """
    Bank migration velocity calculation based on curvature-driven dynamics and vegetation influence.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        channel_width: Channel width array
        channel_thickness: Channel thickness array
        curvature_array: Curvature array
        us0: Base migration velocity parameter
        curvature_factor: Curvature influence factor
        vegetation_factor: Vegetation influence factor
        out_bank_velocity: Output bank velocity array

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_points = centerline_x.shape[0]

    if i < n_points:
        # Use input curvature
        curvature = curvature_array[i]

        # Base physical parameters
        depth = channel_thickness[i]
        width = channel_width[i]

        # Base curvature response
        curvature_intensity = abs(curvature)
        base_lateral_velocity = curvature_factor * curvature_intensity * us0

        # Vegetation influence
        relative_position = float(i) / float(max(n_points - 1, 1))

        # Vegetation spatial pattern
        vegetation_pattern = 0.5 * (
                0.6 * math.sin(relative_position * 6.28 * 2.0) +
                0.3 * math.sin(relative_position * 6.28 * 7.0) +
                0.1 * math.sin(relative_position * 6.28 * 23.0) + 1.0
        )

        # Vegetation resistance always ≥1.0
        vegetation_resistance = 1.0 + vegetation_factor * vegetation_pattern

        # Vegetation asymmetric effects at bends
        if abs(curvature) > 1e-6:
            curvature_asymmetry = math.tanh(curvature * width * 0.01)
            asymmetry_factor = 1.0 + vegetation_factor * 0.2 * abs(curvature_asymmetry)
            vegetation_resistance *= asymmetry_factor

        # Ensure vegetation resistance lower bound
        vegetation_resistance = max(vegetation_resistance, 1.0)

        # Calculate final migration velocity
        if abs(curvature) > 1e-10:
            # Apply vegetation resistance
            final_velocity = base_lateral_velocity / vegetation_resistance
        else:
            final_velocity = 0.0

        # Basic physical constraints
        max_allowed_velocity = abs(curvature) * width * 0.1 + us0 * 0.1
        bank_velocity = min(abs(final_velocity), max_allowed_velocity)

        out_bank_velocity[i] = bank_velocity


def calculate_upstream_integral_effects_cpu(centerline_x, centerline_y, curvature_array,
                                            bank_velocity, width_array,
                                            integral_length_factor=3.0, decay_factor=0.8):
    """
    CPU algorithm to calculate upstream integral effects for each point,
    including integrals of curvature, direction and velocity.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        curvature_array: Curvature array
        bank_velocity: Bank velocity array
        width_array: Width array
        integral_length_factor: Integral length factor
        decay_factor: Decay factor

    Returns:
        tuple: Integral effect results (integrated curvature, integrated direction X,
               integrated direction Y, integrated velocity)
    """
    import numpy as np

    n_points = len(centerline_x)
    mean_width = np.mean(width_array)

    # Calculate integral length (in number of points)
    segment_lengths = []
    for i in range(1, n_points):
        dx = centerline_x[i] - centerline_x[i - 1]
        dy = centerline_y[i] - centerline_y[i - 1]
        segment_lengths.append(np.sqrt(dx * dx + dy * dy))

    avg_segment_length = np.mean(segment_lengths) if segment_lengths else mean_width / 10
    integral_length = mean_width * integral_length_factor
    integral_points = max(3, int(integral_length / avg_segment_length))

    print(f"    Integral parameters: integral length={integral_length:.1f}, integral points={integral_points}")

    # Initialize integral effect arrays
    integrated_curvature = np.zeros(n_points)
    integrated_direction_x = np.zeros(n_points)
    integrated_direction_y = np.zeros(n_points)
    integrated_velocity = np.zeros(n_points)

    for i in range(n_points):
        # Determine integration start point (upstream)
        start_idx = max(0, i - integral_points)

        # Weight normalization factor
        total_weight = 0.0

        # Temporary accumulation variables
        curvature_sum = 0.0
        direction_x_sum = 0.0
        direction_y_sum = 0.0
        velocity_sum = 0.0

        for j in range(start_idx, i + 1):
            # Calculate distance weight (closer points have higher weight)
            distance_factor = float(i - j)
            weight = decay_factor ** distance_factor
            total_weight += weight

            # Accumulate curvature effects
            curvature_sum += curvature_array[j] * weight

            # Accumulate velocity effects
            velocity_sum += bank_velocity[j] * weight

            # Calculate local direction vector
            if j == 0:
                tangent_x = centerline_x[1] - centerline_x[0] if n_points > 1 else 1.0
                tangent_y = centerline_y[1] - centerline_y[0] if n_points > 1 else 0.0
            elif j == n_points - 1:
                tangent_x = centerline_x[j] - centerline_x[j - 1]
                tangent_y = centerline_y[j] - centerline_y[j - 1]
            else:
                tangent_x = centerline_x[j + 1] - centerline_x[j - 1]
                tangent_y = centerline_y[j + 1] - centerline_y[j - 1]

            # Normalize tangent
            tangent_length = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
            if tangent_length > 1e-10:
                tangent_x /= tangent_length
                tangent_y /= tangent_length

            # Determine normal vector direction based on curvature
            local_curvature = curvature_array[j]
            if abs(local_curvature) > 1e-10:
                if local_curvature > 0:
                    # Positive curvature: right normal (clockwise 90° rotation)
                    normal_x = tangent_y
                    normal_y = -tangent_x
                else:
                    # Negative curvature: left normal (counterclockwise 90° rotation)
                    normal_x = -tangent_y
                    normal_y = tangent_x

                # Accumulate direction effects, weighted by curvature intensity
                direction_strength = abs(local_curvature) * weight
                direction_x_sum += normal_x * direction_strength
                direction_y_sum += normal_y * direction_strength

        # Normalize integral results
        if total_weight > 1e-10:
            integrated_curvature[i] = curvature_sum / total_weight
            integrated_velocity[i] = velocity_sum / total_weight

            # Direction vector normalization
            direction_magnitude = np.sqrt(direction_x_sum ** 2 + direction_y_sum ** 2)
            if direction_magnitude > 1e-10:
                integrated_direction_x[i] = direction_x_sum / direction_magnitude
                integrated_direction_y[i] = direction_y_sum / direction_magnitude
            else:
                # Use current point's local normal as default direction
                if i == 0:
                    tx = centerline_x[1] - centerline_x[0] if n_points > 1 else 1.0
                    ty = centerline_y[1] - centerline_y[0] if n_points > 1 else 0.0
                elif i == n_points - 1:
                    tx = centerline_x[i] - centerline_x[i - 1]
                    ty = centerline_y[i] - centerline_y[i - 1]
                else:
                    tx = centerline_x[i + 1] - centerline_x[i - 1]
                    ty = centerline_y[i + 1] - centerline_y[i - 1]

                t_len = np.sqrt(tx ** 2 + ty ** 2)
                if t_len > 1e-10:
                    tx, ty = tx / t_len, ty / t_len

                # Default to right normal
                integrated_direction_x[i] = ty
                integrated_direction_y[i] = -tx
        else:
            # No upstream data, use local values
            integrated_curvature[i] = curvature_array[i]
            integrated_velocity[i] = bank_velocity[i]
            integrated_direction_x[i] = 0.0
            integrated_direction_y[i] = 0.0

    # Check validity of direction vectors
    valid_directions = 0
    for i in range(n_points):
        if abs(integrated_direction_x[i]) > 1e-10 or abs(integrated_direction_y[i]) > 1e-10:
            valid_directions += 1

    return integrated_curvature, integrated_direction_x, integrated_direction_y, integrated_velocity

@cuda.jit
def migrate_river_points_with_integral_effects(centerline_x, centerline_y, thickness, width, relpos,
                                               curvature_array, bank_velocity,
                                               vegetation_factors,
                                               # Integral effect inputs
                                               integrated_curvature, integrated_direction_x,
                                               integrated_direction_y, integrated_velocity,
                                               # Other parameters
                                               migration_rate, migration_factor, migration_limiter,
                                               allow_endpoint_migration, lateral_factor_base,
                                               downstream_factor_base, endpoint_damping_factor,
                                               courant_factor, boundary_damping_zone, migration_time_step,
                                               out_centerline_x, out_centerline_y,
                                               original_x, original_y, average_width,
                                               integral_weight, local_weight):
    """
    River point migration CUDA kernel using upstream integral effects,
    combining integral direction and local direction.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        thickness, width, relpos: Channel geometry parameter arrays
        curvature_array: Curvature array
        bank_velocity: Bank velocity array
        vegetation_factors: Vegetation influence factor array
        integrated_curvature, integrated_direction_x, integrated_direction_y, integrated_velocity: Integral effect arrays
        migration_rate, migration_factor, migration_limiter: Migration control parameters
        allow_endpoint_migration: Whether to allow endpoint migration
        lateral_factor_base, downstream_factor_base, endpoint_damping_factor: Other control factors
        courant_factor, boundary_damping_zone, migration_time_step: Numerical stability parameters
        out_centerline_x, out_centerline_y: Output coordinate arrays
        original_x, original_y: Original coordinate arrays
        average_width: Average width
        integral_weight, local_weight: Integral and local effect weights

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_points = centerline_x.shape[0]

    if i < n_points:
        # Default output as input coordinates
        out_centerline_x[i] = centerline_x[i]
        out_centerline_y[i] = centerline_y[i]

        # Boundary damping zone calculation
        boundary_zone_size = int(n_points * boundary_damping_zone)
        in_boundary_zone = (i < boundary_zone_size) or (i >= n_points - boundary_zone_size)

        boundary_damping_factor_local = 1.0
        if in_boundary_zone:
            if i < boundary_zone_size:
                boundary_damping_factor_local = float(i) / float(boundary_zone_size)
            else:
                dist_from_end = n_points - 1 - i
                boundary_damping_factor_local = float(dist_from_end) / float(boundary_zone_size)
            boundary_damping_factor_local = 0.1 + 0.9 * (boundary_damping_factor_local ** 2)

        # Endpoint processing determination
        is_endpoint = (i == 0 or i == n_points - 1)
        is_startpoint = (i == 0)
        is_endpoint_not_start = (i == n_points - 1)
        should_migrate = True

        if not allow_endpoint_migration and is_endpoint:
            should_migrate = False

        if should_migrate:
            # Calculate migration direction vector
            # Get integral direction vector
            integral_dir_x = integrated_direction_x[i]
            integral_dir_y = integrated_direction_y[i]

            # Calculate local direction vector
            if i == 0:
                dx = centerline_x[1] - centerline_x[0] if n_points > 1 else 1.0
                dy = centerline_y[1] - centerline_y[0] if n_points > 1 else 0.0
            elif i == n_points - 1:
                dx = centerline_x[n_points - 1] - centerline_x[n_points - 2]
                dy = centerline_y[n_points - 1] - centerline_y[n_points - 2]
            else:
                dx = centerline_x[i + 1] - centerline_x[i - 1]
                dy = centerline_y[i + 1] - centerline_y[i - 1]

            # Normalize tangent vector
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-10:
                dx, dy = 1.0, 0.0
                length = 1.0
            else:
                dx, dy = dx / length, dy / length

            # Calculate local normal vector
            local_curvature = curvature_array[i]
            local_dir_x = 0.0
            local_dir_y = 0.0

            if abs(local_curvature) > 1e-10:
                if local_curvature > 0:
                    # Positive curvature: right normal
                    local_dir_x = dy
                    local_dir_y = -dx
                else:
                    # Negative curvature: left normal
                    local_dir_x = -dy
                    local_dir_y = dx

            # Combine integral direction and local direction
            if abs(integral_dir_x) > 1e-10 or abs(integral_dir_y) > 1e-10:
                # Valid integral direction exists, perform weighted combination
                direction_x = integral_weight * integral_dir_x + local_weight * local_dir_x
                direction_y = integral_weight * integral_dir_y + local_weight * local_dir_y
            else:
                # No integral direction, use local direction
                direction_x = local_dir_x
                direction_y = local_dir_y

            # Normalize combined direction
            dir_length = math.sqrt(direction_x ** 2 + direction_y ** 2)
            if dir_length > 1e-10:
                direction_x /= dir_length
                direction_y /= dir_length

            # Add downstream component (optional)
            if abs(local_curvature) > 1e-10 and downstream_factor_base > 0:
                curvature_magnitude = abs(local_curvature)
                downstream_factor = downstream_factor_base * curvature_magnitude
                direction_x += dx * downstream_factor
                direction_y += dy * downstream_factor

                # Re-normalize
                dir_length = math.sqrt(direction_x ** 2 + direction_y ** 2)
                if dir_length > 1e-10:
                    direction_x /= dir_length
                    direction_y /= dir_length

            # Calculate migration velocity and distance
            # Use integral velocity effects
            effective_velocity = integral_weight * integrated_velocity[i] + local_weight * bank_velocity[i]

            # Calculate base migration distance
            base_migration_dist = effective_velocity * migration_rate * migration_factor

            # Courant number limit (numerical stability)
            characteristic_velocity = max(abs(effective_velocity), 1e-10)
            grid_scale = length
            courant_limited_distance = courant_factor * grid_scale

            # Physical limit based on time step
            time_limited_distance = characteristic_velocity * migration_time_step
            max_migration_distance = min(time_limited_distance, courant_limited_distance)

            # Apply Courant limit
            if abs(base_migration_dist) > max_migration_distance:
                migration_dist = math.copysign(max_migration_distance, base_migration_dist)
            else:
                migration_dist = base_migration_dist

            # Global migration distance limit
            if migration_limiter > 0:
                max_distance = average_width * migration_limiter
                if abs(migration_dist) > max_distance:
                    migration_dist = math.copysign(max_distance, migration_dist)

            # Apply vegetation influence
            vegetation_influence = vegetation_factors[i]

            if vegetation_influence > 1e-10:  # Has vegetation influence
                # Use resistance coefficient method
                vegetation_resistance = vegetation_influence
                migration_dist = migration_dist - vegetation_resistance * migration_dist
                print(vegetation_resistance)

            # Apply boundary damping
            migration_dist *= boundary_damping_factor_local

            # Calculate final position (special handling for endpoints)
            if is_startpoint and allow_endpoint_migration:
                # Start point: only allow migration along main direction, constrain Y coordinate
                x_direction = 1.0 if direction_x > 0 else -1.0
                out_centerline_x[i] = centerline_x[i] + x_direction * migration_dist * endpoint_damping_factor
                out_centerline_y[i] = original_y[i]  # Maintain original Y coordinate
            elif is_endpoint_not_start and allow_endpoint_migration:
                # End point: allow full migration but with damping
                out_centerline_x[i] = centerline_x[i] + direction_x * migration_dist * endpoint_damping_factor
                out_centerline_y[i] = centerline_y[i] + direction_y * migration_dist * endpoint_damping_factor
            elif not is_endpoint:
                # Middle points: full migration
                out_centerline_x[i] = centerline_x[i] + direction_x * migration_dist
                out_centerline_y[i] = centerline_y[i] + direction_y * migration_dist

        # Endpoint fixing
        if not allow_endpoint_migration and is_endpoint:
            out_centerline_x[i] = original_x[i]
            out_centerline_y[i] = original_y[i]


def adjust_point_density_gpu(centerline_x, centerline_y, thickness, width, relpos,
                             threshold_factor=2.0, max_total_points=2000,
                             global_min_thickness=None, global_min_width=None,
                             global_max_thickness=None, global_max_width=None):
    """
    GPU-accelerated point density adjustment using global fixed property protection mechanism.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        thickness, width, relpos: Channel geometry parameter arrays
        threshold_factor: Density threshold factor
        max_total_points: Maximum total points
        global_min_thickness, global_min_width: Global minimum values
        global_max_thickness, global_max_width: Global maximum values

    Returns:
        tuple: Adjusted coordinate and parameter arrays
    """
    if not _GPU_AVAILABLE:
        print("    GPU unavailable, skipping density adjustment")
        return centerline_x, centerline_y, thickness, width, relpos

    n_points = len(centerline_x)

    # Check if maximum point limit is already reached
    if n_points >= max_total_points:
        print(f"    Maximum point limit reached ({max_total_points})")
        return centerline_x, centerline_y, thickness, width, relpos

    if (global_min_thickness is None or global_max_thickness is None or
        global_min_width is None or global_max_width is None):
        raise ValueError(
            "Global fixed property range parameters missing! All global range parameters must be provided:\n"
            f"global_min_thickness: {global_min_thickness}\n"
            f"global_max_thickness: {global_max_thickness}\n"
            f"global_min_width: {global_min_width}\n"
            f"global_max_width: {global_max_width}"
        )
    try:
        # Transfer data to GPU
        d_centerline_x = cp.array(centerline_x, dtype=cp.float64)
        d_centerline_y = cp.array(centerline_y, dtype=cp.float64)

        # Calculate segment lengths
        n_segments = n_points - 1
        d_distances = cp.zeros(n_segments, dtype=cp.float64)

        threads_per_block = 256
        blocks_per_grid = (n_segments + threads_per_block - 1) // threads_per_block

        # GPU calculate distances between adjacent points
        calculate_segment_distances_kernel[blocks_per_grid, threads_per_block](
            d_centerline_x, d_centerline_y, d_distances
        )

        # Calculate average distance
        distances_cpu = cp.asnumpy(d_distances)
        average_distance = np.mean(distances_cpu)

        print(f"    Current average point spacing: {average_distance:.3f}")

        # Find segments with excessive spacing
        max_gaps = min(n_segments, 100)
        d_gap_indices = cp.full(max_gaps, -1, dtype=cp.int32)
        d_gap_distances = cp.zeros(max_gaps, dtype=cp.float64)
        d_gap_count = cp.array([0], dtype=cp.int32)

        find_large_gaps_kernel[blocks_per_grid, threads_per_block](
            d_distances, average_distance, threshold_factor,
            d_gap_indices, d_gap_distances, d_gap_count
        )

        # Transfer results back to CPU
        gap_count = int(cp.asnumpy(d_gap_count)[0])

        if gap_count > 0:
            gap_indices = cp.asnumpy(d_gap_indices)[:gap_count]
            gap_distances = cp.asnumpy(d_gap_distances)[:gap_count]

            # Execute interpolation and insertion operations on CPU
            adjusted_x, adjusted_y, adjusted_thickness, adjusted_width, adjusted_relpos = \
                interpolate_and_insert_points_with_protection(
                    centerline_x, centerline_y, thickness, width, relpos,
                    gap_indices, gap_distances, gap_count, average_distance,
                    global_min_thickness, global_min_width,
                    global_max_thickness, global_max_width
                )

            # Check maximum point limit
            if len(adjusted_x) > max_total_points:
                print(f"    Adjusted point count ({len(adjusted_x)}) exceeds limit, truncating to {max_total_points}")
                adjusted_x = adjusted_x[:max_total_points]
                adjusted_y = adjusted_y[:max_total_points]
                adjusted_thickness = adjusted_thickness[:max_total_points]
                adjusted_width = adjusted_width[:max_total_points]
                adjusted_relpos = adjusted_relpos[:max_total_points]

            return adjusted_x, adjusted_y, adjusted_thickness, adjusted_width, adjusted_relpos
        else:
            print(f"    Point density is good")
            return centerline_x, centerline_y, thickness, width, relpos

    except Exception as e:
        print(f"    GPU density adjustment failed: {e}")
        return centerline_x, centerline_y, thickness, width, relpos

@cuda.jit
def find_large_gaps_kernel(distances, average_distance, threshold_factor,
                          out_gap_indices, out_gap_distances, out_gap_count):
    """
    GPU kernel to find adjacent point pairs with distances greater than threshold times average distance.

    Args:
        distances: Segment distance array
        average_distance: Average distance
        threshold_factor: Threshold factor
        out_gap_indices: Output gap index array
        out_gap_distances: Output gap distance array
        out_gap_count: Output gap count

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_segments = distances.shape[0]

    if i < n_segments:
        if distances[i] > threshold_factor * average_distance:
            # Use atomic operation to safely add to gap list
            idx = cuda.atomic.add(out_gap_count, 0, 1)
            if idx < out_gap_indices.shape[0]:
                out_gap_indices[idx] = i
                out_gap_distances[idx] = distances[i]

@cuda.jit
def calculate_segment_distances_kernel(centerline_x, centerline_y, out_distances):
    """
    GPU kernel to calculate distances between adjacent points.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        out_distances: Output distance array

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_points = centerline_x.shape[0]

    if i < n_points - 1:
        dx = centerline_x[i + 1] - centerline_x[i]
        dy = centerline_y[i + 1] - centerline_y[i]
        out_distances[i] = math.sqrt(dx * dx + dy * dy)


def interpolate_and_insert_points_with_protection(centerline_x, centerline_y, thickness, width, relpos,
                                                  gap_indices, gap_distances, gap_count, average_distance,
                                                  min_thickness_limit, min_width_limit,
                                                  max_thickness_limit, max_width_limit):
    """
    Interpolation function with global fixed property protection mechanism
    to prevent property values from exceeding preset ranges.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        thickness, width, relpos: Channel geometry parameter arrays
        gap_indices: Gap index array
        gap_distances: Gap distance array
        gap_count: Gap count
        average_distance: Average distance
        min_thickness_limit, min_width_limit: Property minimum value limits
        max_thickness_limit, max_width_limit: Property maximum value limits

    Returns:
        tuple: Interpolated coordinate and parameter arrays
    """
    if gap_count == 0:
        return centerline_x, centerline_y, thickness, width, relpos

    # Validate global limit parameters
    if (min_thickness_limit is None or max_thickness_limit is None or
        min_width_limit is None or max_width_limit is None):
        raise ValueError(
            "Global property limit parameters missing! All limit parameters are required:\n"
            f"min_thickness_limit: {min_thickness_limit}\n"
            f"max_thickness_limit: {max_thickness_limit}\n"
            f"min_width_limit: {min_width_limit}\n"
            f"max_width_limit: {max_width_limit}"
        )

    # Collect information for all points to be inserted
    insertions = []

    for i in range(gap_count):
        segment_idx = gap_indices[i]
        segment_distance = gap_distances[i]

        # Calculate number of points to insert
        num_inserts = int(segment_distance / average_distance) - 1

        if num_inserts > 0:
            # Limit maximum insertions per segment to prevent over-subdivision
            max_inserts_per_segment = 10
            num_inserts = min(num_inserts, max_inserts_per_segment)

            insertions.append({
                'after_index': segment_idx,
                'num_points': num_inserts,
                'segment_distance': segment_distance
            })

    if not insertions:
        return centerline_x, centerline_y, thickness, width, relpos

    # Sort by index in descending order, insert from back to front to avoid index offset issues
    insertions.sort(key=lambda x: x['after_index'], reverse=True)

    # Execute insertion operations
    current_x = centerline_x.copy()
    current_y = centerline_y.copy()
    current_thickness = thickness.copy()
    current_width = width.copy()
    current_relpos = relpos.copy()

    total_inserted = 0
    total_protected_adjustments = 0

    for insertion in insertions:
        after_idx = insertion['after_index']
        num_points = insertion['num_points']

        # Get two endpoints for insertion position
        x1, y1 = current_x[after_idx], current_y[after_idx]
        x2, y2 = current_x[after_idx + 1], current_y[after_idx + 1]

        thickness1 = current_thickness[after_idx]
        thickness2 = current_thickness[after_idx + 1]
        width1 = current_width[after_idx]
        width2 = current_width[after_idx + 1]
        relpos1 = current_relpos[after_idx]
        relpos2 = current_relpos[after_idx + 1]

        # Generate interpolated points
        new_x_points = []
        new_y_points = []
        new_thickness_points = []
        new_width_points = []
        new_relpos_points = []

        for j in range(1, num_points + 1):
            # Linear interpolation factor
            factor = j / (num_points + 1)

            # Coordinate interpolation (maintain linearity)
            new_x = x1 + factor * (x2 - x1)
            new_y = y1 + factor * (y2 - y1)

            # Property protection mechanism multi-layer protection strategy
            # Basic linear interpolation
            base_thickness = thickness1 + factor * (thickness2 - thickness1)
            base_width = width1 + factor * (width2 - width1)

            # First layer protection: prevent interpolation results from being too small
            min_endpoint_thickness = min(thickness1, thickness2)
            min_endpoint_width = min(width1, width2)

            conservative_thickness = base_thickness
            conservative_width = base_width

            # If interpolation result is much smaller than minimum endpoint value, use conservative strategy
            if base_thickness < min_endpoint_thickness * 0.9:
                conservative_thickness = min_endpoint_thickness * 0.95

            if base_width < min_endpoint_width * 0.9:
                conservative_width = min_endpoint_width * 0.95

            # Second layer protection: apply global minimum value constraints
            protected_thickness = max(conservative_thickness, min_thickness_limit)
            protected_width = max(conservative_width, min_width_limit)

            # Third layer protection: apply global maximum value constraints
            clamped_thickness = min(protected_thickness, max_thickness_limit)
            clamped_width = min(protected_width, max_width_limit)

            # Fourth layer protection: reasonableness check
            thickness_range = abs(thickness2 - thickness1)
            width_range = abs(width2 - width1)

            final_thickness = clamped_thickness
            final_width = clamped_width

            # If endpoint range is very large, limit interpolation points from deviating too far from mean
            if thickness_range > (max_thickness_limit - min_thickness_limit) * 0.5:
                avg_thickness = (thickness1 + thickness2) / 2.0
                max_deviation = thickness_range * 0.3
                range_limited_thickness = max(min(final_thickness, avg_thickness + max_deviation),
                                            avg_thickness - max_deviation)
                final_thickness = range_limited_thickness

            if width_range > (max_width_limit - min_width_limit) * 0.5:
                avg_width = (width1 + width2) / 2.0
                max_deviation = width_range * 0.3
                range_limited_width = max(min(final_width, avg_width + max_deviation),
                                        avg_width - max_deviation)
                final_width = range_limited_width

            # Final mandatory protection ensures absolutely no exceeding global ranges
            final_thickness = np.clip(final_thickness, min_thickness_limit, max_thickness_limit)
            final_width = np.clip(final_width, min_width_limit, max_width_limit)

            # Count protection adjustment times
            if (abs(final_thickness - base_thickness) > 1e-6 or
                abs(final_width - base_width) > 1e-6):
                total_protected_adjustments += 1

            # relpos still uses standard linear interpolation
            new_relpos = relpos1 + factor * (relpos2 - relpos1)

            new_x_points.append(new_x)
            new_y_points.append(new_y)
            new_thickness_points.append(final_thickness)
            new_width_points.append(final_width)
            new_relpos_points.append(new_relpos)

        # Insert new points into arrays
        insert_pos = after_idx + 1

        current_x = np.concatenate([
            current_x[:insert_pos],
            np.array(new_x_points),
            current_x[insert_pos:]
        ])

        current_y = np.concatenate([
            current_y[:insert_pos],
            np.array(new_y_points),
            current_y[insert_pos:]
        ])

        current_thickness = np.concatenate([
            current_thickness[:insert_pos],
            np.array(new_thickness_points),
            current_thickness[insert_pos:]
        ])

        current_width = np.concatenate([
            current_width[:insert_pos],
            np.array(new_width_points),
            current_width[insert_pos:]
        ])

        current_relpos = np.concatenate([
            current_relpos[:insert_pos],
            np.array(new_relpos_points),
            current_relpos[insert_pos:]
        ])

        total_inserted += num_points

    current_thickness = np.clip(current_thickness, min_thickness_limit, max_thickness_limit)
    current_width = np.clip(current_width, min_width_limit, max_width_limit)

    return current_x, current_y, current_thickness, current_width, current_relpos


@cuda.jit
def calculate_curvature(centerline_x, centerline_y, out_curvature, out_radius):
    """
    Calculate curvature and radius of curvature at each point along river centerline
    using three-point angle method.

    Args:
        centerline_x, centerline_y: Centerline coordinate arrays
        out_curvature: Output curvature array
        out_radius: Output radius of curvature array

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_points = centerline_x.shape[0]

    if i < n_points:
        # Initialize default values
        curvature = 0.0
        radius = 1e10

        # Only middle points can calculate curvature (need previous and next points)
        if i > 0 and i < n_points - 1:
            # Get three consecutive points
            x1, y1 = centerline_x[i - 1], centerline_y[i - 1]
            x2, y2 = centerline_x[i], centerline_y[i]
            x3, y3 = centerline_x[i + 1], centerline_y[i + 1]

            # Calculate two vectors
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x3 - x2, y3 - y2

            # Calculate vector lengths
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

            if len1 > 1e-10 and len2 > 1e-10:
                # Calculate cross product and dot product
                cross_product = dx1 * dy2 - dy1 * dx2
                dot_product = dx1 * dx2 + dy1 * dy2
                magnitude_product = len1 * len2

                if magnitude_product > 1e-10:
                    # Calculate angle cosine and limit range
                    cos_angle = dot_product / magnitude_product
                    cos_angle = max(-1.0, min(1.0, cos_angle))

                    # Calculate angle
                    angle = math.acos(cos_angle)

                    # Calculate average length
                    avg_length = (len1 + len2) / 2.0

                    if avg_length > 1e-10:
                        # Calculate curvature magnitude
                        curvature_magnitude = angle / avg_length

                        # Determine curvature sign based on cross product
                        curvature = math.copysign(curvature_magnitude, cross_product)

                        # Calculate radius of curvature
                        radius = 1.0 / max(curvature_magnitude, 1e-10)

        # Output results
        out_curvature[i] = curvature
        out_radius[i] = radius

@cuda.jit
def smooth_centerline_kernel_guss(input_x, input_y, output_x, output_y, window_size,
                             original_x, original_y, allow_endpoint_migration,
                             smoothing_iterations, boundary_factor_param):
    """
    Smooth river centerline using Gaussian weights, ensuring start point Y coordinate remains fixed.

    Args:
        input_x, input_y: Input coordinate arrays
        output_x, output_y: Output coordinate arrays
        window_size: Smoothing window size
        original_x, original_y: Original coordinate arrays
        allow_endpoint_migration: Whether to allow endpoint migration
        smoothing_iterations: Number of smoothing iterations
        boundary_factor_param: Boundary factor parameter

    Returns:
        None: GPU kernel function has no return value
    """
    i = cuda.grid(1)
    n_points = input_x.shape[0]

    if i < n_points:
        temp_x = input_x[i]
        temp_y = input_y[i]

        is_endpoint = (i == 0 or i == n_points - 1)
        is_startpoint = (i == 0)

        for iter_count in range(smoothing_iterations):
            new_x = temp_x
            new_y = temp_y

            if is_endpoint:
                boundary_factor = boundary_factor_param if allow_endpoint_migration else 0.0
            else:
                boundary_factor = 1.0

            if boundary_factor > 0.0:
                half_window = window_size // 2
                win_start = max(0, i - half_window)
                win_end = min(n_points, i + half_window + 1)

                sum_x = 0.0
                sum_y = 0.0
                weight_sum = 0.0

                for j in range(win_start, win_end):
                    dist = abs(j - i)
                    sigma = window_size / 6.0
                    weight = math.exp(-0.5 * (dist / sigma) ** 2)

                    sum_x += input_x[j] * weight
                    sum_y += input_y[j] * weight
                    weight_sum += weight

                if weight_sum > 1e-10:
                    smoothed_x = sum_x / weight_sum
                    smoothed_y = sum_y / weight_sum

                    # Start point special handling: only allow X direction smoothing
                    if is_startpoint:
                        # Start point: only smooth X coordinate, Y coordinate remains original value
                        new_x = temp_x + boundary_factor * (smoothed_x - temp_x)
                        new_y = original_y[i]  # Y coordinate forced to remain original value
                    else:
                        # Other points: normal smoothing
                        new_x = temp_x + boundary_factor * (smoothed_x - temp_x)
                        new_y = temp_y + boundary_factor * (smoothed_y - temp_y)

            temp_x = new_x
            temp_y = new_y

        # Final ensure start point Y coordinate is fixed
        if is_startpoint:
            output_x[i] = temp_x
            output_y[i] = original_y[i]  # Force maintain original Y coordinate
        else:
            output_x[i] = temp_x
            output_y[i] = temp_y