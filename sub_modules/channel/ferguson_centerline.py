"""
ferguson_centerline.py

This module implements the Ferguson model-based river channel centerline generation algorithm,
utilizing physically-constrained random walk methods to simulate the natural meandering evolution
process of rivers. The algorithm integrates sinuosity feedback correction mechanisms, supporting
precise control of target sinuosity and multi-scale smoothing processing, suitable for numerical
reconstruction of realistic river morphologies.

Algorithm Overview:
The Ferguson model is based on curvature evolution equations from river dynamics, generating
river centerlines with natural meandering characteristics through second-order autoregressive
processes and perturbation term superposition. The algorithm employs a strategy combining
physical constraints and statistical control to achieve probabilistic generation of river paths
while satisfying boundary conditions and sinuosity targets.

Core Generation Workflow:
1. Ferguson Physical Evolution: Angle recursion and perturbation superposition based on second-order autoregressive model
2. Directional Deviation Control: Exponential decay change resistance and convergence control based on target angle
3. Angle Limitation System: Dual constraints on single-step turning and cumulative turning to prevent excessive meandering
4. Upstream Memory Effect: Calculation and application of meandering bias forces based on angle change history
5. Distance Boundary Checking: Path termination conditions and length control under model space constraints
6. Sinuosity Feedback Correction: Iterative optimization and convergence guarantee for target sinuosity

Physical Modeling Features:
- Autoregressive Evolution: Discretized implementation and numerical stability control of second-order Ferguson equations
- Perturbation Parameterization: Adaptive adjustment mechanism for perturbation intensity based on target sinuosity
- Constraint Force System: Multiple constraints including directional deviation resistance, angle limitations, and memory bias
- Boundary Conditions: Boundary handling strategy with fixed starting point, free endpoint, and distance limitations
- Physical Reasonability: Constraint mechanisms preventing non-physical angle changes and excessive meandering
"""

import numpy as np
import math
from ..engine import constants as const


def generate_ferguson_centerline(params, channel_params, icc, ic, verbose):
    """
    Generate Ferguson model centerline.

    This function implements the complete Ferguson model workflow for river channel centerline
    generation, including physical evolution, constraint application, and sinuosity correction.

    Args:
        params (dict): Global parameter dictionary containing simulation configuration and channel complex information
        channel_params (dict): Channel parameter dictionary containing target sinuosity, step size, and other settings
        icc (int): Channel complex number
        ic (int): Channel number
        verbose (bool): Boolean flag for outputting detailed debugging information

    Returns:
        tuple: (centerline coordinate array, weight dictionary, actual step count)
    """
    target_sinuosity = channel_params['target_sinuosity']
    actual_steps = channel_params['actual_steps']
    step_size = channel_params['step_size']
    start_x_global = channel_params['start_x_global']
    start_y_global = channel_params['start_y_global']
    model_ly = channel_params['model_ly']

    if verbose:
        print(f" Generating 2D centerline using Ferguson model")

    individual_weights = {
        'algorithm_type': 'ferguson',
    }

    # Set global parameters
    params['current_complex_angle'] = params['cco'][icc - 1]
    params['channel_start_x'] = start_x_global
    params['ymn'] = start_y_global

    # Generate centerline and obtain force decomposition data and actual sinuosity
    centerline_x, centerline_y, forces_data, actual_sinuosity = generate_simplified_ferguson_centerline_2d(
        actual_steps, step_size, target_sinuosity, const.get_channel_seed(icc, ic),
        start_x_global, start_y_global, params['cco'][icc - 1], model_ly, verbose
    )

    # Create final centerline array
    actual_generated_steps = len(centerline_x)
    final_centerline = np.zeros((actual_generated_steps, 2))
    final_centerline[:, 0] = centerline_x
    final_centerline[:, 1] = centerline_y

    # Apply smoothing processing
    final_centerline[:, 0] = apply_local_smoothing(final_centerline[:, 0], actual_generated_steps)
    final_centerline[:, 1] = apply_local_smoothing(final_centerline[:, 1], actual_generated_steps)

    actual_steps = actual_generated_steps

    # Save actual sinuosity values
    if 'actual_channel_properties' not in params:
        params['actual_channel_properties'] = {}

    channel_key = f"{icc}_{ic}"
    params['actual_channel_properties'][channel_key] = {
        'actual_sinuosity': actual_sinuosity,
        'target_sinuosity': target_sinuosity,
        'algorithm_type': 'ferguson',
        'generated_steps': actual_steps
    }

    if verbose:
        print(f"    Ferguson sinuosity: Target={target_sinuosity:.4f}, Actual={actual_sinuosity:.4f}")

    # Store force decomposition data
    if 'ferguson_forces_data' not in params:
        params['ferguson_forces_data'] = {}
    params['ferguson_forces_data'][channel_key] = forces_data

    return final_centerline, individual_weights, actual_steps

def generate_simplified_ferguson_centerline_2d(actual_steps, step_size, target_sinuosity, channel_seed,
                                               start_x_global, start_y_global, complex_angle, ly, verbose=True):
    """
    Generate 2D centerline using Ferguson model with distance constraints replacing Y-coordinate constraints.

    This function implements the core Ferguson algorithm with physical constraints, memory effects,
    and adaptive perturbation control to generate realistic river meandering patterns.

    Args:
        actual_steps (int): Actual number of steps
        step_size (float): Step size magnitude
        target_sinuosity (float): Target sinuosity value
        channel_seed (int): Channel random seed
        start_x_global (float): Global starting X coordinate
        start_y_global (float): Global starting Y coordinate
        complex_angle (float): Complex angle in degrees
        ly (float): Model Y-direction length
        verbose (bool): Whether to output detailed information, defaults to True

    Returns:
        tuple: (X coordinate array, Y coordinate array, force decomposition data list, final sinuosity value)
    """
    # Ferguson model parameters
    k = 0.3
    h = 0.8
    phi = np.arcsin(h)
    b1 = 2.0 * np.exp(-k * h) * np.cos(k * np.cos(phi))
    b2 = -1.0 * np.exp(-2.0 * k * h)

    # Calculate perturbation intensity
    s_raw = calculate_sinuosity_based_perturbation_strength(target_sinuosity)
    s = min(s_raw, 13)

    # Ferguson algorithm parameters
    target_direction = np.radians(complex_angle)
    base_change_resistance = 0.2 + (target_sinuosity - 1.0) * 0.4
    base_change_resistance = max(0.2, min(0.8, base_change_resistance))
    max_change_resistance = 0.9

    # Upstream memory and meandering bias force parameters
    memory_window = max(3, min(45, int(5 + (target_sinuosity - 1.0) * 8)))
    bias_force_coefficient = (target_sinuosity - 1.0) * 0.1
    bias_force_coefficient = max(0.1, min(2.0, bias_force_coefficient))
    angle_threshold = 2.0

    # Angle limitation parameters
    max_single_turn = 45
    max_cumulative_turn = 180.0
    cumulative_window = 15

    # Set random seed
    np.random.seed(channel_seed)

    # Calculate distance constraints
    if abs(complex_angle) < 45:
        target_directiony = np.radians(complex_angle)
        max_distance_from_start = 1.1 * ly / max(math.cos(target_directiony), 0.1)
    else:
        adjusted_angle = 90 - complex_angle
        target_directiony = np.radians(adjusted_angle)
        max_distance_from_start = 1.1 * ly / max(math.cos(target_directiony), 0.1)

    if verbose:
        print(f"      Ferguson distance constraint: Target direction={complex_angle:.1f}°, Max distance={max_distance_from_start:.1f}")

    # Generate perturbation sequence
    base_perturbations = np.random.normal(0, s, actual_steps * 2)

    # Execute Ferguson algorithm
    angles = []
    centerline_x = []
    centerline_y = []
    forces_data = []

    initial_angle = complex_angle
    ang1 = initial_angle
    ang2 = initial_angle

    start_cx, start_cy = 0.0, -100
    cx, cy = start_cx, start_cy
    step_count = 0

    angle_history = []
    angle_changes_history = []

    # Statistical information
    single_limited_turns = 0
    cumulative_limited_turns = 0
    direction_corrections = 0
    bias_force_applications = 0
    max_observed_single_turn = 0.0
    max_observed_cumulative_turn = 0.0
    max_observed_deviation = 0.0
    max_observed_bias_magnitude = 0.0
    total_ferguson_force_magnitude = 0.0
    total_bias_force_magnitude = 0.0

    # Add starting point
    centerline_x.append(cx)
    centerline_y.append(cy)
    angles.append(ang1)
    angle_history.append(ang1)

    # Starting point force data
    forces_data.append({
        'ferguson_dx': 0.0,
        'ferguson_dy': 0.0,
        'bias_dx': 0.0,
        'bias_dy': 0.0,
        'statistical_dx': 0.0,
        'statistical_dy': 0.0,
        'alpha': 0.0,
        'raw_angle': ang1,
        'final_angle': ang1,
        'deviation_from_target': 0.0,
        'change_resistance': base_change_resistance,
        'progress_ratio': 0.0,
        'upstream_angle_sum': 0.0,
        'bias_force_magnitude': 0.0,
        'distance_from_start': 0.0
    })

    # Ferguson main loop
    while step_count < actual_steps * 2:
        step_count += 1

        # Ferguson physical evolution
        if step_count == 1:
            raw_ferguson_angle = ang2
        else:
            perturbation_idx = min(step_count - 1, len(base_perturbations) - 1)
            raw_ferguson_angle = b1 * ang1 + b2 * ang2 + base_perturbations[perturbation_idx]

        # Directional change control based on target direction deviation
        current_direction_deg = ang1
        direction_deviation_deg = abs(current_direction_deg - complex_angle)

        while direction_deviation_deg > 180:
            direction_deviation_deg = 360 - direction_deviation_deg

        normalized_deviation = direction_deviation_deg / 180.0
        max_observed_deviation = max(max_observed_deviation, normalized_deviation)

        exponential_factor = 1.0 + (target_sinuosity - 1.0) * 2.0
        exponential_factor = max(1.0, min(4.0, exponential_factor))

        exponential_change_rate = np.exp(-exponential_factor * normalized_deviation)
        change_resistance = base_change_resistance * (1.0 - exponential_change_rate * 0.8)
        change_resistance = max(0.05, min(max_change_resistance, change_resistance))

        adjusted_ferguson_angle = (1.0 - change_resistance) * raw_ferguson_angle + change_resistance * ang1

        # Angle limitation system
        if step_count > 1:
            raw_single_turn = adjusted_ferguson_angle - ang1
            max_observed_single_turn = max(max_observed_single_turn, abs(raw_single_turn))

            if abs(raw_single_turn) > max_single_turn:
                limited_single_turn = np.sign(raw_single_turn) * max_single_turn
                single_limited_turns += 1
            else:
                limited_single_turn = raw_single_turn

            current_angle = ang1 + limited_single_turn

            # Cumulative turning angle check
            if len(angle_history) >= cumulative_window:
                window_angles = angle_history[-cumulative_window:] + [current_angle]
                cumulative_turn = 0.0

                for i in range(1, len(window_angles)):
                    turn = window_angles[i] - window_angles[i - 1]
                    while turn > 180:
                        turn -= 360
                    while turn < -180:
                        turn += 360
                    cumulative_turn += turn

                max_observed_cumulative_turn = max(max_observed_cumulative_turn, abs(cumulative_turn))

                if abs(cumulative_turn) > max_cumulative_turn:
                    excess_turn = abs(cumulative_turn) - max_cumulative_turn
                    reduction_factor = excess_turn / abs(limited_single_turn) if abs(limited_single_turn) > 0 else 0
                    reduction_factor = min(reduction_factor, 0.8)

                    adjusted_turn = limited_single_turn * (1 - reduction_factor)
                    current_angle = ang1 + adjusted_turn
                    cumulative_limited_turns += 1
        else:
            current_angle = adjusted_ferguson_angle

        # Record angle change history
        if step_count > 1:
            angle_change = current_angle - angle_history[-1]
            while angle_change > 180:
                angle_change -= 360
            while angle_change < -180:
                angle_change += 360
            angle_changes_history.append(angle_change)
        else:
            angle_changes_history.append(0.0)

        # Update angle history
        ang2 = ang1
        ang1 = current_angle
        angle_history.append(current_angle)

        if len(angle_history) > max(cumulative_window, memory_window) + 2:
            angle_history.pop(0)
        if len(angle_changes_history) > memory_window + 2:
            angle_changes_history.pop(0)

        # Ferguson force calculation
        angle_rad = np.radians(current_angle)
        dx_ferguson = step_size * np.sin(angle_rad)
        dy_ferguson = step_size * np.cos(angle_rad)

        # Upstream memory and meandering bias force calculation
        bias_dx, bias_dy = 0.0, 0.0
        upstream_angle_sum = 0.0
        bias_force_magnitude = 0.0

        memory_start = max(0, len(angle_changes_history) - memory_window)
        upstream_angles = angle_changes_history[memory_start:]

        if len(upstream_angles) > 1:
            for angle_change in upstream_angles:
                if abs(angle_change) > angle_threshold:
                    upstream_angle_sum += abs(angle_change)

            if upstream_angle_sum > angle_threshold * 2:
                bias_force_magnitude = bias_force_coefficient * upstream_angle_sum

                recent_changes = upstream_angles[-3:] if len(upstream_angles) >= 3 else upstream_angles
                avg_recent_change = np.mean(recent_changes)

                if avg_recent_change > 0:
                    bias_direction = current_angle + 90
                else:
                    bias_direction = current_angle - 90

                bias_direction_rad = np.radians(bias_direction)
                bias_dx = bias_force_magnitude * step_size * np.sin(bias_direction_rad) * 0.1
                bias_dy = bias_force_magnitude * step_size * np.cos(bias_direction_rad) * 0.1

                bias_force_applications += 1
                max_observed_bias_magnitude = max(max_observed_bias_magnitude, bias_force_magnitude)

        # Displacement calculation
        final_dx = dx_ferguson + bias_dx
        final_dy = dy_ferguson + bias_dy

        # Update position
        new_cx = cx + final_dx
        new_cy = cy + final_dy

        # Check distance constraints
        distance_from_start = np.sqrt((new_cx - start_cx) ** 2 + (new_cy - start_cy) ** 2)

        if distance_from_start >= max_distance_from_start:
            if verbose:
                print(f"      Ferguson algorithm reached distance limit: Distance={distance_from_start:.1f} >= {max_distance_from_start:.1f}")
            break

        # Update coordinates
        cx, cy = new_cx, new_cy

        centerline_x.append(cx)
        centerline_y.append(cy)

        # Statistical force magnitudes
        ferguson_force_magnitude = np.sqrt(dx_ferguson ** 2 + dy_ferguson ** 2)
        total_ferguson_force_magnitude += ferguson_force_magnitude
        total_bias_force_magnitude += np.sqrt(bias_dx ** 2 + bias_dy ** 2)

        # Record detailed force decomposition data
        forces_data.append({
            'ferguson_dx': dx_ferguson,
            'ferguson_dy': dy_ferguson,
            'bias_dx': bias_dx,
            'bias_dy': bias_dy,
            'statistical_dx': 0.0,
            'statistical_dy': 0.0,
            'alpha': 0.0,
            'raw_angle': raw_ferguson_angle,
            'final_angle': current_angle,
            'deviation_from_target_deg': direction_deviation_deg,
            'normalized_deviation': normalized_deviation,
            'exponential_change_rate': exponential_change_rate,
            'change_resistance': change_resistance,
            'progress_ratio': step_count / max(actual_steps, 1),
            'target_direction_deg': complex_angle,
            'current_direction_deg': current_angle,
            'upstream_angle_sum': upstream_angle_sum,
            'bias_force_magnitude': bias_force_magnitude,
            'distance_from_start': distance_from_start
        })

        if abs(adjusted_ferguson_angle - raw_ferguson_angle) > 0.1:
            direction_corrections += 1

    # Convert to numpy arrays
    centerline_x = np.array(centerline_x)
    centerline_y = np.array(centerline_y)
    final_steps = len(centerline_x)

    # Sinuosity feedback correction mechanism
    if verbose:
        print(f"      Starting sinuosity feedback correction...")

    # Calculate initial sinuosity of Ferguson algorithm
    initial_sinuosity = calculate_actual_sinuosity_from_coordinates(centerline_x, centerline_y)

    if verbose:
        print(f"      Ferguson initial sinuosity: {initial_sinuosity:.4f}, Target: {target_sinuosity:.4f}")
        print(f"      Final distance: {distance_from_start:.1f}, Limit: {max_distance_from_start:.1f}")

    # Apply sinuosity feedback correction
    corrected_x, corrected_y = apply_sinuosity_feedback_correction(
        centerline_x, centerline_y, target_sinuosity,
        tolerance=0.01, max_iterations=300, verbose=verbose
    )

    # Calculate final corrected sinuosity
    final_sinuosity = calculate_actual_sinuosity_from_coordinates(corrected_x, corrected_y)

    # Update force data with final position information and actual sinuosity
    for i, force_data in enumerate(forces_data):
        if i < len(corrected_x):
            force_data['corrected_x'] = corrected_x[i]
            force_data['corrected_y'] = corrected_y[i]
        force_data['final_sinuosity'] = final_sinuosity
        force_data['initial_sinuosity'] = initial_sinuosity
        force_data['target_sinuosity'] = target_sinuosity
        force_data['sinuosity_correction_applied'] = True

    # Calculate final deviation and statistics
    total_path_length = 0.0
    for i in range(len(corrected_x) - 1):
        dx = corrected_x[i + 1] - corrected_x[i]
        dy = corrected_y[i + 1] - corrected_y[i]
        total_path_length += np.sqrt(dx * dx + dy * dy)

    final_direction = np.arctan2(corrected_x[-1] - corrected_x[0],
                                 corrected_y[-1] - corrected_y[0])
    final_deviation = abs(final_direction - target_direction)
    while final_deviation > np.pi:
        final_deviation = 2 * np.pi - final_deviation
    final_deviation_deg = np.degrees(final_deviation)

    total_force = total_ferguson_force_magnitude + total_bias_force_magnitude
    ferguson_ratio = (total_ferguson_force_magnitude / total_force * 100) if total_force > 0 else 100
    bias_ratio = (total_bias_force_magnitude / total_force * 100) if total_force > 0 else 0

    if verbose:
        print(f"      Ferguson + Feedback correction verification:")
        print(f"        Ferguson sinuosity: {initial_sinuosity:.4f} -> Final sinuosity: {final_sinuosity:.4f}")
        print(f"        Target sinuosity: {target_sinuosity:.4f}, Final error: {abs(final_sinuosity - target_sinuosity):.4f}")
        print(f"        Target direction: {complex_angle:.1f}°, Final direction: {np.degrees(final_direction):.1f}°")
        print(f"        Final deviation: {final_deviation_deg:.1f}°")
        print(f"        Final distance: {distance_from_start:.1f}, Limit: {max_distance_from_start:.1f}")
        print(f"        Ferguson force: {ferguson_ratio:.1f}%, Bias force: {bias_ratio:.1f}%")
        print(f"        Bias force applications: {bias_force_applications} times")

    return corrected_x, corrected_y, forces_data, final_sinuosity

def calculate_sinuosity_based_perturbation_strength(target_sinuosity):
    """
    Calculate Ferguson model perturbation strength parameter based on target sinuosity.

    This function establishes a linear mapping between target sinuosity and perturbation
    intensity to ensure appropriate meandering characteristics for different sinuosity targets.

    Args:
        target_sinuosity (float): Target sinuosity value

    Returns:
        float: Calculated perturbation strength parameter s
    """
    # Linear mapping parameters
    min_sinuosity = 1.0
    max_sinuosity = 2.0
    min_s = 1.0
    max_s = 13

    # Constrain sinuosity to reasonable range
    clamped_sinuosity = max(min_sinuosity, min(max_sinuosity, target_sinuosity))

    # Linear mapping formula
    if clamped_sinuosity <= min_sinuosity:
        s = min_s
    elif clamped_sinuosity >= max_sinuosity:
        s = max_s
    else:
        # Linear interpolation
        normalized = (clamped_sinuosity - min_sinuosity) / (max_sinuosity - min_sinuosity)
        s = min_s + (max_s - min_s) * normalized

    return s

def apply_sinuosity_feedback_correction(x_coords, y_coords, target_sinuosity,
                                        tolerance=0.01, max_iterations=300, verbose=False):
    """
    Sinuosity feedback correction mechanism achieving target sinuosity through targeted oscillation attenuation.

    This function implements an adaptive correction algorithm that alternates between smoothing
    and migration operations with targeted parameter decay to converge on the desired sinuosity.

    Args:
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
        target_sinuosity (float): Target sinuosity value
        tolerance (float): Error tolerance, defaults to 0.01
        max_iterations (int): Maximum iteration count, defaults to 300
        verbose (bool): Whether to output detailed debugging information, defaults to False

    Returns:
        tuple: (corrected X coordinate array, corrected Y coordinate array)
    """
    current_x = x_coords.copy()
    current_y = y_coords.copy()
    best_x = current_x.copy()
    best_y = current_y.copy()
    best_error = float('inf')

    # Initial parameter settings
    initial_smooth_strength = 3.8
    initial_migration_distance = 10
    min_smooth_strength = 0.005
    min_migration_distance = 0.02

    # Maintain separate parameters
    current_smooth_strength = initial_smooth_strength
    current_migration_distance = initial_migration_distance

    # Oscillation detection and statistics
    last_operation = None
    total_jumps = 0
    smooth_to_migrate_jumps = 0
    migrate_to_smooth_jumps = 0
    total_smooth_operations = 0
    total_migrate_operations = 0

    # Decay factor
    decay_factor = 0.85

    # Initial check
    initial_sinuosity = calculate_actual_sinuosity_from_coordinates(current_x, current_y)

    print(f"        Correction start: Initial={initial_sinuosity:.4f}, Target={target_sinuosity:.4f}")
    print(f"        Initial error: {abs(initial_sinuosity - target_sinuosity):.4f}, Tolerance: {tolerance}")

    if abs(initial_sinuosity - target_sinuosity) <= tolerance:
        print(f"        Ferguson algorithm quality is good, already within threshold range, no correction needed")
        return current_x, current_y

    print(f"        Correction needed, starting iterations...")
    print(f"        Decay strategy: Targeted decay on oscillation, decay factor={decay_factor}")

    # Record last error for detecting improvement stagnation
    stagnation_count = 0

    # Main correction loop
    for iteration in range(max_iterations):
        actual_sinuosity = calculate_actual_sinuosity_from_coordinates(current_x, current_y)
        error = abs(actual_sinuosity - target_sinuosity)

        # Update best result
        if error < best_error:
            best_error = error
            best_x = current_x.copy()
            best_y = current_y.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Detailed debugging output
        if iteration % 15 == 0 or iteration < 8:
            print(f"        Iteration {iteration:3d}: Sinuosity={actual_sinuosity:.4f}, Error={error:.4f}")
            print(f"            Current parameters: Smooth strength={current_smooth_strength:.4f}, Migration distance={current_migration_distance:.4f}")
            print(f"            Jump statistics: Total jumps={total_jumps}, Smooth to migrate={smooth_to_migrate_jumps}, Migrate to smooth={migrate_to_smooth_jumps}")
            print(f"            Operation statistics: Smooth={total_smooth_operations} times, Migrate={total_migrate_operations} times, Stagnant={stagnation_count}")

        # Check convergence
        if error <= tolerance:
            print(f"        Correction converged at iteration {iteration}: Final sinuosity={actual_sinuosity:.4f}")
            print(f"        Final jump statistics: Total jumps={total_jumps} times")
            print(f"        Jump breakdown: Smooth to migrate={smooth_to_migrate_jumps} times, Migrate to smooth={migrate_to_smooth_jumps} times")
            print(f"        Final parameters: Smooth strength={current_smooth_strength:.4f}, Migration distance={current_migration_distance:.4f}")
            return current_x, current_y

        # Check for stagnation
        if stagnation_count > 40:
            print(f"        Stagnation detected ({stagnation_count} times), using best result")
            break

        # Calculate required adjustment magnitude
        sinuosity_gap = actual_sinuosity - target_sinuosity

        # Determine current required operation type
        if sinuosity_gap > 0:
            current_operation = 'smooth'  # Need smoothing operation (reduce sinuosity)
        else:
            current_operation = 'migrate'  # Need migration operation (increase sinuosity)

        # Core oscillation detection and targeted decay logic
        if last_operation is not None and current_operation != last_operation:
            total_jumps += 1

            if last_operation == 'smooth' and current_operation == 'migrate':
                # Jump from smooth to migrate: smoothing operation was excessive, decay smooth parameter
                old_smooth_strength = current_smooth_strength
                current_smooth_strength *= decay_factor
                current_smooth_strength = max(min_smooth_strength, current_smooth_strength)
                smooth_to_migrate_jumps += 1

                print(f"        Jump #{total_jumps}: Smooth to migrate (excessive smoothing)")
                print(f"            Decay smooth strength: {old_smooth_strength:.4f} -> {current_smooth_strength:.4f}")

            elif last_operation == 'migrate' and current_operation == 'smooth':
                # Jump from migrate to smooth: migration operation was excessive, decay migration parameter
                old_migration_distance = current_migration_distance
                current_migration_distance *= decay_factor
                current_migration_distance = max(min_migration_distance, current_migration_distance)
                migrate_to_smooth_jumps += 1

                print(f"        Jump #{total_jumps}: Migrate to smooth (excessive migration)")
                print(f"            Decay migration distance: {old_migration_distance:.4f} -> {current_migration_distance:.4f}")

        # Execute corresponding operation
        if current_operation == 'smooth':
            # Execute smoothing operation
            if abs(sinuosity_gap) > 0.3:
                dynamic_smooth_strength = current_smooth_strength * 1.5
            elif abs(sinuosity_gap) > 0.15:
                dynamic_smooth_strength = current_smooth_strength
            else:
                dynamic_smooth_strength = current_smooth_strength * 0.6

            # Limit maximum strength
            dynamic_smooth_strength = min(dynamic_smooth_strength, 0.2)

            # Execute progressive smoothing
            temp_x, temp_y = apply_progressive_smoothing(
                current_x, current_y, dynamic_smooth_strength,
                target_gap=sinuosity_gap, target_sinuosity=target_sinuosity
            )

            # Check for over-adjustment
            temp_sinuosity = calculate_actual_sinuosity_from_coordinates(temp_x, temp_y)
            if temp_sinuosity < target_sinuosity and actual_sinuosity > target_sinuosity:
                # Over-adjusted, re-adjust with smaller strength
                reduced_strength = dynamic_smooth_strength * 0.3
                current_x, current_y = apply_progressive_smoothing(
                    current_x, current_y, reduced_strength,
                    target_gap=sinuosity_gap, target_sinuosity=target_sinuosity
                )
                if iteration % 20 == 0:
                    print(f"            Over-smoothing detected, reducing strength to {reduced_strength:.4f}")
            else:
                current_x, current_y = temp_x, temp_y

            total_smooth_operations += 1

        else:  # current_operation == 'migrate'
            # Execute migration operation
            if abs(sinuosity_gap) > 0.3:
                dynamic_migration_distance = current_migration_distance * 1.5
            elif abs(sinuosity_gap) > 0.15:
                dynamic_migration_distance = current_migration_distance
            else:
                dynamic_migration_distance = current_migration_distance * 0.6

            current_x, current_y = apply_coordinate_migration(
                current_x, current_y, dynamic_migration_distance
            )
            total_migrate_operations += 1

        # Update operation history
        last_operation = current_operation

        # Early exit condition: if both parameters are very small and oscillations are excessive
        if (current_smooth_strength <= min_smooth_strength * 1.1 and
                current_migration_distance <= min_migration_distance * 1.1 and
                total_jumps > 15):
            print(f"        Both parameters reached minimum values and excessive oscillation ({total_jumps} times), using best result")
            break

    # Final statistics
    final_sinuosity = calculate_actual_sinuosity_from_coordinates(best_x, best_y)
    print(f"        Correction complete: Best error={best_error:.4f}, Final sinuosity={final_sinuosity:.4f}")
    print(f"        Total iterations: {iteration + 1}")
    print(f"        Total jumps: {total_jumps} times")
    print(f"        Jump breakdown: Smooth to migrate={smooth_to_migrate_jumps} times, Migrate to smooth={migrate_to_smooth_jumps} times")
    print(f"        Operation statistics: Smooth={total_smooth_operations} times, Migrate={total_migrate_operations} times")
    print(f"        Final parameters: Smooth strength={current_smooth_strength:.4f}, Migration distance={current_migration_distance:.4f}")

    # Parameter decay effect analysis
    smooth_decay_ratio = current_smooth_strength / initial_smooth_strength
    migrate_decay_ratio = current_migration_distance / initial_migration_distance
    print(f"        Parameter decay effect: Smooth strength retained {smooth_decay_ratio:.1%}, Migration distance retained {migrate_decay_ratio:.1%}")

    return best_x, best_y

def calculate_actual_sinuosity_from_coordinates(x_coords, y_coords):
    """
    Calculate actual sinuosity from coordinate arrays.

    Sinuosity is defined as the ratio of channel length to valley length,
    providing a quantitative measure of channel meandering.

    Args:
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array

    Returns:
        float: Calculated sinuosity value (path length to straight distance ratio)
    """
    if len(x_coords) < 2:
        return 1.0

    # Calculate total path length
    total_length = 0.0
    for i in range(len(x_coords) - 1):
        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]
        segment_length = np.sqrt(dx * dx + dy * dy)
        total_length += segment_length

    # Calculate straight-line distance
    straight_distance = np.sqrt(
        (x_coords[-1] - x_coords[0]) ** 2 +
        (y_coords[-1] - y_coords[0]) ** 2
    )

    if straight_distance < 1e-10:
        return 1.0

    sinuosity = total_length / straight_distance
    return max(1.0, sinuosity)

def apply_coordinate_migration(x_coords, y_coords, migration_distance):
    """
    Apply curvature-based coordinate migration to increase river sinuosity.

    This function implements lateral migration of channel points based on local
    curvature, simulating natural channel evolution processes.

    Args:
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
        migration_distance (float): Migration distance parameter

    Returns:
        tuple: (migrated X coordinate array, migrated Y coordinate array)
    """
    migrated_x = x_coords.copy()
    migrated_y = y_coords.copy()
    n = len(x_coords)

    if n < 5:
        return migrated_x, migrated_y

    # Calculate curvature
    curvature = calculate_coordinate_curvature(x_coords, y_coords)

    # Migration parameters
    curvature_threshold = 1e-7
    max_migration_factor = 10

    for i in range(3, n - 3):
        local_curvature = curvature[i]

        if abs(local_curvature) < curvature_threshold:
            continue

        # Calculate normal vector (perpendicular to tangent)
        if i > 1 and i < n - 2:
            # Use longer tangent calculation for improved stability
            tangent_x = x_coords[i + 2] - x_coords[i - 2]
            tangent_y = y_coords[i + 2] - y_coords[i - 2]
            tangent_length = np.sqrt(tangent_x ** 2 + tangent_y ** 2)

            if tangent_length > 1e-10:
                # Normalize tangent vector
                tangent_x /= tangent_length
                tangent_y /= tangent_length

                # Normal vector (90-degree left rotation)
                normal_x = -tangent_y
                normal_y = tangent_x

                # Determine migration direction and distance based on curvature
                curvature_sign = np.sign(local_curvature)
                curvature_magnitude = abs(local_curvature)

                # Migration distance calculation
                actual_migration = migration_distance * min(curvature_magnitude * 50, max_migration_factor)

                # Migrate toward convex bank exterior
                migrated_x[i] += curvature_sign * normal_x * actual_migration
                migrated_y[i] += curvature_sign * normal_y * actual_migration

    return migrated_x, migrated_y

def apply_progressive_smoothing(x_coords, y_coords, smooth_strength, target_gap, target_sinuosity):
    """
    Apply progressive smoothing processing to avoid over-adjustment of sinuosity.

    This function implements adaptive smoothing with intermediate checking to prevent
    excessive reduction of channel sinuosity beyond the target value.

    Args:
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
        smooth_strength (float): Smoothing strength parameter
        target_gap (float): Target sinuosity minus current sinuosity difference
        target_sinuosity (float): Target sinuosity value

    Returns:
        tuple: (smoothed X coordinate array, smoothed Y coordinate array)
    """
    current_x = x_coords.copy()
    current_y = y_coords.copy()
    n = len(x_coords)

    if n < 5:
        return current_x, current_y

    # Window size and iteration count
    base_window = max(2, int(n * 0.03))
    window_size = max(1, int(base_window * smooth_strength))

    # Smoothing iteration count
    num_passes = max(1, min(2, int(2 * smooth_strength)))

    for pass_idx in range(num_passes):
        temp_x = current_x.copy()
        temp_y = current_y.copy()

        # Use decreasing strength for each iteration
        current_strength = smooth_strength * (0.7 ** pass_idx)

        for i in range(window_size, n - window_size):
            x_neighbor_sum = 0.0
            y_neighbor_sum = 0.0
            weight_sum = 0.0

            # Use smaller neighborhood range
            for j in range(-window_size, window_size + 1):
                if j == 0:
                    continue

                idx = i + j
                if 0 <= idx < n:
                    # Use gentle weights to avoid over-smoothing
                    weight = np.exp(-0.8 * (j / max(1, window_size / 2)) ** 2)
                    x_neighbor_sum += weight * current_x[idx]
                    y_neighbor_sum += weight * current_y[idx]
                    weight_sum += weight

            if weight_sum > 0:
                x_neighbor_avg = x_neighbor_sum / weight_sum
                y_neighbor_avg = y_neighbor_sum / weight_sum

                temp_x[i] = (1.0 - current_strength) * current_x[i] + current_strength * x_neighbor_avg
                temp_y[i] = (1.0 - current_strength) * current_y[i] + current_strength * y_neighbor_avg

        # Intermediate check: check if approaching target after each smoothing
        intermediate_sinuosity = calculate_actual_sinuosity_from_coordinates(temp_x, temp_y)
        intermediate_gap = intermediate_sinuosity - target_sinuosity

        # Stop further smoothing if already close to target or over-adjusted
        if abs(intermediate_gap) < abs(target_gap) * 0.3:
            current_x, current_y = temp_x, temp_y
            break
        elif target_gap > 0 and intermediate_gap <= 0:
            # Use interpolation to return to appropriate position
            interpolation_factor = target_gap / (target_gap - intermediate_gap)
            interpolation_factor = max(0.0, min(1.0, interpolation_factor))
            current_x = current_x + interpolation_factor * (temp_x - current_x)
            current_y = current_y + interpolation_factor * (temp_y - current_y)
            break
        else:
            current_x, current_y = temp_x, temp_y

    return current_x, current_y


def calculate_coordinate_curvature(x_coords, y_coords):
    """
    Calculate curvature of coordinate sequence.

    This function computes local curvature at each point using a three-point
    method, providing a measure of channel bending intensity.

    Args:
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array

    Returns:
        numpy.ndarray: Curvature value array for each point
    """
    n = len(x_coords)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        # Three-point method for curvature calculation
        x1, y1 = x_coords[i - 1], y_coords[i - 1]
        x2, y2 = x_coords[i], y_coords[i]
        x3, y3 = x_coords[i + 1], y_coords[i + 1]

        # Two vectors
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2

        len1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
        len2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

        if len1 > 1e-10 and len2 > 1e-10:
            # Cross product calculation of curvature
            cross_product = dx1 * dy2 - dy1 * dx2
            avg_len = (len1 + len2) / 2
            curvature[i] = cross_product / (avg_len ** 3)

    # Handle boundaries
    curvature[0] = curvature[1] if n > 1 else 0
    curvature[-1] = curvature[-2] if n > 1 else 0

    # Mild smoothing
    smoothed_curvature = np.zeros_like(curvature)
    for i in range(n):
        start_idx = max(0, i - 1)
        end_idx = min(n, i + 2)
        smoothed_curvature[i] = np.mean(curvature[start_idx:end_idx])

    return smoothed_curvature

def apply_local_smoothing(deviation, ny3, num_passes=3):
    """
    Apply local smoothing processing to reduce jagged effects in river centerlines.

    This function implements multi-pass moving average smoothing to improve
    the geometric quality of generated centerlines.

    Args:
        deviation (np.ndarray): Deviation array or coordinate array
        ny3 (int): Array length
        num_passes (int): Number of smoothing iterations, defaults to 3

    Returns:
        numpy.ndarray: Smoothed array
    """
    for _ in range(num_passes):
        temp_deviation = np.copy(deviation)
        for i in range(ny3):
            win_start = max(0, i - 4)
            win_end = min(ny3, i + 5)
            if win_end > win_start:
                deviation[i] = np.mean(temp_deviation[win_start:win_end])
    return deviation