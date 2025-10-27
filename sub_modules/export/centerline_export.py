"""
centerline_export.py
"""

import os
import csv
import numpy as np
import time


def calculate_global_curvature(global_x, global_y):
    """
    Calculate curvature in global coordinate system.

    Args:
        global_x: Global x-coordinate array
        global_y: Global y-coordinate array

    Returns:
        Curvature array in global coordinate system
    """
    n = len(global_x)
    global_curvature = np.zeros(n)

    if len(global_y) != n:
        return global_curvature

    if n < 3:
        return global_curvature

    for i in range(1, n - 1):
        # Calculate adjacent vectors
        dx1 = global_x[i] - global_x[i - 1]
        dy1 = global_y[i] - global_y[i - 1]
        dx2 = global_x[i + 1] - global_x[i]
        dy2 = global_y[i + 1] - global_y[i]

        # Calculate lengths
        len1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = np.sqrt(dx2 * dx2 + dy2 * dy2)

        if len1 < 1e-10 or len2 < 1e-10:
            global_curvature[i] = 0
            continue

        # Calculate unit vectors
        ux1, uy1 = dx1 / len1, dy1 / len1
        ux2, uy2 = dx2 / len2, dy2 / len2

        # Calculate angle change
        dot_product = ux1 * ux2 + uy1 * uy2
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_change = np.arccos(dot_product)

        # Determine bending direction
        cross_product = ux1 * uy2 - uy1 * ux2
        sign = -np.sign(cross_product)

        # Calculate curvature
        avg_len = (len1 + len2) / 2
        if avg_len > 0:
            global_curvature[i] = sign * angle_change / avg_len

    # Boundary point handling
    global_curvature[0] = global_curvature[1] if n > 1 else 0
    global_curvature[-1] = global_curvature[-2] if n > 1 else 0

    return global_curvature


def gpu_rotc_equivalent(xorig, yorig, angle, xx, yy):
    """
    Implementation consistent with gpu_rotc function in GPU rendering code.

    Args:
        xorig: Rotation center x-coordinate
        yorig: Rotation center y-coordinate
        angle: Rotation angle (radians)
        xx: Local x-coordinate
        yy: Local y-coordinate

    Returns:
        xx_new, yy_new: Global coordinates
    """
    yy_new = yorig + yy * np.cos(angle) + xx * np.sin(angle)
    xx_new = xorig - yy * np.sin(angle) + xx * np.cos(angle)
    return xx_new, yy_new


def get_channel_z_top(channel_id, params):
    """
    Get channel top Z value.

    Args:
        channel_id: Channel ID in format 'icc_ic'
        params: Simulation parameter dictionary

    Returns:
        z_top: Channel top Z value
    """
    try:
        icc_str, ic_str = channel_id.split('_')
        icc = int(icc_str)
        ic = int(ic_str)
    except ValueError:
        icc = 1
        ic = 1

    # Search in main channel information
    z_top = None
    for channel_info in params.get('main_channels_info', []):
        channel_icc = int(channel_info.get('icc', 0))
        channel_ic = int(channel_info.get('ic', 0))

        if channel_icc == icc and channel_ic == ic:
            z_top = channel_info.get('z_top')
            break

    # Use default calculation method
    if z_top is None:
        zmn = params.get('zmn', 0)
        nz = params.get('nz', 100)
        zsiz = params.get('zsiz', 1.0)
        z_top = zmn + nz * zsiz * 0.5

    return z_top


def process_centerline_data(centerline_data, params):
    """
    Process centerline data using the same coordinate system as voxel rendering.

    Args:
        centerline_data: List of centerline data
        params: Simulation parameter dictionary

    Returns:
        List of processed centerline data
    """
    processed_data = []
    ymn = params.get('ymn', 0)

    has_y_data = params is not None and 'centerline_y_data' in params

    for channel in centerline_data:
        channel_id = channel['channel_id']
        complex_x = channel['complex_x']
        channel_x = channel['channel_x']
        x_center = complex_x + channel_x
        backup_z = channel['channel_z']
        global_angle = channel['global_angle'] * (np.pi / 180.0)

        z_top = get_channel_z_top(channel_id, params)

        # Try using data from channel
        if abs(z_top - params.get('zmn', 0) - params.get('nz', 100) * params.get('zsiz', 1.0) * 0.5) < 0.001:
            zmn = params.get('zmn', 0)
            z_top_from_channel = zmn + backup_z

            if z_top_from_channel > zmn and z_top_from_channel < params.get('nz', 100) * params.get('zsiz', 1.0) + zmn:
                z_top = z_top_from_channel

        try:
            icc, ic = map(int, channel_id.split('_'))
        except ValueError:
            icc, ic = 1, 1

        processed_points = []

        for point_idx, point in enumerate(channel['points']):
            local_x = point['local_x']

            # Get Y coordinate
            if has_y_data and icc-1 < len(params['centerline_y_data']) and ic-1 < len(params['centerline_y_data'][icc-1]):
                local_y = params['centerline_y_data'][icc-1, ic-1, point_idx] - ymn
            else:
                local_y = point.get('local_y', point_idx * params.get('ysiz', 1.0))

            # Coordinate transformation
            voxel_x, voxel_y = gpu_rotc_equivalent(
                x_center, ymn, global_angle, local_x, local_y
            )

            thickness = point['thickness']
            global_z = z_top
            base_z = global_z - thickness

            processed_point = point.copy()
            processed_point['global_x'] = voxel_x
            processed_point['global_y'] = voxel_y
            processed_point['global_z'] = global_z
            processed_point['base_z'] = base_z
            processed_point['local_y'] = local_y
            processed_points.append(processed_point)

        # Calculate global curvature
        global_x = [p['global_x'] for p in processed_points]
        global_y = [p['global_y'] for p in processed_points]
        global_curvature = calculate_global_curvature(global_x, global_y)

        for i, point in enumerate(processed_points):
            point['global_curvature'] = global_curvature[i]

        processed_channel = channel.copy()
        processed_channel['points'] = processed_points
        processed_data.append(processed_channel)

    return processed_data


def process_migration_data(migration_data, params):
    """
    Process migration data using the same coordinate system as voxel rendering.

    Args:
        migration_data: Migration data dictionary
        params: Simulation parameter dictionary

    Returns:
        Processed migration data dictionary
    """
    processed_data = {}
    ymn = params.get('ymn', 0)

    for channel_id, steps in migration_data.items():
        processed_steps = []

        try:
            icc_str, ic_str = channel_id.split('_')
            icc = int(icc_str)
            ic = int(ic_str)
        except ValueError:
            continue

        # Search for channel information
        x_center = None
        angle = None
        z_top = None

        for channel_info in params.get('main_channels_info', []):
            channel_icc = int(channel_info.get('icc', 0))
            channel_ic = int(channel_info.get('ic', 0))

            if channel_icc == icc and channel_ic == ic:
                x_center = channel_info.get('x_center')
                angle = channel_info.get('angle')
                z_top = channel_info.get('z_top')
                break

        # Use default values
        if x_center is None:
            x_center = params.get('xmn', 0) + params.get('nx', 100) * params.get('xsiz', 1.0) * 0.5
        if angle is None:
            angle = 0.0
        if z_top is None:
            zmn = params.get('zmn', 0)
            nz = params.get('nz', 100)
            zsiz = params.get('zsiz', 1.0)
            z_top = zmn + nz * zsiz * 0.5

        for step_data in steps:
            step = step_data['step']
            centerline_x = step_data['centerline']
            centerline_y = step_data['centerline_y']
            thickness = step_data.get('thickness', [])
            width = step_data.get('width', [])
            relpos = step_data.get('relpos', [])

            n_points = len(centerline_x)
            global_x = []
            global_y = []
            global_z = []
            base_z = []

            for i in range(n_points):
                vx, vy = gpu_rotc_equivalent(
                    x_center, ymn, angle, centerline_x[i], centerline_y[i] - ymn
                )
                global_x.append(vx)
                global_y.append(vy)
                global_z.append(z_top)

                if i < len(thickness):
                    base_z.append(z_top - thickness[i])
                else:
                    base_z.append(z_top - 1.0)

            global_curvature = calculate_global_curvature(global_x, global_y)

            processed_step = {
                'step': step,
                'centerline': global_x,
                'centerline_y': global_y,
                'global_z': global_z,
                'base_z': base_z,
                'thickness': thickness,
                'width': width,
                'relpos': relpos,
                'global_curvature': global_curvature
            }

            processed_steps.append(processed_step)

        processed_data[channel_id] = processed_steps

    return processed_data


def export_centerlines_to_csv(centerline_data, output_dir='fluvpy_centerlines', prefix='', params=None):
    """
    Export river channel centerline data to CSV files.

    Args:
        centerline_data: List of centerline data
        output_dir: Output directory
        prefix: File name prefix
        params: Simulation parameter dictionary

    Returns:
        Export directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    if prefix:
        prefix = f"{prefix}_"

    channels_file = os.path.join(output_dir, f"{prefix}channels.csv")
    points_file = os.path.join(output_dir, f"{prefix}centerline_points.csv")
    distribution_file = os.path.join(output_dir, f"{prefix}channel_distribution.csv")

    channel_id_map = {}
    unique_id = 0

    processed_channels = process_centerline_data(centerline_data, params)
    actual_channel_properties = params.get('actual_channel_properties', {}) if params else {}

    # Export channel overview information
    with open(channels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'channel_id',
            'complex_x',
            'complex_z',
            'channel_x',
            'channel_z',
            'global_angle',
            'num_points',
            'channel_sinuosity',
            'target_sinuosity',
            'fcco',
            'algorithm_used',
            'sinuosity_error'
        ])

        for i, channel in enumerate(processed_channels):
            channel_id = channel['channel_id']
            channel_id_map[channel_id] = unique_id
            unique_id += 1

            actual_props = actual_channel_properties.get(channel_id, {})

            if actual_props:
                channel_sinuosity = actual_props.get('actual_sinuosity', 1.0)
                target_sinuosity = actual_props.get('target_sinuosity', channel_sinuosity)
                fcco = actual_props.get('actual_complex_angle', 0.0)
                algorithm_used = actual_props.get('algorithm_used', 'unknown')
                sinuosity_error = abs(channel_sinuosity - target_sinuosity)
            else:
                channel_sinuosity = -1.0
                target_sinuosity = -1.0
                fcco = -999.0
                algorithm_used = 'missing_data'
                sinuosity_error = -1.0

            writer.writerow([
                unique_id - 1,
                channel_id,
                channel['complex_x'],
                channel['complex_z'],
                channel['channel_x'],
                channel['channel_z'],
                channel['global_angle'],
                len(channel['points']),
                channel_sinuosity,
                target_sinuosity,
                fcco,
                algorithm_used,
                sinuosity_error
            ])

    # Export detailed point information
    with open(points_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'channel_id',
            'point_id',
            'local_x',
            'local_y',
            'global_x',
            'global_y',
            'global_z',
            'base_z',
            'width',
            'thickness',
            'local_curvature',
            'global_curvature',
            'relative_position',
            'channel_sinuosity',
            'target_sinuosity',
            'fcco',
            'algorithm_used',
            'sinuosity_error'
        ])

        for channel in processed_channels:
            channel_id = channel['channel_id']
            unique_channel_id = channel_id_map[channel_id]

            actual_props = actual_channel_properties.get(channel_id, {})

            if actual_props:
                channel_sinuosity = actual_props.get('actual_sinuosity', 1.0)
                target_sinuosity = actual_props.get('target_sinuosity', channel_sinuosity)
                fcco = actual_props.get('actual_complex_angle', 0.0)
                algorithm_used = actual_props.get('algorithm_used', 'unknown')
                sinuosity_error = abs(channel_sinuosity - target_sinuosity)
            else:
                channel_sinuosity = -1.0
                target_sinuosity = -1.0
                fcco = -999.0
                algorithm_used = 'missing_data'
                sinuosity_error = -1.0

            for i, point in enumerate(channel['points']):
                writer.writerow([
                    unique_channel_id,
                    channel_id,
                    i,
                    point['local_x'],
                    point['local_y'],
                    point['global_x'],
                    point['global_y'],
                    point['global_z'],
                    point['base_z'],
                    point['width'],
                    point['thickness'],
                    point.get('curvature', 0.0),
                    point['global_curvature'],
                    point.get('rel_position', 0.5),
                    channel_sinuosity,
                    target_sinuosity,
                    fcco,
                    algorithm_used,
                    sinuosity_error
                ])

    # Export channel distribution file
    with open(distribution_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'channel_id',
            'complex_id',
            'channel_index',
            'channel_sinuosity',
            'target_sinuosity',
            'sinuosity_error',
            'fcco',
            'complex_x',
            'complex_z',
            'channel_x',
            'channel_z',
            'avg_width',
            'avg_thickness',
            'actual_width_thickness_ratio',
            'num_points',
            'algorithm_used',
            'data_quality'
        ])

        channel_distribution_data = []

        for channel in processed_channels:
            channel_id = channel['channel_id']
            unique_channel_id = channel_id_map[channel_id]

            try:
                icc_str, ic_str = channel_id.split('_')
                complex_id = int(icc_str)
                channel_index = int(ic_str)
            except ValueError:
                complex_id = 1
                channel_index = 1

            actual_props = actual_channel_properties.get(channel_id, {})

            if actual_props:
                channel_sinuosity = actual_props.get('actual_sinuosity', 1.0)
                target_sinuosity = actual_props.get('target_sinuosity', channel_sinuosity)
                fcco = actual_props.get('actual_complex_angle', 0.0)
                algorithm_used = actual_props.get('algorithm_used', 'unknown')
                actual_width_thickness_ratio = actual_props.get('actual_width_thickness_ratio', 0.0)
                sinuosity_error = abs(channel_sinuosity - target_sinuosity)
                data_quality = 'complete'
            else:
                channel_sinuosity = -1.0
                target_sinuosity = -1.0
                fcco = -999.0
                algorithm_used = 'missing_data'
                actual_width_thickness_ratio = -1.0
                sinuosity_error = -1.0
                data_quality = 'missing_properties'

            # Calculate average width and thickness
            avg_width = 0.0
            avg_thickness = 0.0
            num_points = len(channel['points'])

            if num_points > 0:
                total_width = sum(point['width'] for point in channel['points'])
                total_thickness = sum(point['thickness'] for point in channel['points'])
                avg_width = total_width / num_points
                avg_thickness = total_thickness / num_points

            distribution_row = [
                unique_channel_id,
                channel_id,
                complex_id,
                channel_index,
                channel_sinuosity,
                target_sinuosity,
                sinuosity_error,
                fcco,
                channel['complex_x'],
                channel['complex_z'],
                channel['channel_x'],
                channel['channel_z'],
                avg_width,
                avg_thickness,
                actual_width_thickness_ratio,
                num_points,
                algorithm_used,
                data_quality
            ]

            channel_distribution_data.append(distribution_row)

        # Sort by sinuosity for output
        channel_distribution_data.sort(key=lambda x: x[4] if x[4] >= 0 else 999)

        for row in channel_distribution_data:
            writer.writerow(row)

    return output_dir


def export_migration_data_to_csv(migration_data, output_dir, prefix='', params=None):
    """
    Export river channel migration process data to CSV files.

    Args:
        migration_data: Migration data dictionary
        output_dir: Output directory
        prefix: File name prefix
        params: Simulation parameter dictionary
    """
    if not migration_data:
        return

    os.makedirs(output_dir, exist_ok=True)

    if prefix:
        prefix = f"{prefix}_"

    migration_summary_file = os.path.join(output_dir, f"{prefix}migration_summary.csv")
    migration_points_file = os.path.join(output_dir, f"{prefix}migration_points.csv")

    channel_id_map = {}
    unique_id = 0

    processed_data = process_migration_data(migration_data, params)

    # Export migration summary information
    with open(migration_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'channel_id',
            'migration_step',
            'num_points'
        ])

        for channel_id, steps in processed_data.items():
            for step_data in steps:
                step = step_data['step']
                num_points = len(step_data['centerline'])

                key = (channel_id, step)
                if key not in channel_id_map:
                    channel_id_map[key] = unique_id
                    unique_id += 1

                writer.writerow([
                    channel_id_map[key],
                    channel_id,
                    step,
                    num_points
                ])

    # Export migration point detailed data
    with open(migration_points_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'channel_id',
            'migration_step',
            'point_id',
            'global_x',
            'global_y',
            'global_z',
            'base_z',
            'width',
            'thickness',
            'relative_position',
            'global_curvature'
        ])

        for channel_id, steps in processed_data.items():
            for step_data in steps:
                step = step_data['step']
                key = (channel_id, step)
                unique_channel_id = channel_id_map[key]

                global_x = step_data['centerline']
                global_y = step_data['centerline_y']
                global_z = step_data['global_z']
                base_z = step_data['base_z']

                width = step_data.get('width', [])
                thickness = step_data.get('thickness', [])
                relpos = step_data.get('relpos', [])
                global_curvature = step_data.get('global_curvature', [])

                n_points = len(global_x)

                for i in range(n_points):
                    writer.writerow([
                        unique_channel_id,
                        channel_id,
                        step,
                        i,
                        global_x[i],
                        global_y[i],
                        global_z[i],
                        base_z[i],
                        width[i] if i < len(width) else 0,
                        thickness[i] if i < len(thickness) else 0,
                        relpos[i] if i < len(relpos) else 0.5,
                        global_curvature[i] if i < len(global_curvature) else 0
                    ])


def process_migrated_channels(migrated_channels, params):
    """
    Process migrated channel information using the same coordinate system as voxel rendering.

    Args:
        migrated_channels: List of migrated channels
        params: Simulation parameter dictionary

    Returns:
        List of processed migrated channels
    """
    processed_channels = []
    ymn = params.get('ymn', 0)

    for channel in migrated_channels:
        icc = channel['icc']
        ic = channel['ic']
        step = channel['step']
        x_center = channel['x_center']
        z_top = channel['z_top']
        angle = channel['angle']

        centerline_x = channel['centerline']
        centerline_y = channel['centerline_y']
        width = channel.get('width', [])
        thickness = channel.get('thickness', [])

        n_points = len(centerline_x)
        global_x = []
        global_y = []
        global_z = []
        base_z = []

        for i in range(n_points):
            vx, vy = gpu_rotc_equivalent(
                x_center, ymn, angle, centerline_x[i], centerline_y[i] - ymn
            )
            global_x.append(vx)
            global_y.append(vy)
            global_z.append(z_top)

            if i < len(thickness):
                base_z.append(z_top - thickness[i])
            else:
                base_z.append(z_top - 1.0)

        global_curvature = calculate_global_curvature(global_x, global_y)

        processed_channel = channel.copy()
        processed_channel['global_x'] = global_x
        processed_channel['global_y'] = global_y
        processed_channel['global_z'] = global_z
        processed_channel['base_z'] = base_z
        processed_channel['global_curvature'] = global_curvature

        processed_channels.append(processed_channel)

    return processed_channels


def export_migrated_channels_to_csv(migrated_channels, output_dir, prefix='', params=None):
    """
    Export migrated channel information to CSV files.

    Args:
        migrated_channels: List of migrated channel information
        output_dir: Output directory
        prefix: File name prefix
        params: Simulation parameter dictionary
    """
    if not migrated_channels:
        return

    os.makedirs(output_dir, exist_ok=True)

    if prefix:
        prefix = f"{prefix}_"

    migrated_channels_file = os.path.join(output_dir, f"{prefix}migrated_channels.csv")
    migrated_points_file = os.path.join(output_dir, f"{prefix}migrated_points.csv")

    channel_id_map = {}
    unique_id = 0

    processed_channels = process_migrated_channels(migrated_channels, params)

    # Export migrated channel summary information
    with open(migrated_channels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'complex_id',
            'channel_id',
            'migration_step',
            'x_center',
            'z_top',
            'angle',
            'num_points'
        ])

        for channel in processed_channels:
            key = (channel['icc'], channel['ic'], channel['step'])
            if key not in channel_id_map:
                channel_id_map[key] = unique_id
                unique_id += 1

            writer.writerow([
                channel_id_map[key],
                channel['icc'],
                channel['ic'],
                channel['step'],
                channel['x_center'],
                channel['z_top'],
                channel['angle'],
                len(channel['global_x'])
            ])

    # Export migrated channel point data
    with open(migrated_points_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'unique_channel_id',
            'complex_id',
            'channel_id',
            'migration_step',
            'point_id',
            'global_x',
            'global_y',
            'global_z',
            'base_z',
            'width',
            'thickness',
            'global_curvature'
        ])

        for channel in processed_channels:
            key = (channel['icc'], channel['ic'], channel['step'])
            unique_channel_id = channel_id_map[key]

            global_x = channel['global_x']
            global_y = channel['global_y']
            global_z = channel['global_z']
            base_z = channel['base_z']

            width = channel.get('width', [])
            thickness = channel.get('thickness', [])
            global_curvature = channel['global_curvature']

            n_points = len(global_x)

            for i in range(n_points):
                writer.writerow([
                    unique_channel_id,
                    channel['icc'],
                    channel['ic'],
                    channel['step'],
                    i,
                    global_x[i],
                    global_y[i],
                    global_z[i],
                    base_z[i],
                    width[i] if i < len(width) else 0,
                    thickness[i] if i < len(thickness) else 0,
                    global_curvature[i]
                ])


def export_results_centerlines(results, export_dir='fluvpy_centerlines'):
    """
    Export river channel centerline data and migration process data from simulation results.

    Args:
        results: Simulation results dictionary
        export_dir: Export directory
    """
    start_time = time.time()
    os.makedirs(export_dir, exist_ok=True)

    total_channels = 0
    total_points = 0
    total_migration_channels = 0
    total_migration_steps = 0

    for i in range(1, results.get('params', {}).get('nsim', 1) + 1):
        realization_key = f'realization_{i}'

        if realization_key not in results:
            continue

        realization_data = results[realization_key]
        export_dir_i = os.path.join(export_dir, f"realization_{i}")

        params = realization_data.get('params', {}).copy()

        # Process migrated_channels, extract Z values as main channel information
        if 'migration_info' in realization_data and 'migrated_channels_info' in realization_data['migration_info']:
            main_channels_info = []

            for channel in realization_data['migration_info']['migrated_channels_info']:
                main_channels_info.append({
                    'icc': int(channel['icc']),
                    'ic': int(channel['ic']),
                    'x_center': channel['x_center'],
                    'z_top': channel['z_top'],
                    'angle': channel['angle']
                })

            params['main_channels_info'] = main_channels_info

        # Export centerline data
        if 'centerline_data' in realization_data:
            centerline_data = realization_data['centerline_data']
            export_centerlines_to_csv(centerline_data, export_dir_i, params=params)

            total_channels += len(centerline_data)
            for channel in centerline_data:
                total_points += len(channel['points'])

        # Export migration data
        if 'migration_info' in realization_data and 'migration_data' in realization_data['migration_info']:
            migration_data = realization_data['migration_info']['migration_data']

            total_migration_channels += len(migration_data)
            for channel_id, steps in migration_data.items():
                total_migration_steps += len(steps)

            migration_dir = os.path.join(export_dir_i, "migration")
            export_migration_data_to_csv(migration_data, migration_dir, params=params)

        # Export migrated channel information
        if 'migration_info' in realization_data and 'migrated_channels_info' in realization_data['migration_info']:
            migrated_channels = realization_data['migration_info']['migrated_channels_info']
            migrated_dir = os.path.join(export_dir_i, "migrated_channels")
            export_migrated_channels_to_csv(migrated_channels, migrated_dir, params=params)

    end_time = time.time()
    print(f"Export complete - Channels: {total_channels}, Points: {total_points}, Time: {end_time - start_time:.2f}s")