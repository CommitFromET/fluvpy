"""
fluvpy_export.py
"""
import json,struct
import numpy as np
import os
import csv
import time
import glob
import re
from typing import Dict, Any, Optional, Tuple, List

def get_highest_voxel_number(output_dir='fluvpy_csv_outputs'):
    """
    Get the highest index number from existing voxels_*.csv files.

    Args:
        output_dir: Output directory path

    Returns:
        int: Highest voxel file index number
    """
    if not os.path.exists(output_dir):
        return 0

    pattern = os.path.join(output_dir, "voxels_*.csv")
    files = glob.glob(pattern)

    highest_num = 0
    for file in files:
        match = re.search(r'voxels_(\d+)\.csv', file)
        if match:
            num = int(match.group(1))
            highest_num = max(highest_num, num)

    return highest_num

def export_normalized_voxels(results, output_dir='fluvpy_csv_outputs', include_non_channel=False):
    """
    Export all realization voxel data to standardized CSV files.

    Args:
        results: Dictionary containing simulation results
        output_dir: Output directory
        include_non_channel: Whether to include non-channel region voxels

    Returns:
        int: Number of exported realizations
    """
    import pandas as pd

    total_start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Get current highest index
    start_num = get_highest_voxel_number(output_dir) + 1

    # Check for specified starting index in environment variables
    if "fluvpy_START_NUM" in os.environ:
        try:
            env_start_num = int(os.environ["fluvpy_START_NUM"])
            start_num = max(start_num, env_start_num)
        except ValueError:
            pass

    print(f"Starting CSV export from index {start_num}")
    if not include_non_channel:
        print("Non-channel region filtering mode enabled")

    implementations_exported = 0
    total_exported_cells = 0

    # Export each realization
    for i in range(1, results.get('params', {}).get('nsim', 1) + 1):
        realization_key = f'realization_{i}'
        file_index = start_num + i - 1

        if realization_key not in results or 'channel' not in results[realization_key]:
            continue

        # Get data and parameters
        channel = results[realization_key]['channel']
        params = results[realization_key].get('params', results.get('params', {}))
        porosity = results[realization_key].get('porosity', None)

        # Extract grid parameters
        nx, ny, nz = params['nx'], params['ny'], params['nz']

        filename = os.path.join(output_dir, f'voxels_{file_index}.csv')
        print(f"Exporting realization {i} to {os.path.basename(filename)}")

        # Generate coordinate grids
        z_coords, y_coords, x_coords = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij'
        )

        # Rearrange dimensions to match original data layout
        x_coords = x_coords.transpose(2, 1, 0)
        y_coords = y_coords.transpose(2, 1, 0)
        z_coords = z_coords.transpose(2, 1, 0)

        # Calculate g values
        g_values = np.zeros_like(channel, dtype=np.int8)
        g_values = np.where(channel == 0, 0,
                            np.where(channel >= 20000, 3,
                                     np.where(channel >= 10000, 2, 1)))

        # Process porosity data
        has_porosity = False
        working_porosity = None

        if porosity is not None and params.get('ipor', 0) == 1 and porosity.shape == channel.shape:
            has_porosity = True
            working_porosity = np.copy(porosity)

            # Set non-channel regions to 0, preserve original values in channel regions
            river_mask = channel > 0
            working_porosity[~river_mask] = 0.0

            # Count invalid porosity values
            invalid_river_mask = (working_porosity <= -9.0) & river_mask
            if np.any(invalid_river_mask):
                invalid_count = np.sum(invalid_river_mask)
                print(f"  Found {invalid_count} invalid porosity values (preserving original)")

        # Apply filtering
        if not include_non_channel:
            valid_mask = g_values > 0
        else:
            valid_mask = np.ones_like(g_values, dtype=bool)

        # Build data dictionary
        valid_indices = np.where(valid_mask)

        data_dict = {
            'x': x_coords[valid_indices],
            'y': y_coords[valid_indices],
            'z': z_coords[valid_indices],
            'g': g_values[valid_indices],
            'channel_id': channel[valid_indices]
        }

        if has_porosity:
            data_dict['porosity'] = working_porosity[valid_indices]

        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False, float_format='%.6f', chunksize=50000)

        # Statistics
        total_cells = nx * ny * nz
        exported_cells = len(df)
        total_exported_cells += exported_cells

        unique_g, counts_g = np.unique(data_dict['g'], return_counts=True)
        g_stats = dict(zip(unique_g, counts_g))

        river_cells = g_stats.get(1, 0)
        levee_cells = g_stats.get(2, 0)
        crevasse_cells = g_stats.get(3, 0)
        non_channel_cells = g_stats.get(0, 0) if include_non_channel else 0

        # Output statistics
        if include_non_channel:
            print(f"  Complete: {exported_cells:,}/{total_cells:,} cells")
            print(f"  Statistics: Channel {river_cells:,}, Levee {levee_cells:,}, Crevasse {crevasse_cells:,}, Non-channel {non_channel_cells:,}")
        else:
            compression_ratio = (1 - exported_cells / total_cells) * 100
            print(f"  Complete: {exported_cells:,} cells exported, {total_cells - exported_cells:,} skipped")
            print(f"  Statistics: Channel {river_cells:,}, Levee {levee_cells:,}, Crevasse {crevasse_cells:,}")
            print(f"  Compression: Reduced data volume by {compression_ratio:.1f}%")

        # Porosity statistics
        if has_porosity and 'porosity' in data_dict:
            por_values = data_dict['porosity']
            por_valid = por_values[por_values > 0]

            if len(por_valid) > 0:
                print(f"  Porosity: {len(por_valid):,} valid values, range [{por_valid.min():.3f}, {por_valid.max():.3f}], mean {por_valid.mean():.3f}")

        implementations_exported += 1

    # Time statistics
    total_elapsed = time.time() - total_start_time
    mode_desc = "(channel types only)" if not include_non_channel else "(all types included)"

    print(f"All voxel data export complete {mode_desc}, exported {implementations_exported} realizations")
    print(f"Total exported voxels: {total_exported_cells:,}")
    print(f"Total elapsed time: {total_elapsed:.2f}s")

    if total_elapsed > 0:
        overall_speed = total_exported_cells / total_elapsed
        print(f"Average export speed: {overall_speed:,.0f} voxels/s")

    if not include_non_channel:
        print("Non-channel region (g=0) voxels have been filtered out")

    return implementations_exported

def export_vegetation_distribution(vegetation_generator, params, update_step=0,
                                   output_dir='vegetation_distributions',
                                   prefix='vegetation'):
    """
    Export vegetation distribution data to CSV file.

    Args:
        vegetation_generator: VegetationPatchGenerator instance
        params: Simulation parameters dictionary
        update_step: Current update step number
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        str: Path to exported file, None if failed
    """
    if vegetation_generator is None:
        print("Vegetation generator is empty, skipping vegetation data export")
        return None

    vegetation_map = vegetation_generator.get_vegetation_map()
    if vegetation_map is None:
        print("Vegetation distribution data is empty, skipping vegetation data export")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Get grid parameters
    nx = params.get('nx', 250)
    ny = params.get('ny', 250)
    xmn = params.get('xmn', 0)
    ymn = params.get('ymn', 0)
    xsiz = params.get('xsiz', 24)
    ysiz = params.get('ysiz', 24)

    global_step = params.get('global_migration_step_count', 0)

    # Build filename
    if update_step == 0:
        filename = f"{prefix}_initial_global{global_step:04d}.csv"
    else:
        filename = f"{prefix}_update{update_step:03d}_global{global_step:04d}.csv"

    filepath = os.path.join(output_dir, filename)

    print(f"Exporting vegetation distribution data to: {filename}")
    print(f"  Grid size: {nx} × {ny}")
    print(f"  Coordinate range: x[{xmn:.1f}, {xmn + (nx - 1) * xsiz:.1f}], y[{ymn:.1f}, {ymn + (ny - 1) * ysiz:.1f}]")

    start_time = time.time()

    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'vegetation_influence', 'grid_i', 'grid_j'])

            total_points = nx * ny
            written_count = 0

            # Iterate through all grid points
            for i in range(nx):
                for j in range(ny):
                    x = xmn + i * xsiz
                    y = ymn + j * ysiz
                    vegetation_value = vegetation_map[i, j]

                    writer.writerow([
                        f"{x:.6f}",
                        f"{y:.6f}",
                        f"{vegetation_value:.6f}",
                        i,
                        j
                    ])

                    written_count += 1

                # Progress display
                if (i + 1) % 25 == 0 or i == nx - 1:
                    progress = (i + 1) / nx * 100
                    print(f"  Export progress: {progress:.1f}% ({written_count}/{total_points} points)")

        end_time = time.time()

        veg_stats = vegetation_generator.get_vegetation_statistics()

        print(f"Vegetation distribution export complete:")
        print(f"  File: {filename}")
        print(f"  Points: {written_count:,}")
        print(f"  Time elapsed: {end_time - start_time:.2f}s")
        if veg_stats:
            print(f"  Influence value range: [{veg_stats['min']:.4f}, {veg_stats['max']:.4f}]")
            print(f"  Statistics: mean={veg_stats['mean']:.4f}, std={veg_stats['std']:.4f}")

        return filepath

    except Exception as e:
        print(f"Vegetation distribution export failed: {e}")
        return None

def export_vegetation_evolution_summary(vegetation_generator, params,
                                        output_dir='vegetation_distributions'):
    """
    Export vegetation evolution summary statistics.

    Args:
        vegetation_generator: VegetationPatchGenerator instance
        params: Simulation parameters dictionary
        output_dir: Output directory

    Returns:
        None
    """
    if vegetation_generator is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    global_step = params.get('global_migration_step_count', 0)

    filename = f"vegetation_evolution_summary_global{global_step:04d}.csv"
    filepath = os.path.join(output_dir, filename)

    print(f"Exporting vegetation evolution summary to: {filename}")

    try:
        veg_stats = vegetation_generator.get_vegetation_statistics()

        if veg_stats is None:
            print("Unable to obtain vegetation statistics")
            return

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['parameter', 'value'])

            # Write basic parameters
            writer.writerow(['global_migration_step', global_step])
            writer.writerow(['nx', params.get('nx', 250)])
            writer.writerow(['ny', params.get('ny', 250)])
            writer.writerow(['xmin', params.get('xmn', 0)])
            writer.writerow(['ymin', params.get('ymn', 0)])
            writer.writerow(['xsize', params.get('xsiz', 24)])
            writer.writerow(['ysize', params.get('ysiz', 24)])

            # Write vegetation parameters
            writer.writerow(['vegetation_influence_strength',
                             params.get('vegetation_influence_strength', 5.0)])
            writer.writerow(['vegetation_update_interval',
                             params.get('vegetation_update_interval', 50)])
            writer.writerow(['vegetation_evolution_factor',
                             params.get('vegetation_evolution_factor', 0.15)])

            # Write statistics
            for key, value in veg_stats.items():
                writer.writerow([f'vegetation_{key}', f'{value:.6f}'])

        print(f"Vegetation evolution summary export complete: {filename}")

    except Exception as e:
        print(f"Vegetation evolution summary export failed: {e}")

def export_vegetation_comparison(vegetation_maps_history, params,
                                 output_dir='vegetation_distributions'):
    """
    Export vegetation distribution change comparison data.

    Args:
        vegetation_maps_history: List of vegetation distribution history
        params: Simulation parameters dictionary
        output_dir: Output directory

    Returns:
        None
    """
    if not vegetation_maps_history or len(vegetation_maps_history) < 2:
        print("Insufficient vegetation history data, skipping comparison export")
        return

    os.makedirs(output_dir, exist_ok=True)
    global_step = params.get('global_migration_step_count', 0)

    filename = f"vegetation_changes_global{global_step:04d}.csv"
    filepath = os.path.join(output_dir, filename)

    print(f"Exporting vegetation change comparison to: {filename}")

    try:
        nx = params.get('nx', 250)
        ny = params.get('ny', 250)
        xmn = params.get('xmn', 0)
        ymn = params.get('ymn', 0)
        xsiz = params.get('xsiz', 24)
        ysiz = params.get('ysiz', 24)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Build dynamic header
            header = ['x', 'y', 'grid_i', 'grid_j']
            for step, _ in vegetation_maps_history:
                if step == 0:
                    header.append('initial_vegetation')
                else:
                    header.append(f'update{step:03d}_vegetation')

            # Add change columns
            if len(vegetation_maps_history) >= 2:
                header.append('total_change')
                header.append('max_change_step')

            writer.writerow(header)

            # Iterate through all grid points
            for i in range(nx):
                for j in range(ny):
                    x = xmn + i * xsiz
                    y = ymn + j * ysiz

                    row = [f"{x:.6f}", f"{y:.6f}", i, j]

                    # Collect all vegetation values for this point
                    vegetation_values = []
                    for step, veg_map in vegetation_maps_history:
                        veg_value = veg_map[i, j]
                        row.append(f"{veg_value:.6f}")
                        vegetation_values.append(veg_value)

                    # Calculate change amounts
                    if len(vegetation_values) >= 2:
                        initial_value = vegetation_values[0]
                        final_value = vegetation_values[-1]
                        total_change = abs(final_value - initial_value)
                        row.append(f"{total_change:.6f}")

                        # Find step with maximum change
                        max_change = 0
                        max_change_step = 0
                        for k in range(1, len(vegetation_values)):
                            change = abs(vegetation_values[k] - vegetation_values[k - 1])
                            if change > max_change:
                                max_change = change
                                max_change_step = vegetation_maps_history[k][0]
                        row.append(max_change_step)

                    writer.writerow(row)

        print(f"Vegetation change comparison export complete: {filename}")
        print(f"  Comparison steps: {len(vegetation_maps_history)}")

    except Exception as e:
        print(f"Vegetation change comparison export failed: {e}")


def save_results_for_web(results, output_dir="web_data", isim=1):
    """
    Save fluvpy results in format required for web visualization (consistent with PyVista function).
    Fixed Infinity/NaN values causing JSON serialization issues.

    Parameters:
        results: fluvpy results dictionary
        output_dir: Output directory
        isim: Realization number
    """
    import os
    import json
    import numpy as np

    def clean_numeric_value(value):
        """Clean numeric values, replace infinity and NaN values"""
        if np.isinf(value):
            return 1e10 if value > 0 else -1e10  # Replace infinity with large numbers
        elif np.isnan(value):
            return 0.0  # Replace NaN with 0
        else:
            return float(value)

    def clean_data_structure(obj):
        """Recursively clean all invalid numeric values in data structure"""
        if isinstance(obj, dict):
            return {key: clean_data_structure(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [clean_data_structure(item) for item in obj]
        elif isinstance(obj, (int, float, np.number)):
            return clean_numeric_value(float(obj))
        else:
            return obj

    os.makedirs(output_dir, exist_ok=True)

    # Get data for specified realization (consistent with PyVista function)
    realization = results[f'realization_{isim}']
    channel = realization['channel']

    # Get parameters (consistent with PyVista function)
    if 'params' in results:
        params = results['params']
    elif 'params' in realization:
        params = realization['params']
    else:
        # Use default parameters
        params = {
            'xmn': 0, 'ymn': 0, 'zmn': 0,
            'xsiz': 15.0, 'ysiz': 15.0, 'zsiz': 2.5
        }

    # Extract grid dimensions
    nx, ny, nz = channel.shape

    # Get grid parameters and ensure numeric validity
    xmn = clean_numeric_value(params.get('xmn', 0))
    ymn = clean_numeric_value(params.get('ymn', 0))
    zmn = clean_numeric_value(params.get('zmn', 0))
    xsiz = clean_numeric_value(params.get('xsiz', 15.0))
    ysiz = clean_numeric_value(params.get('ysiz', 15.0))
    zsiz = clean_numeric_value(params.get('zsiz', 2.5))

    # Prevent division by zero errors
    if xsiz <= 0: xsiz = 15.0
    if ysiz <= 0: ysiz = 15.0
    if zsiz <= 0: zsiz = 2.5

    # Calculate physical dimensions
    x_length = nx * xsiz
    y_length = ny * ysiz
    z_length = nz * zsiz

    # Calculate Z-axis scaling factor (consistent with PyVista function), prevent division by zero
    if z_length > 0:
        z_scale_factor = x_length / (2 * z_length)
    else:
        z_scale_factor = 1.0

    adjusted_z_siz = zsiz * z_scale_factor

    # Ensure all calculation results are valid numerics
    x_length = clean_numeric_value(x_length)
    y_length = clean_numeric_value(y_length)
    z_length = clean_numeric_value(z_length)
    z_scale_factor = clean_numeric_value(z_scale_factor)
    adjusted_z_siz = clean_numeric_value(adjusted_z_siz)

    print(f"Grid dimensions: {nx}×{ny}×{nz}")
    print(f"Physical size: {x_length:.0f}×{y_length:.0f}×{z_length:.0f} meters")
    print(f"Z-axis scale factor: {z_scale_factor:.3f}")
    print(f"Adjusted Z spacing: {adjusted_z_siz:.3f}")

    # Extract voxel data (classification consistent with PyVista function)
    channel_voxels = []
    levee_voxels = []
    crevasse_voxels = []

    # Statistics counters
    channel_count = 0
    levee_count = 0
    crevasse_count = 0

    # Iterate through 3D array, extract non-zero voxels
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value = channel[i, j, k]

                # Check and clean values
                if np.isinf(value) or np.isnan(value):
                    continue  # Skip invalid values

                if value > 0:  # Only process non-zero values
                    # Calculate physical coordinates
                    x_pos = xmn + i * xsiz
                    y_pos = ymn + j * ysiz
                    z_pos = zmn + k * adjusted_z_siz  # Use adjusted z spacing

                    # Clean all coordinate values
                    x_pos = clean_numeric_value(x_pos)
                    y_pos = clean_numeric_value(y_pos)
                    z_pos = clean_numeric_value(z_pos)
                    value = clean_numeric_value(value)

                    voxel_data = {
                        'i': int(i), 'j': int(j), 'k': int(k),  # Grid indices
                        'x': x_pos, 'y': y_pos, 'z': z_pos,  # Physical coordinates
                        'value': value,
                        'size': [xsiz, ysiz, adjusted_z_siz]  # Voxel dimensions
                    }

                    # Classification according to PyVista function standards
                    if value < 10000:  # Channel voxels
                        channel_voxels.append(voxel_data)
                        channel_count += 1
                    elif value < 20000:  # Levee voxels (10000 <= value < 20000)
                        levee_voxels.append(voxel_data)
                        levee_count += 1
                    else:  # Crevasse voxels (value >= 20000)
                        crevasse_voxels.append(voxel_data)
                        crevasse_count += 1

    print(f"Voxel counts - Channel: {channel_count}, Levee: {levee_count}, Crevasse: {crevasse_count}")

    # Build data structure (consistent with PyVista function)
    web_data = {
        'metadata': {
            'realization': isim,
            'grid_dimensions': {'nx': nx, 'ny': ny, 'nz': nz},
            'physical_size': {
                'x_length': x_length,
                'y_length': y_length,
                'z_length': z_length
            },
            'grid_spacing': {
                'xsiz': xsiz,
                'ysiz': ysiz,
                'zsiz': zsiz,
                'adjusted_zsiz': adjusted_z_siz  # Adjusted z spacing
            },
            'origin': {
                'xmn': xmn,
                'ymn': ymn,
                'zmn': zmn
            },
            'scale_factors': {
                'z_scale_factor': z_scale_factor
            },
            'statistics': {
                'total_grid_cells': nx * ny * nz,
                'channel_count': channel_count,
                'levee_count': levee_count,
                'crevasse_count': crevasse_count,
                'total_active_voxels': channel_count + levee_count + crevasse_count,
                'coverage_percent': clean_numeric_value(
                    (channel_count + levee_count + crevasse_count) / (nx * ny * nz) * 100)
            },
            'value_ranges': {
                'channel_range': [1, 9999],
                'levee_range': [10000, 19999],
                'crevasse_range': [20000, 1000000]  # Use specific numbers instead of inf
            },
            'colors': {
                'channel': '#2E86AB',  # Blue
                'levee': '#A23B72',  # Purple
                'crevasse': '#F18F01'  # Orange
            }
        },
        'voxel_data': {
            'channels': channel_voxels,
            'levees': levee_voxels,
            'crevasses': crevasse_voxels
        }
    }

    # Clean entire data structure
    web_data = clean_data_structure(web_data)

    # Save as JSON file
    output_file = os.path.join(output_dir, 'fluvpy_data.json')

    try:
        # Test JSON serialization, use custom encoder to handle special values
        class SafeJSONEncoder(json.JSONEncoder):
            def encode(self, obj):
                if isinstance(obj, float):
                    if np.isinf(obj):
                        return str(1e10 if obj > 0 else -1e10)
                    elif np.isnan(obj):
                        return "0.0"
                return super().encode(obj)

        # Use safe encoder
        json_str = json.dumps(web_data, indent=2, cls=SafeJSONEncoder)
        file_size = len(json_str.encode('utf-8'))

        print(f"JSON validation successful, file size: {file_size / 1024 / 1024:.1f}MB")

        if file_size > 100 * 1024 * 1024:  # Over 100MB
            print(f"Warning: Large file size ({file_size / 1024 / 1024:.1f}MB)")

            # Save metadata and voxel data separately
            metadata_file = os.path.join(output_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(web_data['metadata'], f, indent=2, cls=SafeJSONEncoder)

            for voxel_type, voxel_list in web_data['voxel_data'].items():
                if voxel_list:
                    voxel_file = os.path.join(output_dir, f'{voxel_type}.json')
                    with open(voxel_file, 'w') as f:
                        json.dump(voxel_list, f, cls=SafeJSONEncoder)
                    print(f"Saved {voxel_type}: {len(voxel_list)} voxels")
        else:
            with open(output_file, 'w') as f:
                f.write(json_str)
            print(f"Data saved to: {output_file} (Size: {file_size / 1024 / 1024:.1f}MB)")

        # Verify generated JSON file
        try:
            with open(output_file, 'r') as f:
                test_load = json.load(f)
            print("JSON validation: SUCCESS - File can be loaded correctly")
        except Exception as e:
            print(f"JSON validation FAILED: {e}")

        # Print statistics
        total_voxels = channel_count + levee_count + crevasse_count
        print(f"\nVisualization Data Summary:")
        print(f"  Realization: {isim}")
        print(f"  Grid: {nx}×{ny}×{nz} = {nx * ny * nz:,} cells")
        print(f"  Active voxels: {total_voxels:,} ({(total_voxels / (nx * ny * nz) * 100):.1f}%)")
        print(f"  Channel voxels: {channel_count:,}")
        print(f"  Levee voxels: {levee_count:,}")
        print(f"  Crevasse voxels: {crevasse_count:,}")

        if total_voxels > 0:
            # Calculate value ranges
            all_values = [v['value'] for voxel_list in web_data['voxel_data'].values() for v in voxel_list]
            if all_values:
                min_val, max_val = min(all_values), max(all_values)
                unique_values = len(set(all_values))
                print(f"  Value range: {min_val:.0f} - {max_val:.0f}")
                print(f"  Unique values: {unique_values}")

    except Exception as e:
        print(f"Save failed: {e}")
        print(f"Error details: {type(e).__name__}")

        # Try to identify problematic data
        try:
            # Test each part
            json.dumps(web_data['metadata'])
            print("Metadata JSON: OK")
        except Exception as meta_e:
            print(f"Metadata JSON error: {meta_e}")

        for voxel_type, voxel_list in web_data['voxel_data'].items():
            try:
                json.dumps(voxel_list[:10])  # Test first 10 items
                print(f"{voxel_type} JSON: OK")
            except Exception as voxel_e:
                print(f"{voxel_type} JSON error: {voxel_e}")

        return None

    return web_data


def save_optimized_envelope_surfaces(results: Dict[str, Any], output_dir: str = "web_data", isim: int = 1,
                                     use_binary: bool = True):
    """
    Envelope surface export.

    Parameters:
        results: fluvpy results dictionary
        output_dir: Output directory
        isim: Realization number
        use_binary: Whether to use binary format
    """
    import os
    import json
    import numpy as np
    import struct

    def analyze_data_first(channel):
        """First analyze data to understand actual complexity"""
        nx, ny, nz = channel.shape
        print(f"Analyzing data structure: {nx}×{ny}×{nz}")

        # Classify data
        channel_mask = (channel > 0) & (channel < 10000)
        levee_mask = (channel >= 10000) & (channel < 20000)
        crevasse_mask = channel >= 20000

        total_voxels = np.sum(channel_mask) + np.sum(levee_mask) + np.sum(crevasse_mask)
        sparsity = total_voxels / (nx * ny * nz)

        print(f"Total active voxels: {total_voxels:,}")
        print(f"Grid sparsity: {sparsity:.1%}")

        # Estimate surface complexity
        def estimate_surface_complexity(mask, name):
            if not np.any(mask):
                return 0, 0

            positions = np.where(mask)
            sample_size = min(1000, len(positions[0]))
            sample_indices = np.random.choice(len(positions[0]), sample_size, replace=False)

            exposed_faces = 0
            neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

            for idx in sample_indices:
                x, y, z = positions[0][idx], positions[1][idx], positions[2][idx]
                for dx, dy, dz in neighbors:
                    nx_pos, ny_pos, nz_pos = x + dx, y + dy, z + dz
                    is_same = False
                    if (0 <= nx_pos < nx and 0 <= ny_pos < ny and 0 <= nz_pos < nz):
                        is_same = mask[nx_pos, ny_pos, nz_pos]
                    if not is_same:
                        exposed_faces += 1

            avg_exposed = exposed_faces / sample_size
            total_exposed = avg_exposed * np.sum(mask)

            print(f"{name}: {np.sum(mask):,} voxels, ~{total_exposed:,.0f} exposed faces")
            return int(total_exposed), avg_exposed

        channel_faces, _ = estimate_surface_complexity(channel_mask, "Channel")
        levee_faces, _ = estimate_surface_complexity(levee_mask, "Levee")
        crevasse_faces, _ = estimate_surface_complexity(crevasse_mask, "Crevasse")

        total_faces = channel_faces + levee_faces + crevasse_faces
        print(f"Total estimated faces: {total_faces:,}")
        print(f"As triangles: {total_faces * 2:,}")

        # File size estimation
        if use_binary:
            # Binary format: 12 bytes per vertex, 12 bytes per face
            estimated_size = (total_faces * 1.5 * 12 + total_faces * 2 * 12) / (1024 * 1024)
            print(f"Estimated binary file size: {estimated_size:.1f} MB")
        else:
            # JSON format: much larger overhead
            estimated_size = (total_faces * 1.5 * 50 + total_faces * 2 * 30) / (1024 * 1024)
            print(f"Estimated JSON file size: {estimated_size:.1f} MB")

        return total_faces, estimated_size

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    realization = results[f'realization_{isim}']
    channel = realization['channel']
    nx, ny, nz = channel.shape

    print("=" * 60)
    print("OPTIMIZED ENVELOPE SURFACE EXPORT")
    print("=" * 60)

    # Analyze data first
    estimated_faces, estimated_size = analyze_data_first(channel)

    if estimated_size > 100:
        print(f"\n⚠️  WARNING: File will be very large ({estimated_size:.1f}MB)")
        print("This is normal for complex river systems!")
        print("Consider using binary format or mesh simplification.")

    proceed = input(f"\nProceed with export? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Export cancelled.")
        return None

    # Reclassify data
    channel_classified = np.zeros_like(channel, dtype=np.int32)
    channel_classified[(channel > 0) & (channel < 10000)] = 1
    channel_classified[(channel >= 10000) & (channel < 20000)] = 2
    channel_classified[channel >= 20000] = 3

    # Define voxel face vertices (optimized version - using integer coordinates)
    face_vertices = {
        'front': [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],  # Z-
        'back': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)],  # Z+
        'left': [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],  # X-
        'right': [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],  # X+
        'bottom': [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],  # Y-
        'top': [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]  # Y+
    }

    neighbor_offsets = {
        'front': (0, 0, -1), 'back': (0, 0, 1), 'left': (-1, 0, 0),
        'right': (1, 0, 0), 'bottom': (0, -1, 0), 'top': (0, 1, 0)
    }

    def extract_envelope_optimized(facies_id, facies_name):
        """Optimized envelope extraction"""
        print(f"\nExtracting {facies_name} envelope...")

        # Find all voxels of this facies
        voxel_mask = (channel_classified == facies_id)
        if not np.any(voxel_mask):
            return [], []

        positions = np.where(voxel_mask)
        total_voxels = len(positions[0])
        print(f"Processing {total_voxels:,} voxels...")

        # Use dictionary for fast neighbor lookup
        voxel_set = set(zip(positions[0], positions[1], positions[2]))

        vertices = []
        faces = []
        vertex_dict = {}  # Vertex deduplication

        processed = 0
        for i, j, k in zip(positions[0], positions[1], positions[2]):
            processed += 1
            if processed % 10000 == 0:
                print(f"  Progress: {processed:,}/{total_voxels:,} ({processed / total_voxels * 100:.1f}%)")

            # Check 6 faces
            for face_name, face_verts in face_vertices.items():
                # Check neighbor
                di, dj, dk = neighbor_offsets[face_name]
                neighbor_pos = (i + di, j + dj, k + dk)

                # If neighbor is not of same type, this face is exposed
                is_neighbor_same = (neighbor_pos in voxel_set)
                if not is_neighbor_same:
                    # Add this face
                    face_indices = []
                    for vi, vj, vk in face_verts:
                        world_pos = (i + vi, j + vj, k + vk)
                        if world_pos not in vertex_dict:
                            vertex_dict[world_pos] = len(vertices)
                            vertices.append(world_pos)
                        face_indices.append(vertex_dict[world_pos])

                    # Convert to triangles (optimized order to ensure correct normals)
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])

        print(f"  Result: {len(vertices):,} vertices, {len(faces):,} triangles")
        return vertices, faces

    # Process each facies
    facies_data = []
    facies_defs = [
        {"id": 1, "name": "Channel", "color": "#2E86AB"},
        {"id": 2, "name": "Levee", "color": "#A23B72"},
        {"id": 3, "name": "Crevasse", "color": "#F18F01"},
    ]

    for facies in facies_defs:
        vertices, faces = extract_envelope_optimized(facies["id"], facies["name"])
        if vertices and faces:
            facies_data.append({
                "id": facies["id"],
                "name": facies["name"],
                "color": facies["color"],
                "vertices": vertices,
                "faces": faces
            })

    if not facies_data:
        print("No data to export!")
        return None

    # Save data
    if use_binary:
        # Binary format - significantly reduces file size
        output_file = os.path.join(output_dir, f'envelope_{isim}.bin')
        save_binary_format(facies_data, output_file, isim)
    else:
        # JSON format
        output_file = os.path.join(output_dir, f'envelope_{isim}.json')
        save_json_format(facies_data, output_file, isim)

    return output_file


def save_binary_format(facies_data, output_file, isim):
    """Save in compact binary format"""
    with open(output_file, 'wb') as f:
        # File header
        f.write(b'FLVP')  # Magic number
        f.write(struct.pack('I', 1))  # Version
        f.write(struct.pack('I', isim))  # Realization number
        f.write(struct.pack('I', len(facies_data)))  # Number of surfaces

        for surface in facies_data:
            vertices = surface['vertices']
            faces = surface['faces']

            # Surface header
            name_bytes = surface['name'].encode('utf-8')[:32].ljust(32, b'\0')
            f.write(name_bytes)
            f.write(struct.pack('I', surface['id']))
            f.write(struct.pack('I', len(vertices)))
            f.write(struct.pack('I', len(faces)))

            # Color (RGB)
            color_hex = surface['color'].lstrip('#')
            r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
            f.write(struct.pack('BBB', r, g, b))
            f.write(b'\0')  # Alignment

            # Vertex data (3 floats per vertex)
            for vertex in vertices:
                f.write(struct.pack('fff', float(vertex[0]), float(vertex[1]), float(vertex[2])))

            # Face data (3 uint32 per face)
            for face in faces:
                f.write(struct.pack('III', face[0], face[1], face[2]))

    file_size = os.path.getsize(output_file)
    print(f"\n✅ Binary export complete:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size / 1024 / 1024:.2f} MB")


def save_json_format(facies_data, output_file, isim):
    """Save in JSON format (compatible with existing viewers)"""
    json_data = {
        "metadata": {
            "realization": isim,
            "data_type": "optimized_envelope_surfaces",
            "format_version": "1.0"
        },
        "surfaces": []
    }

    total_vertices = 0
    total_faces = 0

    for surface in facies_data:
        surface_data = {
            "facies_id": surface["id"],
            "facies_name": surface["name"],
            "color": surface["color"],
            "vertices": [[float(v[0]), float(v[1]), float(v[2])] for v in surface["vertices"]],
            "faces": surface["faces"],
            "statistics": {
                "vertex_count": len(surface["vertices"]),
                "face_count": len(surface["faces"])
            }
        }
        json_data["surfaces"].append(surface_data)
        total_vertices += len(surface["vertices"])
        total_faces += len(surface["faces"])

    json_data["metadata"]["total_statistics"] = {
        "total_vertices": total_vertices,
        "total_faces": total_faces
    }

    with open(output_file, 'w') as f:
        json.dump(json_data, f, separators=(',', ':'))  # Compact format

    file_size = os.path.getsize(output_file)
    print(f"\n✅ JSON export complete:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
    print(f"   Vertices: {total_vertices:,}")
    print(f"   Faces: {total_faces:,}")