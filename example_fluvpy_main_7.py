"""
main.py

"""
import time
import argparse

try:
    from sub_modules.engine import engine
    from sub_modules.visualize import fluvpy_visualize
    from sub_modules.export import fluvpy_export
    from sub_modules.visualize.fluvpy_visualize import visualize_fluvpy_results
    from sub_modules.export.fluvpy_export import export_normalized_voxels,save_optimized_envelope_surfaces
    from sub_modules.export.centerline_export import export_results_centerlines

    print("All modules imported successfully!")
except ImportError as e:
    print(f"Module import failed: {e}")
    print("Please check sub_modules package structure and __init__.py files")
    exit(1)

def export_simulation_results(results, output_dir='fluvpy_csv_outputs', export_centerlines=True, centerlines_dir='fluvpy_centerlines'):
    """Export simulation results to CSV files"""
    print("\nStarting simulation results export...")
    import os
    os.makedirs(output_dir, exist_ok=True)
    print("Exporting normalized voxel data...")
    #num_exported = export_normalized_voxels(results, output_dir)
    #web_data = save_optimized_envelope_surfaces(results)
    #print(f"Exported {num_exported} realization normalized voxel datasets")
    print("\nAll export tasks completed")


def main(interactive_vis=False, vis_figsize=(10, 8), vis_dpi=100, save_path=None,
         journal_style='nature', language='zh', export_csv=True, export_dir='fluvpy_csv_outputs',
         generate_parameter_distribution=False,
         export_centerlines=False, centerlines_dir='fluvpy_centerlines',
         visualize_porosity=False,
         visualize_porosity_distribution=False,
         use_gpu_migration=True,
         export_individual_plots=False,
         enable_distribution_analysis=False,
         vegetation_seed_offset=45678,
         export_vegetation=False,
         vegetation_export_dir='vegetation_distributions',
         export_vegetation_evolution=False,
         export_vegetation_summary=True
         ):
    """
    fluvpy main function for running simulation and visualizing results

    Parameters:
        interactive_vis: Enable interactive visualization
        vis_figsize: Visualization figure size
        vis_dpi: Visualization DPI
        save_path: Result save path
        journal_style: Visualization style, options: 'nature', 'science', 'geology'
        language: Display language, 'zh' for Chinese, 'en' for English
        export_csv: Export CSV data
        export_dir: CSV export directory
        generate_parameter_distribution: Generate parameter distribution plots
        export_centerlines: Export channel centerline data
        centerlines_dir: Centerline data export directory
        visualize_porosity: Display porosity visualization
        visualize_porosity_distribution: Display porosity distribution statistics plots
        use_gpu_migration: Use GPU for channel migration computation
        export_individual_plots: Export individual subplots
        enable_distribution_analysis: Enable distribution analysis
        vegetation_seed_offset: Vegetation seed offset parameter
        export_vegetation: Export vegetation distribution data
        vegetation_export_dir: Vegetation data export directory
        export_vegetation_evolution: Export vegetation evolution comparison data
        export_vegetation_summary: Export vegetation evolution summary
    """
    # Model parameter settings
    nx = 250  # Number of grids in X direction (model width direction grid count)
    ny = 250  # Number of grids in Y direction (model length direction grid count)
    nz = 250  # Number of grids in Z direction (model depth direction grid count)
    lx = 6000  # Physical length in X direction (meters)
    ly = 6000  # Physical length in Y direction (meters)
    lz = 250  # Physical thickness in Z direction (meters)

    xsiz = lx / nx  # Grid spacing in X direction (meters/grid)
    ysiz = ly / ny  # Grid spacing in Y direction (meters/grid)
    zsiz = lz / nz  # Grid spacing in Z direction (meters/grid)

    # Parameter distribution definitions
    # Triangular distribution parameters [minimum, mode, maximum]
    fcct_tri = [10, 12, 13]  # Complex thickness distribution
    fccwtr_tri = [1.5, 1.6, 1.7]  # Complex width-to-thickness ratio distribution
    fccntg_tri = [1, 1, 1]  # Complex net-to-gross ratio distribution
    fcco_tri0 = [0, 0, 0]  # Complex orientation angle
    sinuosity_custom = {  # 河道弯曲度自定义区间分布
        'type': 'custom',
        'ranges': [(1.3, 1.4), (1.4, 1.5), (1.5, 1.6)],  # 弯曲度区间
        'weights': [0.45, 0.45, 0.1]  # 选择每个区间的权重
    }
    sinuosity_tri = [1.3, 1.4, 1.5]  # Channel sinuosity triangular distribution
    fcpor_base_tri = [0.1, 0.225, 0.35]  # Porosity base value distribution
    fcpor_range_tri = [0.01, 0.015, 0.02]  # Porosity range distribution

    base_seed = 2981325  # Base seed
    vegetation_seed = base_seed + vegetation_seed_offset  # Independent vegetation seed

    # Build simulation parameter dictionary
    params = {
        'seed': base_seed,
        'use_gpu': True,  # Enable GPU acceleration
        'nx': nx, 'ny': ny, 'nz': nz,
        'xmn': 0, 'ymn': 0, 'zmn': 1,# Starting coordinates in each direction (meters)
        'xsiz': xsiz, 'ysiz': ysiz, 'zsiz': zsiz,
        'nsim': 1,  # Number of simulation realizations
        'mxcc': 500,  # Maximum number of channel complexes
        'mxc': 1,  # Maximum number of channels per complex

        'enable_distribution_analysis': enable_distribution_analysis,
        'avgthick': lz,  # Average thickness
        'ipor': 1,  # Generate porosity model
        'visualize_porosity': visualize_porosity,
        # Channel porosity parameters
        'fcpor_base': fcpor_base_tri,# Channel porosity base value distribution
        'fcpor_range': fcpor_range_tri,# Channel porosity range distribution
        'use_channel_specific_porosity': True,# Use channel-specific porosity calculation


        'collect_centerline_data': True,# Centerline data collection flag
        'source_channel_mapping': {},# Source channel mapping dictionary

        # Z distribution control parameters

        'z_distribution_mode': 'custom', # Z distribution mode ('custom'=custom)
        'z_min_depth': 0.01 * lz, # Minimum depth in Z direction (meters)
        'z_max_depth': 1 * lz, # Maximum depth in Z direction (meters)
        'z_num_strata': 10,# Number of strata
        'z_variation': 0.1,  # Z direction variation degree
        'z_custom_weights': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# Custom weight array
        'z_cycles': 2.0, # Number of cycles in Z direction
        'z_exponent': 2.0,  # Z direction exponent

        # Channel complex parameter distributions
        'fcco': fcco_tri0,  # Orientation angle triangular distribution
        'fcct': fcct_tri,  # Complex thickness triangular distribution
        'fccwtr': fccwtr_tri,  # Complex width-to-thickness ratio triangular distribution
        'fccntg': fccntg_tri,  # Complex net-to-gross ratio triangular distribution

        # Channel sinuosity control parameters
        'channel_sinuosity': sinuosity_tri,# Channel sinuosity control parameter
        # Channel parameter distributions
        'fcat': fcct_tri,  # Channel average thickness triangular distribution
        'fcau': [1.0, 1.0, 1.0],  # Channel thickness undulation triangular distribution
        'fcwtr': fccwtr_tri,  # Channel width-to-thickness ratio triangular distribution
        'fcawu': [13, 15, 19],  # Channel width control, larger values mean greater width

        # Partition control parameters
        'partition_axis': 'x',  # Partition axis, 'x' or 'y'
        'num_regions': 11,  # Number of partitions

        # Regional control configurations
        'region_configs': {
            0: {  # Region 0 configuration
                'density_factor': 0.05,  # Density factor
                'fcco': fcco_tri0,  # Regional orientation angle distribution
                'fcct': fcct_tri,  # Regional thickness distribution
                'fcat': fcct_tri,  # Regional channel thickness distribution
                'channel_sinuosity': sinuosity_custom,  # Regional channel sinuosity
                'z_distribution_params': {  # Regional Z distribution parameters
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2, 1]
                }
            },
            1: {  # Region 1
                'density_factor': 0.2,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2,1, 1]
                }
            },
            2: {  # Region 2
                'density_factor': 0.4,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2, 1, 1, 1]
                }
            },
            3: {  # Region 3
                'density_factor': 0.6,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0.05, 0.08, 0.1, 0.2, 1, 1, 1, 1, 1]
                }
            },
            4: {  # Region 3
                'density_factor': 0.8,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0.05, 0.08, 0.1, 0.2,1, 1, 1, 1, 1, 1, 1]
                }
            },
            5: {  # Region 3
                'density_factor': 1,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0.05, 0.1, 0.2, 1, 1, 1, 1, 1, 1, 1, 1]
                }
            },
            6: {  # Region 3
                'density_factor': 0.8,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0.05, 0.08, 0.1, 0.2, 1, 1, 1, 1, 1, 1, 1]
                }
            },
            7: {  # Region 3
                'density_factor': 0.6,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0.05, 0.08, 0.1, 0.2, 1, 1, 1, 1, 1]
                }
            },
            8: {  # Region 3
                'density_factor': 0.4,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights':[0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2, 1, 1, 1]
                }
            },
            9: {  # Region 3
                'density_factor': 0.2,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2,1, 1]
                }
            },
            10: {  # Region 3
                'density_factor': 0.05,
                'fcco': fcco_tri0,
                'fcct': fcct_tri,
                'fcat': fcct_tri,
                'channel_sinuosity': sinuosity_custom,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 11,
                    'z_custom_weights': [0, 0, 0, 0, 0, 0, 0.05, 0.08, 0.1, 0.2,1]
                }
            },

        },

        # Channel migration parameters
        'vegetation_seed': vegetation_seed,
        'enable_migration': False,# Enable channel migration simulation
        'migration_steps':40,# Migration steps
        'migration_rate': 100,# Migration rate
        'cutoff_factor': 1,# Cutoff factor
        'enable_cutoff': True,# Enable cutoff
        'allow_endpoint_migration': True,# Allow endpoint migration
        'migration_z_increment': 0,  # Migration Z increment
        'use_gpu_migration': use_gpu_migration,  # Use GPU migration computation
        # Numerical stability parameters
        'courant_factor': 0.4,  # Courant number factor (numerical stability)
        'migration_limiter': 1.1,  # Migration limiter
        'smoothing_iterations': 1,  # Smoothing iterations
        'boundary_damping_zone': 0,  # Boundary damping zone

        'enable_integral_effects': True,# Enable integral effects calculation
        'integral_length_factor': 12,# Integral length factor
        'integral_decay_factor': 0.5,# Integral decay factor
        'integral_weight': 0.7,# Integral weight
        'local_weight': 0.3,# Local weight

        'friction_coefficient': 0.1,# Friction coefficient
        'curvature_factor': 50,# Curvature influence factor
        'migration_time_step': 8640000,# Migration time step (seconds)

        # Vegetation spatial heterogeneity parameters
        'vegetation_enabled': False,# Vegetation influence enable flag
        'vegetation_patch_count': 12,# Vegetation patch count
        'vegetation_update_interval': 50,# Vegetation update interval
        'vegetation_smoothing_sigma': 1,# Vegetation smoothing parameter
        'vegetation_value_range': (0, 1),# Vegetation value range
        'vegetation_natural_variability': 0.25,# Vegetation natural variability
        'vegetation_evolution_factor': 0,# Vegetation evolution factor

        'vegetation_influence_strength': 0.6,# Vegetation influence strength
        'vegetation_river_influence_enabled': True,# Vegetation river influence enable
        'vegetation_factor': 0,# Vegetation factor

        # Vegetation export control parameters
        'export_vegetation_enabled': export_vegetation,# Vegetation export enable
        'vegetation_export_dir': vegetation_export_dir,# Vegetation export directory
        'export_vegetation_evolution': export_vegetation_evolution,# Vegetation evolution export
        'export_vegetation_summary': export_vegetation_summary,# Vegetation summary export

        # Channel angle variation parameters
        'allow_channel_angle_variation': 30,# Channel angle variation allowance (degrees)
        'channel_angle_variation': 30.0,# Channel angle variation amplitude (degrees)

        'enable_river_trends': False,# Enable river trend control
        'width_downstream_trend': 2.1, # Width downstream trend
        'depth_downstream_trend': 0.5,# Depth downstream trend
        'z_downstream_trend': 0,# Z downstream trend
        'width_curvature_factor': 0,# Width curvature factor

        # Sedimentary facies type control parameters
        'facies_types': ['channel', 'levee', 'crevasse'],

        # Natural levee parameters
        'levee_enabled': False,# Natural levee enable
        'levee_width_ratio': [1.6, 1.7, 2.0], # Natural levee width ratio distribution
        'levee_height_ratio': [0.25, 0.29, 0.32],# Natural levee height ratio distribution
        'levee_depth_ratio': [1.8, 1.9, 2.0], # Natural levee depth ratio distribution
        'levee_asymmetry': [0.2, 0.5, 0.8], # Natural levee asymmetry distribution
        'levee_thickness_smoothing_enabled': True,# Natural levee thickness smoothing enable
        'levee_thickness_smoothing_iterations': 6,# Natural levee thickness smoothing iterations
        'levee_thickness_smoothing_strength': 0.6,# Natural levee thickness smoothing strength

        # Crevasse splay parameters
        'crevasse_enabled': False,# Crevasse splay enable
        'crevasse_per_channel': [0, 0, 0.1],# Number of crevasse splays per channel distribution
        'crevasse_angle': [30, 60, 90],# Crevasse splay angle distribution (degrees)
        'crevasse_height_ratio': [1.2, 1.3, 1.5],# Crevasse splay height ratio distribution
        'crevasse_length_ratio': [0.3, 0.5, 0.7],# Crevasse splay length ratio distribution
        'crevasse_width_ratio': [0.7, 0.8, 0.9], # Crevasse splay width ratio distribution
        'crevasse_sinuosity': [1.6, 1.8, 2.6], # Crevasse splay sinuosity distribution
    }

    # Display GPU settings
    print(f"\nSimulation GPU settings: {'Enabled' if params['use_gpu'] else 'Disabled'}")

    # Display centerline data collection settings
    if params['collect_centerline_data']:
        print(f"\nCenterline data collection: Enabled (will export to {centerlines_dir})")
    else:
        print("\nCenterline data collection: Disabled")

    print("Starting channel simulation...")

    # Run simulation
    start_time = time.time()
    results = engine.fluvpy(params)
    results['params'] = params
    # Add parameters to each realization as well
    for i in range(1, params['nsim'] + 1):
        key = f'realization_{i}'
        if key in results:
            results[key]['params'] = params
    end_time = time.time()

    # Output results
    print(f"Simulation completed, generated {len(results)} realizations")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # Output result information for each realization
    for i in range(1, params['nsim'] + 1):
        key = f'realization_{i}'
        if key in results:
            print(f"Realization {i} global proportion: {results[key]['global_proportion']:.4f}")

    if export_csv:
        export_simulation_results(results, export_dir)

    # Export centerline data as CSV
    if export_centerlines:
        export_results_centerlines(results, centerlines_dir)

    # Visualize results
    try:
        print("Starting visualization...")

        if interactive_vis:
            print("Using interactive visualization mode...")
            visualize_fluvpy_results(
                results=results,
                isim=1,
                slice_z=None,
                figsize=vis_figsize,
                dpi=vis_dpi,
                save_path=save_path,
                journal_style=journal_style,
                language=language,
                force_partitioning=True,
                interactive=True,
                export_individual_plots=export_individual_plots
            )
        else:
            print("Using standard visualization mode...")

        if generate_parameter_distribution:
            print("Generating channel parameter distribution statistics plots...")
            fluvpy_visualize.visualize_channel_parameters_distribution(
                results=results,
                isim=1,
                journal_style=journal_style,
                language=language,
                figsize=vis_figsize,
                dpi=vis_dpi,
                save_path=save_path[:-4] + "_params_distribution.png" if save_path else None
            )

        # Visualization of porosity distribution statistics plots
        if visualize_porosity_distribution:
            print("Generating channel porosity distribution statistics plots...")
            porosity_dist_save_path = save_path[:-4] + "_porosity_distribution.png" if save_path else None
            fluvpy_visualize.visualize_porosity_distribution(
                results=results,
                isim=1,
                journal_style=journal_style,
                language=language,
                figsize=vis_figsize,
                dpi=vis_dpi,
                save_path=porosity_dist_save_path
            )

    except Exception as e:
        print(f"Visualization failed: {e}")
    fluvpy_visualize.visualize_fluvpy_3d(results)  # 3D voxel display, showing colors based on channel body ID
    return results


if __name__ == "__main__":
    # Command line argument support
    parser = argparse.ArgumentParser(description='fluvpy channel simulation')
    parser.add_argument('--interactive', action='store_true', default=False, help='Enable interactive visualization')
    parser.add_argument('--no-interactive', action='store_false', dest='interactive', help='Disable interactive visualization')
    parser.add_argument('--export-individual-plots', action='store_true', default=False, help='Export each subplot as separate PNG files')
    parser.add_argument('--no-export-individual-plots', action='store_false', dest='export_individual_plots', help='Do not export separate subplot files')

    parser.add_argument('--parameter-distribution', action='store_true', default=False, help='Generate channel parameter distribution statistics plots')
    parser.add_argument('--no-parameter-distribution', action='store_false', dest='parameter_distribution', help='Do not generate channel parameter distribution statistics plots')
    parser.add_argument('--figwidth', type=int, default=10, help='Visualization figure width')
    parser.add_argument('--figheight', type=int, default=8, help='Visualization figure height')

    parser.add_argument('--dpi', type=int, default=100, help='Visualization resolution')
    parser.add_argument('--save', type=str, default=None, help='Save result file path')

    parser.add_argument('--style', type=str, default='nature', choices=['nature', 'science', 'geology'],
                        help='Journal style (nature/science/geology)')

    parser.add_argument('--lang', type=str, default='en', choices=['zh', 'en'], help='Display language (zh=Chinese/en=English)')

    parser.add_argument('--export-csv', action='store_true', default=False, help='Export CSV data')
    parser.add_argument('--no-export-csv', action='store_false', dest='export_csv', help='Do not export CSV data')
    parser.add_argument('--export-dir', type=str, default='fluvpy_csv_outputs', help='CSV data export directory')

    parser.add_argument('--export-centerlines', action='store_true', default=False, help='Export channel centerline data as CSV')
    parser.add_argument('--no-export-centerlines', action='store_false', dest='export_centerlines',
                        help='Do not export channel centerline data')
    parser.add_argument('--centerlines-dir', type=str, default='fluvpy_centerlines', help='Centerline data export directory')
    parser.add_argument('--porosity-min', type=float, default=0.01, help='Minimum porosity value')
    parser.add_argument('--porosity-max', type=float, default=0.50, help='Maximum porosity value')
    parser.add_argument('--visualize-porosity-distribution', action='store_true', default=False,
                        help='Generate porosity distribution statistics plots')
    # Channel migration GPU control parameters
    parser.add_argument('--gpu-migration', action='store_true', default=True,
                        help='Enable GPU-accelerated channel migration computation (enabled by default)')
    parser.add_argument('--no-gpu-migration', action='store_false', dest='gpu_migration',
                        help='Disable GPU-accelerated channel migration computation, use CPU algorithm')

    # Vegetation seed offset parameter
    parser.add_argument('--vegetation-seed-offset', type=int, default=56789,
                        help='Vegetation seed offset for generating independent vegetation random seed (default: 12345)')
    # Vegetation export related parameters
    parser.add_argument('--export-vegetation', action='store_true', default=True,
                        help='Export vegetation distribution data to CSV files')
    parser.add_argument('--no-export-vegetation', action='store_false', dest='export_vegetation',
                        help='Do not export vegetation distribution data')
    parser.add_argument('--vegetation-export-dir', type=str, default='vegetation_distributions',
                        help='Vegetation data export directory, default is vegetation_distributions')
    parser.add_argument('--export-vegetation-evolution', action='store_true', default=True,
                        help='Export vegetation evolution comparison data')
    parser.add_argument('--export-vegetation-summary', action='store_true', default=True,
                        help='Export vegetation evolution summary')

    args = parser.parse_args()

    # Run main function
    main(
        interactive_vis=args.interactive,
        export_individual_plots=args.export_individual_plots,
        vis_figsize=(args.figwidth, args.figheight),
        vis_dpi=args.dpi,
        save_path=args.save,
        journal_style=args.style,
        language=args.lang,
        export_csv=args.export_csv,
        export_dir=args.export_dir,
        generate_parameter_distribution=args.parameter_distribution,
        export_centerlines=args.export_centerlines,
        centerlines_dir=args.centerlines_dir,
        visualize_porosity_distribution=args.visualize_porosity_distribution,
        use_gpu_migration=args.gpu_migration,
        vegetation_seed_offset=args.vegetation_seed_offset,
        export_vegetation=args.export_vegetation,
        vegetation_export_dir=args.vegetation_export_dir,
        export_vegetation_evolution=args.export_vegetation_evolution,
        export_vegetation_summary=args.export_vegetation_summary
    )