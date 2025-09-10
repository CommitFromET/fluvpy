"""
main.py

"""
import time
import argparse

try:
    from sub_modules.engine import engine
    from sub_modules.visualize import fluvpy_visualize
    from sub_modules.visualize.fluvpy_visualize import visualize_fluvpy_results
    from sub_modules.export.fluvpy_export import export_normalized_voxels
    from sub_modules.export.centerline_export import export_results_centerlines

    print("所有模块导入成功！")
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请检查 sub_modules 包结构和 __init__.py 文件")
    exit(1)

def export_simulation_results(results, output_dir='fluvpy_csv_outputs', export_centerlines=True, centerlines_dir='fluvpy_centerlines'):
    """导出模拟结果到CSV文件"""
    print("\n开始导出模拟结果...")
    import os
    os.makedirs(output_dir, exist_ok=True)
    print("导出标准化体素数据...")
    num_exported = export_normalized_voxels(results, output_dir)
    print(f"共导出了 {num_exported} 个实现的标准化体素数据")
    print("\n所有导出任务完成")


def main(interactive_vis=False, vis_figsize=(10, 8), vis_dpi=100, save_path=None,
         journal_style='nature', language='zh', export_csv=True, export_dir='fluvpy_csv_outputs',
         generate_parameter_distribution=False,
         export_centerlines=False, centerlines_dir='fluvpy_centerlines',
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
    fluvpy主函数，执行模拟并可视化结果
    """
    # 模型参数设置
    nx = 300  # X方向网格数量（模型宽度方向网格数）
    ny = 300  # Y方向网格数量（模型长度方向网格数）
    nz = 200  # Z方向网格数量（模型深度方向网格数）
    lx = 3000  # X方向物理长度（米）
    ly = 3000  # Y方向物理长度（米）
    lz = 100  # Z方向物理厚度（米）

    xsiz = lx / nx  # X方向网格间距（米/网格）
    ysiz = ly / ny  # Y方向网格间距（米/网格）
    zsiz = lz / nz  # Z方向网格间距（米/网格）

    # 参数分布定义
    fcct_tri1 = [2, 3, 4]  # 复合体厚度分布
    fcct_tri2 = [4, 5, 6]  # 复合体厚度分布
    fcct_tri3= [6, 7, 8]  # 复合体厚度分布

    fcct_normal = {
        'type': 'normal', 'mean': 5, 'std': 3, 'min_limit': 2, 'max_limit': 8,
    }
    fccwtr_tri = [1.7, 1.7, 1.71]  # 复合体宽厚比分布
    fccntg_tri = [1, 1, 1]  # 复合体毛净比分布
    fcco_tri0 = [0, 0, 0]  # 复合体角度

    sinuosity_tri = [1.3, 1.4, 1.8]  # 河道弯曲度三角分布
    fcpor_base_tri = [0.1, 0.225, 0.35]  # 孔隙度基准值分布
    fcpor_range_tri = [0.01, 0.015, 0.02]  # 孔隙度范围分布

    base_seed = 2981325  # 基础种子
    vegetation_seed = base_seed + vegetation_seed_offset  # 植被独立种子

    # 构建模拟参数字典
    params = {
        'seed': base_seed,
        'use_gpu': True,  # 是否使用GPU加速
        'nx': nx, 'ny': ny, 'nz': nz,
        'xmn': 0, 'ymn': 0, 'zmn': 1,#方向起始坐标（米）
        'xsiz': xsiz, 'ysiz': ysiz, 'zsiz': zsiz,
        'nsim': 1,  # 模拟实现数量
        'mxcc': 20,  # 最大河道复合体数量
        'mxc': 1,  # 每个复合体中的最大河道数量

        'enable_distribution_analysis': enable_distribution_analysis,
        'avgthick': lz,  # 平均厚度
        'ipor': 1,  # 是否生成孔隙率模型
        # 河道孔隙度参数
        'fcpor_base': fcpor_base_tri,# 河道孔隙度基准值分布
        'fcpor_range': fcpor_range_tri,# 河道孔隙度范围分布
        'use_channel_specific_porosity': True,# 使用河道特定孔隙度计算

        'collect_centerline_data': True,# 中心线数据收集标志
        'source_channel_mapping': {},# 源河道映射字典

        # Z分布控制参数
        'z_distribution_mode': 'custom', # Z分布模式（'custom'=自定义）
        'z_min_depth': 0.01 * lz, # Z方向最小深度（米）
        'z_max_depth': 1 * lz, # Z方向最大深度（米）
        'z_num_strata': 10,# 地层数量
        'z_variation': 0.1,  # Z方向变异程度
        'z_custom_weights': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 自定义权重数组
        'z_cycles': 2.0, # Z方向周期数
        'z_exponent': 2.0,  # Z方向指数

        # 河道复合体参数分布
        'fcco': fcco_tri0,  # 方向角三角分布
        'fccwtr': fccwtr_tri,  # 复合体宽厚比三角分布
        'fccntg': fccntg_tri,  # 复合体净毛比三角分布

        # 河道弯曲度控制参数
        'channel_sinuosity': sinuosity_tri,# 河道弯曲度控制参数
        # 河道参数分布
        'fcau': [1.0, 1.0, 1.0],  # 河道厚度起伏三角分布
        'fcwtr': fccwtr_tri,  # 河道宽厚比三角分布
        'fcawu': [13, 15, 19],  # 河道宽度控制，数值越大，宽度越大

        # 分区控制参数
        'partition_axis': 'x',  # 分区轴，'x'或'y'
        'num_regions': 3,  # 分区数量

        # 区域控制配置
        'region_configs': {
            0: {  # 区域0配置
                'density_factor': 1,  # 密度因子
                'fcco': fcco_tri0,  # 区域方向角分布
                'fcct': fcct_tri1,  # 区域复合体厚度分布
                'fcat': fcct_tri1,  # 区域河道厚度分布
                'channel_sinuosity': sinuosity_tri,  # 区域河道弯曲度
                'z_distribution_params': {  # 区域Z分布参数
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 6,
                    'z_custom_weights': [1, 1, 1, 1, 1, 1]
                }
            },
            1: {  # 区域1
                'density_factor': 1,
                'fcco': fcco_tri0,
                'fcct': fcct_tri2,
                'fcat': fcct_tri2,
                'channel_sinuosity': sinuosity_tri,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 6,
                    'z_custom_weights': [1, 1, 1, 1, 1, 1]
                }
            },
            2: {  # 区域2
                'density_factor': 1,
                'fcco': fcco_tri0,
                'fcct': fcct_tri3,
                'fcat': fcct_tri3,
                'channel_sinuosity': sinuosity_tri,
                'z_distribution_params': {
                    'z_distribution_mode': 'custom',
                    'z_num_strata': 6,
                    'z_custom_weights': [1, 1, 1, 1, 1, 1]
                }
            },
        },

        # 河道迁移参数
        'vegetation_seed': vegetation_seed,
        'enable_migration': False,# 启用河道迁移模拟
        'migration_steps': 12,# 迁移步数
        'migration_rate': 100,# 迁移速率
        'cutoff_factor': 0.6,# 截弯取直因子
        'enable_cutoff': True,# 启用截弯取直
        'allow_endpoint_migration': True,# 允许端点迁移
        'migration_z_increment': 0,  # 迁移Z增量
        'use_gpu_migration': use_gpu_migration,  # 使用GPU迁移计算

        'enable_integral_effects': True,# 启用积分效应计算
        'integral_length_factor': 12,# 积分长度因子
        'integral_decay_factor': 0.5,# 积分衰减因子
        'integral_weight': 0.7,# 积分权重
        'local_weight': 0.3,# 局部权重

        'friction_coefficient': 0.1,# 摩擦系数
        'curvature_factor': 50,# 曲率影响因子
        'migration_time_step': 8640000,# 迁移时间步长（秒）

        # 植被空间异质性参数
        'vegetation_enabled': False,# 植被影响启用标志
        'vegetation_patch_count': 12,# 植被斑块数量
        'vegetation_update_interval': 50,# 植被更新间隔
        'vegetation_smoothing_sigma': 1,# 植被平滑化参数
        'vegetation_value_range': (0, 1),# 植被值范围
        'vegetation_natural_variability': 0.25,# 植被自然变异度
        'vegetation_evolution_factor': 0,# 植被演化因子

        'vegetation_influence_strength': 0.6,# 植被影响强度
        'vegetation_river_influence_enabled': True,# 植被河道影响启用
        'vegetation_factor': 0,# 植被因子

        # 植被导出控制参数
        'export_vegetation_enabled': export_vegetation,# 植被导出启用
        'vegetation_export_dir': vegetation_export_dir,# 植被导出目录
        'export_vegetation_evolution': export_vegetation_evolution,# 植被演化导出
        'export_vegetation_summary': export_vegetation_summary,# 植被总结导出

        # 数值稳定性参数
        'courant_factor': 0.4, # Courant数因子（数值稳定性）
        'migration_limiter': 0.4, # 迁移限制器
        'smoothing_iterations': 1,# 平滑迭代次数
        'boundary_damping_zone': 0,  # 边界阻尼区

        # 河道角度变异参数
        'allow_channel_angle_variation': 30,# 河道角度变异允许度（度）
        'channel_angle_variation': 30.0,# 河道角度变异幅度（度）

        'enable_river_trends': False,# 启用河道趋势控制
        'width_downstream_trend': 3.1, # 宽度下游趋势
        'depth_downstream_trend': 0.5,# 深度下游趋势
        'z_downstream_trend': 0,# Z下游趋势
        'width_curvature_factor': 0,# 宽度曲率因子

        # 沉积相类型控制参数
        'facies_types': ['channel', 'levee', 'crevasse'],

        # 天然堤参数
        'levee_enabled': True,# 天然堤启用
        'levee_proportion': 0.15, # 天然堤比例
        'levee_width_ratio': [3.6, 4.7, 5.0], # 天然堤宽度比例分布
        'levee_height_ratio': [0.25, 0.29, 0.32],# 天然堤高度比例分布
        'levee_depth_ratio': [1.3, 1.5, 1.9], # 天然堤深度比例分布
        'levee_asymmetry': [0.2, 0.5, 0.8], # 天然堤不对称度分布
        'levee_thickness_smoothing_enabled': True,# 天然堤厚度平滑启用
        'levee_thickness_smoothing_iterations': 6,# 天然堤厚度平滑迭代次数
        'levee_thickness_smoothing_strength': 0.6,# 天然堤厚度平滑强度

        # 决口扇参数
        'crevasse_enabled': True,# 决口扇启用
        'crevasse_proportion': 0.1,# 决口扇比例
        'crevasse_per_channel': [0, 1, 2],# 每个河道的决口扇数量分布
        'crevasse_angle': [30, 60, 90],# 决口扇角度分布（度）
        'crevasse_height_ratio': [1.2, 1.3, 1.5],# 决口扇高度比例分布
        'crevasse_length_ratio': [0.3, 0.5, 0.7],# 决口扇长度比例分布
        'crevasse_width_ratio': [0.7, 0.8, 0.9], # 决口扇宽度比例分布
        'crevasse_sinuosity': [1.6, 1.8, 2.6], # 决口扇弯曲度分布
    }

    # 显示GPU设置
    print(f"\n模拟GPU设置: {'启用' if params['use_gpu'] else '禁用'}")

    # 显示中心线数据收集设置
    if params['collect_centerline_data']:
        print(f"\n中心线数据收集: 已启用 (将导出到 {centerlines_dir})")
    else:
        print("\n中心线数据收集: 未启用")

    print("启动河道模拟...")

    # 运行模拟
    start_time = time.time()
    results = engine.fluvpy(params)
    results['params'] = params
    # 为每个realization也添加参数
    for i in range(1, params['nsim'] + 1):
        key = f'realization_{i}'
        if key in results:
            results[key]['params'] = params
    end_time = time.time()

    # 输出结果
    print(f"模拟完成，共生成 {len(results)} 个实现")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # 为每个实现输出结果信息
    for i in range(1, params['nsim'] + 1):
        key = f'realization_{i}'
        if key in results:
            print(f"实现 {i} 全局比例: {results[key]['global_proportion']:.4f}")

    if export_csv:
        export_simulation_results(results, export_dir)

    # 导出中心线数据为CSV
    if export_centerlines:
        export_results_centerlines(results, centerlines_dir)

    # 可视化结果
    try:
        print("开始可视化...")

        if interactive_vis:
            print("使用性能优化的交互式可视化模式...")
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
            print("使用标准可视化模式...")

        if generate_parameter_distribution:
            print("生成河道参数分布统计图...")
            fluvpy_visualize.visualize_channel_parameters_distribution(
                results=results,
                isim=1,
                journal_style=journal_style,
                language=language,
                figsize=vis_figsize,
                dpi=vis_dpi,
                save_path=save_path[:-4] + "_params_distribution.png" if save_path else None
            )

        # 孔隙度分布统计图的可视化
        if visualize_porosity_distribution:
            print("生成河道孔隙度分布统计图...")
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
        print(f"可视化失败: {e}")
    fluvpy_visualize.visualize_fluvpy_3d(results)  # 三维体素显示，根据河道体ID显示颜色
    return results

if __name__ == "__main__":
    # 命令行参数支持
    parser = argparse.ArgumentParser(description='fluvpy河道模拟')
    parser.add_argument('--interactive', action='store_true', default=True, help='启用交互式可视化')
    parser.add_argument('--no-interactive', action='store_false', dest='interactive', help='禁用交互式可视化')
    parser.add_argument('--export-individual-plots', action='store_true', default=True, help='导出每个子图为单独的PNG文件')
    parser.add_argument('--no-export-individual-plots', action='store_false', dest='export_individual_plots', help='不导出单独的子图文件')

    parser.add_argument('--parameter-distribution', action='store_true', default=True, help='生成河道参数分布统计图')
    parser.add_argument('--no-parameter-distribution', action='store_false', dest='parameter_distribution', help='不生成河道参数分布统计图')
    parser.add_argument('--figwidth', type=int, default=10, help='可视化图宽度')
    parser.add_argument('--figheight', type=int, default=8, help='可视化图高度')

    parser.add_argument('--dpi', type=int, default=100, help='可视化分辨率')
    parser.add_argument('--save', type=str, default=None, help='保存结果文件路径')

    parser.add_argument('--style', type=str, default='nature', choices=['nature', 'science', 'geology'],
                        help='期刊风格 (nature/science/geology)')

    parser.add_argument('--lang', type=str, default='en', choices=['zh', 'en'], help='显示语言 (zh=中文/en=英文)')

    parser.add_argument('--export-csv', action='store_true', default=True, help='导出CSV数据')
    parser.add_argument('--no-export-csv', action='store_false', dest='export_csv', help='不导出CSV数据')
    parser.add_argument('--export-dir', type=str, default='fluvpy_csv_outputs', help='CSV数据导出目录')

    parser.add_argument('--export-centerlines', action='store_true', default=True, help='导出河道中心线数据为CSV')
    parser.add_argument('--no-export-centerlines', action='store_false', dest='export_centerlines',
                        help='不导出河道中心线数据')
    parser.add_argument('--centerlines-dir', type=str, default='fluvpy_centerlines', help='中心线数据导出目录')
    parser.add_argument('--porosity-min', type=float, default=0.01, help='孔隙度最小值')
    parser.add_argument('--porosity-max', type=float, default=0.50, help='孔隙度最大值')
    parser.add_argument('--visualize-porosity-distribution', action='store_true', default=True,
                        help='生成孔隙度分布统计图')
    # 河道迁移GPU控制参数
    parser.add_argument('--gpu-migration', action='store_true', default=True,
                        help='启用GPU加速河道迁移计算（默认启用）')
    parser.add_argument('--no-gpu-migration', action='store_false', dest='gpu_migration',
                        help='禁用GPU加速河道迁移计算，使用CPU算法')

    # 物理模型参数
    parser.add_argument('--friction-coefficient', type=float, default=0.025,
                        help='河床摩擦系数 (默认: 0.025)')
    parser.add_argument('--curvature-factor', type=float, default=2.5,
                        help='曲率影响因子 (默认: 2.5)')

    # 植被种子偏移量参数
    parser.add_argument('--vegetation-seed-offset', type=int, default=56789,
                        help='植被种子偏移量，用于生成独立的植被随机种子 (默认: 12345)')
    # 植被导出相关参数
    parser.add_argument('--export-vegetation', action='store_true', default=True,
                        help='导出植被分布数据到CSV文件')
    parser.add_argument('--no-export-vegetation', action='store_false', dest='export_vegetation',
                        help='不导出植被分布数据')
    parser.add_argument('--vegetation-export-dir', type=str, default='vegetation_distributions',
                        help='植被数据导出目录，默认为vegetation_distributions')
    parser.add_argument('--export-vegetation-evolution', action='store_true', default=True,
                        help='导出植被演化对比数据')
    parser.add_argument('--export-vegetation-summary', action='store_true', default=True,
                        help='导出植被演化总结')

    args = parser.parse_args()

    # 运行主函数
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