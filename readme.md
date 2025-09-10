# Fluvpy

> Fluvpy是一款专注于河流沉积相三维训练图像生成的Python程序包。Fluvpy算法的主要目的是实现生成的训练图像的各项分布特征与研究区域已知的沉积相分布统计数据的高度吻合，确保训练图像足够的代表性。

## 📖 算法特性

- **精确的弯曲度控制**：基于Ferguson算法的增强框架，通过引入弯曲度反馈校正机制，在保持河道物理演化特性的前提下实现了弯曲度的精确控制
- **精确的密度分布异质性**：提出轮询+概率双重机制（PPDM），有效控制河道在XZ方向上的分区密度，实现河道体群在横纵空间上的宏观非均质性模拟
- **先进的渲染方案**：提出候选者收集+延迟决策的渲染方案，基于该方案构建了渲染优先级算法，有效保证了沉积体空间叠置关系的地质合理性，同时解决了GPU渲染时的并行时窗冲突问题
- **多样化的几何参数分布**：通过参数字典到概率分布函数的智能映射，实现了多个参数的多种分布模式
- **集成河道迁移模式**：引入了植被加固影响机制，实现非均质植被对河道迁移阻力的影响

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用Anaconda或Miniconda进行环境管理

### 安装指南

#### 第一步：下载源码

```bash
# 方法1: 使用git克隆（如果你有git）
git clone https://github.com/yourusername/fluvpy.git
cd fluvpy

# 方法2: 直接下载（推荐新手使用）
# 1. 访问GitHub项目页面
# 2. 点击绿色的"Code"按钮
# 3. 选择"Download ZIP"
# 4. 解压到你想要的目录
# 5. 进入解压后的文件夹
```

#### 第二步：安装依赖

```bash
# 推荐：创建虚拟环境（可选但强烈建议）
conda create -n fluvpy python=3.8
conda activate fluvpy

# 或者使用pip创建虚拟环境
python -m venv fluvpy_env
# Windows激活
fluvpy_env\Scripts\activate
# Linux/Mac激活
source fluvpy_env/bin/activate

# 安装所有依赖库
pip install cupy==13.4.0
pip install matplotlib==3.10.6
pip install numba==0.60.0
pip install numpy==1.26.4
pip install pandas==2.3.2
pip install Pillow==11.3.0
pip install psutil==5.9.0
pip install pyvista==0.44.2
pip install scikit-learn==1.7.1
pip install scipy==1.16.1
```

#### 一键安装（推荐）

在项目根目录创建 `requirements.txt` 文件，内容如下：

```txt
cupy==13.4.0
matplotlib==3.10.6
numba==0.60.0
numpy==1.26.4
pandas==2.3.2
Pillow==11.3.0
psutil==5.9.0
pyvista==0.44.2
scikit-learn==1.7.1
scipy==1.16.1
```

然后执行：

```bash
pip install -r requirements.txt
```

## 📚 典型示例

### 示例一
多相模型的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic1.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_1.html)

### 示例二
河道迁移模型的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic2.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_2.html)

### 示例三
冲洪积扇模型的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic3.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_3.html)

### 示例四
平行趋势性控制的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic4.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_4.html)

### 示例五
分区密度控制的示例，点击图片可查看三维模型

[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic5-1.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_5_1.html)

[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic5-2.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_5_2.html)

[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic5-3.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_5_3.html)

### 示例六
分区弯曲度控制的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic6-1.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_6_1.html)

### 示例七
分区厚度控制的示例，点击图片可查看三维模型
[![Fluvpy 3D 河流沉积相可视化示例](https://raw.githubusercontent.com/commitfromet/fluvpy/master/png/pic7-1.png)](https://commitfromet.github.io/fluvpy/web_view/web_viewer_7_1.html)

```python

```
## 🛠️ 主要参数
### 1. 基础网格参数
```python
# 网格数量（决定模型分辨率）
'nx': 300,            # X方向网格数量（模型宽度方向）
'ny': 300,            # Y方向网格数量（模型长度方向）
'nz': 200,            # Z方向网格数量（模型深度方向）

# 物理尺寸（米）
'lx': 3000,           # X方向物理长度（模型宽度）
'ly': 3000,           # Y方向物理长度（模型长度）
'lz': 100,            # Z方向物理厚度（地层厚度）

# 起始坐标（米）
'xmn': 0,             # X方向起始坐标
'ymn': 0,             # Y方向起始坐标
'zmn': 1,             # Z方向起始坐标
```

### 2. 模拟控制参数

```python
'seed': 2981325,          # 随机种子，控制模拟的可重现性
'nsim': 1,                # 模拟实现数量
'use_gpu': True,          # 是否使用GPU加速
'avgthick': 100,          # 平均厚度（米）
'mxcc': 20,               # 最大河道复合体数量
'mxc': 1,                 # 每个复合体中的最大河道数量
```

### 3. 河道几何参数分布


```python
# 复合体参数（三角分布：[最小值, 众数, 最大值]）
'fcco': [0, 0, 0],              # 复合体方向角分布（度）
'fcct': [2, 3, 4],              # 复合体厚度分布（米）
'fccwtr': [1.7, 1.7, 1.71],     # 复合体宽厚比分布
'fccntg': [1, 1, 1],            # 复合体净毛比分布

# 河道参数
'fcat': [2, 3, 4],              # 河道厚度分布（米）
'fcau': [1.0, 1.0, 1.0],        # 河道厚度起伏分布
'fcwtr': [1.7, 1.7, 1.71],      # 河道宽厚比分布
'fcawu': [13, 15, 19],          # 河道宽度控制参数

# 弯曲度控制
'channel_sinuosity': [1.3, 1.4, 1.8],  # 河道弯曲度分布
```

### 4. 孔隙度参数

```python
'fcpor_base': [0.1, 0.225, 0.35],        # 河道孔隙度基准值分布
'fcpor_range': [0.01, 0.015, 0.02],      # 河道孔隙度范围分布
```

### 5. 分区控制参数
分区控制参数如下，通过设置分区轴，选择分区垂直方向
每个区域内可以实现多种参数的单独配置。
```python
'partition_axis': 'x',           # 分区轴（'x'或'y'）
'num_regions': 3,                # 分区数量

# 区域控制配置
'region_configs': {
    0: {  # 区域0配置
        'density_factor': 1,                         # 密度因子
        'fcco': [0, 0, 0],                           # 区域方向角分布
        'fcct': [2, 3, 4],                           # 区域复合体厚度分布
        'fcat': [2, 3, 4],                           # 区域河道厚度分布
        'channel_sinuosity': [1.3, 1.4, 1.8],        # 区域河道弯曲度
        'z_distribution_params': {                   # 区域Z分布参数
            'z_distribution_mode': 'custom',         # 区域Z分布模式-自定义
            'z_num_strata': 6,                       # Z分区数量
            'z_custom_weights': [1, 1, 1, 1, 1, 1]   # 各Z层权重
        }                                            # 区域Z分布参数
    },                                               # 区域Z分布参数
                                                     # ... 其他区域配置形式相同
},
```

### 6. 河道迁移参数

```python
'enable_migration': False,              # 启用河道迁移模拟
'migration_steps': 12,                  # 迁移步数
'migration_rate': 100,                  # 迁移速率
'cutoff_factor': 1,                     # 截弯取直因子
'migration_z_increment': 0,             # 迁移Z增量

# 物理参数
'curvature_factor': 50,                 # 曲率影响因子
'migration_time_step': 8640000,         # 迁移时间步长（秒）

# 积分效应参数
'integral_length_factor': 12,           # 积分长度因子
'integral_decay_factor': 0.5,           # 积分衰减因子
'integral_weight': 0.7,                 # 积分权重
'local_weight': 0.3,                    # 局部权重
```

### 7. 植被参数

```python
'vegetation_seed': 3026003,                    # 植被独立种子
'vegetation_enabled': False,                   # 植被影响启用标志
'vegetation_patch_count': 12,                  # 植被斑块数量
'vegetation_update_interval': 50,              # 植被更新间隔
'vegetation_smoothing_sigma': 1,               # 植被平滑化参数
'vegetation_value_range': (0, 1),              # 植被值范围
'vegetation_natural_variability': 0.25,        # 植被自然变异度
'vegetation_influence_strength': 0.6,          # 植被影响强度
'vegetation_river_influence_enabled': True,    # 植被河道影响启用
```

### 8. 沉积相类型参数

```python
# 天然堤参数
'levee_enabled': False,                        # 天然堤启用
'levee_width_ratio': [3.6, 4.7, 5.0],         # 天然堤宽度比例分布
'levee_height_ratio': [0.25, 0.29, 0.32],     # 天然堤高度比例分布
'levee_depth_ratio': [1.3, 1.5, 1.9],         # 天然堤深度比例分布
'levee_asymmetry': [0.2, 0.5, 0.8],           # 天然堤不对称度分布
'levee_thickness_smoothing_enabled': True,     # 天然堤厚度平滑启用
'levee_thickness_smoothing_iterations': 6,     # 天然堤厚度平滑迭代次数
'levee_thickness_smoothing_strength': 0.6,     # 天然堤厚度平滑强度

# 决口扇参数
'crevasse_enabled': False,                     # 决口扇启用
'crevasse_per_channel': [0, 1, 2],             # 每个河道的决口扇数量分布
'crevasse_angle': [30, 60, 90],                # 决口扇角度分布（度）
'crevasse_height_ratio': [1.2, 1.3, 1.5],     # 决口扇高度比例分布
'crevasse_length_ratio': [0.3, 0.5, 0.7],     # 决口扇长度比例分布
'crevasse_width_ratio': [0.7, 0.8, 0.9],      # 决口扇宽度比例分布
'crevasse_sinuosity': [1.6, 1.8, 2.6],        # 决口扇弯曲度分布
```

### 9. 河道趋势控制参数

```python
'enable_river_trends': False,            # 启用河道趋势控制
'width_downstream_trend': 3.1,           # 宽度下游趋势
'depth_downstream_trend': 0.5,           # 深度下游趋势
'z_downstream_trend': 0,                 # Z下游趋势
'width_curvature_factor': 0,             # 宽度曲率因子
```

### 10. 数据收集与导出参数

```python
'collect_centerline_data': True,                # 中心线数据收集标志

# 植被导出控制参数
'export_vegetation_enabled': False,             # 植被导出启用
'vegetation_export_dir': 'vegetation_distributions',  # 植被导出目录

```

## 📝 使用示例

### 示例1：基础河道生成

```python

```




## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)

## 📞 联系方式

- **邮箱**: 1249069981@qq.com/etdaizai@gmail.com
- - **项目地址**: https://github.com/yourusername/fluvpy
- **问题反馈**: [Issues](https://github.com/commitfromet/fluvpy/issues)

---

<div align="center">
如果这个项目对你有帮助，请给它一个 ⭐️
</div>