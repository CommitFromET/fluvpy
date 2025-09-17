# 🌊 Fluvpy

<div align="center">

[![Research](https://img.shields.io/badge/Research-Geological%20Modeling-blue.svg)](https://github.com/CommitFromET/fluvpy)
[![Open Source](https://img.shields.io/badge/Open%20Source-❤-red.svg)](https://github.com/CommitFromET/fluvpy)
[![Training Images](https://img.shields.io/badge/Training%20Images-Generation-purple.svg)](https://github.com/CommitFromET/fluvpy)

[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Supported-green.svg)](https://developer.nvidia.com/cuda-zone)
[![3D Visualization](https://img.shields.io/badge/3D-Visualization-orange.svg)](https://github.com/CommitFromET/fluvpy)

[![GitHub Stars](https://img.shields.io/github/stars/CommitFromET/fluvpy.svg?style=social)](https://github.com/CommitFromET/fluvpy/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/CommitFromET/fluvpy.svg?style=social)](https://github.com/CommitFromET/fluvpy/network)
[![GitHub Issues](https://img.shields.io/github/issues/CommitFromET/fluvpy)](https://github.com/CommitFromET/fluvpy/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/CommitFromET/fluvpy)](https://github.com/CommitFromET/fluvpy/commits/master)

</div>

> Fluvpy is a Python package specifically designed for generating three-dimensional training images of fluvial sedimentary facies.
> The primary goal of the Fluvpy algorithm is to achieve high consistency between the distribution characteristics of generated training images and the known statistical data of sedimentary facies distribution in study areas, ensuring training images have sufficient representativeness.

## 📋 Table of Contents

- [📝 Project Overview](#-project-overview)
- [✨ Algorithm Features](#-algorithm-features)
- [🚀 Quick Start](#-quick-start)
- [🎨 Usage Guide](#-usage-guide)
  - [📁 Project File Structure](#-project-file-structure)
  - [📋 Usage Workflow](#-usage-workflow)
  - [⚠️ Important Notes](#-important-notes)
- [🖼️ Typical Examples](#️-typical-examples)
- [🛠️ Main Parameters](#️-main-parameters)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact](#-contact)

## 📝 Project Overview

> The effective application of deep learning algorithms in channel sand body identification relies on obtaining high-quality training images. However, existing channel body training image generation algorithms struggle to effectively reproduce the statistical distribution characteristics of target regions.
> This paper presents Fluvpy, a Python program for unconditional stochastic simulation of channel bodies, which can achieve heterogeneity control and diverse parameter distribution of subsurface channel bodies at macroscopic spatial scales.
> The Fluvpy algorithm adopts a hierarchical and modular architecture design with high robustness and scalability. The algorithm focuses on training image generation for fluvial sedimentary facies and belongs to the category of unconditional simulation algorithms.

## ✨ Algorithm Features

- **Precise Sinuosity Control**: Based on the enhanced Ferguson algorithm framework, introduces a sinuosity feedback correction mechanism to achieve precise sinuosity control while maintaining channel physical evolution characteristics
- **Precise Density Distribution Heterogeneity**: Proposes the Polling + Probability Dual Mechanism (PPDM) to effectively control channel regional density in XZ directions, achieving macroscopic heterogeneity simulation of channel body groups in horizontal and vertical spatial dimensions
- **Advanced Rendering Scheme**: Proposes a candidate set + deferred decision rendering scheme, based on which a rendering priority algorithm is constructed to effectively ensure geological rationality of sedimentary body spatial superposition relationships while solving parallel time window conflicts in GPU rendering processes
- **Diverse Geometric Parameter Distribution**: Through intelligent mapping of parameter dictionaries to probability distribution functions, achieves various distribution modes for different parameters
- **Integrated Channel Migration Patterns**: Introduces vegetation reinforcement influence mechanism to achieve the effect of heterogeneous vegetation on channel migration resistance

## 🚀 Quick Start

### 💻 System Requirements

- Python 3.12.3

### ⚙️ Installation Guide

#### Step 1: Download Source Code

```bash
# Method 1: Use git clone (if you have git installed)
git clone https://github.com/CommitFromET/fluvpy.git
cd fluvpy

# Method 2: Direct download (recommended for beginners)
# 1. Visit the GitHub project page
https://github.com/CommitFromET/fluvpy
# 2. Click the green "Code" button
# 3. Select "Download ZIP"
# 4. Extract to your desired directory
# 5. Navigate to the extracted folder
```

#### Step 2: Install Dependencies

```bash
# Recommended: Create virtual environment (optional but strongly recommended)
conda create -n fluvpy python=3.12.3
conda activate fluvpy

# Or use pip to create virtual environment
python -m venv fluvpy_env
# Windows activation
fluvpy_env\Scripts\activate
# Linux/Mac activation
source fluvpy_env/bin/activate

# Install all required libraries
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

#### One-Click Installation (Recommended)

Create a `requirements.txt` file in the project root directory with the following content:

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

Then execute:
```bash
pip install -r requirements.txt
```

## 🎨 Usage Guide

### 📁 Project File Structure

Fluvpy adopts a modular design, with all core algorithm modules located in the `sub_modules` folder:

```
fluvpy/
├── 📁 sub_modules/                  # Core algorithm modules directory
│   ├── 📁 channel/                  # Channel generation module
│   │   ├──  channel.py              # Data flow for channel centerline step point generation
│   │   ├──  channel_complex.py      # Complex spatial generation, responsible for zone control and position generation
│   │   ├──  ferguson_centerline.py  # Ferguson algorithm-based channel centerline generation with precise sinuosity control
│   │   └──  intersection_cutoff.py  # Self-intersection detection and processing of centerline step points, ensuring channel geometric rationality
│   │
│   ├── 📁 engine/                   # Core engine module
│   │   ├──  constants.py            # System seed management
│   │   ├──  engine.py               # System main engine, responsible for process scheduling of all major algorithms
│   │   └──  process_executor.py     # Program parallel control
│   │
│   ├── 📁 export/                   # Data export module
│   │   ├──  centerline_export.py    # Centerline step point data export program
│   │   └──  fluvpy_export.py        # Voxel data and example data export
│   │
│   ├── 📁 facies/                   # Sedimentary facies generation module
│   │   └──  facies_generator.py     # Natural levee and crevasse splay sedimentary facies geometry generation
│   │
│   ├── 📁 migration/                # Channel migration module
│   │   ├──  gpu_migration.py        # GPU-accelerated channel migration algorithm
│   │   ├──  neck_cutoff.py          # Neck cutoff detection and processing during channel migration
│   │   └──  vegetation_patches.py   # Heterogeneous vegetation distribution generator
│   │
│   ├── 📁 render/                   # Voxel rendering module
│   │   ├──  render_channel.py       # Channel voxel rendering module
│   │   ├──  render_facies.py        # Natural levee and crevasse splay voxel rendering module
│   │   └──  Voxel_removes_module.py# Voxel cleaning module, processing abnormal voxels that don't conform to sedimentary patterns
│   │
│   ├── 📁 utils/                    # Utility module
│   │   └──  utils.py                # Common function module, storing public functions
│   │
│   └── 📁 visualize/                # Visualization module
│       └──  fluvpy_visualize.py     # Visualization program module
│
├──  fluvpy_main.py                  # Main program entry point
└──  sequential_fluvpy_runner.py     # Batch execution program
```

### 📋 Usage Workflow

#### 🔸 Single Simulation Workflow

1. **Parameter Configuration Phase**

>Configure all simulation parameters in fluvpy_main.py
   
2. **Simulation Execution Phase**
>Execute fluvpy_main.py after configuring parameters

3. **Result Analysis Phase**
   - View 3D visualization results
   - Analyze parameter distribution statistics

4. **Parameter Adjustment Phase**
>Adjust parameters to target requirements

#### 🔸 Batch Generation Workflow

1. **Batch Configuration**

>All visualization parameters must be turned off for batch execution
   ```python
   # Set seed parameter to random in fluvpy_main.py
   base_seed = random.randint(10000, 999999)  # Random seed
   ```

2. **Batch Execution**
   ```
   # Execute batch generation (visualization parameters must be turned off)
   sequential_fluvpy_runner.py
   ```

### ⚠️ Important Notes

1. **Batch Execution Requirements**:
   - All visualization parameters must be turned off
   - Set random seed: `base_seed = random.randint(10000, 999999)`
   - Ensure sufficient disk space

2. **GPU Support**:
   - Some algorithms in this program require mandatory GPU usage

3. **Parameter Debugging Recommendations**:
   - Test parameter effects with small-scale grids first
   - Gradually increase complexity to verify stability
   - Use visualization features to verify geological rationality

## 🖼️ Typical Examples
>Click images to view interactive 3D models (Enable VPN if unable to open)

### Example 1
>Multi-stage model (demonstrates multi-stage model rendering priorities in non-migration mode)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic1.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_1.html)

### Example 2
>Channel migration model (demonstrates multi-stage model rendering priorities in migration mode)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic2.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_2.html)

### Example 3
>Alluvial fan model (alluvial fan simulation through trend control algorithm, regional control algorithm, and angle settings)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic3.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_3.html)

### Example 4
>Parallel trend model (parallel trend descent generation through trend control algorithm, simulating non-stationary images similar to coastal sedimentary facies)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic4.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_4.html)

### Example 5
>Regional density control - density decreasing towards one side
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic5-1.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_5_1.html)
>Regional density control - density decreasing from center to both sides
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic5-2.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_5_2.html)
>Regional density control - vertical direction density heterogeneity control
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic5-3.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_5_3.html)

### Example 6
>Regional sinuosity control (unidirectional sinuosity decreasing trend)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic6-1.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_6_1.html)
>Regional sinuosity control (bidirectional sinuosity decreasing trend)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic6-2.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_6_2.html)

### Example 7
>Regional thickness control (unidirectional thickness decreasing trend)
[![Fluvpy 3D](https://raw.githubusercontent.com/commitfromet/fluvpy-example/master/png/pic7-1.png)](https://commitfromet.github.io/fluvpy-example/web_view/web_viewer_7_1.html)

## 🛠️ Main Parameters

### 1. Basic Grid Parameters
```python
# Grid count (determines model resolution)
'nx': 300,            # X-direction grid count (model width direction)
'ny': 300,            # Y-direction grid count (model length direction)
'nz': 200,            # Z-direction grid count (model depth direction)

# Physical dimensions (meters)
'lx': 3000,           # X-direction physical length (model width)
'ly': 3000,           # Y-direction physical length (model length)
'lz': 100,            # Z-direction physical thickness (stratigraphic thickness)

# Starting coordinates (meters)
'xmn': 0,             # X-direction starting coordinate
'ymn': 0,             # Y-direction starting coordinate
'zmn': 1,             # Z-direction starting coordinate
```

### 2. Simulation Control Parameters

```python
'seed': 2981325,          # Random seed, controls simulation reproducibility
'nsim': 1,                # Number of simulation realizations
'mxcc': 20,               # Maximum number of channel complexes
'mxc': 1,                 # Maximum number of channels per complex
```

### 3. Channel Geometry Parameter Distribution
>Channels inherit from complexes. Complex generation in this algorithm is actually only used for spatial positioning.<br>
> Channel geometry parameters are independent. The model generates channels, so attention should be paid to all channel geometry parameter settings.
```python
# Complex parameters (triangular distribution: [minimum, mode, maximum])
'fcco': [0, 0, 0],              # Complex orientation angle distribution (degrees)
'fcct': [2, 3, 4],              # Complex thickness distribution (meters)
'fccwtr': [1.7, 1.7, 1.71],     # Complex width-to-thickness ratio distribution

# Channel parameters
'fcat': [2, 3, 4],              # Channel thickness distribution (meters)
'fcau': [1.0, 1.0, 1.0],        # Channel thickness undulation distribution
'fcwtr': [1.7, 1.7, 1.71],      # Channel width-to-thickness ratio distribution
'fcawu': [13, 15, 19],          # Channel width control parameter

# Sinuosity control
'channel_sinuosity': [1.3, 1.4, 1.8],  # Channel sinuosity distribution
```

### 4. Porosity Parameters
>Range distribution refers to random fluctuation range based on base value.
```python
'fcpor_base': [0.1, 0.225, 0.35],        # Channel porosity base value distribution
'fcpor_range': [0.01, 0.015, 0.02],      # Channel porosity range distribution
```

### 5. Regional Control Parameters
>Regional control parameters are as follows. By setting the partition axis, select the vertical direction of partitioning.<br>
>Multiple parameters can be independently configured within each region.
```python
'partition_axis': 'x',           # Partition axis ('x' or 'y')
'num_regions': 3,                # Number of partitions
# Regional control configuration
'region_configs': {
    0: {  # Region 0 configuration
        'density_factor': 1,                         # Density factor
        'fcco': [0, 0, 0],                           # Regional orientation angle distribution
        'fcct': [2, 3, 4],                           # Regional complex thickness distribution
        'fcat': [2, 3, 4],                           # Regional channel thickness distribution
        'channel_sinuosity': [1.3, 1.4, 1.8],        # Regional channel sinuosity
        'z_distribution_params': {                   # Regional Z distribution parameters
            'z_distribution_mode': 'custom',         # Regional Z distribution mode - custom
            'z_num_strata': 6,                       # Number of Z partitions
            'z_custom_weights': [1, 1, 1, 1, 1, 1]   # Weight of each Z layer
        }                                            # Regional Z distribution parameters
    },                                               # Regional Z distribution parameters
                                                     # ... Other regional configurations follow the same format
},
```

### 6. Channel Migration Parameters
```python
'enable_migration': False,              # Enable channel migration simulation
'migration_steps': 12,                  # Number of migration steps
'migration_rate': 100,                  # Migration rate
'cutoff_factor': 1,                     # Cutoff factor
'migration_z_increment': 0,             # Migration Z increment

# Physical parameters
'curvature_factor': 50,                 # Curvature influence factor
'migration_time_step': 8640000,         # Migration time step (seconds)

# Integral effect parameters
'integral_length_factor': 12,           # Integral length factor
'integral_decay_factor': 0.5,           # Integral decay factor
'integral_weight': 0.7,                 # Integral weight
'local_weight': 0.3,                    # Local weight
```

### 7. Vegetation Parameters
```python
'vegetation_seed': 3026003,                    # Independent vegetation seed
'vegetation_enabled': False,                   # Vegetation influence enable flag
'vegetation_patch_count': 12,                  # Number of vegetation patches
'vegetation_update_interval': 50,              # Vegetation update interval
'vegetation_smoothing_sigma': 1,               # Vegetation smoothing parameter
'vegetation_value_range': (0, 1),              # Vegetation value range
'vegetation_natural_variability': 0.25,        # Vegetation natural variability
'vegetation_influence_strength': 0.6,          # Vegetation influence strength
'vegetation_river_influence_enabled': True,    # Vegetation river influence enable
```

### 8. Sedimentary Facies Type Parameters

```python
# Natural levee parameters
'levee_enabled': False,                        # Natural levee enable
'levee_width_ratio': [3.6, 4.7, 5.0],         # Natural levee width ratio distribution
'levee_height_ratio': [0.25, 0.29, 0.32],     # Natural levee height ratio distribution
'levee_depth_ratio': [1.3, 1.5, 1.9],         # Natural levee depth ratio distribution
'levee_asymmetry': [0.2, 0.5, 0.8],           # Natural levee asymmetry distribution
'levee_thickness_smoothing_enabled': True,     # Natural levee thickness smoothing enable
'levee_thickness_smoothing_iterations': 6,     # Natural levee thickness smoothing iterations
'levee_thickness_smoothing_strength': 0.6,     # Natural levee thickness smoothing strength

# Crevasse splay parameters
'crevasse_enabled': False,                     # Crevasse splay enable
'crevasse_per_channel': [0, 1, 2],             # Number of crevasse splays per channel distribution
'crevasse_angle': [30, 60, 90],                # Crevasse splay angle distribution (degrees)
'crevasse_height_ratio': [1.2, 1.3, 1.5],     # Crevasse splay height ratio distribution
'crevasse_length_ratio': [0.3, 0.5, 0.7],     # Crevasse splay length ratio distribution
'crevasse_width_ratio': [0.7, 0.8, 0.9],      # Crevasse splay width ratio distribution
'crevasse_sinuosity': [1.6, 1.8, 2.6],        # Crevasse splay sinuosity distribution
```

### 9. Channel Trend Control Parameters

```python
'enable_river_trends': False,            # Enable river trend control
'width_downstream_trend': 3.1,           # Width downstream trend (controls ratio of endpoint width to starting point width)
'depth_downstream_trend': 0.5,           # Depth downstream trend (controls ratio of endpoint depth to starting point depth)
'z_downstream_trend': 0,                 # Z value downstream trend
'width_curvature_factor': 0,             # Width curvature factor
```

### 10. Data Collection and Export Parameters

```python
'collect_centerline_data': True,                # Centerline data collection flag
# Vegetation export control parameters
'export_vegetation_enabled': False,             # Vegetation export enable
'vegetation_export_dir': 'vegetation_distributions',  # Vegetation export directory
```

```python
parser.add_argument('--export-csv', action='store_true', default=True, 
                    help='Export CSV voxel data')
parser.add_argument('--export-centerlines', action='store_true', default=False, 
                    help='Export channel centerline data to CSV')
parser.add_argument('--export-vegetation', action='store_true', default=True,
                    help='Export vegetation distribution data to CSV files')
```

### 11. Visualization Parameters

```python
parser.add_argument('--interactive', action='store_true', default=False, 
                    help='Enable interactive visualization')
parser.add_argument('--parameter-distribution', action='store_true', default=False, 
                    help='Generate channel parameter distribution statistics')
parser.add_argument('--visualize-porosity-distribution', action='store_true', default=False,
                    help='Generate porosity distribution statistics')
fluvpy_visualize.visualize_fluvpy_3d(results)  # 3D voxel display, shows colors based on channel body ID
```

## 🤝 Contributing

Issues and improvement suggestions are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE)

## 📞 Contact

- **Email**: 1249069981@qq.com/etdaizai@gmail.com
- **Project Website**: https://github.com/CommitFromET/fluvpy
- **Issue Reports**: [Issues](https://github.com/commitfromet/fluvpy/issues)

---

<div align="center">
If this project helps you, please give it a ⭐️
</div>