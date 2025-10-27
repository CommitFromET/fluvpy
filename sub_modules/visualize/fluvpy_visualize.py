
"""
fluvpy_visualize.py
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# Global constants for base line width and font size
BASE_LINEWIDTH = 0.8
BASE_FONTSIZE = 8


def get_visualization_output_config():
    """
    Centralized configuration function for fluvpy visualization output parameters
    Contains all parameters that affect actual file output

    Returns:
        config (dict): Configuration dictionary containing all output-related parameters
    """
    config = {

        # ==================== Global Base Parameters ====================
        'base_parameters': {
            'BASE_FONTSIZE': 8,  # Base font size
            'BASE_LINEWIDTH': 0.8,  # Base line width
        },

        # ==================== Figure Overall Parameters ====================
        'figure_parameters': {
            'figsize': (5, 8),  # Figure size (width, height) in inches
            'dpi': 500,  # Resolution, dots per inch
            'facecolor': 'white',  # Figure background color
            'edgecolor': 'none',  # Figure edge color
        },

        # ==================== Layout Grid Parameters ====================
        'layout_parameters': {
            # GridSpec main layout parameters
            'main_grid': {
                'nrows': 3,  # Main grid rows
                'ncols': 3,  # Main grid columns
                'width_ratios': [1, 1, 0.7],  # Column width ratios [left, middle, right]
                'height_ratios': [0.4, 0.25, 0.35],  # Row height ratios [top, middle, bottom]
                'wspace': 0.25,  # Column spacing ratio
                'hspace': 0.30,  # Row spacing ratio
            },

            # Regional distribution sub-grid parameters
            'region_subgrid': {
                'wspace': 0.1,  # Region spacing ratio
            },

            # Overall layout adjustment parameters
            'layout_adjust': {
                'tight_layout_rect': [0, 0, 1, 0.96],  # tight_layout rectangle [left, bottom, right, top]
                'subplots_adjust': {  # Alternative layout adjustment parameters
                    'top': 0.94,  # Top margin
                    'bottom': 0.05,  # Bottom margin
                    'left': 0.07,  # Left margin
                    'right': 0.93,  # Right margin
                    'wspace': 0.2,  # Subplot horizontal spacing
                    'hspace': 0.3,  # Subplot vertical spacing
                }
            },

            # Interactive mode layout parameters
            'interactive_layout': {
                'plot_area_rect': [0.05, 0.15, 0.9, 0.8],  # Plot area [left, bottom, width, height]
                'line_slider_rect': [0.2, 0.05, 0.3, 0.03],  # Line width slider position
                'font_slider_rect': [0.6, 0.05, 0.3, 0.03],  # Font slider position
            }
        },

        # ==================== Font Parameters ====================
        'font_parameters': {
            # Font size scaling ratios (relative to BASE_FONTSIZE)
            'size_scales': {
                'main_title': 1.5,  # Main title font scaling ratio
                'subplot_title': 1.5,  # Subplot title font scaling ratio
                'axis_label': 2.0,  # Axis label font scaling ratio
                'tick_label': 1.5,  # Tick label font scaling ratio
                'colorbar_label': 1.5,  # Colorbar label font scaling ratio
                'colorbar_tick': 1.5,  # Colorbar tick font scaling ratio
                'region_label': 1.5,  # Region label font scaling ratio
                'info_text': 1.5,  # Info text font scaling ratio
                'heatmap_text': 1.5,  # Heatmap text annotation font scaling ratio
            },

            # Font style parameters
            'styles': {
                'title_weight': 'bold',  # Title font weight
                'label_weight': 'normal',  # Label font weight
                'family': 'Arial',  # Default font family
                'monospace_family': 'monospace',  # Monospace font family
            },

            # Chinese font settings
            'chinese_fonts': [
                'Source Han Sans CN',
                'Microsoft YaHei',
                'SimHei',
                'STHeiti',
                'FangSong',
                'SimSun',
                'NSimSun',
                'Arial Unicode MS',
                'PingFang SC'
            ]
        },

        # ==================== Line and Border Parameters ====================
        'line_parameters': {
            # Line width scaling ratios (relative to BASE_LINEWIDTH)
            'width_scales': {
                'axis_line': 1.0,  # Axis line width scaling
                'tick_major': 1.0,  # Major tick line width scaling
                'tick_minor': 0.8,  # Minor tick line width scaling
                'grid_line': 0.5,  # Grid line width scaling
                'boundary_line': 1.0,  # Boundary line width scaling
                'region_line': 0.2,  # Region distribution line width scaling
                'global_line': 0.2,  # Global distribution line width scaling
                'colorbar_outline': 0.3,  # Colorbar outline line width scaling
            },

            # Tick line length parameters
            'tick_lengths': {
                'major_length': 3,  # Major tick line length base
                'minor_length': 2,  # Minor tick line length base
            },

            # Line style parameters
            'styles': {
                'boundary_style': '--',  # Boundary line style
                'grid_style': '--',  # Grid line style
                'region_style': '-',  # Region line style
            }
        },

        # ==================== Color and Transparency Parameters ====================
        'color_parameters': {
            # Transparency settings
            'alpha_values': {
                'fill_area': 0.3,  # Fill area transparency
                'boundary_line': 0.7,  # Boundary line transparency
                'text_box': 0.7,  # Text box transparency
                'region_label_box': 0.7,  # Region label box transparency
                'info_text_box': 0.8,  # Info text box transparency
                'grid_line': 0.5,  # Grid line transparency
                'grid_minor': 0.1,  # Minor grid line transparency
            },

            # Channel color mapping
            'channel_colors': [
                '#9E0142', '#D53E4F', '#F46D43', '#FDAE61', '#FEE08B', '#FFFFBF',
                '#E6F598', '#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2'
            ],

            # Journal style color schemes
            'journal_colors': {
                'nature': [
                    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
                    '#56B4E9', '#E69F00', '#000000'
                ],
                'science': [
                    '#3b4cc0', '#b40426', '#f1a340', '#6dbf45', '#662506',
                    '#5e3c99', '#e66101', '#1a1a1a'
                ],
                'geology': [
                    '#8c510a', '#bf812d', '#dfc27d', '#2166ac', '#4393c3',
                    '#92c5de', '#053061', '#67001f'
                ]
            },

            # Background color settings
            'background_colors': {
                'figure_bg': 'white',  # Figure background color
                'axes_bg': 'white',  # Axes background color
                'science_axes_bg': '#f8f8f8',  # Science style axes background
                'geology_axes_bg': '#f9f9f9',  # Geology style axes background
                'geology_figure_bg': '#fcfcfc',  # Geology style figure background
                'channel_bg': [0.95, 0.95, 0.95],  # Channel background color
            },

            # Grid line colors
            'grid_colors': {
                'default': 'gray',  # Default grid color
                'science': 'white',  # Science style grid color
                'geology': '#cccccc',  # Geology style grid color
            }
        },

        # ==================== Colorbar Parameters ====================
        'colorbar_parameters': {
            # General colorbar parameters
            'size': "5%",  # Colorbar width percentage
            'pad': 0.05,  # Colorbar to subplot spacing
            'tick_length': 3,  # Colorbar tick length base
            'tick_width_scale': 0.8,  # Colorbar tick width scaling
            'outline_width_scale': 0.3,  # Colorbar outline line width scaling

            # New: Individual subplot colorbar position configuration
            'positions': {
                # Areal channel distribution (ax1) colorbar position
                'areal_proportion': {
                    'location': 'right',  # Position: 'top', 'bottom', 'left', 'right'
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'vertical',  # Orientation: 'horizontal', 'vertical'
                    'aspect': 20,  # Aspect ratio
                    'shrink': 1.0,  # Shrink ratio
                    'anchor': (0.0, 0.5),  # Anchor position (x, y)
                    'panchor': (1.0, 0.5),  # Parent anchor position (x, y)
                },

                # Channel cross-section (ax3) colorbar position
                'cross_section': {
                    'location': 'right',  # Position
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'vertical',  # Orientation
                    'aspect': 20,  # Aspect ratio
                    'shrink': 1.0,  # Shrink ratio
                    'anchor': (0.0, 0.5),  # Anchor position
                    'panchor': (1.0, 0.5),  # Parent anchor position
                },

                # Channel density heatmap (ax_heatmap) colorbar position
                'channel_density_heatmap': {
                    'location': 'right',  # Position
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'vertical',  # Orientation
                    'aspect': 20,  # Aspect ratio
                    'shrink': 1.0,  # Shrink ratio
                    'anchor': (0.0, 0.5),  # Anchor position
                    'panchor': (1.0, 0.5),  # Parent anchor position
                },

                # Vegetation distribution (ax_vegetation) colorbar position
                'vegetation_distribution': {
                    'location': 'right',  # Position
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'vertical',  # Orientation
                    'aspect': 20,  # Aspect ratio
                    'shrink': 1.0,  # Shrink ratio
                    'anchor': (0.0, 0.5),  # Anchor position
                    'panchor': (1.0, 0.5),  # Parent anchor position
                },

                # Porosity distribution colorbar position (if added)
                'porosity_distribution': {
                    'location': 'bottom',  # Position
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'horizontal',  # Orientation
                    'aspect': 30,  # Aspect ratio
                    'shrink': 0.8,  # Shrink ratio
                    'anchor': (0.5, 1.0),  # Anchor position
                    'panchor': (0.5, 0.0),  # Parent anchor position
                },

                # Permeability distribution colorbar position (if added)
                'permeability_distribution': {
                    'location': 'bottom',  # Position
                    'size': '5%',  # Colorbar size
                    'pad': 0.05,  # Spacing from subplot
                    'orientation': 'horizontal',  # Orientation
                    'aspect': 30,  # Aspect ratio
                    'shrink': 0.8,  # Shrink ratio
                    'anchor': (0.5, 1.0),  # Anchor position
                    'panchor': (0.5, 0.0),  # Parent anchor position
                },
            },

            # Default colorbar position parameters (for plots not specified in positions)
            'default_position': {
                'location': 'right',  # Default position
                'size': '5%',  # Default size
                'pad': 0.05,  # Default spacing
                'orientation': 'vertical',  # Default orientation
                'aspect': 20,  # Default aspect ratio
                'shrink': 1.0,  # Default shrink ratio
                'anchor': (0.0, 0.5),  # Default anchor position
                'panchor': (1.0, 0.5),  # Default parent anchor position
            },

            # 3D colorbar parameters
            '3d_colorbar': {
                'position_x': 0.93,  # 3D colorbar horizontal position
                'position_y': 0,  # 3D colorbar vertical position
                'height': 1,  # 3D colorbar height
                'width': 0.09,  # 3D colorbar width
                'title_font_size': 20,  # 3D colorbar title font size
                'label_font_size': 20,  # 3D colorbar label font size
                'n_labels': 5,  # 3D colorbar label count
                'font_family': "arial",  # 3D colorbar font family
                'vertical': True,  # 3D colorbar vertical orientation
                'fmt': "%.0f",  # 3D colorbar number format
            }
        },

        # ==================== Image Display Parameters ====================
        'image_parameters': {
            'interpolation': 'nearest',  # Image interpolation method
            'aspect': 'auto',  # Image aspect ratio
            'origin': 'lower',  # Image origin position
        },

        # ==================== Text Box and Annotation Parameters ====================
        'text_parameters': {
            # Text box styles
            'box_styles': {
                'region_label': 'round,pad=0.2',  # Region label box style
                'info_text': 'round',  # Info text box style
            },

            # Text alignment parameters
            'alignment': {
                'region_label_ha': 'center',  # Region label horizontal alignment
                'region_label_va': 'top',  # Region label vertical alignment
                'info_text_ha': 'left',  # Info text horizontal alignment
                'info_text_va': 'top',  # Info text vertical alignment
                'heatmap_text_ha': 'center',  # Heatmap text horizontal alignment
                'heatmap_text_va': 'center',  # Heatmap text vertical alignment
            },

            # Text position parameters
            'positions': {
                'info_text_x': 0.05,  # Info text X position (relative coordinates)
                'info_text_y': 0.95,  # Info text Y position (relative coordinates)
                'region_label_y_offset': 10,  # Region label Y offset
                'suptitle_y': 0.98,  # Main title Y position
            }
        },

        # ==================== Heatmap Special Parameters ====================
        'heatmap_parameters': {
            'colormap': 'YlOrRd',  # Heatmap color mapping
            'grid_color': 'white',  # Heatmap grid line color
            'grid_linewidth': 1,  # Heatmap grid line width
            'text_color_threshold': 0.5,  # Text color switching threshold
            'text_colors': {
                'light': 'black',  # Light background text color
                'dark': 'white',  # Dark background text color
            }
        },

        # ==================== 3D Visualization Parameters ====================
        '3d_parameters': {
            'window_size': (1000, 500),  # 3D window size
            'z_scale_factor_ratio': 2,  # Z-axis scaling factor (x-axis width / z-axis height ratio)
            'camera_distance_factor': 1.5,  # Camera distance factor
            'camera_height_factor': 0.5,  # Camera height factor
            'upward_offset_factor': 2.5,  # Upward offset factor
            'threshold_value': 0.5,  # Channel threshold
            'opacity': 0.5,  # Sample grid transparency
            'sample_box_range': {  # Sample box range
                'x_factor': (0.4, 0.6),  # X direction range factor
                'y_factor': (0.4, 0.6),  # Y direction range factor
                'z_factor': (0.4, 0.6),  # Z direction range factor
            }
        },

        # ==================== Slider Widget Parameters ====================
        'slider_parameters': {
            'line_width_slider': {
                'valmin': 0.5,  # Line width slider minimum value
                'valmax': 3.0,  # Line width slider maximum value
                'valinit': 1.0,  # Line width slider initial value
                'valstep': 0.1,  # Line width slider step size
            },
            'font_size_slider': {
                'valmin': 0.5,  # Font slider minimum value
                'valmax': 3.0,  # Font slider maximum value
                'valinit': 1.0,  # Font slider initial value
                'valstep': 0.1,  # Font slider step size
            }
        },

        # ==================== Journal Style Specific Parameters ====================
        'journal_style_parameters': {
            'nature': {
                'grid_enabled': False,  # Enable grid
                'axis_line_scale': 0.8,  # Axis line scaling
                'tick_line_scale': 0.8,  # Tick line scaling
            },
            'science': {
                'grid_enabled': True,  # Enable grid
                'grid_alpha': 0.1,  # Grid transparency
                'axis_line_scale': 0.9,  # Axis line scaling
                'grid_linestyle': '-',  # Grid line style
                'grid_linewidth_scale': 0.5,  # Grid line width scaling
            },
            'geology': {
                'grid_enabled': True,  # Enable grid
                'grid_linestyle': '--',  # Grid line style
                'grid_alpha': 0.3,  # Grid transparency
                'grid_linewidth_scale': 0.6,  # Grid line width scaling
            }
        },

        # ==================== Output File Parameters ====================
        'output_parameters': {
            'main_figure': {
                'dpi': 500,  # Main figure output DPI
                'bbox_inches': 'tight',  # Bounding box setting
                'facecolor': 'white',  # Output background color
                'edgecolor': 'none',  # Output edge color
            },
            'individual_plots': {
                'dpi': 500,  # Individual subplot output DPI
                'bbox_inches': 'tight',  # Bounding box setting
                'figsize': (6, 6),  # Individual subplot size
                'facecolor': 'white',  # Output background color
                'folder_suffix': '_individual_plots',  # Folder suffix
            }
        },

        # ==================== Special Position and Offset Parameters ====================
        'position_parameters': {
            'region_boundary_offset': 0,  # Region boundary offset
            'legend_position': 'best',  # Legend position
            'colorbar_shrink': 0.50,  # Colorbar shrink ratio
            'text_box_pad': 0.2,  # Text box inner padding
            'axis_label_pad': 4,  # Axis label spacing
            'title_pad': 20,  # Title spacing
        }
    }

    return config


class fluvpy3DVisualizer:
    """fluvpy visualization class with 3D voxel plots replacing vertical proportion plots"""

    def __init__(
            self,
            results: Dict[str, Any],
            isim: int = 1,
            slice_z: Optional[int] = None,
            slice_y: Optional[int] = None,
            figsize: Tuple[int, int] = None,
            dpi: int = None,
            journal_style: str = 'nature',
            language: str = 'zh',
            force_partitioning: bool = False,
            export_individual_plots: bool = False
    ):
        """
        Initialize visualization object
        """
        # Get configuration parameters
        self.config = get_visualization_output_config()

        self.results = results
        self.isim = isim
        self.journal_style = journal_style
        self.language = language
        self.force_partitioning = force_partitioning
        self.slice_y = slice_y
        self.export_individual_plots = export_individual_plots

        # Use configuration parameters to set figure parameters
        self.figsize = figsize or self.config['figure_parameters']['figsize']
        self.dpi = dpi or self.config['figure_parameters']['dpi']

        # Get text dictionary
        self.text = self._get_language_text_enhanced(language)

        # Extract basic data
        self._extract_data()

        # Determine Z slice
        self.nz = self.channel.shape[2]
        if slice_z is None:
            self.slice_z = self.nz // 2
        else:
            self.slice_z = slice_z

        # Calculate partition data
        self._compute_region_data()

        # Save references to visualization elements
        self.artists = {}
        self.axes = {}
        self.cbar_objects = {}
        self.text_objects = {}

        # Create colormap object
        self.channel_cmap = self._create_channel_colormap()

        # Create 3D voxel plot color mapping
        self._create_custom_3d_colormap()
        self.voxel_norm = plt.Normalize(vmin=1, vmax=605)

    def _get_scaled_fontsize(self, scale_key, font_size_scale=1.0):
        """Get scaled font size"""
        base_size = self.config['base_parameters']['BASE_FONTSIZE']
        scale_factor = self.config['font_parameters']['size_scales'][scale_key]
        return base_size * scale_factor * font_size_scale

    def _get_scaled_linewidth(self, scale_key, line_width_scale=1.0):
        """Get scaled line width"""
        base_width = self.config['base_parameters']['BASE_LINEWIDTH']
        scale_factor = self.config['line_parameters']['width_scales'][scale_key]
        return base_width * scale_factor * line_width_scale

    def _get_tick_length(self, tick_type, line_width_scale=1.0):
        """Get scaled tick length"""
        base_length = self.config['line_parameters']['tick_lengths'][f'{tick_type}_length']
        return base_length * line_width_scale

    def _get_alpha_value(self, alpha_key):
        """Get transparency value"""
        return self.config['color_parameters']['alpha_values'][alpha_key]

    def _draw_vegetation_distribution(self, gs, line_width_scale, font_size_scale):
        """
        Draw vegetation distribution plot
        """
        # Check if vegetation data exists
        vegetation_map = self.params.get('final_vegetation_map')
        if vegetation_map is None:
            return

        # Create vegetation distribution subplot
        ax_vegetation = self.fig.add_subplot(gs[0, 2])
        self.axes['ax_vegetation'] = ax_vegetation

        # Draw vegetation distribution
        im_vegetation = ax_vegetation.imshow(vegetation_map.T, cmap='Greens',
                                             origin=self.config['image_parameters']['origin'],
                                             aspect=self.config['image_parameters']['aspect'])
        self.artists['vegetation_img'] = im_vegetation

        # Add colorbar
        cbar_vegetation = self._add_colorbar(self.fig, ax_vegetation, im_vegetation,
                                             self.text.get('vegetation_value', 'Vegetation Value'),
                                             font_size_scale, colorbar_type='vegetation_distribution')
        self.cbar_objects['cbar_vegetation'] = cbar_vegetation

        # Set title and labels
        title_vegetation = ax_vegetation.set_title(f"(e) {self.text.get('vegetation_distribution', 'Vegetation Distribution')}",
                                                   fontweight=self.config['font_parameters']['styles']['title_weight'],
                                                   fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
        self.text_objects['title_vegetation'] = title_vegetation

        xlabel_vegetation = ax_vegetation.set_xlabel(self.text['x_direction'],
                                                     fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['xlabel_vegetation'] = xlabel_vegetation

        ylabel_vegetation = ax_vegetation.set_ylabel(self.text['y_direction'],
                                                     fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['ylabel_vegetation'] = ylabel_vegetation

        # Set tick styles
        ax_vegetation.tick_params(direction='out',
                                  length=self._get_tick_length('major', line_width_scale),
                                  width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                  labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

    def visualize_3d_with_pyvista(self):
        """
        Create 3D visualization of channel model using PyVista

        Returns:
            Returns True and image if successful, False and None if failed
        """
        try:
            import pyvista as pv
            import numpy as np
        except ImportError:
            return False, None

        # Get channel data
        channel = self.channel

        # Check if data has values above threshold
        threshold = self.config['3d_parameters']['threshold_value']
        has_channels = np.any(channel > threshold)

        # Extract grid dimensions
        nx, ny, nz = channel.shape

        # Get grid parameters
        x_siz = self.params.get('xsiz', 10.0)
        y_siz = self.params.get('ysiz', 10.0)
        z_siz = self.params.get('zsiz', 10.0)

        # Get origin coordinates
        x_origin = self.params.get('xmn', 0)
        y_origin = self.params.get('ymn', 0)
        z_origin = self.params.get('zmn', 0)

        # Calculate physical dimensions of grid
        x_length = nx * x_siz
        y_length = ny * y_siz
        z_length = nz * z_siz

        # Calculate z-direction scale factor to make z-axis height half of x-axis width
        z_scale_factor_ratio = self.config['3d_parameters']['z_scale_factor_ratio']
        z_scale_factor = x_length / (z_scale_factor_ratio * z_length)

        # Apply z-axis scale factor to z_siz
        adjusted_z_siz = z_siz * z_scale_factor

        # Create PyVista grid
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(x_siz, y_siz, adjusted_z_siz),
            origin=(x_origin, y_origin, z_origin)
        )

        # Add channel data to grid
        grid.cell_data["Channel"] = channel.flatten(order='F')

        # Create boundary box representing complete grid
        outline = grid.outline()

        # Create plotting object
        window_size = self.config['3d_parameters']['window_size']
        plotter = pv.Plotter(off_screen=True)
        plotter.window_size = window_size

        # Set background color to white
        plotter.background_color = self.config['color_parameters']['background_colors']['figure_bg']

        # Add boundary box to show complete grid extent
        plotter.add_mesh(outline, color='black', line_width=1)

        # Simple grid display
        plotter.show_grid()

        # If channel data exists, add channel isosurface
        if has_channels:
            try:
                # Create channel isosurface
                channel_mesh = grid.threshold(threshold, scalars="Channel")

                # Add channel isosurface
                plotter.add_mesh(channel_mesh, cmap=self.voxel_cmap, show_edges=False, show_scalar_bar=False)

                # Manually add vertical colorbar
                colorbar_config = self.config['colorbar_parameters']['3d_colorbar']
                sargs = dict(
                    position_x=colorbar_config['position_x'],
                    position_y=colorbar_config['position_y'],
                    height=colorbar_config['height'],
                    width=colorbar_config['width'],
                    title="Facies Code",
                    title_font_size=colorbar_config['title_font_size'],
                    label_font_size=colorbar_config['label_font_size'],
                    n_labels=colorbar_config['n_labels'],
                    italic=False,
                    bold=False,
                    shadow=False,
                    fmt=colorbar_config['fmt'],
                    font_family=colorbar_config['font_family'],
                    vertical=colorbar_config['vertical']
                )
                plotter.add_scalar_bar(**sargs)

            except Exception as e:
                plotter.add_text("No channels in data", position='upper_edge', font_size=14, color='red')
        else:
            # If no channel data, add hint text
            plotter.add_text("No channels in data", position='upper_edge', font_size=14, color='red')

            # Add sample grid to show scale
            sample_range = self.config['3d_parameters']['sample_box_range']
            box = pv.Box(bounds=(
                x_origin + x_length * sample_range['x_factor'][0], x_origin + x_length * sample_range['x_factor'][1],
                y_origin + y_length * sample_range['y_factor'][0], y_origin + y_length * sample_range['y_factor'][1],
                z_origin + z_length * sample_range['z_factor'][0] * z_scale_factor,
                z_origin + z_length * sample_range['z_factor'][1] * z_scale_factor
            ))
            box.cell_data["Channel"] = np.ones(box.n_cells)
            opacity = self.config['3d_parameters']['opacity']
            plotter.add_mesh(box, color='gray', style='wireframe', opacity=opacity, show_scalar_bar=False)

        # Calculate adjusted grid center
        center = [
            x_origin + x_length / 2,
            y_origin + y_length / 2,
            z_origin + (nz * adjusted_z_siz) / 2
        ]

        # Set camera position and focal point
        camera_distance_factor = self.config['3d_parameters']['camera_distance_factor']
        camera_height_factor = self.config['3d_parameters']['camera_height_factor']
        upward_offset_factor = self.config['3d_parameters']['upward_offset_factor']

        camera_distance = max(x_length, y_length) * camera_distance_factor
        upward_offset = z_length * camera_height_factor * upward_offset_factor
        camera_position = [
            center[0],
            center[1] + camera_distance,
            center[2] + camera_distance * camera_height_factor - upward_offset
        ]
        focal_point = [
            center[0],
            center[1],
            center[2] - upward_offset
        ]

        # Set camera up direction as positive Z-axis
        view_up = [0, 0, 1]

        # Apply camera settings
        plotter.camera_position = [camera_position, focal_point, view_up]

        # Try to capture rendered image and return
        try:
            # Render image
            plotter.show(title=f"3D Fluvial Channel Model - Realization {self.isim}", auto_close=False)

            # Save image
            image = plotter.screenshot()
            plotter.close()

            # Convert to PIL image
            from PIL import Image
            pil_img = Image.fromarray(image)

            return True, pil_img
        except Exception as e:
            return False, None

    def _extract_data(self):
        """Extract simulation results data"""
        # Get results for specified realization
        realization = self.results[f'realization_{self.isim}']
        self.channel = realization['channel']
        self.pcurvea = realization.get('vertical_proportion', np.zeros(self.channel.shape[2]))
        self.pmapa = realization.get('areal_proportion', np.zeros((self.channel.shape[0], self.channel.shape[1])))
        self.prop = realization.get('global_proportion', 0.0)

        # Try to get params from results
        self.params = {}
        if 'params' in self.results:
            self.params = self.results['params']
        # If realization has params, use those with priority
        if 'params' in realization:
            self.params = realization['params']

        # Add necessary default parameter values
        self._set_default_params()

        # Process partitioning information
        self._setup_partitioning()

    def _set_default_params(self):
        """Set default parameter values"""
        # Ensure basic grid parameters exist
        if 'nx' not in self.params:
            self.params['nx'] = self.channel.shape[0]
        if 'ny' not in self.params:
            self.params['ny'] = self.channel.shape[1]
        if 'nz' not in self.params:
            self.params['nz'] = self.channel.shape[2]

        # Add default grid origin and size parameters
        for key, default_value in [('xmn', 0.0), ('ymn', 0.0), ('zmn', 0.0),
                                   ('xsiz', 15.0), ('ysiz', 15.0), ('zsiz', 2.5)]:
            if key not in self.params:
                self.params[key] = default_value

        # Ensure parameters include default channel parameters
        if 'fcco' not in self.params:
            self.params['fcco'] = [0, 0, 0]
        if 'fcct' not in self.params:
            self.params['fcct'] = [3, 5, 8]
        if 'fccwtr' not in self.params:
            self.params['fccwtr'] = [1.5, 1.6, 1.7]
        if 'fccntg' not in self.params:
            self.params['fccntg'] = [0.3, 0.5, 0.7]

    def _setup_partitioning(self):
        """Set up partition control related parameters"""
        # Import directly from fluvpy_engine and check if using partitioning
        self.use_partitioning = self.params.get('use_partitioning', False)

        # If force partitioning is specified, override parameter setting
        if self.force_partitioning:
            self.use_partitioning = True
            self.params['use_partitioning'] = True

        # Try multiple ways to get region controller
        self.region_controller = None

        # Method 1: Get directly from params
        if 'region_controller' in self.params:
            self.region_controller = self.params['region_controller']

        # Method 2: If region_controller not in params but partitioning enabled, try to rebuild
        elif self.use_partitioning:
            try:
                # Try to import RegionController and recreate
                from channel_complex import RegionController
                axis = self.params.get('partition_axis', 'x')
                num_regions = self.params.get('num_regions', 3)
                self.region_controller = RegionController(self.params, axis, num_regions)

                # Apply region configurations
                if 'region_configs' in self.params:
                    for region_id, config in self.params['region_configs'].items():
                        self.region_controller.configure_region(int(region_id), **config)
            except ImportError:
                # Method 3: Manually build partitioning info from parameters
                if 'partition_axis' in self.params and 'num_regions' in self.params:
                    axis = self.params.get('partition_axis', 'x')
                    num_regions = self.params.get('num_regions', 3)

                    # Create custom region list
                    regions = []
                    if axis == 'x':
                        xmn = self.params.get('xmn', 0)
                        nx = self.params.get('nx', 300)
                        xsiz = self.params.get('xsiz', 10)
                        total_length = nx * xsiz

                        for i in range(num_regions):
                            start = xmn + i * (total_length / num_regions)
                            end = xmn + (i + 1) * (total_length / num_regions)
                            regions.append({
                                'id': i,
                                'start': start,
                                'end': end,
                                'density_factor': 1.0
                            })

                    # Create temporary region controller object
                    class TempRegionController:
                        def __init__(self, regions, axis):
                            self.regions = regions
                            self.axis = axis
                            self.num_regions = len(regions)

                    self.region_controller = TempRegionController(regions, axis)

        # Confirm region controller status and unify partition count
        if self.region_controller:
            actual_num_regions = len(self.region_controller.regions)
            param_num_regions = self.params.get('num_regions', 3)

            # Use actual region count, but give warning if mismatch
            if actual_num_regions != param_num_regions:
                print(f"Warning: Actual region count ({actual_num_regions}) differs from parameter setting ({param_num_regions}), using actual count")

            self.num_regions = actual_num_regions
        else:
            self.num_regions = self.params.get('num_regions', 3)

        # Initialize region boundaries and labels
        self.region_boundaries = []
        self.region_labels = []

        # If region controller exists, get region boundaries
        if self.region_controller:
            nx = self.channel.shape[0]
            if self.region_controller.axis == 'x':
                for i, region in enumerate(self.region_controller.regions):
                    # Convert physical coordinates to indices
                    start_idx = max(0, int(np.floor(
                        (region['start'] - self.params.get('xmn', 0)) / self.params.get('xsiz', 15.0))))
                    end_idx = min(nx, int(np.ceil(
                        (region['end'] - self.params.get('xmn', 0)) / self.params.get('xsiz', 15.0))))

                    if start_idx < nx:
                        self.region_boundaries.append(start_idx)

                    # Add region labels
                    mid_x = start_idx + (end_idx - start_idx) // 2
                    if 0 <= mid_x < nx:
                        self.region_labels.append((mid_x, f"R{i + 1}"))
            else:  # y-axis partitioning
                ny = self.channel.shape[1]
                for i, region in enumerate(self.region_controller.regions):
                    # Convert physical coordinates to indices
                    start_idx = max(0, int(np.floor(
                        (region['start'] - self.params.get('ymn', 0)) / self.params.get('ysiz', 15.0))))

                    if i == 0:  # First region
                        self.region_boundaries.append(start_idx)

                    # Add region labels
                    end_idx = min(ny, int(np.ceil(
                        (region['end'] - self.params.get('ymn', 0)) / self.params.get('ysiz', 15.0))))
                    mid_x = nx // 2  # x position at center
                    mid_y = start_idx + (end_idx - start_idx) // 2
                    if 0 <= mid_y < ny:
                        self.region_labels.append((mid_x, f"R{i + 1}"))

        # If no region information obtained, create default partitioning
        if not self.region_boundaries and self.use_partitioning:
            nx = self.channel.shape[0]
            region_size = nx // self.num_regions

            for i in range(self.num_regions):
                start_idx = i * region_size
                self.region_boundaries.append(start_idx)

                # Add region labels
                mid_x = start_idx + region_size // 2
                self.region_labels.append((mid_x, f"R{i + 1}"))

    def _compute_channel_density_heatmap(self):
        """
        Compute channel density heatmap based on partitioned grid

        Returns:
            heatmap_data: 2D array representing channel density for each XZ grid
            x_boundaries: X direction partition boundaries
            z_boundaries: Z direction partition boundaries
            grid_info: Grid information dictionary
        """
        # 1. Get partitioning information
        num_regions = self.params.get('num_regions', 3)
        region_controller = self.params.get('region_controller', None)

        # Get grid parameters
        nx = self.params.get('nx', 250)
        nz = self.params.get('nz', 250)
        xmn = self.params.get('xmn', 0)
        zmn = self.params.get('zmn', 0)
        xsiz = self.params.get('xsiz', 24)
        zsiz = self.params.get('zsiz', 1)

        # Calculate physical ranges
        x_max = xmn + nx * xsiz
        z_max = zmn + nz * zsiz

        # 2. Determine horizontal partition boundaries
        x_boundaries = []
        x_region_size = (x_max - xmn) / num_regions
        for i in range(num_regions + 1):
            x_boundaries.append(xmn + i * x_region_size)

        # 3. Determine vertical partition information
        z_weights_list = []
        max_z_regions = 0

        if region_controller and hasattr(region_controller, 'regions'):
            # Get z weights for each horizontal partition from region_controller
            for i, region in enumerate(region_controller.regions):
                z_params = region.get('z_distribution_params', {})
                z_weights = z_params.get('z_custom_weights', [1])
                z_weights_list.append(z_weights)
                max_z_regions = max(max_z_regions, len(z_weights))
        else:
            # Get from global parameters or use defaults
            if 'region_configs' in self.params:
                for i in range(num_regions):
                    if i in self.params['region_configs']:
                        z_params = self.params['region_configs'][i].get('z_distribution_params', {})
                        z_weights = z_params.get('z_custom_weights', [1, 1, 1, 1, 1])
                    else:
                        z_weights = [1, 1, 1, 1, 1]  # Default 5 vertical partitions
                    z_weights_list.append(z_weights)
                    max_z_regions = max(max_z_regions, len(z_weights))
            else:
                # Use unified defaults
                default_z_weights = [1, 1, 1, 1, 1]
                for i in range(num_regions):
                    z_weights_list.append(default_z_weights)
                max_z_regions = len(default_z_weights)

        # 4. Calculate vertical partition boundaries (based on maximum vertical partitions)
        z_boundaries = []
        z_region_size = (z_max - zmn) / max_z_regions
        for i in range(max_z_regions + 1):
            z_boundaries.append(zmn + i * z_region_size)

        # 5. Initialize density grid
        heatmap_data = np.zeros((num_regions, max_z_regions))

        # 6. Get channel data directly from simulation results array
        try:
            realization = self.results[f'realization_{self.isim}']

            # Try to get channel grid data
            channel_grid = None
            if 'channel' in realization:
                channel_grid = realization['channel']

            # Method 1: If channel grid exists, count non-zero starting positions
            if channel_grid is not None:
                # Find first occurrence position of each channel (different values) as starting point
                unique_channels = np.unique(channel_grid)
                unique_channels = unique_channels[unique_channels > 0]  # Exclude background value 0

                channel_count_total = 0
                for channel_id in unique_channels:
                    # Find all positions of this channel
                    channel_positions = np.where(channel_grid == channel_id)

                    if len(channel_positions[0]) > 0:
                        # Find Y direction minimum position as starting point (Y=0 direction is starting point)
                        min_y_idx = np.argmin(channel_positions[1])

                        start_ix = channel_positions[0][min_y_idx]
                        start_iy = channel_positions[1][min_y_idx]
                        start_iz = channel_positions[2][min_y_idx]

                        # Convert to physical coordinates
                        start_x = xmn + start_ix * xsiz
                        start_z = zmn + start_iz * zsiz

                        # Determine starting point's horizontal partition
                        x_region_idx = -1
                        for i in range(num_regions):
                            if x_boundaries[i] <= start_x < x_boundaries[i + 1]:
                                x_region_idx = i
                                break

                        # Handle boundary cases
                        if x_region_idx == -1 and start_x >= x_boundaries[-1]:
                            x_region_idx = num_regions - 1

                        # Determine starting point's vertical partition
                        z_region_idx = -1
                        for i in range(max_z_regions):
                            if z_boundaries[i] <= start_z < z_boundaries[i + 1]:
                                z_region_idx = i
                                break

                        # Handle boundary cases
                        if z_region_idx == -1 and start_z >= z_boundaries[-1]:
                            z_region_idx = max_z_regions - 1

                        # If coordinates are valid, increment corresponding grid count
                        if 0 <= x_region_idx < num_regions and 0 <= z_region_idx < max_z_regions:
                            heatmap_data[x_region_idx, z_region_idx] += 1
                            channel_count_total += 1

            # Method 2: If Method 1 found no data, try to get from centerline_data
            if np.sum(heatmap_data) == 0:
                centerline_data_list = realization.get('centerline_data', [])

                if centerline_data_list:
                    channel_count_total = 0
                    for channel_data in centerline_data_list:
                        if not isinstance(channel_data, dict) or 'points' not in channel_data:
                            continue

                        points = channel_data['points']
                        if len(points) == 0:
                            continue

                        # Get channel starting point coordinates
                        start_point = points[0]
                        if 'global_x' not in start_point or 'global_z' not in start_point:
                            continue

                        start_x = start_point['global_x']
                        start_z = start_point['global_z']

                        # Partition calculation logic same as Method 1
                        x_region_idx = -1
                        for i in range(num_regions):
                            if x_boundaries[i] <= start_x < x_boundaries[i + 1]:
                                x_region_idx = i
                                break

                        if x_region_idx == -1 and start_x >= x_boundaries[-1]:
                            x_region_idx = num_regions - 1

                        z_region_idx = -1
                        for i in range(max_z_regions):
                            if z_boundaries[i] <= start_z < z_boundaries[i + 1]:
                                z_region_idx = i
                                break

                        if z_region_idx == -1 and start_z >= z_boundaries[-1]:
                            z_region_idx = max_z_regions - 1

                        if 0 <= x_region_idx < num_regions and 0 <= z_region_idx < max_z_regions:
                            heatmap_data[x_region_idx, z_region_idx] += 1
                            channel_count_total += 1

            # Method 3: If first two methods found no data, try to rebuild from main_channels_info
            if np.sum(heatmap_data) == 0:
                main_channels_info = realization.get('main_channels_info', [])

                if main_channels_info:
                    channel_count_total = 0
                    for channel_info in main_channels_info:
                        # Calculate channel starting point global coordinates
                        x_center = channel_info.get('x_center', 0)
                        z_top = channel_info.get('z_top', 0)

                        # If centerline data exists, use first point offset
                        centerline_offset = 0
                        if 'centerline' in channel_info and len(channel_info['centerline']) > 0:
                            centerline_offset = channel_info['centerline'][0]

                        # Calculate starting point coordinates
                        start_x = x_center + centerline_offset
                        start_z = z_top

                        # Partition calculation logic same as before
                        x_region_idx = -1
                        for i in range(num_regions):
                            if x_boundaries[i] <= start_x < x_boundaries[i + 1]:
                                x_region_idx = i
                                break

                        if x_region_idx == -1 and start_x >= x_boundaries[-1]:
                            x_region_idx = num_regions - 1

                        z_region_idx = -1
                        for i in range(max_z_regions):
                            if z_boundaries[i] <= start_z < z_boundaries[i + 1]:
                                z_region_idx = i
                                break

                        if z_region_idx == -1 and start_z >= z_boundaries[-1]:
                            z_region_idx = max_z_regions - 1

                        if 0 <= x_region_idx < num_regions and 0 <= z_region_idx < max_z_regions:
                            heatmap_data[x_region_idx, z_region_idx] += 1
                            channel_count_total += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            return heatmap_data, x_boundaries, z_boundaries, {
                'num_x_regions': num_regions,
                'num_z_regions': max_z_regions,
                'x_range': (xmn, x_max),
                'z_range': (zmn, z_max)
            }

        # 8. Normalize density data
        max_density = np.max(heatmap_data)
        if max_density > 0:
            heatmap_data_normalized = heatmap_data / max_density
        else:
            heatmap_data_normalized = heatmap_data

        # 10. Prepare grid information
        grid_info = {
            'num_x_regions': num_regions,
            'num_z_regions': max_z_regions,
            'x_range': (xmn, x_max),
            'z_range': (zmn, z_max),
            'x_region_size': x_region_size,
            'z_region_size': z_region_size,
            'total_channels': int(np.sum(heatmap_data)),
            'max_density': max_density,
            'z_weights_per_region': z_weights_list,
            'raw_data': heatmap_data,
            'x_boundaries': x_boundaries,
            'z_boundaries': z_boundaries
        }

        return heatmap_data_normalized, x_boundaries, z_boundaries, grid_info

    def _draw_channel_density_heatmap(self, gs, line_width_scale, font_size_scale):
        """
        Draw channel density heatmap
        """
        # Calculate heatmap data
        heatmap_data, x_boundaries, z_boundaries, grid_info = self._compute_channel_density_heatmap()

        if grid_info['total_channels'] == 0:
            return

        # Create heatmap subplot
        ax_heatmap = self.fig.add_subplot(gs[:, 2])
        self.axes['ax_heatmap'] = ax_heatmap

        # Get heatmap parameters
        heatmap_config = self.config['heatmap_parameters']

        # Draw heatmap
        im_heatmap = ax_heatmap.imshow(heatmap_data.T,
                                       cmap=heatmap_config['colormap'],
                                       origin=self.config['image_parameters']['origin'],
                                       aspect=self.config['image_parameters']['aspect'],
                                       interpolation=self.config['image_parameters']['interpolation'])

        self.artists['heatmap_img'] = im_heatmap

        # Add colorbar
        cbar_heatmap = self._add_colorbar(self.fig, ax_heatmap, im_heatmap,
                                          self.text.get('channel_density', 'Channel Density'),
                                          font_size_scale, colorbar_type='channel_density_heatmap')
        self.cbar_objects['cbar_heatmap'] = cbar_heatmap

        # Set axis labels
        num_x_regions = grid_info['num_x_regions']
        num_z_regions = grid_info['num_z_regions']

        # X-axis ticks and labels
        x_tick_positions = np.arange(num_x_regions)
        x_tick_labels = [f"X{i}" for i in range(num_x_regions)]
        ax_heatmap.set_xticks(x_tick_positions)
        ax_heatmap.set_xticklabels(x_tick_labels)

        # Z-axis ticks and labels
        z_tick_positions = np.arange(num_z_regions)
        z_tick_labels = [f"Z{i}" for i in range(num_z_regions)]
        ax_heatmap.set_yticks(z_tick_positions)
        ax_heatmap.set_yticklabels(z_tick_labels)

        # Add grid lines
        ax_heatmap.set_xticks(np.arange(-0.5, num_x_regions, 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-0.5, num_z_regions, 1), minor=True)
        ax_heatmap.grid(which='minor',
                        color=heatmap_config['grid_color'],
                        linestyle='-',
                        linewidth=heatmap_config['grid_linewidth'])

        # Add numerical annotations in each grid cell
        threshold = heatmap_config['text_color_threshold']
        for i in range(num_x_regions):
            for j in range(num_z_regions):
                density_value = heatmap_data[i, j]
                if density_value > 0:
                    # Choose text color based on density value
                    text_color = (heatmap_config['text_colors']['dark'] if density_value > threshold
                                  else heatmap_config['text_colors']['light'])
                    text_obj = ax_heatmap.text(i, j, f'{density_value:.2f}',
                                               ha=self.config['text_parameters']['alignment']['heatmap_text_ha'],
                                               va=self.config['text_parameters']['alignment']['heatmap_text_va'],
                                               color=text_color,
                                               fontsize=self._get_scaled_fontsize('heatmap_text', font_size_scale),
                                               fontweight=self.config['font_parameters']['styles']['title_weight'])
                    self.text_objects[f'heatmap_text_{i}_{j}'] = text_obj

        # Set title
        title_text = f"(d) {self.text.get('channel_density_heatmap', 'Channel Density Heatmap')}"
        title_heatmap = ax_heatmap.set_title(title_text,
                                             fontweight=self.config['font_parameters']['styles']['title_weight'],
                                             fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
        self.text_objects['title_heatmap'] = title_heatmap

        # Set axis labels
        xlabel_heatmap = ax_heatmap.set_xlabel(self.text.get('x_partition', 'X Direction Partitions'),
                                               fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['xlabel_heatmap'] = xlabel_heatmap

        ylabel_heatmap = ax_heatmap.set_ylabel(self.text.get('z_partition', 'Z Direction Partitions'),
                                               fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['ylabel_heatmap'] = ylabel_heatmap

        # Set tick styles
        ax_heatmap.tick_params(direction='out',
                               length=self._get_tick_length('major', line_width_scale),
                               width=self._get_scaled_linewidth('tick_major', line_width_scale),
                               labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

        # Save heatmap information for individual export
        self.heatmap_info = {
            'data': heatmap_data,
            'grid_info': grid_info,
            'x_boundaries': x_boundaries,
            'z_boundaries': z_boundaries
        }

    def _compute_region_data(self):
        """
        Compute all partition data and statistics
        """
        # Create dictionary to store partition statistics
        self.region_stats = {}
        self.region_pcurves = []  # Store vertical distribution for each region

        # Dynamically generate sufficient color schemes
        def generate_colors(num_colors, style='nature'):
            """Dynamically generate specified number of colors"""
            colors_config = self.config['color_parameters']['journal_colors']

            if style in colors_config:
                base_colors = colors_config[style]
            else:
                base_colors = colors_config['nature']

            # If needed colors exceed predefined colors, use matplotlib colormaps to generate more
            if num_colors <= len(base_colors):
                return base_colors[:num_colors]
            else:
                # Use colormap to generate more colors
                if style == 'geology':
                    cmap = plt.cm.Set3
                elif style == 'science':
                    cmap = plt.cm.tab20
                else:
                    cmap = plt.cm.tab10

                # Combine predefined colors with dynamically generated colors
                additional_colors = [cmap(i / max(1, num_colors - len(base_colors)))
                                     for i in range(num_colors - len(base_colors))]
                # Convert matplotlib colors to hexadecimal
                additional_colors = ['#{:02x}{:02x}{:02x}'.format(
                    int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in additional_colors]

                return base_colors + additional_colors

        # Dynamically generate color scheme
        self.colors = generate_colors(self.num_regions, self.journal_style)

        # Calculate global statistics
        try:
            self.region_stats['Global'] = {
                'mean': float(np.nanmean(self.pcurvea)),
                'max': float(np.nanmax(self.pcurvea)),
                'min': float(np.nanmin(self.pcurvea)),
                'std': float(np.nanstd(self.pcurvea)),
                'cells': int(np.sum(self.channel > 0))
            }
        except Exception as e:
            self.region_stats['Global'] = {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'cells': 0}

        # Determine Y cross-section position
        try:
            if self.slice_y is None:
                # Find Y position containing most channels
                self.y_position = np.argmax(np.sum(self.channel[:, :, :], axis=(0, 2)))
                if np.isnan(self.y_position) or self.y_position == 0:
                    self.y_position = self.channel.shape[1] // 2
            else:
                # Use specified Y position, ensure within valid range
                self.y_position = max(0, min(self.slice_y, self.channel.shape[1] - 1))
        except Exception as e:
            self.y_position = self.channel.shape[1] // 2

        # Initialize list to save all region maximum proportion values
        all_max_proportions = []
        if 'Global' in self.region_stats and 'max' in self.region_stats['Global']:
            all_max_proportions.append(self.region_stats['Global']['max'])

        # If region controller exists, calculate partition data
        if self.use_partitioning and self.region_controller:
            # Get partitioning information
            regions = self.region_controller.regions

            # Calculate vertical distribution for each partition
            for i, region in enumerate(regions):
                # Get partition boundaries
                try:
                    if self.region_controller.axis == 'x':
                        # Convert physical coordinates to grid indices
                        start_idx = max(0, int(np.floor((region['start'] - self.params['xmn']) / self.params['xsiz'])))
                        end_idx = min(self.channel.shape[0],
                                      int(np.ceil((region['end'] - self.params['xmn']) / self.params['xsiz'])))

                        # Calculate valid index range
                        if start_idx >= self.channel.shape[0] or end_idx <= 0 or start_idx >= end_idx:
                            continue

                        # Extract data for this region and calculate vertical proportion
                        region_channel = self.channel[start_idx:end_idx, :, :]
                        if region_channel.size > 0:
                            # Calculate vertical proportion - average along x and y directions
                            region_pcurve = np.mean((region_channel > 0).astype(float), axis=(0, 1))
                            self.region_pcurves.append(region_pcurve)

                            # Calculate statistics for this region
                            region_name = f"Region{i + 1}"
                            self.region_stats[region_name] = {
                                'mean': float(np.nanmean(region_pcurve)),
                                'max': float(np.nanmax(region_pcurve)),
                                'min': float(np.nanmin(region_pcurve)),
                                'std': float(np.nanstd(region_pcurve)),
                                'relative': float(np.nanmean(region_pcurve) / self.region_stats['Global']['mean'])
                                if self.region_stats['Global']['mean'] > 0 else 0,
                                'cells': int(np.sum(region_channel > 0))
                            }

                            # Record region's maximum proportion value
                            all_max_proportions.append(self.region_stats[region_name]['max'])

                    else:  # y-axis partitioning
                        # Convert physical coordinates to grid indices
                        start_idx = max(0, int(np.floor((region['start'] - self.params['ymn']) / self.params['ysiz'])))
                        end_idx = min(self.channel.shape[1],
                                      int(np.ceil((region['end'] - self.params['ymn']) / self.params['ysiz'])))

                        # Calculate valid index range
                        if start_idx >= self.channel.shape[1] or end_idx <= 0 or start_idx >= end_idx:
                            continue

                        # Extract data for this region and calculate vertical proportion
                        region_channel = self.channel[:, start_idx:end_idx, :]
                        if region_channel.size > 0:
                            # Calculate vertical proportion - average along x and y directions
                            region_pcurve = np.mean((region_channel > 0).astype(float), axis=(0, 1))
                            self.region_pcurves.append(region_pcurve)

                            # Calculate statistics for this region
                            region_name = f"Region{i + 1}"
                            self.region_stats[region_name] = {
                                'mean': float(np.nanmean(region_pcurve)),
                                'max': float(np.nanmax(region_pcurve)),
                                'min': float(np.nanmin(region_pcurve)),
                                'std': float(np.nanstd(region_pcurve)),
                                'relative': float(np.nanmean(region_pcurve) / self.region_stats['Global']['mean'])
                                if self.region_stats['Global']['mean'] > 0 else 0,
                                'cells': int(np.sum(region_channel > 0))
                            }

                            # Record region's maximum proportion value
                            all_max_proportions.append(self.region_stats[region_name]['max'])
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        # If partition data not successfully obtained but we want partition visualization, manually partition
        if (self.use_partitioning and len(self.region_pcurves) == 0 and self.num_regions > 0):
            # Equally divide data along X-axis
            nx = self.channel.shape[0]
            region_size = nx // self.num_regions

            # Manually create partitions
            for i in range(self.num_regions):
                start_idx = i * region_size
                end_idx = (i + 1) * region_size if i < self.num_regions - 1 else nx

                # Extract data for this region and calculate vertical proportion
                region_channel = self.channel[start_idx:end_idx, :, :]
                if region_channel.size > 0:
                    # Calculate vertical proportion - average along x and y directions
                    region_pcurve = np.mean((region_channel > 0).astype(float), axis=(0, 1))
                    self.region_pcurves.append(region_pcurve)

                    # Calculate statistics for this region
                    region_name = f"Region{i + 1}"
                    self.region_stats[region_name] = {
                        'mean': float(np.nanmean(region_pcurve)),
                        'max': float(np.nanmax(region_pcurve)),
                        'min': float(np.nanmin(region_pcurve)),
                        'std': float(np.nanstd(region_pcurve)),
                        'relative': float(np.nanmean(region_pcurve) / self.region_stats['Global']['mean'])
                        if self.region_stats['Global']['mean'] > 0 else 0,
                        'cells': int(np.sum(region_channel > 0))
                    }

                    # Record region's maximum proportion value
                    all_max_proportions.append(self.region_stats[region_name]['max'])

        # Ensure all region maximum values are considered
        for region_name, stats in self.region_stats.items():
            if region_name != 'Global' and 'max' in stats and stats['max'] not in all_max_proportions:
                all_max_proportions.append(stats['max'])

        # Calculate x-axis range (for vertical distribution plots)
        if all_max_proportions:
            # Find maximum proportion value among all regions
            max_of_all = max(all_max_proportions)
            # Add 50% safety margin to ensure complete display
            self.max_proportion = max_of_all * 1.5
        else:
            self.max_proportion = 0.15  # Default value

    def _save_individual_plots(self, base_save_path: str):
        """
        Save each subplot as individual PNG file
        """
        import os

        # Parse base path
        base_dir = os.path.dirname(base_save_path)
        base_name = os.path.splitext(os.path.basename(base_save_path))[0]

        # Get output configuration
        individual_config = self.config['output_parameters']['individual_plots']

        # Create folder for individual plots
        folder_suffix = individual_config['folder_suffix']
        individual_plots_dir = os.path.join(base_dir, f"{base_name}{folder_suffix}")
        os.makedirs(individual_plots_dir, exist_ok=True)

        # Set font and line width scaling factors for individual subplots (use default 1.0)
        font_size_scale = 1.0
        line_width_scale = 1.0

        # Save each subplot
        saved_count = 0

        for ax_name, ax in self.axes.items():
            try:
                # Create new figure containing only current subplot
                individual_figsize = individual_config['figsize']
                individual_dpi = individual_config['dpi']
                fig_individual = plt.figure(figsize=individual_figsize, dpi=individual_dpi)
                ax_new = fig_individual.add_subplot(111)

                # Handle different subplot types differently
                if ax_name == 'ax1':  # Areal channel distribution
                    im = ax_new.imshow(self.pmapa.T, cmap='RdYlBu_r',
                                       origin=self.config['image_parameters']['origin'])
                    # Remove "(a)" prefix
                    ax_new.set_title(f"{self.text['areal_proportion']}",
                                     fontweight=self.config['font_parameters']['styles']['title_weight'],
                                     fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                    ax_new.set_xlabel(self.text['x_direction'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                    ax_new.set_ylabel(self.text['y_direction'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                    # Set tick styles
                    ax_new.tick_params(direction='out',
                                       length=self._get_tick_length('major', line_width_scale),
                                       width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                       labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                    # Add colorbar
                    cbar = self._add_colorbar(fig_individual, ax_new, im, self.text['channel_proportion'],
                                              font_size_scale, colorbar_type='areal_proportion')
                    cbar.set_label(self.text['channel_proportion'],
                                   fontsize=self._get_scaled_fontsize('colorbar_label', font_size_scale))
                    cbar.ax.tick_params(labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale))

                    save_name = "01_areal_proportion.png"

                elif ax_name == 'ax2':  # 3D voxel plot
                    # Try to regenerate PyVista image
                    success, pyvista_image = self.visualize_3d_with_pyvista()
                    if success:
                        ax_new.imshow(np.array(pyvista_image))
                        ax_new.axis('off')
                    else:
                        ax_new.text(0.5, 0.5, "PyVista not installed\nUnable to display 3D visualization",
                                    ha='center', va='center',
                                    fontsize=self._get_scaled_fontsize('info_text', font_size_scale))
                        ax_new.axis('off')

                    # Remove "(b)" prefix
                    ax_new.set_title(f"{self.text['3d_channel_model']}",
                                     fontweight=self.config['font_parameters']['styles']['title_weight'],
                                     fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
                    save_name = "02_3d_channel_model.png"

                elif ax_name == 'ax3':  # Channel cross-section
                    # Use same linear mapping as main plot
                    cross_section_data = self.channel[:, self.y_position, :].T

                    # Use masked array to handle background
                    masked_data = np.ma.masked_where(cross_section_data == 0, cross_section_data)

                    # First draw background
                    ax_new.imshow(np.where(cross_section_data == 0, 1, 0),
                                  cmap='gray',
                                  origin=self.config['image_parameters']['origin'],
                                  aspect=self.config['image_parameters']['aspect'],
                                  vmin=0, vmax=1, alpha=1.0)

                    # Then draw channel data
                    im = ax_new.imshow(masked_data,
                                       cmap=self.channel_cmap,
                                       origin=self.config['image_parameters']['origin'],
                                       aspect=self.config['image_parameters']['aspect'],
                                       interpolation=self.config['image_parameters']['interpolation'],
                                       vmin=self.channel_min,
                                       vmax=self.channel_max)

                    # Remove "(c)" prefix
                    ax_new.set_title(f"{self.text['cross_section']} (Y={self.y_position})",
                                     fontweight=self.config['font_parameters']['styles']['title_weight'],
                                     fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                    ax_new.set_xlabel(self.text['x_direction'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                    ax_new.set_ylabel(self.text['z_direction'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                    # Set tick styles
                    ax_new.tick_params(direction='out',
                                       length=self._get_tick_length('major', line_width_scale),
                                       width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                       labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                    # Add partition boundary lines
                    boundary_style = self.config['line_parameters']['styles']['boundary_style']
                    boundary_alpha = self._get_alpha_value('boundary_line')
                    for boundary_x in self.region_boundaries:
                        ax_new.axvline(x=boundary_x, color='r', linestyle=boundary_style, alpha=boundary_alpha,
                                       linewidth=self._get_scaled_linewidth('boundary_line', line_width_scale))

                    # Add region labels
                    region_label_offset = self.config['text_parameters']['positions']['region_label_y_offset']
                    region_label_ha = self.config['text_parameters']['alignment']['region_label_ha']
                    region_label_va = self.config['text_parameters']['alignment']['region_label_va']
                    region_label_alpha = self._get_alpha_value('region_label_box')
                    box_style = self.config['text_parameters']['box_styles']['region_label']

                    for mid_x, label in self.region_labels:
                        ax_new.text(mid_x, self.channel.shape[2] - region_label_offset, label,
                                    color='black', ha=region_label_ha, va=region_label_va,
                                    fontsize=self._get_scaled_fontsize('region_label', font_size_scale),
                                    bbox=dict(facecolor='white', alpha=region_label_alpha, boxstyle=box_style))

                    # Add colorbar - use fixed _add_colorbar
                    cbar = self._add_colorbar(fig_individual, ax_new, im, self.text['facies_code'],
                                              font_size_scale, colorbar_type='cross_section')
                    cbar.set_label(self.text['facies_code'],
                                   fontsize=self._get_scaled_fontsize('colorbar_label', font_size_scale))
                    cbar.ax.tick_params(labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale))

                    # Set colorbar range and labels
                    cbar.set_ticks(np.linspace(self.channel_min, self.channel_max, 6))
                    cbar.set_ticklabels([f'{int(val)}' for val in np.linspace(self.channel_min, self.channel_max, 6)])

                    save_name = "03_cross_section.png"

                elif ax_name == 'ax_heatmap':  # Channel density heatmap
                    if hasattr(self, 'heatmap_info'):
                        heatmap_data = self.heatmap_info['data']
                        grid_info = self.heatmap_info['grid_info']
                        heatmap_config = self.config['heatmap_parameters']

                        im = ax_new.imshow(heatmap_data.T,
                                           cmap=heatmap_config['colormap'],
                                           origin=self.config['image_parameters']['origin'],
                                           aspect=self.config['image_parameters']['aspect'],
                                           interpolation=self.config['image_parameters']['interpolation'])

                        # Set axis labels
                        num_x_regions = grid_info['num_x_regions']
                        num_z_regions = grid_info['num_z_regions']

                        # X-axis ticks and labels
                        x_tick_positions = np.arange(num_x_regions)
                        x_tick_labels = [f"X{i}" for i in range(num_x_regions)]
                        ax_new.set_xticks(x_tick_positions)
                        ax_new.set_xticklabels(x_tick_labels)

                        # Z-axis ticks and labels
                        z_tick_positions = np.arange(num_z_regions)
                        z_tick_labels = [f"Z{i}" for i in range(num_z_regions)]
                        ax_new.set_yticks(z_tick_positions)
                        ax_new.set_yticklabels(z_tick_labels)

                        # Set tick styles
                        ax_new.tick_params(direction='out',
                                           length=self._get_tick_length('major', line_width_scale),
                                           width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                           labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                        # Add grid lines
                        ax_new.set_xticks(np.arange(-0.5, num_x_regions, 1), minor=True)
                        ax_new.set_yticks(np.arange(-0.5, num_z_regions, 1), minor=True)
                        ax_new.grid(which='minor',
                                    color=heatmap_config['grid_color'],
                                    linestyle='-',
                                    linewidth=heatmap_config['grid_linewidth'])

                        # Add numerical annotations in each grid cell
                        threshold = heatmap_config['text_color_threshold']
                        for i in range(num_x_regions):
                            for j in range(num_z_regions):
                                density_value = heatmap_data[i, j]
                                if density_value > 0:
                                    # Choose text color based on density value
                                    text_color = (heatmap_config['text_colors']['dark'] if density_value > threshold
                                                  else heatmap_config['text_colors']['light'])
                                    ax_new.text(i, j, f'{density_value:.2f}',
                                                ha=self.config['text_parameters']['alignment']['heatmap_text_ha'],
                                                va=self.config['text_parameters']['alignment']['heatmap_text_va'],
                                                color=text_color,
                                                fontsize=self._get_scaled_fontsize('heatmap_text', font_size_scale),
                                                fontweight=self.config['font_parameters']['styles']['title_weight'])

                        # Remove "(d)" prefix
                        ax_new.set_title(f"{self.text['channel_density_heatmap']}",
                                         fontweight=self.config['font_parameters']['styles']['title_weight'],
                                         fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                        ax_new.set_xlabel(self.text['x_partition'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                        ax_new.set_ylabel(self.text['z_partition'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                        # Add colorbar - use fixed _add_colorbar
                        cbar = self._add_colorbar(fig_individual, ax_new, im, self.text['channel_density'],
                                                  font_size_scale, colorbar_type='channel_density_heatmap')
                        cbar.set_label(self.text['channel_density'],
                                       fontsize=self._get_scaled_fontsize('colorbar_label', font_size_scale))
                        cbar.ax.tick_params(labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale))

                        save_name = "04_channel_density_heatmap.png"
                    else:
                        continue

                elif ax_name == 'ax_vegetation':  # Vegetation distribution
                    # Check if vegetation data exists
                    vegetation_map = self.params.get('final_vegetation_map')
                    if vegetation_map is not None:
                        im = ax_new.imshow(vegetation_map.T, cmap='Greens',
                                           origin=self.config['image_parameters']['origin'],
                                           aspect=self.config['image_parameters']['aspect'])

                        # Remove prefix
                        ax_new.set_title(f"{self.text['vegetation_distribution']}",
                                         fontweight=self.config['font_parameters']['styles']['title_weight'],
                                         fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                        ax_new.set_xlabel(self.text['x_direction'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                        ax_new.set_ylabel(self.text['y_direction'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                        # Set tick styles
                        ax_new.tick_params(direction='out',
                                           length=self._get_tick_length('major', line_width_scale),
                                           width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                           labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                        # Add colorbar - use fixed _add_colorbar
                        cbar = self._add_colorbar(fig_individual, ax_new, im, self.text['vegetation_value'],
                                                  font_size_scale, colorbar_type='vegetation_distribution')
                        cbar.set_label(self.text['vegetation_value'],
                                       fontsize=self._get_scaled_fontsize('colorbar_label', font_size_scale))
                        cbar.ax.tick_params(labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale))

                        save_name = "05_vegetation_distribution.png"
                    else:
                        continue

                elif ax_name == 'ax_global':  # Global vertical distribution
                    ax_new.plot(self.pcurvea, range(len(self.pcurvea)), 'k-',
                                linewidth=self._get_scaled_linewidth('global_line', line_width_scale))
                    ax_new.fill_betweenx(range(len(self.pcurvea)), 0, self.pcurvea,
                                         color='lightgray', alpha=self._get_alpha_value('fill_area'))

                    # Remove prefix, keep only core title
                    ax_new.set_title(f"{self.text['global']} - {self.text['vertical_proportion']}",
                                     fontweight=self.config['font_parameters']['styles']['title_weight'],
                                     fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                    ax_new.set_xlabel(self.text['proportion'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                    ax_new.set_ylabel(self.text['z_direction'],
                                      fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                    # Set tick styles
                    ax_new.tick_params(direction='out',
                                       length=self._get_tick_length('major', line_width_scale),
                                       width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                       labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                    ax_new.grid(True, linestyle=self.config['line_parameters']['styles']['grid_style'],
                                alpha=self._get_alpha_value('grid_line'),
                                linewidth=self._get_scaled_linewidth('grid_line', line_width_scale))
                    ax_new.set_xlim(0, self.max_proportion * 1.2)

                    save_name = "06_global_vertical_distribution.png"

                elif ax_name.startswith('ax_region_'):  # Regional vertical distribution
                    region_idx = int(ax_name.split('_')[-1])
                    if region_idx < len(self.region_pcurves):
                        region_pcurve = self.region_pcurves[region_idx]
                        color = self.colors[region_idx] if region_idx < len(self.colors) else 'blue'

                        ax_new.plot(region_pcurve, range(len(region_pcurve)), '-',
                                    linewidth=self._get_scaled_linewidth('region_line', line_width_scale),
                                    color=color)
                        ax_new.fill_betweenx(range(len(region_pcurve)), 0, region_pcurve,
                                             color=color, alpha=self._get_alpha_value('fill_area'))

                        region_name = f"{self.text['region']} {region_idx + 1}"
                        # Remove prefix, keep only core title
                        ax_new.set_title(f"{region_name} - {self.text['vertical_proportion']}",
                                         fontweight=self.config['font_parameters']['styles']['title_weight'],
                                         fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))

                        ax_new.set_xlabel(self.text['proportion'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
                        ax_new.set_ylabel(self.text['z_direction'],
                                          fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))

                        # Set tick styles
                        ax_new.tick_params(direction='out',
                                           length=self._get_tick_length('major', line_width_scale),
                                           width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                           labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                        ax_new.grid(True, linestyle=self.config['line_parameters']['styles']['grid_style'],
                                    alpha=self._get_alpha_value('grid_line'),
                                    linewidth=self._get_scaled_linewidth('grid_line', line_width_scale))
                        ax_new.set_xlim(0, self.max_proportion * 1.2)
                        ax_new.set_ylim(0, self.nz)

                        save_name = f"07_region_{region_idx + 1}_vertical_distribution.png"
                    else:
                        continue

                else:
                    # Skip other subplot types
                    plt.close(fig_individual)
                    continue

                # Save individual plot
                save_path_individual = os.path.join(individual_plots_dir, save_name)
                plt.tight_layout()
                bbox_inches = individual_config['bbox_inches']
                fig_individual.savefig(save_path_individual, dpi=individual_dpi, bbox_inches=bbox_inches)
                plt.close(fig_individual)

                saved_count += 1

            except Exception as e:
                if 'fig_individual' in locals():
                    plt.close(fig_individual)
                continue

    def show(self, save_path=None, interactive=True):
        """
        Display visualization results with interactive control support
        """
        # Create figure
        figure_config = self.config['figure_parameters']
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi,
                              facecolor=figure_config['facecolor'],
                              edgecolor=figure_config['edgecolor'])

        # Reserve space for sliders if using interactive mode
        if interactive:
            interactive_config = self.config['layout_parameters']['interactive_layout']
            self.plot_area = plt.axes(interactive_config['plot_area_rect'])
            plt.sca(self.plot_area)

            # Create slider areas
            line_slider_ax = plt.axes(interactive_config['line_slider_rect'])
            font_slider_ax = plt.axes(interactive_config['font_slider_rect'])

            # Create slider widgets
            line_slider_config = self.config['slider_parameters']['line_width_slider']
            font_slider_config = self.config['slider_parameters']['font_size_slider']

            self.line_slider = Slider(
                ax=line_slider_ax,
                label=self.text['line_width'],
                valmin=line_slider_config['valmin'],
                valmax=line_slider_config['valmax'],
                valinit=line_slider_config['valinit'],
                valstep=line_slider_config['valstep']
            )

            self.font_slider = Slider(
                ax=font_slider_ax,
                label=self.text['font_size'],
                valmin=font_slider_config['valmin'],
                valmax=font_slider_config['valmax'],
                valinit=font_slider_config['valinit'],
                valstep=font_slider_config['valstep']
            )

            # Bind slider update functions
            self.line_slider.on_changed(self._update_style)
            self.font_slider.on_changed(self._update_style)

        # Initial drawing
        self._set_journal_style(self.journal_style, 1.0, 1.0)
        self._set_language_font(self.language)
        self._draw_figure(1.0, 1.0)

        # Save main figure (if path is specified)
        if save_path:
            output_config = self.config['output_parameters']['main_figure']
            plt.savefig(save_path,
                        dpi=output_config['dpi'],
                        bbox_inches=output_config['bbox_inches'],
                        facecolor=output_config['facecolor'],
                        edgecolor=output_config['edgecolor'])

        # Check if individual subplot export is needed
        if self.export_individual_plots:
            # If no main figure save path is specified, create a default path for individual subplots
            if save_path is None:
                import os
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"fluvpy_visualization_{timestamp}.png"

            self._save_individual_plots(save_path)

        # Display figure
        plt.show()

    def _update_style(self, val=None):
        """
        Update visual style
        """
        # Get current slider values
        line_width_scale = self.line_slider.val
        font_size_scale = self.font_slider.val

        # Apply style
        self._set_journal_style(self.journal_style, line_width_scale, font_size_scale)

        # Update all visual elements
        self._update_visuals(line_width_scale, font_size_scale)

        # Redraw figure
        self.fig.canvas.draw_idle()

    def _update_visuals(self, line_width_scale, font_size_scale):
        """
        Update all visual element styles without recalculating data
        """
        # Update line widths and marker sizes
        for key, artist in self.artists.items():
            if hasattr(artist, 'set_linewidth'):
                if 'line' in key:
                    if 'region' in key:
                        linewidth = self._get_scaled_linewidth('region_line', line_width_scale)
                    else:
                        linewidth = self._get_scaled_linewidth('global_line', line_width_scale)
                    artist.set_linewidth(linewidth)
                elif 'boundary' in key:
                    artist.set_linewidth(self._get_scaled_linewidth('boundary_line', line_width_scale))

            # Update fill areas
            if hasattr(artist, 'get_facecolor') and 'fill' in key:
                color = artist.get_facecolor()
                artist.set_alpha(self._get_alpha_value('fill_area'))

        # Update label font sizes and positions
        for key, text_obj in self.text_objects.items():
            if 'title' in key:
                if 'suptitle' in key:
                    text_obj.set_fontsize(self._get_scaled_fontsize('main_title', font_size_scale))
                else:
                    text_obj.set_fontsize(self._get_scaled_fontsize('subplot_title', font_size_scale))
            elif 'label' in key or 'region_name' in key:
                text_obj.set_fontsize(self._get_scaled_fontsize('axis_label', font_size_scale))
            elif 'heatmap_text' in key:
                text_obj.set_fontsize(self._get_scaled_fontsize('heatmap_text', font_size_scale))

        # Update axis styles
        for ax_name, ax in self.axes.items():
            # Update tick parameters
            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self._get_scaled_fontsize('tick_label', font_size_scale),
                width=self._get_scaled_linewidth('tick_major', line_width_scale),
                length=self._get_tick_length('major', line_width_scale)
            )

            # Update grid lines
            if ax.get_xgridlines() or ax.get_ygridlines():
                grid_style = self.config['line_parameters']['styles']['grid_style']
                grid_linewidth = self._get_scaled_linewidth('grid_line', line_width_scale)
                grid_alpha = self._get_alpha_value('grid_line')
                ax.grid(True, linestyle=grid_style,
                        linewidth=grid_linewidth,
                        alpha=grid_alpha, color='gray')

        # Update colorbars
        for key, cbar in self.cbar_objects.items():
            cbar.ax.tick_params(
                labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale),
                length=self._get_tick_length('major', line_width_scale),
                width=self._get_scaled_linewidth('tick_major', line_width_scale)
            )

            if hasattr(cbar, 'outline'):
                cbar.outline.set_linewidth(self._get_scaled_linewidth('colorbar_outline', line_width_scale))

            # Update colorbar labels
            if hasattr(cbar, 'ax') and hasattr(cbar.ax, 'get_ylabel'):
                cbar.ax.yaxis.label.set_fontsize(self._get_scaled_fontsize('colorbar_label', font_size_scale))

    def _draw_figure(self, line_width_scale, font_size_scale):
        """
        Draw complete figure
        """
        # Get layout configuration
        layout_config = self.config['layout_parameters']
        main_grid_config = layout_config['main_grid']
        region_subgrid_config = layout_config['region_subgrid']

        # Create GridSpec layout
        if self.use_partitioning and len(self.region_pcurves) > 0:
            # Define layout
            gs = gridspec.GridSpec(main_grid_config['nrows'], main_grid_config['ncols'],
                                   width_ratios=main_grid_config['width_ratios'],
                                   height_ratios=main_grid_config['height_ratios'],
                                   wspace=main_grid_config['wspace'],
                                   hspace=main_grid_config['hspace'])

            # Create subgrid for regional vertical distribution
            gs_regions = gridspec.GridSpecFromSubplotSpec(1, self.num_regions + 1,
                                                          subplot_spec=gs[2, :2],
                                                          wspace=region_subgrid_config['wspace'])
        else:
            # Use same layout but without partitioning
            gs = gridspec.GridSpec(main_grid_config['nrows'], main_grid_config['ncols'],
                                   width_ratios=main_grid_config['width_ratios'],
                                   height_ratios=main_grid_config['height_ratios'],
                                   wspace=main_grid_config['wspace'],
                                   hspace=main_grid_config['hspace'])

            gs_regions = gridspec.GridSpecFromSubplotSpec(1, self.num_regions + 1,
                                                          subplot_spec=gs[2, :2],
                                                          wspace=region_subgrid_config['wspace'])

        # ===== 1. Horizontal channel proportion map (upper left) =====
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.axes['ax1'] = ax1

        # Draw planar channel distribution
        image_config = self.config['image_parameters']
        im1 = ax1.imshow(self.pmapa.T, cmap='RdYlBu_r',
                         origin=image_config['origin'],
                         aspect=image_config['aspect'])
        self.artists['pmapa_img'] = im1

        # Add colorbar
        cbar1 = self._add_colorbar(self.fig, ax1, im1, self.text['channel_proportion'],
                                   font_size_scale, colorbar_type='areal_proportion')
        self.cbar_objects['cbar1'] = cbar1

        # Add title and labels
        title1 = ax1.set_title(f"(a) {self.text['areal_proportion']}",
                               fontweight=self.config['font_parameters']['styles']['title_weight'],
                               fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
        self.text_objects['title1'] = title1

        xlabel1 = ax1.set_xlabel(self.text['x_direction'],
                                 fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['xlabel1'] = xlabel1

        ylabel1 = ax1.set_ylabel(self.text['y_direction'],
                                 fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['ylabel1'] = ylabel1

        # Set tick styles
        ax1.tick_params(direction='out',
                        length=self._get_tick_length('major', line_width_scale),
                        width=self._get_scaled_linewidth('tick_major', line_width_scale),
                        labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

        # ===== 2. Three-dimensional voxel plot (upper right) =====
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.axes['ax2'] = ax2

        # Attempt to use PyVista for visualization
        success, pyvista_image = self.visualize_3d_with_pyvista()

        if success:
            # Display PyVista rendered image in matplotlib subplot
            ax2.imshow(np.array(pyvista_image))
            ax2.axis('off')  # Turn off axes for better visual effect

            # Set title
            title2 = ax2.set_title(f"(b) {self.text['3d_channel_model']}",
                                   fontweight=self.config['font_parameters']['styles']['title_weight'],
                                   fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
            self.text_objects['title2'] = title2
        else:
            # PyVista unavailable, display message
            ax2.text(0.5, 0.5, "PyVista not installed\nUnable to display 3D visualization\nPlease install PyVista: pip install pyvista",
                     ha='center', va='center',
                     fontsize=self._get_scaled_fontsize('info_text', font_size_scale))
            ax2.axis('off')

            # Set title
            title2 = ax2.set_title(f"(b) {self.text['3d_channel_model']}",
                                   fontweight=self.config['font_parameters']['styles']['title_weight'],
                                   fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
            self.text_objects['title2'] = title2

        # ===== 3. Channel cross-section view (middle) =====
        ax3 = self.fig.add_subplot(gs[1, :2])
        self.axes['ax3'] = ax3

        # Draw cross-section plot
        cross_section_data = self.channel[:, self.y_position, :].T

        # Use masked array to handle background
        masked_data = np.ma.masked_where(cross_section_data == 0, cross_section_data)

        # First draw background
        bg_colors = self.config['color_parameters']['background_colors']['channel_bg']
        ax3.imshow(np.where(cross_section_data == 0, 1, 0),
                   cmap='gray',
                   origin=image_config['origin'],
                   aspect=image_config['aspect'],
                   vmin=0, vmax=1, alpha=1.0)

        # Then draw channel data
        im3 = ax3.imshow(masked_data,
                         cmap=self.channel_cmap,
                         origin=image_config['origin'],
                         aspect=image_config['aspect'],
                         interpolation=image_config['interpolation'],
                         vmin=self.channel_min,
                         vmax=self.channel_max)

        self.artists['cross_section_img'] = im3

        # Add colorbar
        cbar3 = self._add_colorbar(self.fig, ax3, im3, self.text['facies_code'],
                                   font_size_scale, colorbar_type='cross_section')

        # Set colorbar range and labels
        cbar3.set_ticks(np.linspace(self.channel_min, self.channel_max, 6))
        cbar3.set_ticklabels([f'{int(val)}' for val in np.linspace(self.channel_min, self.channel_max, 6)])

        self.cbar_objects['cbar3'] = cbar3

        # Add title and labels
        title3 = ax3.set_title(
            f"(c) {self.text['cross_section']} (Y={self.y_position}, Channel range: {self.channel_min}-{self.channel_max})",
            fontweight=self.config['font_parameters']['styles']['title_weight'],
            fontsize=self._get_scaled_fontsize('subplot_title', font_size_scale))
        self.text_objects['title3'] = title3

        xlabel3 = ax3.set_xlabel(self.text['x_direction'],
                                 fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['xlabel3'] = xlabel3

        ylabel3 = ax3.set_ylabel(self.text['z_direction'],
                                 fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
        self.text_objects['ylabel3'] = ylabel3

        # Set tick styles
        ax3.tick_params(direction='out',
                        length=self._get_tick_length('major', line_width_scale),
                        width=self._get_scaled_linewidth('tick_major', line_width_scale),
                        labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

        # If partitioning boundaries exist, add them to the plot
        boundary_style = self.config['line_parameters']['styles']['boundary_style']
        boundary_alpha = self._get_alpha_value('boundary_line')
        for i, boundary_x in enumerate(self.region_boundaries):
            boundary_line = ax3.axvline(x=boundary_x, color='r', linestyle=boundary_style,
                                        linewidth=self._get_scaled_linewidth('boundary_line', line_width_scale),
                                        alpha=boundary_alpha)
            self.artists[f'boundary_line_{i}'] = boundary_line

        # Add region labels
        text_config = self.config['text_parameters']
        region_label_offset = text_config['positions']['region_label_y_offset']
        region_label_ha = text_config['alignment']['region_label_ha']
        region_label_va = text_config['alignment']['region_label_va']
        region_label_alpha = self._get_alpha_value('region_label_box')
        box_style = text_config['box_styles']['region_label']
        text_box_pad = self.config['position_parameters']['text_box_pad']

        for i, (mid_x, label) in enumerate(self.region_labels):
            region_label = ax3.text(mid_x, self.channel.shape[2] - region_label_offset, label,
                                    color='black', ha=region_label_ha, va=region_label_va,
                                    fontsize=self._get_scaled_fontsize('region_label', font_size_scale),
                                    bbox=dict(facecolor='white', alpha=region_label_alpha,
                                              boxstyle=f'round,pad={text_box_pad}'))
            self.text_objects[f'region_label_{i}'] = region_label

        # ===== 4. Channel density heatmap (right side) =====
        try:
            self._draw_channel_density_heatmap(gs, line_width_scale, font_size_scale)
        except Exception as e:
            import traceback
            traceback.print_exc()

        # ===== 5. Regional vertical distribution plots (bottom) =====
        if self.use_partitioning and len(self.region_pcurves) > 0:
            region_axes = []

            # Add global vertical distribution
            ax_global = self.fig.add_subplot(gs_regions[0])
            self.axes['ax_global'] = ax_global

            global_detail_line, = ax_global.plot(self.pcurvea, range(len(self.pcurvea)), 'k-',
                                                 linewidth=self._get_scaled_linewidth('global_line', line_width_scale))
            self.artists['global_detail_line'] = global_detail_line

            fill_alpha = self._get_alpha_value('fill_area')
            global_detail_fill = ax_global.fill_betweenx(range(len(self.pcurvea)), 0, self.pcurvea,
                                                         color='lightgray', alpha=fill_alpha)
            self.artists['global_detail_fill'] = global_detail_fill

            # Set title
            global_title = ax_global.set_title(f"{self.text['global']}",
                                               fontsize=self._get_scaled_fontsize('axis_label', font_size_scale),
                                               fontweight=self.config['font_parameters']['styles']['title_weight'])
            self.text_objects['global_title'] = global_title

            # Set axis range
            ax_global.set_xlim(0, self.max_proportion * 1.2)

            # Set grid
            grid_style = self.config['line_parameters']['styles']['grid_style']
            grid_linewidth = self._get_scaled_linewidth('grid_line', line_width_scale)
            grid_alpha = self._get_alpha_value('grid_line')
            ax_global.grid(True, linestyle=grid_style,
                           linewidth=grid_linewidth,
                           alpha=grid_alpha, color='gray')

            # Set Y-axis label
            global_ylabel = ax_global.set_ylabel(self.text['z_direction'],
                                                 fontsize=self._get_scaled_fontsize('axis_label', font_size_scale))
            self.text_objects['global_ylabel'] = global_ylabel

            # Set tick styles
            ax_global.tick_params(axis='y', direction='out',
                                  length=self._get_tick_length('major', line_width_scale),
                                  width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                  labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

            region_axes.append(ax_global)

            # Create vertical distribution plots for each region
            for i, region_pcurve in enumerate(self.region_pcurves):
                ax_region = self.fig.add_subplot(gs_regions[i + 1])
                self.axes[f'ax_region_{i}'] = ax_region

                region_detail_line, = ax_region.plot(region_pcurve, range(len(region_pcurve)), '-',
                                                     linewidth=self._get_scaled_linewidth('region_line',
                                                                                          line_width_scale),
                                                     color=self.colors[i])
                self.artists[f'region_detail_line_{i}'] = region_detail_line

                region_detail_fill = ax_region.fill_betweenx(range(len(region_pcurve)), 0, region_pcurve,
                                                             color=self.colors[i], alpha=fill_alpha)
                self.artists[f'region_detail_fill_{i}'] = region_detail_fill

                # Get region name
                region_name = f"{self.text['region']} {i + 1}"
                if self.region_controller and i < len(self.region_controller.regions):
                    region_id = self.region_controller.regions[i]['id']
                    region_name = f"{self.text['region']} {region_id + 1}"

                # Set title
                region_title = ax_region.set_title(region_name,
                                                   fontsize=self._get_scaled_fontsize('axis_label', font_size_scale),
                                                   fontweight=self.config['font_parameters']['styles']['title_weight'])
                self.text_objects[f'region_title_{i}'] = region_title

                # Set axis range
                ax_region.set_xlim(0, self.max_proportion * 1.2)

                # Set grid
                ax_region.grid(True, linestyle=grid_style,
                               linewidth=grid_linewidth,
                               alpha=grid_alpha, color='gray')

                # Show X-axis label only on rightmost subplot
                if i == len(self.region_pcurves) - 1:
                    region_xlabel = ax_region.set_xlabel(self.text['proportion'],
                                                         fontsize=self._get_scaled_fontsize('axis_label',
                                                                                            font_size_scale))
                    self.text_objects[f'region_xlabel_{i}'] = region_xlabel

                # Hide Y-axis tick labels on all subplots except leftmost
                ax_region.tick_params(axis='y', direction='out',
                                      length=self._get_tick_length('major', line_width_scale),
                                      width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                      labelsize=self._get_scaled_fontsize('tick_label', font_size_scale),
                                      labelleft=False)
                ax_region.tick_params(axis='x', direction='out',
                                      length=self._get_tick_length('major', line_width_scale),
                                      width=self._get_scaled_linewidth('tick_major', line_width_scale),
                                      labelsize=self._get_scaled_fontsize('tick_label', font_size_scale))

                region_axes.append(ax_region)

            # Ensure consistent Y-axis range for all regional vertical distribution plots
            y_min = 0
            y_max = self.nz
            for ax in region_axes:
                ax.set_ylim(y_min, y_max)

        else:
            # If no partitioning, add information text area
            ax4 = self.fig.add_subplot(gs[2, :2])
            self.axes['ax4'] = ax4
            ax4.axis('off')

            # Prepare information text content
            info_text = f"{self.text['simulation']} #{self.isim}\n"
            info_text += f"{self.text['global_proportion']}: {self.prop:.4f}\n"
            info_text += f"{self.text['grid_size']}: {self.channel.shape[0]}×{self.channel.shape[1]}×{self.channel.shape[2]}\n"

            # Add key parameters
            fcco = self.params.get('fcco', [0, 0, 0])
            fcct = self.params.get('fcct', [0, 0, 0])
            fccwtr = self.params.get('fccwtr', [0, 0, 0])
            fccntg = self.params.get('fccntg', [0, 0, 0])

            info_text += f"\n{self.text['simulation_parameters']}:\n"
            info_text += f"- {self.text['direction_angle']}: [{fcco[0]:.1f}, {fcco[1]:.1f}, {fcco[2]:.1f}]\n"
            info_text += f"- {self.text['thickness']}: [{fcct[0]:.1f}, {fcct[1]:.1f}, {fcct[2]:.1f}]\n"
            info_text += f"- {self.text['width_thick_ratio']}: [{fccwtr[0]:.1f}, {fccwtr[1]:.1f}, {fccwtr[2]:.1f}]\n"
            info_text += f"- {self.text['net_gross_ratio']}: [{fccntg[0]:.2f}, {fccntg[1]:.2f}, {fccntg[2]:.2f}]\n"

            # Display text information
            text_positions = self.config['text_parameters']['positions']
            text_alignment = self.config['text_parameters']['alignment']
            info_text_alpha = self._get_alpha_value('info_text_box')
            info_box_style = self.config['text_parameters']['box_styles']['info_text']

            info_text_obj = ax4.text(text_positions['info_text_x'], text_positions['info_text_y'],
                                     info_text, transform=ax4.transAxes,
                                     fontsize=self._get_scaled_fontsize('info_text', font_size_scale),
                                     verticalalignment=text_alignment['info_text_va'],
                                     horizontalalignment=text_alignment['info_text_ha'],
                                     fontfamily=self.config['font_parameters']['styles']['family'],
                                     bbox=dict(boxstyle=info_box_style, facecolor='white', alpha=info_text_alpha))
            self.text_objects['info_text'] = info_text_obj

        # Add vegetation distribution at appropriate location in _draw_figure method
        try:
            self._draw_vegetation_distribution(gs, line_width_scale, font_size_scale)
        except Exception as e:
            pass

        # ===== Add main title =====
        suptitle_y = self.config['text_parameters']['positions']['suptitle_y']
        suptitle = self.fig.suptitle(f"{self.text['title']} (Z={self.slice_z})",
                                     fontsize=self._get_scaled_fontsize('main_title', font_size_scale),
                                     y=suptitle_y,
                                     fontweight=self.config['font_parameters']['styles']['title_weight'])
        self.text_objects['suptitle'] = suptitle

        # Adjust layout
        layout_adjust_config = self.config['layout_parameters']['layout_adjust']
        try:
            plt.tight_layout(rect=layout_adjust_config['tight_layout_rect'])
        except:
            # Use simple adjustment as fallback
            subplots_config = layout_adjust_config['subplots_adjust']
            self.fig.subplots_adjust(top=subplots_config['top'],
                                     bottom=subplots_config['bottom'],
                                     left=subplots_config['left'],
                                     right=subplots_config['right'],
                                     wspace=subplots_config['wspace'],
                                     hspace=subplots_config['hspace'])

    def _get_language_text_enhanced(self, language: str) -> Dict[str, str]:
        """
        Get language text
        """
        if language == 'zh':
            base_text = {
                'title': 'Fluvpy Simulation Results Visualization',
                'simulation': 'Simulation',
                'channel_distribution': 'Channel Distribution',
                'porosity_distribution': 'Porosity Distribution',
                'permeability_distribution': 'Permeability Distribution',
                'areal_proportion': 'Areal Channel Proportion',
                'vertical_proportion': 'Vertical Channel Proportion',
                'cross_section': 'Cross Section',
                'facies_code': 'Facies Code',
                'porosity': 'Porosity',
                'permeability': 'Permeability (log10 mD)',
                'channel_proportion': 'Channel Proportion',
                'x_direction': 'X Direction',
                'y_direction': 'Y Direction',
                'z_direction': 'Z Direction',
                'proportion': 'Proportion',
                'global_proportion': 'Global Channel Proportion',
                'grid_size': 'Grid Size',
                'channel_count': 'Channel Count',
                'average_porosity': 'Average Porosity',
                'average_permeability': 'Average Permeability',
                'save_message': 'Figure saved to',
                'region': 'Region',
                'global': 'Global',
                'partition_info': 'Partition Information',
                'partition_axis': 'Partition Axis',
                'region_count': 'Region Count',
                'density': 'Density Factor',
                'region_statistics': 'Regional Statistics',
                'simulation_parameters': 'Simulation Parameters',
                'direction_angle': 'Direction Angle',
                'thickness': 'Thickness',
                'width_thick_ratio': 'Width-Thickness Ratio',
                'net_gross_ratio': 'Net-Gross Ratio',
                'line_width': 'Line Width',
                'font_size': 'Font Size',
                '3d_channel_model': '3D Channel Model',
                'x_axis': 'X Axis',
                'y_axis': 'Y Axis',
                'z_axis': 'Z Axis',
                # Heatmap related text
                'channel_density': 'Channel Density',
                'channel_density_heatmap': 'Channel Density Heatmap',
                'x_partition': 'X Partitions',
                'z_partition': 'Z Partitions',
                'density_value': 'Density Value',
                'normalized_density': 'Normalized Density',
                'heatmap_stats': 'Heatmap Statistics',
                'vegetation_distribution': 'Vegetation Distribution',
                'vegetation_value': 'Vegetation Value',
                'vegetation_influence': 'Vegetation Influence',
            }
        else:  # Default English
            base_text = {
                'title': 'fluvpy Simulation Results',
                'simulation': 'Simulation',
                'channel_distribution': 'Channel Distribution',
                'porosity_distribution': 'Porosity Distribution',
                'permeability_distribution': 'Permeability Distribution',
                'areal_proportion': 'Areal Channel Proportion',
                'vertical_proportion': 'Vertical Channel Proportion',
                'cross_section': 'Cross Section',
                'facies_code': 'Facies Code',
                'porosity': 'Porosity',
                'permeability': 'Permeability (log10 mD)',
                'channel_proportion': 'Channel Proportion',
                'x_direction': 'X Direction',
                'y_direction': 'Y Direction',
                'z_direction': 'Z Direction',
                'proportion': 'Proportion',
                'global_proportion': 'Global Channel Proportion',
                'grid_size': 'Grid Size',
                'channel_count': 'Channel Count',
                'average_porosity': 'Average Porosity',
                'average_permeability': 'Average Permeability',
                'save_message': 'Figure saved to',
                'region': 'Region',
                'global': 'Global',
                'partition_info': 'Partition Information',
                'partition_axis': 'Partition Axis',
                'region_count': 'Region Count',
                'density': 'Density Factor',
                'region_statistics': 'Region Statistics',
                'simulation_parameters': 'Simulation Parameters',
                'direction_angle': 'Direction Angle',
                'thickness': 'Thickness',
                'width_thick_ratio': 'Width-Thickness Ratio',
                'net_gross_ratio': 'Net-Gross Ratio',
                'line_width': 'Line Width',
                'font_size': 'Font Size',
                '3d_channel_model': '3D Channel Model',
                'x_axis': 'X Axis',
                'y_axis': 'Y Axis',
                'z_axis': 'Z Axis',
                # Heatmap related text
                'channel_density': 'Channel Density',
                'channel_density_heatmap': 'Channel Density Heatmap',
                'x_partition': 'X Partitions',
                'z_partition': 'Z Partitions',
                'density_value': 'Density Value',
                'normalized_density': 'Normalized Density',
                'heatmap_stats': 'Heatmap Statistics',
                'vegetation_distribution': 'Vegetation Distribution',
                'vegetation_value': 'Vegetation Value',
                'vegetation_influence': 'Vegetation Influence',
            }

        return base_text

    def _set_journal_style(self, style: str, line_width_scale: float = 1.0, font_size_scale: float = 1.0) -> None:
        """
        Set journal-style plotting parameters
        """
        # First reset to default style
        plt.style.use('default')

        # Get configuration parameters
        base_font_size = self.config['base_parameters']['BASE_FONTSIZE'] * font_size_scale
        base_line_width = self.config['base_parameters']['BASE_LINEWIDTH'] * line_width_scale
        font_config = self.config['font_parameters']['styles']

        mpl.rcParams['font.family'] = font_config['family']
        mpl.rcParams['font.size'] = base_font_size
        mpl.rcParams['axes.labelsize'] = base_font_size
        mpl.rcParams['axes.titlesize'] = base_font_size * self.config['font_parameters']['size_scales']['subplot_title']
        mpl.rcParams['xtick.labelsize'] = base_font_size * self.config['font_parameters']['size_scales']['tick_label']
        mpl.rcParams['ytick.labelsize'] = base_font_size * self.config['font_parameters']['size_scales']['tick_label']
        mpl.rcParams['legend.fontsize'] = base_font_size * self.config['font_parameters']['size_scales']['tick_label']
        mpl.rcParams['figure.titlesize'] = base_font_size * self.config['font_parameters']['size_scales']['main_title']

        # Axis lines and tick lines
        mpl.rcParams['axes.linewidth'] = base_line_width
        mpl.rcParams['xtick.major.width'] = base_line_width
        mpl.rcParams['ytick.major.width'] = base_line_width
        mpl.rcParams['xtick.minor.width'] = base_line_width * self.config['line_parameters']['width_scales'][
            'tick_minor']
        mpl.rcParams['ytick.minor.width'] = base_line_width * self.config['line_parameters']['width_scales'][
            'tick_minor']
        mpl.rcParams['xtick.major.size'] = self.config['line_parameters']['tick_lengths'][
                                               'major_length'] * line_width_scale
        mpl.rcParams['ytick.major.size'] = self.config['line_parameters']['tick_lengths'][
                                               'major_length'] * line_width_scale
        mpl.rcParams['xtick.minor.size'] = self.config['line_parameters']['tick_lengths'][
                                               'minor_length'] * line_width_scale
        mpl.rcParams['ytick.minor.size'] = self.config['line_parameters']['tick_lengths'][
                                               'minor_length'] * line_width_scale

        # Legend style
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = self.config['color_parameters']['alpha_values']['text_box']
        mpl.rcParams['legend.edgecolor'] = '0.8'
        mpl.rcParams['legend.fancybox'] = True

        # Get journal style configuration
        journal_config = self.config['journal_style_parameters'].get(style,
                                                                     self.config['journal_style_parameters']['nature'])
        colors_config = self.config['color_parameters']

        # Specific journal style settings
        if style == 'nature':
            # Nature style: clean, thin lines, no grid
            mpl.rcParams['axes.grid'] = journal_config['grid_enabled']
            mpl.rcParams['axes.linewidth'] = base_line_width * journal_config['axis_line_scale']
            mpl.rcParams['xtick.major.width'] = base_line_width * journal_config['tick_line_scale']
            mpl.rcParams['ytick.major.width'] = base_line_width * journal_config['tick_line_scale']
            mpl.rcParams['xtick.minor.width'] = base_line_width * journal_config['tick_line_scale'] * \
                                                self.config['line_parameters']['width_scales']['tick_minor']
            mpl.rcParams['ytick.minor.width'] = base_line_width * journal_config['tick_line_scale'] * \
                                                self.config['line_parameters']['width_scales']['tick_minor']
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_config['journal_colors']['nature'])
            mpl.rcParams['figure.facecolor'] = colors_config['background_colors']['figure_bg']
            mpl.rcParams['axes.facecolor'] = colors_config['background_colors']['axes_bg']

        elif style == 'science':
            # Science style: with grid, lines slightly thicker than Nature
            mpl.rcParams['axes.grid'] = journal_config['grid_enabled']
            mpl.rcParams['grid.alpha'] = journal_config['grid_alpha']
            mpl.rcParams['axes.linewidth'] = base_line_width * journal_config['axis_line_scale']
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_config['journal_colors']['science'])
            mpl.rcParams['figure.facecolor'] = colors_config['background_colors']['figure_bg']
            mpl.rcParams['axes.facecolor'] = colors_config['background_colors']['science_axes_bg']
            mpl.rcParams['grid.color'] = colors_config['grid_colors']['science']
            mpl.rcParams['grid.linestyle'] = journal_config['grid_linestyle']
            mpl.rcParams['grid.linewidth'] = base_line_width * journal_config['grid_linewidth_scale']

        elif style == 'geology':
            # Geology journal style: dashed grid, light background
            mpl.rcParams['axes.grid'] = journal_config['grid_enabled']
            mpl.rcParams['grid.linestyle'] = journal_config['grid_linestyle']
            mpl.rcParams['grid.alpha'] = journal_config['grid_alpha']
            mpl.rcParams['axes.facecolor'] = colors_config['background_colors']['geology_axes_bg']
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_config['journal_colors']['geology'])
            mpl.rcParams['figure.facecolor'] = colors_config['background_colors']['geology_figure_bg']
            mpl.rcParams['grid.color'] = colors_config['grid_colors']['geology']
            mpl.rcParams['grid.linewidth'] = base_line_width * journal_config['grid_linewidth_scale']

    def _set_language_font(self, language: str) -> None:
        """
        Set language font support
        """
        if language == 'zh':
            # Get Chinese font list
            chinese_fonts = self.config['font_parameters']['chinese_fonts']

            # Check and set Chinese font
            font_found = False
            try:
                import matplotlib.font_manager as fm
                available_fonts = [f.name for f in fm.fontManager.ttflist]

                for font in chinese_fonts:
                    if font in available_fonts:
                        plt.rcParams['font.family'] = [font, 'sans-serif']
                        plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial Unicode MS']
                        font_found = True
                        break
            except:
                pass

            if not font_found:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']

            mpl.rcParams['font.monospace'] = [self.config['font_parameters']['styles']['monospace_family'], 'SimHei',
                                              'monospace']

            # Solve minus sign display issue
            plt.rcParams['axes.unicode_minus'] = False

    def _create_channel_colormap(self):
        """
        Create channel color mapping
        """
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        import matplotlib.colors as mcolors

        # Get channel color configuration
        channel_colors_hex = self.config['color_parameters']['channel_colors']

        # Get channel data range
        channel_mask = self.channel > 0
        if np.any(channel_mask):
            min_channel = np.min(self.channel[channel_mask])
            max_channel = np.max(self.channel[channel_mask])

            # Save channel value range for later use
            self.channel_min = min_channel
            self.channel_max = max_channel
        else:
            self.channel_min = 1
            self.channel_max = 11

        # Create custom color mapping
        cmap = LinearSegmentedColormap.from_list(
            'channel_linear',
            channel_colors_hex,
            N=256
        )

        # Set background color
        bg_color = self.config['color_parameters']['background_colors']['channel_bg']
        cmap.set_under(color=bg_color)

        return cmap

    def _create_custom_3d_colormap(self):
        """
        Create custom colormap for 3D visualization
        """
        from matplotlib.colors import ListedColormap

        # Get configured color scheme
        custom_colors = self.config['color_parameters']['channel_colors']

        # Create custom colormap
        self.voxel_cmap = ListedColormap(custom_colors)

    def _add_colorbar(self, fig: plt.Figure, ax: plt.Axes, im: plt.cm.ScalarMappable,
                      title: str, font_size_scale: float = 1.0, colorbar_type: str = None):
        """
        Add colorbar to subplot with configurable positioning

        Parameters:
            fig: matplotlib figure object
            ax: matplotlib axis object
            im: mappable color object
            title: colorbar title
            font_size_scale: font scaling factor
            colorbar_type: colorbar type, used to determine position configuration
        """
        # Get colorbar configuration
        colorbar_config = self.config['colorbar_parameters']

        # Determine position configuration based on colorbar type
        if colorbar_type and colorbar_type in colorbar_config['positions']:
            pos_config = colorbar_config['positions'][colorbar_type]
        else:
            pos_config = colorbar_config['default_position']

        # Create colorbar
        cbar = None

        # Select creation method based on position configuration
        location = pos_config['location'].lower()

        if location in ['right', 'left', 'top', 'bottom']:
            # Use make_axes_locatable method
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)

            # Create colorbar axis according to configuration - fix pad=0 ineffective issue
            pad_value = pos_config['pad']
            if pad_value == 0:
                pad_value = 0.01  # Use minimal value instead of 0 to avoid pad=0 ineffectiveness

            cax = divider.append_axes(
                location,
                size=pos_config['size'],
                pad=pad_value  # Fixed pad value
            )

            # Create colorbar
            cbar = fig.colorbar(
                im,
                cax=cax,
                orientation=pos_config['orientation'],
                aspect=pos_config['aspect'],
                shrink=pos_config['shrink']
            )

        else:
            # Use matplotlib built-in method (suitable for complex position configurations)
            pad_value = pos_config['pad']
            if pad_value == 0:
                pad_value = 0.01  # Same fix for pad=0 issue

            cbar = fig.colorbar(
                im,
                ax=ax,
                location=location if location in ['right', 'left', 'top', 'bottom'] else 'right',
                orientation=pos_config['orientation'],
                aspect=pos_config['aspect'],
                shrink=pos_config['shrink'],
                pad=pad_value,  # Fixed pad value
                anchor=pos_config['anchor'],
                panchor=pos_config['panchor']
            )

        # If creation fails, use default method
        if cbar is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.01)  # Default also uses 0.01 instead of 0
            cbar = fig.colorbar(im, cax=cax)

        # Set colorbar title
        cbar.set_label(title, fontsize=self._get_scaled_fontsize('colorbar_label', font_size_scale))

        # Set colorbar tick label font size
        cbar.ax.tick_params(labelsize=self._get_scaled_fontsize('colorbar_tick', font_size_scale))

        # Optimize colorbar appearance
        cbar.outline.set_linewidth(self._get_scaled_linewidth('colorbar_outline', font_size_scale))
        cbar.ax.tick_params(
            length=colorbar_config['tick_length'],
            width=self._get_scaled_linewidth('tick_major', font_size_scale) * colorbar_config['tick_width_scale']
        )
        return cbar


def visualize_fluvpy_results(
        results: Dict[str, Any],
        isim: int = 1,
        slice_z: Optional[int] = None,
        slice_y: Optional[int] = None,
        figsize: Tuple[int, int] = None,
        dpi: int = None,
        save_path: Optional[str] = None,
        journal_style: str = 'nature',
        language: str = 'zh',
        force_partitioning: bool = True,
        interactive: bool = True,
        export_individual_plots: bool = False
) -> None:
    """
    Visualize fluvpy simulation results

    Parameters:
        results: Results dictionary returned by fluvpy
        isim: Realization number to visualize
        slice_z: Z-slice index to display, if None shows middle slice
        figsize: Figure size, if None uses default value from configuration
        dpi: Resolution, if None uses default value from configuration
        save_path: Save path, if None does not save
        journal_style: Journal style, options: 'nature', 'science', 'geology'
        language: Display language, 'en' for English, 'zh' for Chinese
        force_partitioning: Whether to force enable partitioning visualization
        interactive: Whether to enable interactive controls
        export_individual_plots: Whether to export individual subplots
    """
    # Get configuration to determine default figure size
    config = get_visualization_output_config()

    # Automatically adjust figure size to fit heatmaps
    if figsize is None:
        default_figsize = config['figure_parameters']['figsize']
        adjusted_figsize = (max(14, default_figsize[0] + 4), max(10, default_figsize[1]))
    else:
        adjusted_figsize = figsize if figsize[0] >= 12 else (max(14, figsize[0] + 4), max(10, figsize[1]))

    # Create visualization object
    visualizer = fluvpy3DVisualizer(
        results=results,
        isim=isim,
        slice_z=slice_z,
        slice_y=slice_y,
        figsize=adjusted_figsize,
        dpi=dpi,
        journal_style=journal_style,
        language=language,
        force_partitioning=force_partitioning,
        export_individual_plots=export_individual_plots
    )

    # Display results
    visualizer.show(save_path=save_path, interactive=interactive)



def visualize_channel_parameters_distribution(results, isim=1, journal_style='nature', language='en', save_path=None,
                                              figsize=(12, 10), dpi=150):
    """
    Visualize channel parameter distributions, including distribution types and statistical analysis

    Parameters:
        results: fluvpy simulation results dictionary
        isim: Realization number to visualize
        journal_style: Journal style ('nature', 'science', 'geology')
        language: Label language ('zh' or 'en')
        save_path: Path to save figure, if None displays directly
        figsize: Figure size (width, height)
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec
    from matplotlib.font_manager import FontProperties
    import os

    # Get realization data and parameters
    realization = results[f'realization_{isim}']
    params = realization.get('params', {})

    # Configure Chinese font support
    if language == 'zh':
        import platform
        system = platform.system()

        if system == 'Windows':
            try:
                font_path = 'C:/Windows/Fonts/msyh.ttc'
                if not os.path.exists(font_path):
                    font_path = 'C:/Windows/Fonts/simsun.ttc'
                if os.path.exists(font_path):
                    chinese_font = FontProperties(fname=font_path)
                else:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        elif system == 'Darwin':  # macOS
            try:
                font_path = '/System/Library/Fonts/PingFang.ttc'
                if not os.path.exists(font_path):
                    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
                if os.path.exists(font_path):
                    chinese_font = FontProperties(fname=font_path)
                else:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        elif system == 'Linux':
            try:
                font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                    '/usr/share/fonts/truetype/arphic/uming.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                ]

                font_found = False
                for path in font_paths:
                    if os.path.exists(path):
                        chinese_font = FontProperties(fname=path)
                        font_found = True
                        break

                if not font_found:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        else:
            chinese_font = FontProperties()
            language = 'en'
    else:
        chinese_font = None

    # Set style
    try:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'
    except Exception as e:
        pass

    # Set color scheme according to journal style
    if journal_style == 'nature':
        color_palette = plt.cm.viridis
        main_color = '#2C3E50'
        accent_color = '#3498DB'
        edge_color = '#7F8C8D'
    elif journal_style == 'science':
        color_palette = plt.cm.plasma
        main_color = '#34495E'
        accent_color = '#E74C3C'
        edge_color = '#BDC3C7'
    elif journal_style == 'geology':
        color_palette = plt.cm.terrain
        main_color = '#5D6D7E'
        accent_color = '#27AE60'
        edge_color = '#95A5A6'
    else:
        color_palette = plt.cm.viridis
        main_color = '#2C3E50'
        accent_color = '#3498DB'
        edge_color = '#7F8C8D'

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[0.2, 1, 1, 1])

    # Determine text based on language
    if language == 'zh':
        title_text = 'Channel Parameter Distribution Statistical Analysis'
        param_title = 'Parameter Distribution Types'
        thickness_title = 'Channel Thickness Distribution'
        wtr_title = 'Width-to-Thickness Ratio Distribution'
        scatter_title = 'Thickness vs Width-to-Thickness Ratio'
        x_label_thickness = 'Thickness (m)'
        x_label_wtr = 'Width-to-Thickness Ratio'
        y_label = 'Frequency'
        dist_type_text = {
            'triangular': 'Triangular Distribution',
            'discrete': 'Discrete Distribution',
            'custom': 'Custom Range Distribution',
            'normal': 'Normal Distribution',
            'uniform': 'Uniform Distribution',
            'lognormal': 'Log-Normal Distribution'
        }
        region_text = 'Region'
    else:  # 'en'
        title_text = 'Channel Parameter Distribution Analysis'
        param_title = 'Parameter Distribution Types'
        thickness_title = 'Channel Thickness Distribution'
        wtr_title = 'Width-to-Thickness Ratio Distribution'
        scatter_title = 'Thickness vs Width-to-Thickness Ratio'
        x_label_thickness = 'Thickness (m)'
        x_label_wtr = 'Width-to-Thickness Ratio'
        y_label = 'Frequency'
        dist_type_text = {
            'triangular': 'Triangular',
            'discrete': 'Discrete',
            'custom': 'Custom Range',
            'normal': 'Normal',
            'uniform': 'Uniform',
            'lognormal': 'Log-Normal'
        }
        region_text = 'Region'

    # Add title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    if chinese_font and language == 'zh':
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16,
                      fontweight='bold', color=main_color, fontproperties=chinese_font)
    else:
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16,
                      fontweight='bold', color=main_color)

    # Function to determine distribution type
    def get_distribution_type(param):
        if isinstance(param, dict) and 'type' in param:
            return param['type']
        elif isinstance(param, list) and len(param) >= 3:
            return 'triangular'
        else:
            return 'unknown'

    # Extract and display parameter distribution types
    ax_params = fig.add_subplot(gs[1, 0])
    ax_params.axis('off')

    # Determine parameter distributions
    fcct_type = get_distribution_type(params.get('fcct', []))
    fccwtr_type = get_distribution_type(params.get('fccwtr', []))
    fccntg_type = get_distribution_type(params.get('fccntg', []))
    fcco_type = get_distribution_type(params.get('fcco', []))

    # Check if region-specific parameters exist
    use_partitioning = params.get('use_partitioning', False)
    region_configs = params.get('region_configs', {})

    # Display parameter distribution types
    if language == 'zh':
        param_info = [
            f"Complex Thickness: {dist_type_text.get(fcct_type, fcct_type)}",
            f"Complex Width/Thickness: {dist_type_text.get(fccwtr_type, fccwtr_type)}",
            f"Complex Net/Gross Ratio: {dist_type_text.get(fccntg_type, fccntg_type)}",
            f"Complex Orientation: {dist_type_text.get(fcco_type, fcco_type)}"
        ]
    else:
        param_info = [
            f"Complex Thickness: {dist_type_text.get(fcct_type, fcct_type)}",
            f"Complex Width/Thickness: {dist_type_text.get(fccwtr_type, fccwtr_type)}",
            f"Complex Net/Gross Ratio: {dist_type_text.get(fccntg_type, fccntg_type)}",
            f"Complex Orientation: {dist_type_text.get(fcco_type, fcco_type)}"
        ]

    # Add region-specific information if available
    if use_partitioning and region_configs:
        param_info.append("\nRegion-specific parameters:" if language == 'zh' else "\nRegion-specific parameters:")
        for region_id, config in region_configs.items():
            param_info.append(f"- {region_text} {region_id}:")
            if 'fcct' in config:
                reg_type = get_distribution_type(config['fcct'])
                param_info.append(
                    f"  - {'Thickness' if language == 'zh' else 'Thickness'}: {dist_type_text.get(reg_type, reg_type)}")
            if 'fccwtr' in config:
                reg_type = get_distribution_type(config['fccwtr'])
                param_info.append(
                    f"  - {'Width/Thickness' if language == 'zh' else 'Width/Thickness'}: {dist_type_text.get(reg_type, reg_type)}")

    # Display parameter information text
    if chinese_font and language == 'zh':
        ax_params.text(0.05, 0.95, '\n'.join(param_info), va='top', ha='left',
                       fontsize=9, color=main_color, fontproperties=chinese_font)
        ax_params.set_title(param_title, fontsize=12, color=main_color, fontproperties=chinese_font)
    else:
        ax_params.text(0.05, 0.95, '\n'.join(param_info), va='top', ha='left',
                       fontsize=9, color=main_color)
        ax_params.set_title(param_title, fontsize=12, color=main_color)

    # Extract channel data for visualization
    thickness_data = []
    wtr_data = []

    if 'main_channels_info' in realization:
        for channel_info in realization['main_channels_info']:
            # Extract thickness and width arrays
            thickness_array = channel_info.get('thickness', [])
            width_array = channel_info.get('width', [])

            # Calculate average thickness and width-to-thickness ratio
            if len(thickness_array) > 0:
                avg_thickness = np.mean(thickness_array)
                thickness_data.append(avg_thickness)

                if len(width_array) == len(thickness_array) and len(width_array) > 0:
                    # Calculate width-to-thickness ratio
                    wtr_values = [w / t if t > 0 else 0 for w, t in zip(width_array, thickness_array)]
                    avg_wtr = np.mean(wtr_values)
                    wtr_data.append(avg_wtr)

    # If unable to extract from main_channels_info, try using centerlines
    if not thickness_data and 'centerlines' in realization:
        centerlines = realization['centerlines']
        if 'main' in centerlines:
            for centerline in centerlines['main']:
                if 'thickness' in centerline and len(centerline['thickness']) > 0:
                    avg_thickness = np.mean(centerline['thickness'])
                    thickness_data.append(avg_thickness)

                if 'width' in centerline and 'thickness' in centerline and \
                        len(centerline['width']) == len(centerline['thickness']) and \
                        len(centerline['width']) > 0:
                    # Calculate width-to-thickness ratio
                    width_array = centerline['width']
                    thickness_array = centerline['thickness']
                    wtr_values = [w / t if t > 0 else 0 for w, t in zip(width_array, thickness_array)]
                    avg_wtr = np.mean(wtr_values)
                    wtr_data.append(avg_wtr)

    # If we still have no data, use available raw arrays in realization
    if not thickness_data:
        ct_array = params.get('ct', [])
        cw_array = params.get('cw', [])

        if isinstance(ct_array, np.ndarray) and ct_array.size > 0:
            # Extract valid thickness and width data from 3D array
            for icc in range(len(params.get('ccx', []))):
                for ic in range(len(params.get('cx', [])[icc])):
                    thickness_values = ct_array[icc, ic, :]
                    valid_indices = np.where(thickness_values > 0)[0]

                    if len(valid_indices) > 0:
                        avg_thickness = np.mean(thickness_values[valid_indices])
                        thickness_data.append(avg_thickness)

                        if isinstance(cw_array, np.ndarray) and cw_array.size > 0:
                            width_values = cw_array[icc, ic, valid_indices]
                            wtr_values = width_values / thickness_values[valid_indices]
                            avg_wtr = np.mean(wtr_values)
                            wtr_data.append(avg_wtr)

    # If still no data, generate sample data using distribution information from parameters
    if not thickness_data:
        # Generate sample thickness data from fcct parameter
        def generate_sample_data(dist_param, n_samples=100):
            if isinstance(dist_param, dict) and 'type' in dist_param:
                if dist_param['type'] == 'normal':
                    mean = dist_param.get('mean', 5.0)
                    std = dist_param.get('std', 1.0)
                    return np.random.normal(mean, std, n_samples)
                elif dist_param['type'] == 'discrete':
                    values = dist_param.get('values', [5.0])
                    probabilities = dist_param.get('probabilities', None)
                    return np.random.choice(values, size=n_samples, p=probabilities)
                elif dist_param['type'] == 'custom':
                    # Simplified handling of custom ranges
                    ranges = dist_param.get('ranges', [(0, 1)])
                    weights = dist_param.get('weights', [1.0])
                    # Select ranges
                    selected_range_idx = np.random.choice(len(ranges), size=n_samples,
                                                          p=np.array(weights) / sum(weights))
                    result = []
                    for idx in selected_range_idx:
                        min_val, max_val = ranges[idx]
                        result.append(np.random.uniform(min_val, max_val))
                    return np.array(result)
            elif isinstance(dist_param, list) and len(dist_param) >= 3:
                # Triangular distribution
                min_val, mode_val, max_val = dist_param[:3]
                return np.random.triangular(min_val, mode_val, max_val, n_samples)
            else:
                # Default uniform distribution
                return np.random.uniform(1.0, 10.0, n_samples)

        # Generate thickness and width-to-thickness ratio samples
        thickness_samples = generate_sample_data(params.get('fcct', [5.0, 7.0, 8.5]), 100)

        # Calculate width-to-thickness ratio
        if isinstance(params.get('fccwtr', []), (list, dict)):
            wtr_samples = generate_sample_data(params.get('fccwtr', [1.2, 1.5, 1.7]), 100)
        else:
            wtr_samples = np.random.uniform(1.0, 2.0, 100)

        thickness_data = thickness_samples
        wtr_data = wtr_samples

    # Convert to numpy arrays for analysis
    thickness_data = np.array(thickness_data)
    wtr_data = np.array(wtr_data)

    # Calculate statistical data
    thickness_stats = {
        'mean': np.mean(thickness_data) if len(thickness_data) > 0 else 0,
        'median': np.median(thickness_data) if len(thickness_data) > 0 else 0,
        'std': np.std(thickness_data) if len(thickness_data) > 0 else 0,
        'min': np.min(thickness_data) if len(thickness_data) > 0 else 0,
        'max': np.max(thickness_data) if len(thickness_data) > 0 else 0
    }

    wtr_stats = {
        'mean': np.mean(wtr_data) if len(wtr_data) > 0 else 0,
        'median': np.median(wtr_data) if len(wtr_data) > 0 else 0,
        'std': np.std(wtr_data) if len(wtr_data) > 0 else 0,
        'min': np.min(wtr_data) if len(wtr_data) > 0 else 0,
        'max': np.max(wtr_data) if len(wtr_data) > 0 else 0
    }

    # Plot thickness histogram
    ax_thickness = fig.add_subplot(gs[1, 1:])
    if len(thickness_data) > 0:
        # Determine number of bins based on data
        n_bins = min(max(5, int(np.sqrt(len(thickness_data)))), 15)

        # Create histogram
        hist, bins, _ = ax_thickness.hist(thickness_data, bins=n_bins, color=accent_color,
                                          alpha=0.7, edgecolor=edge_color, linewidth=1)

        # Add normal distribution curve if sufficient data
        if len(thickness_data) >= 5:
            x = np.linspace(thickness_stats['min'], thickness_stats['max'], 100)
            y = len(thickness_data) * (bins[1] - bins[0]) * \
                np.exp(-(x - thickness_stats['mean']) ** 2 / (2 * thickness_stats['std'] ** 2)) / \
                (thickness_stats['std'] * np.sqrt(2 * np.pi))
            ax_thickness.plot(x, y, linewidth=2, color=main_color)

        # Add statistical text
        if language == 'zh':
            stats_text = f"Mean: {thickness_stats['mean']:.2f}\nMedian: {thickness_stats['median']:.2f}\n" \
                         f"Std Dev: {thickness_stats['std']:.2f}\nRange: [{thickness_stats['min']:.2f}, {thickness_stats['max']:.2f}]"
        else:
            stats_text = f"Mean: {thickness_stats['mean']:.2f}\nMedian: {thickness_stats['median']:.2f}\n" \
                         f"Std Dev: {thickness_stats['std']:.2f}\nRange: [{thickness_stats['min']:.2f}, {thickness_stats['max']:.2f}]"

        # Set font based on language
        if chinese_font and language == 'zh':
            ax_thickness.text(0.95, 0.95, stats_text, transform=ax_thickness.transAxes,
                              fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round',
                                                                          facecolor='white', alpha=0.7),
                              fontproperties=chinese_font)
        else:
            ax_thickness.text(0.95, 0.95, stats_text, transform=ax_thickness.transAxes,
                              fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round',
                                                                          facecolor='white', alpha=0.7))

    # Set thickness plot title and labels
    if chinese_font and language == 'zh':
        ax_thickness.set_title(thickness_title, fontsize=12, color=main_color, fontproperties=chinese_font)
        ax_thickness.set_xlabel(x_label_thickness, fontsize=10, color=main_color, fontproperties=chinese_font)
        ax_thickness.set_ylabel(y_label, fontsize=10, color=main_color, fontproperties=chinese_font)
    else:
        ax_thickness.set_title(thickness_title, fontsize=12, color=main_color)
        ax_thickness.set_xlabel(x_label_thickness, fontsize=10, color=main_color)
        ax_thickness.set_ylabel(y_label, fontsize=10, color=main_color)

    ax_thickness.grid(True, linestyle='--', alpha=0.7)

    # Add regional partitioning information
    ax_region_info = fig.add_subplot(gs[2, 0])
    ax_region_info.axis('off')

    # Display regional partitioning information (if available)
    if use_partitioning and region_configs:
        if language == 'zh':
            region_info = [f"Region Partitioning Information ({len(region_configs)} regions):"]
        else:
            region_info = [f"Region Partitioning ({len(region_configs)} regions):"]

        for region_id, config in region_configs.items():
            region_info.append(f"- {region_text} {region_id}:")
            if 'density_factor' in config:
                region_info.append(
                    f"  - {'Density Factor' if language == 'zh' else 'Density'}: {config['density_factor']:.2f}")

        # Set font based on language
        if chinese_font and language == 'zh':
            ax_region_info.text(0.05, 0.95, '\n'.join(region_info), va='top', ha='left',
                                fontsize=9, color=main_color, fontproperties=chinese_font)
        else:
            ax_region_info.text(0.05, 0.95, '\n'.join(region_info), va='top', ha='left',
                                fontsize=9, color=main_color)

    # Plot width-to-thickness ratio histogram
    ax_wtr = fig.add_subplot(gs[2, 1:])
    if len(wtr_data) > 0:
        # Determine number of bins based on data
        n_bins = min(max(5, int(np.sqrt(len(wtr_data)))), 15)

        # Create histogram
        hist, bins, _ = ax_wtr.hist(wtr_data, bins=n_bins, color=accent_color,
                                    alpha=0.7, edgecolor=edge_color, linewidth=1)

        # Add normal distribution curve if sufficient data
        if len(wtr_data) >= 5:
            x = np.linspace(wtr_stats['min'], wtr_stats['max'], 100)
            y = len(wtr_data) * (bins[1] - bins[0]) * \
                np.exp(-(x - wtr_stats['mean']) ** 2 / (2 * wtr_stats['std'] ** 2)) / \
                (wtr_stats['std'] * np.sqrt(2 * np.pi))
            ax_wtr.plot(x, y, linewidth=2, color=main_color)

        # Add statistical text
        if language == 'zh':
            stats_text = f"Mean: {wtr_stats['mean']:.2f}\nMedian: {wtr_stats['median']:.2f}\n" \
                         f"Std Dev: {wtr_stats['std']:.2f}\nRange: [{wtr_stats['min']:.2f}, {wtr_stats['max']:.2f}]"
        else:
            stats_text = f"Mean: {wtr_stats['mean']:.2f}\nMedian: {wtr_stats['median']:.2f}\n" \
                         f"Std Dev: {wtr_stats['std']:.2f}\nRange: [{wtr_stats['min']:.2f}, {wtr_stats['max']:.2f}]"

        # Set font based on language
        if chinese_font and language == 'zh':
            ax_wtr.text(0.95, 0.95, stats_text, transform=ax_wtr.transAxes,
                        fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round',
                                                                    facecolor='white', alpha=0.7),
                        fontproperties=chinese_font)
        else:
            ax_wtr.text(0.95, 0.95, stats_text, transform=ax_wtr.transAxes,
                        fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round',
                                                                    facecolor='white', alpha=0.7))

    # Set width-to-thickness ratio plot title and labels
    if chinese_font and language == 'zh':
        ax_wtr.set_title(wtr_title, fontsize=12, color=main_color, fontproperties=chinese_font)
        ax_wtr.set_xlabel(x_label_wtr, fontsize=10, color=main_color, fontproperties=chinese_font)
        ax_wtr.set_ylabel(y_label, fontsize=10, color=main_color, fontproperties=chinese_font)
    else:
        ax_wtr.set_title(wtr_title, fontsize=12, color=main_color)
        ax_wtr.set_xlabel(x_label_wtr, fontsize=10, color=main_color)
        ax_wtr.set_ylabel(y_label, fontsize=10, color=main_color)

    ax_wtr.grid(True, linestyle='--', alpha=0.7)

    # Create third information area
    ax_corr_info = fig.add_subplot(gs[3, 0])
    ax_corr_info.axis('off')

    # Add correlation analysis information
    if len(thickness_data) > 0 and len(wtr_data) > 0 and len(thickness_data) == len(wtr_data):
        correlation = np.corrcoef(thickness_data, wtr_data)[0, 1]

        if language == 'zh':
            corr_info = [
                "Correlation Analysis:",
                f"Pearson correlation coefficient: {correlation:.2f}",
                "",
                "Correlation interpretation:",
                f"{'Strong negative correlation' if correlation < -0.7 else 'Moderate negative correlation' if correlation < -0.3 else 'Weak negative correlation' if correlation < 0 else 'No correlation' if abs(correlation) < 0.1 else 'Weak positive correlation' if correlation < 0.3 else 'Moderate positive correlation' if correlation < 0.7 else 'Strong positive correlation'}"
            ]
        else:
            corr_info = [
                "Correlation Analysis:",
                f"Pearson coefficient: {correlation:.2f}",
                "",
                "Interpretation:",
                f"{'Strong negative' if correlation < -0.7 else 'Moderate negative' if correlation < -0.3 else 'Weak negative' if correlation < 0 else 'No correlation' if abs(correlation) < 0.1 else 'Weak positive' if correlation < 0.3 else 'Moderate positive' if correlation < 0.7 else 'Strong positive'}"
            ]

        # Set font based on language
        if chinese_font and language == 'zh':
            ax_corr_info.text(0.05, 0.95, '\n'.join(corr_info), va='top', ha='left',
                              fontsize=9, color=main_color, fontproperties=chinese_font)
        else:
            ax_corr_info.text(0.05, 0.95, '\n'.join(corr_info), va='top', ha='left',
                              fontsize=9, color=main_color)

    # Create scatter plot comparing thickness and width-to-thickness ratio
    ax_scatter = fig.add_subplot(gs[3, 1:])

    if len(thickness_data) > 0 and len(wtr_data) > 0 and len(thickness_data) == len(wtr_data):
        scatter = ax_scatter.scatter(thickness_data, wtr_data, c=np.arange(len(thickness_data)),
                                     cmap=color_palette, alpha=0.8, s=50, edgecolor='white')

        # Add trend line
        if len(thickness_data) >= 2:
            z = np.polyfit(thickness_data, wtr_data, 1)
            p = np.poly1d(z)
            ax_scatter.plot(np.sort(thickness_data), p(np.sort(thickness_data)),
                            linewidth=2, color=main_color)

            # Add correlation coefficient
            correlation = np.corrcoef(thickness_data, wtr_data)[0, 1]
            ax_scatter.text(0.95, 0.05, f"r = {correlation:.2f}", transform=ax_scatter.transAxes,
                            fontsize=10, ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Set scatter plot title
        if chinese_font and language == 'zh':
            ax_scatter.set_title(scatter_title, fontsize=12, color=main_color, fontproperties=chinese_font)
            ax_scatter.set_xlabel(x_label_thickness, fontsize=10, color=main_color, fontproperties=chinese_font)
            ax_scatter.set_ylabel(x_label_wtr, fontsize=10, color=main_color, fontproperties=chinese_font)
        else:
            ax_scatter.set_title(scatter_title, fontsize=12, color=main_color)
            ax_scatter.set_xlabel(x_label_thickness, fontsize=10, color=main_color)
            ax_scatter.set_ylabel(x_label_wtr, fontsize=10, color=main_color)

    ax_scatter.grid(True, linestyle='--', alpha=0.7)

    # Set style elements
    for ax in [ax_thickness, ax_wtr, ax_scatter]:
        for spine in ax.spines.values():
            spine.set_color(edge_color)
        ax.tick_params(colors=main_color)

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    else:
        plt.show()

    # Return statistical data for further analysis
    return {
        'thickness_stats': thickness_stats,
        'wtr_stats': wtr_stats,
        'thickness_data': thickness_data,
        'wtr_data': wtr_data
    }


def visualize_porosity_distribution(results, isim=1, journal_style='nature', language='en', save_path=None,
                                    figsize=(12, 10), dpi=150):
    """
    Visualize porosity distribution of channel voxels

    Parameters:
        results: fluvpy simulation results dictionary
        isim: Realization number to visualize
        journal_style: Journal style ('nature', 'science', 'geology')
        language: Label language ('zh' or 'en')
        save_path: Path to save figure, if None displays directly
        figsize: Figure size (width, height)
        dpi: Figure resolution
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec
    from matplotlib.font_manager import FontProperties
    import os

    # Get realization data
    realization = results[f'realization_{isim}']
    channel = realization['channel']
    porosity = realization['porosity']
    params = realization.get('params', {})

    # Configure Chinese font support
    if language == 'zh':
        import platform
        system = platform.system()

        if system == 'Windows':
            try:
                font_path = 'C:/Windows/Fonts/msyh.ttc'
                if not os.path.exists(font_path):
                    font_path = 'C:/Windows/Fonts/simsun.ttc'
                if os.path.exists(font_path):
                    chinese_font = FontProperties(fname=font_path)
                else:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        elif system == 'Darwin':  # macOS
            try:
                font_path = '/System/Library/Fonts/PingFang.ttc'
                if not os.path.exists(font_path):
                    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
                if os.path.exists(font_path):
                    chinese_font = FontProperties(fname=font_path)
                else:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        elif system == 'Linux':
            try:
                font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                    '/usr/share/fonts/truetype/arphic/uming.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                ]

                font_found = False
                for path in font_paths:
                    if os.path.exists(path):
                        chinese_font = FontProperties(fname=path)
                        font_found = True
                        break

                if not font_found:
                    chinese_font = FontProperties()
                    language = 'en'
            except:
                chinese_font = FontProperties()
                language = 'en'
        else:
            chinese_font = FontProperties()
            language = 'en'
    else:
        chinese_font = None

    # Set style
    try:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'
    except Exception as e:
        pass

    # Set color scheme according to journal style
    if journal_style == 'nature':
        color_palette = plt.cm.viridis
        main_color = '#2C3E50'
        accent_color = '#3498DB'
        edge_color = '#7F8C8D'
    elif journal_style == 'science':
        color_palette = plt.cm.plasma
        main_color = '#34495E'
        accent_color = '#E74C3C'
        edge_color = '#BDC3C7'
    elif journal_style == 'geology':
        color_palette = plt.cm.terrain
        main_color = '#5D6D7E'
        accent_color = '#27AE60'
        edge_color = '#95A5A6'
    else:
        color_palette = plt.cm.viridis
        main_color = '#2C3E50'
        accent_color = '#3498DB'
        edge_color = '#7F8C8D'

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[0.2, 1, 1, 1])

    # Determine text based on language
    if language == 'zh':
        title_text = 'Channel Porosity Distribution Statistical Analysis'
        porosity_title = 'Porosity Parameters'
        histogram_title = 'Porosity Histogram'
        depth_title = 'Porosity Distribution by Depth'
        cumulative_title = 'Porosity Cumulative Distribution'
        x_label_porosity = 'Porosity'
        y_label_freq = 'Frequency'
        y_label_depth = 'Relative Depth'
        y_label_cumulative = 'Cumulative Percentage (%)'
        position_text = 'Position'
        region_text = 'Region'
    else:  # 'en'
        title_text = 'Channel Porosity Distribution Analysis'
        porosity_title = 'Porosity Parameters'
        histogram_title = 'Porosity Histogram'
        depth_title = 'Porosity Distribution by Depth'
        cumulative_title = 'Porosity Cumulative Distribution'
        x_label_porosity = 'Porosity'
        y_label_freq = 'Frequency'
        y_label_depth = 'Relative Depth'
        y_label_cumulative = 'Cumulative Percentage (%)'
        position_text = 'Position'
        region_text = 'Region'

    # Add title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    if chinese_font and language == 'zh':
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16,
                      fontweight='bold', color=main_color, fontproperties=chinese_font)
    else:
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16,
                      fontweight='bold', color=main_color)

    # Extract porosity data
    valid_mask = (channel > 0) & (porosity > 0)
    porosity_data = porosity[valid_mask]

    # Ensure valid data exists
    if len(porosity_data) == 0:
        # Generate sample data for demonstration
        porosity_data = np.random.normal(0.2, 0.05, 1000)
        porosity_data = np.clip(porosity_data, 0.05, 0.35)

    # Calculate statistical data
    porosity_stats = {
        'mean': np.mean(porosity_data),
        'median': np.median(porosity_data),
        'std': np.std(porosity_data),
        'min': np.min(porosity_data),
        'max': np.max(porosity_data),
        'count': len(porosity_data)
    }

    # Display porosity parameter information
    ax_params = fig.add_subplot(gs[1, 0])
    ax_params.axis('off')

    # Get porosity parameters
    fcpor_base = params.get('fcpor_base', [0.15, 0.20, 0.25])
    fcpor_range = params.get('fcpor_range', [0.05, 0.07, 0.10])
    use_specific = params.get('use_channel_specific_porosity', False)

    # Organize parameter information to display
    if language == 'zh':
        param_info = [
            f"Total porosity voxels: {porosity_stats['count']}",
            f"Porosity range: [{porosity_stats['min']:.4f}, {porosity_stats['max']:.4f}]",
            f"Mean porosity: {porosity_stats['mean']:.4f}",
            f"Median porosity: {porosity_stats['median']:.4f}",
            f"Standard deviation: {porosity_stats['std']:.4f}",
            "",
            f"Porosity base parameter: {str(fcpor_base)}",
            f"Porosity range parameter: {str(fcpor_range)}",
            f"Channel-specific porosity: {'Enabled' if use_specific else 'Disabled'}"
        ]
    else:
        param_info = [
            f"Total porosity voxels: {porosity_stats['count']}",
            f"Porosity range: [{porosity_stats['min']:.4f}, {porosity_stats['max']:.4f}]",
            f"Mean porosity: {porosity_stats['mean']:.4f}",
            f"Median porosity: {porosity_stats['median']:.4f}",
            f"Standard deviation: {porosity_stats['std']:.4f}",
            "",
            f"Porosity base parameter: {str(fcpor_base)}",
            f"Porosity range parameter: {str(fcpor_range)}",
            f"Channel-specific porosity: {'Enabled' if use_specific else 'Disabled'}"
        ]

    # Display parameter information text
    if chinese_font and language == 'zh':
        ax_params.text(0.05, 0.95, '\n'.join(param_info), va='top', ha='left',
                       fontsize=9, color=main_color, fontproperties=chinese_font)
        ax_params.set_title(porosity_title, fontsize=12, color=main_color, fontproperties=chinese_font)
    else:
        ax_params.text(0.05, 0.95, '\n'.join(param_info), va='top', ha='left',
                       fontsize=9, color=main_color)
        ax_params.set_title(porosity_title, fontsize=12, color=main_color)

    # Plot porosity histogram
    ax_histogram = fig.add_subplot(gs[1, 1:])
    if len(porosity_data) > 0:
        # Determine number of bins based on data
        n_bins = min(max(5, int(np.sqrt(len(porosity_data)))), 20)

        # Create histogram
        hist, bins, _ = ax_histogram.hist(porosity_data, bins=n_bins, color=accent_color,
                                          alpha=0.7, edgecolor=edge_color, linewidth=1,
                                          density=True)

        # Add normal distribution curve
        x = np.linspace(porosity_stats['min'], porosity_stats['max'], 100)
        y = np.exp(-(x - porosity_stats['mean']) ** 2 / (2 * porosity_stats['std'] ** 2)) / \
            (porosity_stats['std'] * np.sqrt(2 * np.pi))
        ax_histogram.plot(x, y, linewidth=2, color=main_color)

        # Add KDE curve at top of histogram
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(porosity_data)
        x_kde = np.linspace(porosity_stats['min'], porosity_stats['max'], 100)
        y_kde = kde(x_kde)
        ax_histogram.plot(x_kde, y_kde, color=main_color, linestyle='--',
                          label='KDE', linewidth=1.5)

    # Set histogram title and labels
    if chinese_font and language == 'zh':
        ax_histogram.set_title(histogram_title, fontsize=12, color=main_color, fontproperties=chinese_font)
        ax_histogram.set_xlabel(x_label_porosity, fontsize=10, color=main_color, fontproperties=chinese_font)
        ax_histogram.set_ylabel(y_label_freq, fontsize=10, color=main_color, fontproperties=chinese_font)
    else:
        ax_histogram.set_title(histogram_title, fontsize=12, color=main_color)
        ax_histogram.set_xlabel(x_label_porosity, fontsize=10, color=main_color)
        ax_histogram.set_ylabel(y_label_freq, fontsize=10, color=main_color)

    ax_histogram.grid(True, linestyle='--', alpha=0.7)

    # Create and analyze relationship between porosity and depth
    ax_depth = fig.add_subplot(gs[2, 1:])

    # Find valid porosity values and corresponding depths
    if valid_mask.any():
        # Extract valid coordinates
        iz_coords = np.where(valid_mask)[2]  # z coordinates
        porosity_by_depth = porosity[valid_mask]

        # Calculate relative depth (0=top, 1=bottom)
        nz = channel.shape[2]
        relative_depth = iz_coords / nz

        # Create scatter plot
        ax_depth.scatter(porosity_by_depth, 1 - relative_depth, alpha=0.3, s=2,
                         c=porosity_by_depth, cmap=color_palette)

        # Add depth-stratified averages
        depth_bins = 10  # Divide into 10 layers
        depth_ranges = np.linspace(0, 1, depth_bins + 1)

        for i in range(depth_bins):
            depth_min = depth_ranges[i]
            depth_max = depth_ranges[i + 1]

            # Find points within this depth range
            depth_mask = (relative_depth >= depth_min) & (relative_depth < depth_max)
            if np.sum(depth_mask) > 0:
                mean_porosity = np.mean(porosity_by_depth[depth_mask])
                mean_depth = 1 - (depth_min + depth_max) / 2

                # Plot average value points and range lines
                ax_depth.scatter(mean_porosity, mean_depth, color='red', s=30, zorder=5)

                # Add horizontal lines indicating depth ranges
                if i % 2 == 0:
                    ax_depth.axhline(y=1 - depth_min, color='gray', linestyle='--', alpha=0.3)

        # Try adding trend line
        try:
            z = np.polyfit(porosity_by_depth, 1 - relative_depth, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(porosity_stats['min'], porosity_stats['max'], 100)
            ax_depth.plot(x_trend, p(x_trend), color=main_color, linestyle='-', linewidth=2)
        except:
            pass

    # Invert Y-axis so top is at the top
    ax_depth.invert_yaxis()

    # Set title and labels
    if chinese_font and language == 'zh':
        ax_depth.set_title(depth_title, fontsize=12, color=main_color, fontproperties=chinese_font)
        ax_depth.set_xlabel(x_label_porosity, fontsize=10, color=main_color, fontproperties=chinese_font)
        ax_depth.set_ylabel(y_label_depth, fontsize=10, color=main_color, fontproperties=chinese_font)
    else:
        ax_depth.set_title(depth_title, fontsize=12, color=main_color)
        ax_depth.set_xlabel(x_label_porosity, fontsize=10, color=main_color)
        ax_depth.set_ylabel(y_label_depth, fontsize=10, color=main_color)

    # Set Y-axis labels to indicate top and bottom
    y_ticks = np.linspace(0, 1, 6)
    if language == 'zh':
        y_labels = ['Top', '0.2', '0.4', '0.6', '0.8', 'Bottom']
    else:
        y_labels = ['Top', '0.2', '0.4', '0.6', '0.8', 'Bottom']
    ax_depth.set_yticks(y_ticks)
    ax_depth.set_yticklabels(y_labels)

    ax_depth.grid(True, linestyle='--', alpha=0.7)

    # Create porosity cumulative distribution plot
    ax_cumulative = fig.add_subplot(gs[3, 1:])

    if len(porosity_data) > 0:
        # Calculate cumulative distribution
        sorted_data = np.sort(porosity_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

        # Plot cumulative distribution curve
        ax_cumulative.plot(sorted_data, cumulative, color=accent_color, linewidth=2)

        # Add percentile lines
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(porosity_data, percentiles)

        for p, val in zip(percentiles, percentile_values):
            # Vertical line
            ax_cumulative.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
            # Horizontal line
            ax_cumulative.axhline(y=p, color='gray', linestyle='--', alpha=0.5)
            # Add marker points
            ax_cumulative.scatter(val, p, color='red', s=30, zorder=5)
            # Add labels
            if p == 50:  # Only add label to median to avoid crowding
                if language == 'zh':
                    label = f"Median: {val:.3f}"
                else:
                    label = f"Median: {val:.3f}"
                ax_cumulative.annotate(label, (val, p), xytext=(10, 0),
                                       textcoords='offset points', fontsize=8)

    # Set title and labels
    if chinese_font and language == 'zh':
        ax_cumulative.set_title(cumulative_title, fontsize=12, color=main_color, fontproperties=chinese_font)
        ax_cumulative.set_xlabel(x_label_porosity, fontsize=10, color=main_color, fontproperties=chinese_font)
        ax_cumulative.set_ylabel(y_label_cumulative, fontsize=10, color=main_color, fontproperties=chinese_font)
    else:
        ax_cumulative.set_title(cumulative_title, fontsize=12, color=main_color)
        ax_cumulative.set_xlabel(x_label_porosity, fontsize=10, color=main_color)
        ax_cumulative.set_ylabel(y_label_cumulative, fontsize=10, color=main_color)

    ax_cumulative.grid(True, linestyle='--', alpha=0.7)

    # Add porosity distribution type analysis
    ax_dist_info = fig.add_subplot(gs[2:, 0])
    ax_dist_info.axis('off')

    # Analyze porosity distribution type from data
    from scipy import stats

    # Check if approximately normal distribution
    k2, p_normal = stats.normaltest(porosity_data)
    is_normal = p_normal > 0.05

    # Check skewness and kurtosis
    skewness = stats.skew(porosity_data)
    kurtosis = stats.kurtosis(porosity_data)

    # Calculate quartiles to check distribution shape
    q1, q2, q3 = np.percentile(porosity_data, [25, 50, 75])
    iqr = q3 - q1

    # Determine distribution type
    if is_normal:
        dist_type = 'Normal Distribution' if language == 'zh' else 'Normal Distribution'
    elif skewness > 0.5:
        dist_type = 'Right-skewed Distribution' if language == 'zh' else 'Right-skewed Distribution'
    elif skewness < -0.5:
        dist_type = 'Left-skewed Distribution' if language == 'zh' else 'Left-skewed Distribution'
    elif kurtosis > 0.5:
        dist_type = 'Leptokurtic Distribution' if language == 'zh' else 'Leptokurtic Distribution'
    elif kurtosis < -0.5:
        dist_type = 'Platykurtic Distribution' if language == 'zh' else 'Platykurtic Distribution'
    else:
        dist_type = 'Approximately Normal' if language == 'zh' else 'Approximately Normal'

    # Prepare distribution analysis text
    if language == 'zh':
        dist_info = [
            "Porosity Distribution Analysis:",
            f"Distribution type: {dist_type}",
            f"Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})",
            f"Kurtosis: {kurtosis:.4f} ({'Leptokurtic' if kurtosis > 0 else 'Platykurtic' if kurtosis < 0 else 'Mesokurtic'})",
            "",
            "Quantile Analysis:",
            f"1st Quartile (Q1): {q1:.4f}",
            f"Median (Q2): {q2:.4f}",
            f"3rd Quartile (Q3): {q3:.4f}",
            f"Interquartile Range (IQR): {iqr:.4f}",
            "",
            "Coefficient of Variation:",
            f"CV = {porosity_stats['std'] / porosity_stats['mean']:.4f}",
            "",
            "Spatial Distribution Trend:",
            "See right plot for top-to-bottom trend"
        ]
    else:
        dist_info = [
            "Porosity Distribution Analysis:",
            f"Distribution type: {dist_type}",
            f"Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})",
            f"Kurtosis: {kurtosis:.4f} ({'Leptokurtic' if kurtosis > 0 else 'Platykurtic' if kurtosis < 0 else 'Mesokurtic'})",
            "",
            "Quantile Analysis:",
            f"1st Quartile (Q1): {q1:.4f}",
            f"Median (Q2): {q2:.4f}",
            f"3rd Quartile (Q3): {q3:.4f}",
            f"Interquartile Range (IQR): {iqr:.4f}",
            "",
            "Coefficient of Variation:",
            f"CV = {porosity_stats['std'] / porosity_stats['mean']:.4f}",
            "",
            "Spatial Distribution Trend:",
            "See right plot for top-to-bottom trend"
        ]

    # Display distribution analysis text
    if chinese_font and language == 'zh':
        ax_dist_info.text(0.05, 0.95, '\n'.join(dist_info), va='top', ha='left',
                          fontsize=9, color=main_color, fontproperties=chinese_font)
    else:
        ax_dist_info.text(0.05, 0.95, '\n'.join(dist_info), va='top', ha='left',
                          fontsize=9, color=main_color)

    # Set style elements
    for ax in [ax_histogram, ax_depth, ax_cumulative]:
        for spine in ax.spines.values():
            spine.set_color(edge_color)
        ax.tick_params(colors=main_color)

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    else:
        plt.show()

    # Return statistical data for further analysis
    return {
        'porosity_stats': porosity_stats,
        'porosity_data': porosity_data
    }


def visualize_fluvpy_3d(results: Dict[str, Any], isim: int = 1):
    """
    Create 3D visualization of river model using PyVista with interactive controls

    Parameters:
        results: Dictionary of results from fluvpy
        isim: Realization number to visualize
    """
    try:
        import pyvista as pv
        import numpy as np
        from matplotlib.colors import ListedColormap
        import time
    except ImportError:
        print("PyVista not installed, cannot display 3D channel visualization")
        return None

    start_time = time.time()
    print(f"\nStarting 3D channel visualization - Realization {isim}")

    # Get results for specified realization
    realization = results[f'realization_{isim}']
    channel = realization['channel']

    # Try to get parameters, use defaults if not available
    if 'params' in results:
        params = results['params']
    elif 'params' in realization:
        params = realization['params']
    else:
        # Set default parameters
        params = {
            'xmn': 0, 'ymn': 0, 'zmn': 0,
            'xsiz': 15.0, 'ysiz': 15.0, 'zsiz': 2.5
        }

    # Extract grid dimensions
    nx, ny, nz = channel.shape

    # Get grid parameters
    xmn = params.get('xmn', 0)
    ymn = params.get('ymn', 0)
    zmn = params.get('zmn', 0)
    xsiz = params.get('xsiz', 15.0)
    ysiz = params.get('ysiz', 15.0)
    zsiz = params.get('zsiz', 2.5)

    # Calculate physical dimensions
    x_length = nx * xsiz
    y_length = ny * ysiz
    z_length = nz * zsiz

    # Calculate z-direction scale factor to make z-axis height half of x-axis width
    z_scale_factor = x_length / (2 * z_length)
    adjusted_z_siz = zsiz * z_scale_factor

    # Create PyVista grid
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(xsiz, ysiz, adjusted_z_siz),
        origin=(xmn, ymn, zmn)
    )

    # Add channel data to grid
    grid.cell_data["Channel"] = channel.flatten(order='F')

    # Create plotter object
    plotter = pv.Plotter(off_screen=False)
    plotter.window_size = (1200, 800)
    plotter.background_color = 'white'

    # Add axes and grid
    plotter.show_axes()
    plotter.show_grid()

    # Add outline to show complete grid bounds
    outline = grid.outline()
    plotter.add_mesh(outline, color='black', line_width=2, name='outline')

    # Check if channel data exists
    has_channels = np.any(channel > 0)

    # Store mesh objects and visibility states
    mesh_objects = {}
    visibility_states = {}

    if has_channels:
        try:
            # Get valid channel data
            valid_mask = channel > 0
            valid_values = channel[valid_mask]

            channel_mask = channel < 10000
            levee_mask = (channel >= 10000) & (channel < 20000)
            crevasse_mask = channel >= 20000

            # Channel color scheme
            channel_colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

            # 1. Create channel voxel mesh (G < 10000)
            if np.any(channel_mask & (channel > 0)):
                channel_grid = grid.copy()
                channel_data = np.where(channel_mask, channel, 0)
                channel_grid.cell_data["Channel"] = channel_data.flatten(order='F')

                try:
                    mesh_objects['channel'] = channel_grid.threshold(0.5, scalars="Channel")
                    plotter.add_mesh(
                        mesh_objects['channel'],
                        color=channel_colors[0],
                        opacity=1,
                        show_edges=False,
                        name='channel_mesh'
                    )
                    visibility_states['channel'] = True
                    print(f"Channel voxels created, containing {mesh_objects['channel'].n_cells} cells")
                except Exception as e:
                    print(f"Channel voxel creation failed: {e}")

            # 2. Create levee voxel mesh (10000 <= G < 20000)
            if np.any(levee_mask):
                levee_grid = grid.copy()
                levee_data = np.where(levee_mask, channel, 0)
                levee_grid.cell_data["Channel"] = levee_data.flatten(order='F')

                try:
                    mesh_objects['levee'] = levee_grid.threshold(9999.5, scalars="Channel")
                    plotter.add_mesh(
                        mesh_objects['levee'],
                        color=channel_colors[1],
                        opacity=1,
                        show_edges=False,
                        name='levee_mesh'
                    )
                    visibility_states['levee'] = True
                    print(f"Levee voxels created, containing {mesh_objects['levee'].n_cells} cells")
                except Exception as e:
                    print(f"Levee voxel creation failed: {e}")

            # 3. Create crevasse splay voxel mesh (G >= 20000)
            if np.any(crevasse_mask):
                crevasse_grid = grid.copy()
                crevasse_data = np.where(crevasse_mask, channel, 0)
                crevasse_grid.cell_data["Channel"] = crevasse_data.flatten(order='F')

                try:
                    mesh_objects['crevasse'] = crevasse_grid.threshold(19999.5, scalars="Channel")
                    plotter.add_mesh(
                        mesh_objects['crevasse'],
                        color=channel_colors[2],
                        opacity=1,
                        show_edges=False,
                        name='crevasse_mesh'
                    )
                    visibility_states['crevasse'] = True
                    print(f"Crevasse splay voxels created, containing {mesh_objects['crevasse'].n_cells} cells")
                except Exception as e:
                    print(f"Crevasse splay voxel creation failed: {e}")

        except Exception as e:
            print(f"Channel grid creation failed: {e}")
            plotter.add_text("Channel data rendering failed", position='upper_edge', font_size=14, color='red')
    else:
        plotter.add_text("No channels in data", position='upper_edge', font_size=14, color='red')

        # Add example grid to show scale
        try:
            box = pv.Box(bounds=(
                xmn + x_length * 0.4, xmn + x_length * 0.6,
                ymn + y_length * 0.4, ymn + y_length * 0.6,
                zmn + z_length * 0.4 * z_scale_factor, zmn + z_length * 0.6 * z_scale_factor
            ))
            plotter.add_mesh(box, color='gray', style='wireframe', opacity=0.5, name='example_box')
        except:
            pass

    # Create button control functions
    def toggle_mesh_visibility(mesh_type):
        """Toggle visibility of specified mesh type"""
        if mesh_type in mesh_objects and mesh_type in visibility_states:
            try:
                mesh_name = f'{mesh_type}_mesh'
                current_state = visibility_states[mesh_type]

                # Toggle state
                visibility_states[mesh_type] = not current_state

                # Update display
                if visibility_states[mesh_type]:
                    # Show mesh
                    if mesh_name not in [actor.name for actor in plotter.renderer.actors.values() if
                                         hasattr(actor, 'name')]:
                        colors = {'channel': channel_colors[0], 'levee': channel_colors[1],
                                  'crevasse': channel_colors[2]}
                        opacities = {'channel': 1, 'levee': 1, 'crevasse': 1}
                        plotter.add_mesh(
                            mesh_objects[mesh_type],
                            color=colors[mesh_type],
                            opacity=opacities[mesh_type],
                            show_edges=False,
                            name=mesh_name
                        )
                else:
                    # Hide mesh
                    try:
                        plotter.remove_actor(mesh_name, reset_camera=False)
                    except:
                        pass

                plotter.render()
                status = "Show" if visibility_states[mesh_type] else "Hide"
                mesh_names = {'channel': 'Channel', 'levee': 'Levee', 'crevasse': 'Crevasse'}
                print(f"{mesh_names[mesh_type]}: {status}")

            except Exception as e:
                print(f"{mesh_type} toggle failed: {e}")

    # Add button controls - positioned in upper left, button size reduced by 1/3
    button_size = 33  # Original 50, reduced by 1/3
    try:
        # Channel voxel button - upper left first
        if 'channel' in mesh_objects:
            def toggle_channel_callback(state):
                if state != visibility_states.get('channel', True):
                    toggle_mesh_visibility('channel')

            plotter.add_checkbox_button_widget(
                callback=toggle_channel_callback,
                value=True,
                position=(10, 750),  # Upper left position
                size=button_size,
                border_size=2,
                color_on='blue',
                color_off='lightgray',
                background_color='white'
            )
            # Add button label
            plotter.add_text("Channel", position=(50, 755), font_size=12, color='blue')

        # Levee voxel button - upper left second
        if 'levee' in mesh_objects:
            def toggle_levee_callback(state):
                if state != visibility_states.get('levee', True):
                    toggle_mesh_visibility('levee')

            plotter.add_checkbox_button_widget(
                callback=toggle_levee_callback,
                value=True,
                position=(10, 710),  # Upper left position, below first button
                size=button_size,
                border_size=2,
                color_on='purple',
                color_off='lightgray',
                background_color='white'
            )
            # Add button label
            plotter.add_text("Levee", position=(50, 715), font_size=12, color='purple')

        # Crevasse splay voxel button - upper left third
        if 'crevasse' in mesh_objects:
            def toggle_crevasse_callback(state):
                if state != visibility_states.get('crevasse', True):
                    toggle_mesh_visibility('crevasse')

            plotter.add_checkbox_button_widget(
                callback=toggle_crevasse_callback,
                value=True,
                position=(10, 670),  # Upper left position, below second button
                size=button_size,
                border_size=2,
                color_on='orange',
                color_off='lightgray',
                background_color='white'
            )
            # Add button label
            plotter.add_text("Crevasse", position=(50, 675), font_size=12, color='orange')

        print("Button controls created - Click upper left buttons to toggle show/hide")

    except Exception as e:
        print(f"Button control creation failed: {e}")

    # Calculate statistics
    channel_cells = np.count_nonzero((channel > 0) & (channel < 10000))
    levee_cells = np.count_nonzero((channel >= 10000) & (channel < 20000))
    crevasse_cells = np.count_nonzero(channel >= 20000)
    total_cells = nx * ny * nz

    # Get channel value range
    if has_channels:
        valid_values = channel[channel > 0]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        unique_channels = len(np.unique(valid_values))
    else:
        min_val = max_val = unique_channels = 0

    # Add legend and statistics
    legend_text = [
        f"3D Channel Visualization - Realization {isim}",
        f"Grid Size: {nx}×{ny}×{nz}",
        f"Physical Size: {x_length:.0f}×{y_length:.0f}×{z_length:.0f} meters",
        "",
        "Voxel Classification Statistics:",
        f"Channel Voxels (G<10000): {channel_cells:,} cells",
        f"Levee Voxels (10000≤G<20000): {levee_cells:,} cells",
        f"Crevasse Voxels (G≥20000): {crevasse_cells:,} cells",
        "",
        "Interactive Controls:",
        "Click upper left buttons to toggle show/hide",
        "Blue button - Channel",
        "Purple button - Levee",
        "Orange button - Crevasse",
        "",
        f"Total Voxels: {channel_cells + levee_cells + crevasse_cells:,}",
        f"Coverage: {(channel_cells + levee_cells + crevasse_cells) / total_cells * 100:.1f}%"
    ]

    plotter.add_text(
        "\n".join(legend_text),
        position='upper_right',
        font_size=9,
        color='black',
        shadow=True
    )

    # Calculate adjusted grid center for camera setting
    center = [
        xmn + x_length / 2,
        ymn + y_length / 2,
        zmn + (nz * adjusted_z_siz) / 2
    ]

    # Set camera position and focal point
    camera_distance = max(x_length, y_length) * 1.2
    upward_offset = z_length * 0.3
    camera_position = [
        center[0] + camera_distance * 0.7,
        center[1] + camera_distance * 0.7,
        center[2] + camera_distance * 0.5
    ]
    focal_point = center

    # Set camera up direction to positive Z-axis
    view_up = [0, 0, 1]

    # Apply camera settings
    plotter.camera_position = [camera_position, focal_point, view_up]

    # Add coordinate axis scales and labels
    bounds = [xmn, xmn + x_length, ymn, ymn + y_length, zmn, zmn + nz * adjusted_z_siz]

    # Set coordinate axis range and ticks
    plotter.show_bounds(
        bounds=bounds,
        xlabel='X Coordinate (m)',
        ylabel='Y Coordinate (m)',
        zlabel='Z Coordinate (m)',
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        font_size=13,
        color='black'
    )

    # Display
    try:
        plotter.show(
            title=f"Interactive 3D Channel Model - Realization {isim}",
            interactive=True,
            auto_close=False
        )
    except Exception as e:
        print(f"Display failed: {e}")
        plotter.show(title=f"Interactive 3D Channel Model - Realization {isim}")

    total_time = time.time() - start_time
    print(f"3D channel visualization completed in {total_time:.3f} seconds")

    # Return statistics
    return {
        'channel_count': channel_cells,
        'levee_count': levee_cells,
        'crevasse_count': crevasse_cells,
        'total_voxels': channel_cells + levee_cells + crevasse_cells,
        'coverage_percent': (channel_cells + levee_cells + crevasse_cells) / total_cells * 100,
        'value_range': (min_val, max_val) if has_channels else (0, 0),
        'visualization_time': total_time
    }


def visualize_fluvpy_3d_Polygon(results: Dict[str, Any], isim: int = 1):
    """
    Create 3D visualization of river model using PyVista
    """
    try:
        import pyvista as pv
        import numpy as np
        pv.global_theme.allow_empty_mesh = True
    except ImportError:
        print("PyVista not installed. Please install it using 'pip install pyvista'")
        return

    # Get results for specified realization
    realization = results[f'realization_{isim}']
    channel = realization['channel']

    # Extract grid dimensions
    nx, ny, nz = channel.shape

    # Print statistical information for debugging
    unique_vals = np.unique(channel)
    print(f"Channel grid unique values: {unique_vals}")
    print(f"Value counts:")
    for val in unique_vals:
        count = np.sum(channel == val)
        print(f"  Value {val}: {count} cells ({count / (nx * ny * nz) * 100:.2f}%)")

    # Redefine clear classification IDs (instead of relying on original ID ranges)
    channel_new = np.zeros_like(channel)

    # Explicitly define IDs for various sedimentary facies
    channel_new[channel > 0] = 1  # All sedimentary facies default to main channel
    channel_new[(channel >= 10000) & (channel < 20000)] = 2  # Natural levees
    channel_new[(channel >= 20000) & (channel < 30000)] = 3  # Crevasse splays

    # Create PyVista grid
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(1, 1, 1),
        origin=(0, 0, 0)
    )

    # Add channel data to grid
    grid.cell_data["Channel"] = channel_new.flatten(order='F')

    # Create plotter object
    plotter = pv.Plotter()
    plotter.set_background('white')

    # Record sedimentary facies statistics and store grid objects
    facies_counts = {}
    facies_surfaces = {}

    # More explicit threshold and color definitions
    facies_defs = [
        {"id": 1, "name": "Main Channel", "color": "blue", "opacity": 0.7},
        {"id": 2, "name": "Natural Levees", "color": "cyan", "opacity": 0.7},
        {"id": 3, "name": "Crevasse Splays", "color": "green", "opacity": 0.7},
    ]

    # Extract and visualize different types of sedimentary facies separately
    for facies in facies_defs:
        facies_id = facies["id"]
        facies_name = facies["name"]
        facies_color = facies["color"]
        facies_opacity = facies["opacity"]

        # Set precise threshold range
        facies_mesh = grid.threshold([facies_id - 0.01, facies_id + 0.01], scalars="Channel")

        if facies_mesh.n_cells > 0:
            # Convert to surface and apply smoothing
            try:
                # Try using extract_surface, which usually generates better continuous surfaces
                facies_surface = facies_mesh.extract_surface()

                # Apply different degrees of smoothing based on sedimentary facies type
                if facies_id == 3:  # Natural levees need stronger smoothing
                    facies_surface = facies_surface.smooth(n_iter=25, relaxation_factor=0.4)
                else:
                    facies_surface = facies_surface.smooth(n_iter=15, relaxation_factor=0.2)
            except:
                # If surface extraction fails, fall back to original grid
                facies_surface = facies_mesh

            # Save grid for interactive control
            facies_surfaces[facies_name] = {
                'surface': facies_surface,
                'color': facies_color,
                'opacity': facies_opacity
            }

            # Add to scene, save actor reference
            actor = plotter.add_mesh(facies_surface, color=facies_color, label=facies_name,
                                     opacity=facies_opacity, show_edges=False,
                                     smooth_shading=True, specular=0.5, specular_power=15)

            # Save actor reference for controlling show/hide
            facies_surfaces[facies_name]['actor'] = actor

            facies_counts[facies_name] = facies_mesh.n_cells
            print(f"Added {facies_name}: {facies_mesh.n_cells} cells")

    # Check total sedimentary facies cell count
    total_facies = np.sum(channel > 0)
    print(f"Total sedimentary facies cell count: {total_facies}")
    print(f"Total rendered cell count: {sum(facies_counts.values())}")

    # Add warning if features are missing
    if sum(facies_counts.values()) < total_facies * 0.9:
        print(
            f"Warning: Approximately {total_facies - sum(facies_counts.values())} cells ({(1 - sum(facies_counts.values()) / total_facies) * 100:.1f}%) were not properly rendered")

    # Add clear labels for legend
    plotter.add_legend(bcolor='white', face='circle', size=[0.15, 0.15])

    # Add text information
    info_text = [
        f"Simulation #{isim}",
        f"Global Proportion: {realization['global_proportion']:.4f}",
        "Sedimentary Facies Cell Statistics:"
    ]

    for facies_name, count in facies_counts.items():
        info_text.append(f"- {facies_name}: {count} ({count / total_facies * 100:.1f}%)")

    plotter.add_text("\n".join(info_text), position='upper_left',
                     font_size=12, color='black', shadow=True)

    # Add coordinate axes and grid
    plotter.show_grid()
    plotter.add_axes()

    # Optimize camera position to ensure all content is visible
    plotter.view_isometric()
    plotter.reset_camera()

    # Set view range to ensure all content is visible
    plotter.set_scale(1, 1, 2)  # Enhance Z-direction to make channels more visible

    # Enhance lighting effects to improve depth perception
    plotter.enable_eye_dome_lighting()

    # Add interactive controls - checkbox controls for showing/hiding various sedimentary facies
    def toggle_facies(facies_name):
        def callback(state):
            actor = facies_surfaces[facies_name]['actor']
            if state:
                actor.SetVisibility(1)
            else:
                actor.SetVisibility(0)
            plotter.render()

        return callback

    # Add checkbox controls to interface
    for i, facies_name in enumerate(["Main Channel", "Natural Levees", "Crevasse Splays"]):
        if facies_name in facies_surfaces:
            color = facies_surfaces[facies_name]['color']
            plotter.add_checkbox_button_widget(toggle_facies(facies_name), value=True,
                                               position=(10, 10 + i * 30), size=20,
                                               border_size=1, color_on=color,
                                               color_off='grey', background_color='white')
            plotter.add_text(f"Show {facies_name}", position=(40, 10 + i * 30),
                             font_size=12, color='black')

    # Display rendering
    plotter.show(title=f"3D Fluvial Channel Model - Realization {isim}")
