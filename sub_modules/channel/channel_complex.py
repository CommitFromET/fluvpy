"""
channel_complex.py

This module implements regional control and spatial distribution algorithms for channel complexes.
The algorithm establishes a multi-level spatial partitioning system through the RegionController class,
enabling probabilistic distribution and parametric control of channel complexes within lateral and
vertical partitions.

Core Control Mechanisms:
1. Regional Initialization: Default regional configuration generation based on model scale and partitioning strategy
2. Hierarchical Parameter Resolution: Regional parameters > Global parameters > Default parameters priority system
3. Lateral Partition Management: Spatial division and polling scheduling mechanism along X or Y axis
4. Vertical Partition Control: Z-direction probabilistic distribution and stratified generation based on stratigraphic weights
5. Probabilistic Allocation: Regional generation probability calculation and dynamic adjustment based on density factors
6. Safety Net Mechanism: Forced complex creation guarantee in zero-generation scenarios

Spatial Partitioning Features:
- Multi-axis partition support: Flexible partition configuration and parameter inheritance along X or Y axis
- Regional density control: Complex generation frequency regulation based on relative density factors
- Vertical stratigraphic modeling: Stratigraphic thickness distribution and weight allocation strategies for different regions
- Boundary smoothing: Continuity assurance and gradient optimization for inter-regional parameter transitions
- Dynamic load balancing: Intelligent management of regional generation counts and attempt records
"""
import numpy as np
from typing import Dict, Any
from ..engine import constants as const
from ..utils.utils import calculate_z_position, get_value_from_distribution, is_distribution_dict


class RegionController:
    """
    Region Controller - Manages regional partitioning and parameters for channel complexes.

    This class provides comprehensive control over spatial partitioning, probability-based distribution,
    and parameter management for channel complex generation across multiple regions.
    """

    def __init__(self, params, axis='x', num_regions=3):
        """
        Initialize the region controller.

        Args:
            params (dict): Simulation parameter dictionary containing grid configuration and channel parameters
            axis (str): Partition axis, 'x' or 'y', defaults to 'x'
            num_regions (int): Number of regions, defaults to 3

        Returns:
            None: Constructor has no return value
        """
        self.params = params
        self.axis = axis.lower()
        self.num_regions = num_regions
        self.regions = []
        self.distribution_mode = params.get('partition_distribution', 'probability_based')

        # Lateral partition polling control
        self.current_region_index = 0
        self.attempted_regions = set()
        self.first_complex_generated = False
        self.generated_complexes_count = 0

        # Vertical partition polling control
        self.current_z_region_indices = {}
        self.attempted_z_regions = {}
        self.z_generated_counts = {}

        # Initialize regions
        self._initialize_regions()

        # Apply user-defined regional configurations
        if 'region_configs' in params:
            for region_id, config in params['region_configs'].items():
                self.configure_region(int(region_id), **config)

        # Calculate regional probabilities
        self._calculate_region_probabilities()

        # Initialize vertical partition polling states
        self._initialize_z_region_polling()

    def _get_parameter_with_priority(self, param_name, region_id=None, default_value=None):
        """
        Retrieve parameters with priority: Regional parameters > Global parameters > Default values.

        Args:
            param_name (str): Parameter name string
            region_id (int): Region ID, if provided, prioritize retrieval from corresponding regional configuration
            default_value: Default value used when parameter is not found in other sources

        Returns:
            Any: Retrieved parameter value
        """
        # First priority: retrieve from specified regional configuration
        if region_id is not None and 'region_configs' in self.params:
            region_config = self.params['region_configs'].get(region_id, {})
            if param_name in region_config:
                return region_config[param_name]

        # Second priority: retrieve from global parameters
        if param_name in self.params:
            return self.params[param_name]

        # Third priority: attempt to retrieve from any regional configuration as reference
        if 'region_configs' in self.params:
            for region_config in self.params['region_configs'].values():
                if param_name in region_config:
                    return region_config[param_name]

        # Fourth priority: use default value
        if default_value is not None:
            return default_value

        # If none available, return reasonable default values
        default_values = {
            'fcco': [0, 0, 0],
            'fcct': [3, 5, 8],
            'fccwtr': [1.2, 1.5, 1.8],
            'fccntg': [0.6, 0.7, 0.8],
            'fcat': [3, 5, 8],
            'channel_sinuosity': [1.1, 1.3, 1.5],
            'z_min_depth': 0.01,
            'z_max_depth': 100.0,
            'density_factor': 1.0
        }

        return default_values.get(param_name, [0, 1, 2])

    def _initialize_regions(self):
        """
        Initialize default regional configurations.

        Args:
            None

        Returns:
            None: Directly modifies self.regions list
        """
        # Get grid dimensions
        nx = self.params['nx']
        ny = self.params['ny']
        xmn = self.params['xmn']
        ymn = self.params['ymn']
        xsiz = self.params['xsiz']
        ysiz = self.params['ysiz']

        # Get Z range parameters
        z_min = self.params.get('z_min_depth', 0.01 * self.params['nz'] * self.params['zsiz'])
        z_max = self.params.get('z_max_depth', 1.0 * self.params['nz'] * self.params['zsiz'])
        z_range = z_max - z_min

        # Determine partition direction and dimensions based on axis
        if self.axis == 'x':
            total_size = nx * xsiz
            min_pos = xmn
        else:  # self.axis == 'y'
            total_size = ny * ysiz
            min_pos = ymn

        # Calculate size of each region
        region_size = total_size / self.num_regions

        # Create configuration for each region
        for i in range(self.num_regions):
            # Regional boundaries
            start_pos = min_pos + i * region_size
            end_pos = min_pos + (i + 1) * region_size

            # Retrieve base parameters from partition configuration or global parameters
            base_fcco = self._get_parameter_with_priority('fcco', i, [0, 0, 0])
            base_fcct = self._get_parameter_with_priority('fcct', i, [3, 5, 8])
            base_fccwtr = self._get_parameter_with_priority('fccwtr', i, [1.2, 1.5, 1.8])
            base_fccntg = self._get_parameter_with_priority('fccntg', i, [0.6, 0.7, 0.8])
            base_density = self._get_parameter_with_priority('density_factor', i, 1.0)

            # Regional complex density
            density_factor = base_density if isinstance(base_density, (int, float)) else 0.5 + (
                        i / (self.num_regions - 1)) if self.num_regions > 1 else 1.0

            # Regional angle range
            if isinstance(base_fcco, list) and len(base_fcco) >= 3:
                angle_min = base_fcco[0] + i * 30 / self.num_regions
                angle_max = base_fcco[2] - (self.num_regions - 1 - i) * 30 / self.num_regions
                angle_mid = (angle_min + angle_max) / 2
                fcco_region = [angle_min, angle_mid, angle_max]
            elif is_distribution_dict(base_fcco):
                import copy
                fcco_region = copy.deepcopy(base_fcco)
                if fcco_region.get('type') == 'normal':
                    fcco_region['mean'] = fcco_region.get('mean', 0) + i * 10
            else:
                fcco_region = [i * 15 - 30, i * 15 - 15, i * 15]

            # Regional thickness and width-to-thickness ratio
            thickness_factor = 1.0
            wtr_factor = 0.8 + 0.4 * (
                        (self.num_regions - 1 - i) / (self.num_regions - 1)) if self.num_regions > 1 else 1.0

            # Process thickness parameters
            if isinstance(base_fcct, list):
                fcct_region = [p * thickness_factor for p in base_fcct]
            elif is_distribution_dict(base_fcct):
                import copy
                fcct_region = copy.deepcopy(base_fcct)
                if fcct_region.get('type') == 'normal':
                    fcct_region['mean'] = fcct_region.get('mean', 5.0) * thickness_factor
            else:
                fcct_region = [3 * thickness_factor, 5 * thickness_factor, 8 * thickness_factor]

            # Process width-to-thickness ratio parameters
            if isinstance(base_fccwtr, list):
                fccwtr_region = [p * wtr_factor for p in base_fccwtr]
            elif is_distribution_dict(base_fccwtr):
                import copy
                fccwtr_region = copy.deepcopy(base_fccwtr)
                if fccwtr_region.get('type') == 'normal':
                    fccwtr_region['mean'] = fccwtr_region.get('mean', 1.4) * wtr_factor
            else:
                fccwtr_region = [1.2 * wtr_factor, 1.5 * wtr_factor, 1.8 * wtr_factor]

            # Process net-to-gross ratio parameters
            if isinstance(base_fccntg, list):
                fccntg_region = base_fccntg.copy()
            elif is_distribution_dict(base_fccntg):
                import copy
                fccntg_region = copy.deepcopy(base_fccntg)
            else:
                fccntg_region = [0.6, 0.7, 0.8]

            # Create regional Z distribution parameters
            region_z_params = {
                'z_min_depth': z_min,
                'z_max_depth': z_max,
                'z_distribution_mode': 'custom'
            }

            # Set stratigraphic count and weights
            strata_count = 8 + i * 2
            region_z_params['z_num_strata'] = strata_count

            # Create weight distribution based on regional index
            if i == 0:  # First region biased toward shallow layers
                weights = [strata_count - j for j in range(strata_count)]
            elif i == self.num_regions - 1:  # Last region biased toward deep layers
                weights = [j + 1 for j in range(strata_count)]
            else:  # Middle regions use different distribution patterns
                if i % 3 == 1:  # Bell-shaped distribution
                    mid = strata_count // 2
                    weights = [strata_count - abs(j - mid) for j in range(strata_count)]
                elif i % 3 == 2:  # V-shaped distribution
                    weights = [min(j + 1, strata_count - j) for j in range(strata_count)]
                else:  # Uniform distribution
                    weights = [strata_count // 2 for j in range(strata_count)]

            # Ensure all weights are positive
            weights = [max(0, w) for w in weights]
            region_z_params['z_custom_weights'] = weights

            # Save regional information
            region = {
                'id': i,
                'start': start_pos,
                'end': end_pos,
                'density_factor': density_factor,
                'fcco': fcco_region,
                'fcct': fcct_region,
                'fccwtr': fccwtr_region,
                'fccntg': fccntg_region,
                'z_distribution_params': region_z_params,
                'probability': 0.0
            }

            # Add channel-related parameters, retrieved from partition configuration or global parameters
            region['fcat'] = self._get_parameter_with_priority('fcat', i, fcct_region)  # Channel thickness uses complex thickness as default
            region['channel_sinuosity'] = self._get_parameter_with_priority('channel_sinuosity', i, [1.1, 1.3, 1.5])
            region['fcwtr'] = self._get_parameter_with_priority('fcwtr', i, fccwtr_region)  # Channel width-to-thickness ratio
            region['fcau'] = self._get_parameter_with_priority('fcau', i, [0.9, 1.0, 1.1])
            region['fcawu'] = self._get_parameter_with_priority('fcawu', i, [13, 15, 19])

            self.regions.append(region)

            # Add regional Z distribution parameters to global parameters
            if 'region_z_distribution' not in self.params:
                self.params['region_z_distribution'] = {}
            self.params['region_z_distribution'][str(i)] = region_z_params

    def _initialize_z_region_polling(self):
        """
        Initialize vertical partition polling states.

        Args:
            None

        Returns:
            None: Directly modifies instance variables
        """
        for region_id in range(self.num_regions):
            self.current_z_region_indices[region_id] = 0
            self.attempted_z_regions[region_id] = set()
            self.z_generated_counts[region_id] = 0

    def configure_region(self, region_id, **kwargs):
        """
        Configure parameters for a specific region.

        Args:
            region_id (int): Region ID
            **kwargs: Parameter key-value pairs to configure

        Returns:
            None: Directly modifies regional configuration
        """
        if 0 <= region_id < len(self.regions):
            for key, value in kwargs.items():
                self.regions[region_id][key] = value

    def _calculate_region_probabilities(self):
        """
        Calculate generation probabilities for each region.

        Args:
            None

        Returns:
            None: Directly modifies regional probability attributes
        """
        total_density = sum(region['density_factor'] for region in self.regions)

        if total_density <= 0:
            for region in self.regions:
                region['probability'] = 1.0 / len(self.regions)
        else:
            for region in self.regions:
                region['probability'] = region['density_factor'] / total_density

    def _calculate_z_region_probabilities(self, region_id):
        """
        Calculate generation probabilities for vertical partitions within specified lateral partition.

        Args:
            region_id (int): Lateral partition ID

        Returns:
            list: List of vertical partition probabilities
        """
        if region_id >= len(self.regions):
            return []

        region = self.regions[region_id]
        z_params = region.get('z_distribution_params', {})
        z_weights = z_params.get('z_custom_weights', [1])

        total_weight = sum(z_weights)

        if total_weight <= 0:
            probabilities = [1.0 / len(z_weights) for _ in z_weights]
        else:
            probabilities = [weight / total_weight for weight in z_weights]

        return probabilities

    def get_region_for_position(self, pos):
        """
        Determine the region based on spatial position coordinates.

        Args:
            pos (float): Spatial position coordinate value

        Returns:
            dict: Corresponding regional configuration dictionary
        """
        for region in self.regions:
            if region['start'] <= pos < region['end']:
                return region
        return self.regions[-1]

    def get_region_by_id(self, region_id):
        """
        Retrieve regional configuration based on region ID.

        Args:
            region_id (int): Region ID integer

        Returns:
            dict: Regional configuration dictionary, returns None or first region on failure
        """
        if 0 <= region_id < len(self.regions):
            return self.regions[region_id]
        else:
            return self.regions[0] if self.regions else None

    def get_next_region(self):
        """
        Get the next region to attempt.

        Args:
            None

        Returns:
            dict: Next regional configuration dictionary
        """
        region = self.regions[self.current_region_index]
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        self.attempted_regions.add(region['id'])
        return region

    def should_generate_complex(self, region):
        """
        Probabilistically determine whether to generate complex in current region.

        Args:
            region (dict): Regional configuration dictionary

        Returns:
            bool: Whether complex should be generated
        """
        random_value = np.random.random()
        return random_value < region['probability']

    def reset_attempt_tracking(self):
        """
        Reset attempt tracking state.

        Args:
            None

        Returns:
            None: Directly clears attempt records
        """
        self.attempted_regions.clear()

    def get_region_with_highest_probability(self):
        """
        Get regional configuration with highest probability.

        Args:
            None

        Returns:
            dict: Regional configuration dictionary with highest probability, returns None on failure
        """
        if not self.regions:
            return None

        max_probability = -1
        best_region = None

        for region in self.regions:
            if region['probability'] > max_probability:
                max_probability = region['probability']
                best_region = region

        return best_region

    def get_middle_region(self):
        """
        Get middle regional configuration.

        Args:
            None

        Returns:
            dict: Middle regional configuration dictionary, returns None or first region on failure
        """
        if not self.regions:
            return None

        middle_index = self.num_regions // 2

        if self.num_regions % 2 == 0 and middle_index > 0:
            middle_index = middle_index - 1

        return self.regions[middle_index] if middle_index < len(self.regions) else self.regions[0]

    def get_region_for_complex(self, complex_idx, total_complexes):
        """
        Assign region for specified complex.

        Args:
            complex_idx (int): Complex index
            total_complexes (int): Total number of complexes

        Returns:
            dict: Assigned regional configuration dictionary, returns None on failure
        """
        # Determine if this is the first complex and mxcc < 3
        is_first_complex = (self.generated_complexes_count == 0)
        should_force_first = is_first_complex and total_complexes < 3

        # If all regions have been attempted, reset attempt records
        if len(self.attempted_regions) >= len(self.regions):
            if should_force_first:
                middle_region = self.get_middle_region()
                if middle_region is not None:
                    self.generated_complexes_count += 1
                    self.first_complex_generated = True
                    self.reset_attempt_tracking()
                    return middle_region
                else:
                    return None
            else:
                self.reset_attempt_tracking()
                return None

        # Attempt all regions
        while len(self.attempted_regions) < len(self.regions):
            region = self.get_next_region()

            if should_force_first:
                if region['id'] == self.get_middle_region()['id']:
                    self.generated_complexes_count += 1
                    self.first_complex_generated = True
                    return region
                else:
                    continue

            if self.should_generate_complex(region):
                self.generated_complexes_count += 1
                return region
            else:
                pass

        # Safety net
        if should_force_first:
            middle_region = self.get_middle_region()
            if middle_region is not None:
                self.generated_complexes_count += 1
                self.first_complex_generated = True
                return middle_region

        return None

    def get_next_z_region(self, region_id):
        """
        Get next vertical partition to attempt within specified lateral partition.

        Args:
            region_id (int): Lateral partition ID

        Returns:
            tuple: (vertical partition index, vertical partition probability)
        """
        if region_id not in self.current_z_region_indices:
            self.current_z_region_indices[region_id] = 0
            self.attempted_z_regions[region_id] = set()

        # Get vertical partition parameters
        region = self.regions[region_id]
        z_params = region.get('z_distribution_params', {})
        z_weights = z_params.get('z_custom_weights', [1])
        z_probabilities = self._calculate_z_region_probabilities(region_id)

        num_z_regions = len(z_weights)
        current_z_index = self.current_z_region_indices[region_id]

        # Get probability of current vertical partition
        z_probability = z_probabilities[current_z_index] if current_z_index < len(z_probabilities) else 0.0

        # Update to next vertical partition index
        self.current_z_region_indices[region_id] = (current_z_index + 1) % num_z_regions

        return current_z_index, z_probability

    def should_generate_in_z_region(self, z_probability):
        """
        Probabilistically determine whether to generate complex in current vertical partition.

        Args:
            z_probability (float): Vertical partition generation probability

        Returns:
            bool: Whether complex should be generated in this vertical partition
        """
        random_value = np.random.random()
        will_generate = random_value < z_probability
        return will_generate

    def reset_z_attempt_tracking(self, region_id):
        """
        Reset vertical partition attempt tracking state for specified lateral partition.

        Args:
            region_id (int): Lateral partition ID

        Returns:
            None: Directly clears attempt records for this partition
        """
        if region_id in self.attempted_z_regions:
            self.attempted_z_regions[region_id].clear()

    def get_z_region_for_complex(self, region_id, complex_idx, total_complexes):
        """
        Assign vertical partition for complex within given lateral partition.

        Args:
            region_id (int): Lateral partition ID
            complex_idx (int): Complex index
            total_complexes (int): Total number of complexes

        Returns:
            int: Assigned vertical partition index, returns None on failure
        """
        if region_id >= len(self.regions):
            return None

        region = self.regions[region_id]
        z_params = region.get('z_distribution_params', {})
        z_weights = z_params.get('z_custom_weights', [1])
        num_z_regions = len(z_weights)

        # Ensure vertical partition polling state is initialized
        if region_id not in self.attempted_z_regions:
            self.attempted_z_regions[region_id] = set()
        if region_id not in self.current_z_region_indices:
            self.current_z_region_indices[region_id] = 0

        # Poll through all vertical partitions
        for attempt in range(num_z_regions):
            z_region_idx, z_probability = self.get_next_z_region(region_id)

            # Mark as attempted
            self.attempted_z_regions[region_id].add(z_region_idx)

            if self.should_generate_in_z_region(z_probability):
                self.z_generated_counts[region_id] = self.z_generated_counts.get(region_id, 0) + 1
                return z_region_idx

        # Reset attempt records for next allocation
        self.reset_z_attempt_tracking(region_id)
        return None

    def get_position_for_complex(self, complex_idx, total_complexes, params):
        """
        Generate spatial position coordinates for specified complex within assigned region.

        Args:
            complex_idx (int): Complex index
            total_complexes (int): Total number of complexes
            params (dict): Simulation parameter dictionary

        Returns:
            tuple: (x coordinate, y coordinate)
        """
        region = self.get_region_for_complex(complex_idx, total_complexes)

        if self.axis == 'x':
            x_pos = region['start'] + np.random.random() * (region['end'] - region['start'])
            y_pos = params['ymn'] + np.random.random() * params['ny'] * params['ysiz']
        else:  # self.axis == 'y'
            x_pos = params['xmn'] + np.random.random() * params['nx'] * params['xsiz']
            y_pos = region['start'] + np.random.random() * (region['end'] - region['start'])

        return x_pos, y_pos


def getcc_with_density_control(icc: int, params: Dict[str, Any], ccx: np.ndarray, ccz: np.ndarray,
                               cco: np.ndarray, cct: np.ndarray, ccw: np.ndarray, ccntg: np.ndarray) -> bool:
    """
    Generate channel complex with regional partition control.

    This function implements a sophisticated regional control system for channel complex generation,
    incorporating both lateral and vertical partitioning with probabilistic allocation mechanisms.

    Args:
        icc (int): Complex number (starting from 1)
        params (dict): Simulation parameter dictionary
        ccx (np.ndarray): Complex position array (x-coordinate)
        ccz (np.ndarray): Complex position array (z-coordinate)
        cco (np.ndarray): Complex orientation angle array
        cct (np.ndarray): Complex thickness array
        ccw (np.ndarray): Complex width array
        ccntg (np.ndarray): Complex net-to-gross ratio array

    Returns:
        bool: True if generation successful, False if failed
    """
    # Save and set random state
    current_state = np.random.get_state()
    np.random.set_state(const.random_state)

    # Get region controller
    region_controller = params.get('region_controller')
    if region_controller is None:
        axis = params.get('partition_axis', 'x')
        num_regions = params.get('num_regions', 3)
        region_controller = RegionController(params, axis, num_regions)
        params['region_controller'] = region_controller

    # Get lateral partition for this complex
    region = region_controller.get_region_for_complex(icc - 1, params['mxcc'])

    if region is None:
        const.random_state = np.random.get_state()
        np.random.set_state(current_state)
        return False

    # Attempt to get vertical partition within selected lateral partition
    z_region_index = region_controller.get_z_region_for_complex(region['id'], icc - 1, params['mxcc'])

    if z_region_index is None:
        const.random_state = np.random.get_state()
        np.random.set_state(current_state)
        return False

    print(f"Complex {icc} Region {region['id']} Vertical {z_region_index}")

    # Pass regional configuration parameters to global parameters, ensuring channel generation can access them
    region_params_mapping = {
        'fcat': 'fcat',
        'channel_sinuosity': 'channel_sinuosity',
        'fcwtr': 'fcwtr',
        'fcau': 'fcau',
        'fcawu': 'fcawu'
    }

    # Save current complex's regional configuration to global parameters
    if 'current_region_params' not in params:
        params['current_region_params'] = {}

    params['current_region_params'][icc] = {}
    for global_key, region_key in region_params_mapping.items():
        if region_key in region:
            params['current_region_params'][icc][global_key] = region[region_key]
            # Also add to global parameters as backup
            if global_key not in params:
                params[global_key] = region[region_key]

    # Set complex parameters

    # Generate complex X position within selected lateral partition
    if region_controller.axis == 'x':
        region_width = region['end'] - region['start']
        ccx[icc - 1] = region['start'] + np.random.random() * region_width
    else:  # region_controller.axis == 'y'
        ccx[icc - 1] = params['xmn'] + np.random.random() * params['nx'] * params['xsiz']

    # Create region-specific parameter set
    region_params = params.copy()

    # Use region-specific Z distribution parameters
    if 'z_distribution_params' in region:
        z_params = region['z_distribution_params']
        for key, value in z_params.items():
            region_params[key] = value
        region_params['z_distribution_mode'] = 'custom'

    # Calculate Z value based on vertical partition
    z_weights = region_params.get('z_custom_weights', [1])
    z_min = region_params.get('z_min_depth', 0.01 * params['nz'] * params['zsiz'])
    z_max = region_params.get('z_max_depth', 1.0 * params['nz'] * params['zsiz'])

    # Calculate Z value range for this vertical partition
    num_z_regions = len(z_weights)
    z_range = z_max - z_min
    z_layer_height = z_range / num_z_regions

    z_layer_min = z_min + z_region_index * z_layer_height
    z_layer_max = z_min + (z_region_index + 1) * z_layer_height

    # Randomly generate Z value within this vertical partition
    ccz[icc - 1] = z_layer_min + np.random.random() * (z_layer_max - z_layer_min)

    # Ensure Z value is within reasonable range
    min_z = region_params.get('z_min_bound',
                              region['fcct'][1] if isinstance(region['fcct'], list) else region['fcct'].get('min', 0))
    max_z = region_params.get('z_max_bound', params['avgthick'] * 1.5)
    ccz[icc - 1] = max(min_z, min(max_z, ccz[icc - 1]))

    # Set other parameters

    # Orientation angle
    if is_distribution_dict(region.get('fcco')):
        cco[icc - 1] = get_value_from_distribution(region['fcco'])
    else:
        angle_range = params.get('angle_range', 60.0)
        base_angle = (region['fcco'][0] + region['fcco'][2]) / 2
        angle_variation = np.random.normal(0, angle_range / 3)
        cco[icc - 1] = base_angle + angle_variation
        cco[icc - 1] = max(region['fcco'][0], min(region['fcco'][2], cco[icc - 1]))

    # Thickness
    thickness_variation = params.get('thickness_variation', 0.01)
    cct[icc - 1] = get_value_from_distribution(region['fcct']) * (1 + np.random.normal(0, thickness_variation))

    # Width
    width_variation = params.get('width_variation', 0.25)
    ccw[icc - 1] = get_value_from_distribution(region['fccwtr']) * cct[icc - 1] * (
                1 + np.random.normal(0, width_variation))

    # Net-to-gross ratio
    ntg_variation = params.get('ntg_variation', 0.2)
    ccntg[icc - 1] = get_value_from_distribution(region['fccntg']) * (1 + np.random.normal(0, ntg_variation))
    ccntg[icc - 1] = max(0.05, min(0.95, ccntg[icc - 1]))

    # Update global random state
    const.random_state = np.random.get_state()
    np.random.set_state(current_state)

    return True


def create_fallback_complex(params: Dict[str, Any], ccx: np.ndarray, ccz: np.ndarray,
                            cco: np.ndarray, cct: np.ndarray, ccw: np.ndarray, ccntg: np.ndarray) -> bool:
    """
    Force generation of a complex at the model center (safety net mechanism).

    This function serves as a safety net to ensure at least one complex is generated
    when the probabilistic regional allocation fails to produce any complexes.

    Args:
        params (dict): Simulation parameter dictionary
        ccx (np.ndarray): Complex position array (x-coordinate)
        ccz (np.ndarray): Complex position array (z-coordinate)
        cco (np.ndarray): Complex orientation angle array
        cct (np.ndarray): Complex thickness array
        ccw (np.ndarray): Complex width array
        ccntg (np.ndarray): Complex net-to-gross ratio array

    Returns:
        bool: True if generation successful, False if failed
    """
    print("    Detected 0 complexes, activating safety net mechanism to force complex generation at model center")

    # Save and set random state
    current_state = np.random.get_state()
    np.random.set_state(const.random_state)

    try:
        # Calculate model center position
        nx = params['nx']
        ny = params['ny']
        xmn = params['xmn']
        ymn = params['ymn']
        xsiz = params['xsiz']
        ysiz = params['ysiz']

        # X position: model center
        ccx[0] = xmn + (nx * xsiz) / 2.0

        # Z position: use default depth calculation method
        ccz[0] = calculate_z_position(1, params.get('mxcc', 1000), params)

        # Angle: use parameter median
        if is_distribution_dict(params.get('fcco')):
            cco[0] = get_value_from_distribution(params['fcco'])
        else:
            if isinstance(params['fcco'], list) and len(params['fcco']) >= 3:
                cco[0] = params['fcco'][1]  # Use median
            else:
                cco[0] = 0.0  # Default angle

        # Thickness: use parameter default value
        if is_distribution_dict(params.get('fcct')):
            cct[0] = get_value_from_distribution(params['fcct'])
        else:
            if isinstance(params['fcct'], list) and len(params['fcct']) >= 2:
                cct[0] = params['fcct'][1]  # Use median
            else:
                cct[0] = params.get('avgthick', 5.0)  # Use default value

        # Width
        if is_distribution_dict(params.get('fccwtr')):
            width_ratio = get_value_from_distribution(params['fccwtr'])
        else:
            if isinstance(params['fccwtr'], list) and len(params['fccwtr']) >= 2:
                width_ratio = params['fccwtr'][1]  # Use median
            else:
                width_ratio = 1.4  # Default width-to-thickness ratio

        ccw[0] = width_ratio * cct[0]

        # Net-to-gross ratio
        if is_distribution_dict(params.get('fccntg')):
            ccntg[0] = get_value_from_distribution(params['fccntg'])
        else:
            if isinstance(params['fccntg'], list) and len(params['fccntg']) >= 2:
                ccntg[0] = params['fccntg'][1]  # Use median
            else:
                ccntg[0] = 0.6  # Default net-to-gross ratio

        # Ensure values are within reasonable ranges
        ccntg[0] = max(0.05, min(0.95, ccntg[0]))

        print(
            f"    Safety net complex parameters: X={ccx[0]:.2f}, Z={ccz[0]:.2f}, Angle={cco[0]:.1f}°, Thickness={cct[0]:.2f}, Width={ccw[0]:.2f}")

        # Update global random state
        const.random_state = np.random.get_state()
        np.random.set_state(current_state)

        return True

    except Exception as e:
        print(f"    Safety net complex generation failed: {e}")
        const.random_state = np.random.get_state()
        np.random.set_state(current_state)
        return False