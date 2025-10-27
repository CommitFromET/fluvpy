"""
vegetation_patches.py
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import time


class VegetationPatchGenerator:
    """Efficient natural vegetation patch generator"""

    def __init__(self, nx, ny, xmn, ymn, xsiz, ysiz, seed=None):
        """
        Initialize vegetation patch generator.

        Args:
            nx, ny: Number of grid nodes in X and Y directions
            xmn, ymn: Grid origin coordinates
            xsiz, ysiz: Grid cell size
            seed: Random seed

        Returns:
            None
        """
        self.nx = nx
        self.ny = ny
        self.xmn = xmn
        self.ymn = ymn
        self.xsiz = xsiz
        self.ysiz = ysiz
        self.initial_seed = seed
        self.current_seed = seed
        self.update_count = 0
        self.current_values = None

        # Pre-compute coordinate grids
        self.x_indices = np.arange(nx)
        self.y_indices = np.arange(ny)
        self.xx, self.yy = np.meshgrid(self.x_indices, self.y_indices, indexing='ij')

        if self.initial_seed is not None:
            print(f"Vegetation generator initialized with seed: {self.initial_seed}")
        else:
            print("Vegetation generator initialized with random seed")

    def _natural_fractal_noise(self, octaves=4, persistence=0.5, lacunarity=2.0):
        """
        Generate fractal noise using random value interpolation to create multi-scale noise patterns.

        Args:
            octaves: Number of octaves
            persistence: Persistence parameter
            lacunarity: Lacunarity parameter

        Returns:
            ndarray: Fractal noise array
        """
        if self.current_seed is not None:
            np.random.seed(self.current_seed)

        noise = np.zeros((self.nx, self.ny))

        for i in range(octaves):
            freq = lacunarity ** i
            amp = persistence ** i

            # Calculate grid size for current octave
            grid_size = max(4, int(min(self.nx, self.ny) / (freq * 4)))

            # Generate random value grid
            grid_nx = self.nx // grid_size + 2
            grid_ny = self.ny // grid_size + 2
            random_grid = np.random.random((grid_nx, grid_ny)) * 2 - 1

            # Bilinear interpolation
            octave_noise = np.zeros((self.nx, self.ny))

            for x in range(self.nx):
                for y in range(self.ny):
                    # Calculate position in random grid
                    gx = x / grid_size
                    gy = y / grid_size

                    # Get grid coordinates of four corner points
                    x0, y0 = int(gx), int(gy)
                    x1, y1 = x0 + 1, y0 + 1

                    # Ensure not out of bounds
                    x1 = min(x1, grid_nx - 1)
                    y1 = min(y1, grid_ny - 1)

                    # Calculate interpolation weights
                    wx = gx - x0
                    wy = gy - y0

                    # Bilinear interpolation
                    v00 = random_grid[x0, y0]
                    v10 = random_grid[x1, y0]
                    v01 = random_grid[x0, y1]
                    v11 = random_grid[x1, y1]

                    # Smooth interpolation function (smoothstep)
                    wx = wx * wx * (3 - 2 * wx)
                    wy = wy * wy * (3 - 2 * wy)

                    v0 = v00 * (1 - wx) + v10 * wx
                    v1 = v01 * (1 - wx) + v11 * wx
                    value = v0 * (1 - wy) + v1 * wy

                    octave_noise[x, y] = value

            noise += amp * octave_noise

        return noise

    def _add_natural_river_influence(self, base_map, river_strength=0.3):
        """
        Add river influence using random paths and irregular widths to simulate rivers.

        Args:
            base_map: Base map array
            river_strength: River influence strength

        Returns:
            ndarray: Map array with added river influence
        """
        if self.current_seed is not None:
            np.random.seed(self.current_seed + 1)

        num_rivers = np.random.randint(1, 4)
        river_influence = np.zeros_like(base_map)

        for river_id in range(num_rivers):
            river_seed = self.current_seed + river_id + 10 if self.current_seed else river_id
            np.random.seed(river_seed)

            # Random start and end points
            start_x = np.random.randint(0, self.nx // 4)
            start_y = np.random.uniform(0.2, 0.8) * self.ny
            end_x = np.random.randint(3 * self.nx // 4, self.nx)
            end_y = np.random.uniform(0.2, 0.8) * self.ny

            # Generate river path points
            num_points = 20
            river_points = []

            for i in range(num_points + 1):
                t = i / num_points

                # Basic linear interpolation
                base_x = start_x + t * (end_x - start_x)
                base_y = start_y + t * (end_y - start_y)

                # Add random offset, middle points have larger offset
                deviation_strength = 30 * np.sin(np.pi * t)
                offset_x = np.random.normal(0, deviation_strength)
                offset_y = np.random.normal(0, deviation_strength)

                river_x = np.clip(base_x + offset_x, 0, self.nx - 1)
                river_y = np.clip(base_y + offset_y, 0, self.ny - 1)

                river_points.append((river_x, river_y))

            # Calculate river influence for each grid point
            for i in range(self.nx):
                for j in range(self.ny):
                    min_distance = float('inf')

                    # Find nearest river distance
                    for k in range(len(river_points) - 1):
                        x1, y1 = river_points[k]
                        x2, y2 = river_points[k + 1]

                        # Calculate distance from point to line segment
                        A = i - x1
                        B = j - y1
                        C = x2 - x1
                        D = y2 - y1

                        dot = A * C + B * D
                        len_sq = C * C + D * D

                        if len_sq < 1e-10:
                            distance = np.sqrt(A * A + B * B)
                        else:
                            param = dot / len_sq
                            if param < 0:
                                xx, yy = x1, y1
                            elif param > 1:
                                xx, yy = x2, y2
                            else:
                                xx = x1 + param * C
                                yy = y1 + param * D

                            distance = np.sqrt((i - xx) ** 2 + (j - yy) ** 2)

                        min_distance = min(min_distance, distance)

                    # River influence decays with distance, width varies randomly
                    width_variation = np.random.uniform(0.7, 1.3)
                    base_width = 25 * width_variation
                    influence = np.exp(-min_distance / base_width) * river_strength
                    river_influence[i, j] = max(river_influence[i, j], influence)

        return base_map + river_influence

    def _add_kmeans_patches(self, base_map, num_patches=5, patch_strength=0.4):
        """
        Generate vegetation patches using soft clustering based on K-means algorithm for spatial clustering.

        Args:
            base_map: Base map array
            num_patches: Number of patches
            patch_strength: Patch strength

        Returns:
            ndarray: Map array with added patches
        """
        if self.current_seed is not None:
            np.random.seed(self.current_seed + 2)

        # Create spatial coordinate grid
        x_coords, y_coords = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')

        # Add spatial noise for realism
        spatial_correlation = 0.3
        noise_strength = spatial_correlation * 10
        x_noise = np.random.normal(0, noise_strength, (self.nx, self.ny))
        y_noise = np.random.normal(0, noise_strength, (self.nx, self.ny))

        # Combine coordinate features
        coords = np.column_stack([
            (x_coords + x_noise).flatten(),
            (y_coords + y_noise).flatten()
        ])

        # Add additional feature dimensions for clustering complexity
        heterogeneity_level = 0.8
        if heterogeneity_level > 0.5:
            center_x, center_y = self.nx // 2, self.ny // 2
            dist_feature = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2).flatten()
            dist_feature = dist_feature / np.max(dist_feature)

            angle_feature = np.arctan2(y_coords - center_y, x_coords - center_x).flatten()
            angle_feature = (angle_feature + np.pi) / (2 * np.pi)

            coords = np.column_stack([coords, dist_feature, angle_feature])

        # Get cluster centers
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_patches, random_state=self.current_seed, n_init=10)
            kmeans.fit(coords)
            cluster_centers = kmeans.cluster_centers_
        except ImportError:
            print("sklearn not installed, using grid division method")
            patch_size_x = max(1, self.nx // int(np.sqrt(num_patches)))
            patch_size_y = max(1, self.ny // int(np.sqrt(num_patches)))

            vegetation_patches = np.zeros((self.nx, self.ny), dtype=np.int32)
            patch_id = 0
            for i in range(0, self.nx, patch_size_x):
                for j in range(0, self.ny, patch_size_y):
                    vegetation_patches[i:i + patch_size_x, j:j + patch_size_y] = patch_id % num_patches
                    patch_id += 1

            actual_num_patches = len(np.unique(vegetation_patches))

            if heterogeneity_level > 0.8:
                patch_values = np.random.beta(0.5, 0.5, actual_num_patches)
            elif heterogeneity_level > 0.5:
                patch_values = np.concatenate([
                    np.random.beta(2, 8, actual_num_patches // 2),
                    np.random.beta(8, 2, actual_num_patches - actual_num_patches // 2)
                ])
                np.random.shuffle(patch_values)
            else:
                patch_values = np.random.normal(0.5, 0.2, actual_num_patches)
                patch_values = np.clip(patch_values, 0.1, 0.9)

            patch_values = np.clip(patch_values, 0.0, 1.0)
            patch_values = (patch_values - 0.5) * patch_strength * 2

            patch_influence = np.zeros_like(base_map)
            for i in range(actual_num_patches):
                patch_mask = vegetation_patches == i
                patch_influence[patch_mask] = patch_values[i]

            if spatial_correlation > 0.3:
                patch_influence = gaussian_filter(patch_influence, sigma=1.5)

            return base_map + patch_influence

        # Soft clustering weight calculation
        patch_influence = np.zeros((self.nx, self.ny), dtype=np.float64)

        # Soft clustering parameters
        softness_factor = 2.0
        distance_decay_power = 1.5
        overlap_enhancement = 0.3

        # Generate patch base values
        if heterogeneity_level > 0.8:
            patch_base_values = np.random.beta(0.5, 0.5, num_patches)
        elif heterogeneity_level > 0.5:
            patch_base_values = np.concatenate([
                np.random.beta(2, 8, num_patches // 2),
                np.random.beta(8, 2, num_patches - num_patches // 2)
            ])
            np.random.shuffle(patch_base_values)
        else:
            patch_base_values = np.random.normal(0.5, 0.2, num_patches)
            patch_base_values = np.clip(patch_base_values, 0.1, 0.9)

        patch_base_values = (patch_base_values - 0.5) * patch_strength * 2

        # Distance-based soft weight assignment
        for i in range(self.nx):
            for j in range(self.ny):
                point_coord = [i, j]

                if heterogeneity_level > 0.5:
                    center_x, center_y = self.nx // 2, self.ny // 2
                    dist_feat = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    dist_feat = dist_feat / np.sqrt((self.nx / 2) ** 2 + (self.ny / 2) ** 2)
                    angle_feat = np.arctan2(j - center_y, i - center_x)
                    angle_feat = (angle_feat + np.pi) / (2 * np.pi)
                    point_coord = [i, j, dist_feat, angle_feat]

                # Calculate distances to all cluster centers
                distances = []
                for center in cluster_centers:
                    if len(center) == len(point_coord):
                        dist = np.sqrt(np.sum((np.array(point_coord) - center) ** 2))
                    else:
                        dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                    distances.append(dist)

                distances = np.array(distances)

                # Soft weight calculation
                min_dist = np.min(distances)
                if min_dist < 1e-6:
                    min_dist = 1e-6

                # softmax weights
                inv_distances = 1.0 / (distances + min_dist * 0.1)
                softmax_weights = np.exp(inv_distances * softness_factor)
                softmax_weights = softmax_weights / np.sum(softmax_weights)

                # Distance decay weights
                max_dist = np.max(distances)
                if max_dist > 1e-6:
                    normalized_distances = distances / max_dist
                    decay_weights = np.exp(-normalized_distances ** distance_decay_power * softness_factor)
                    decay_weights = decay_weights / np.sum(decay_weights)
                else:
                    decay_weights = np.ones(len(distances)) / len(distances)

                # Fuse weights
                alpha = 0.6
                beta = 0.4
                final_weights = alpha * softmax_weights + beta * decay_weights

                # Calculate final influence value
                weighted_influence = 0.0
                for k in range(num_patches):
                    weight = final_weights[k]
                    base_value = patch_base_values[k]

                    # Boundary enhancement effect
                    if len(distances) > 1:
                        weight_entropy = -np.sum(final_weights * np.log(final_weights + 1e-10))
                        max_entropy = np.log(len(final_weights))
                        entropy_factor = weight_entropy / max_entropy if max_entropy > 0 else 0
                        boundary_enhancement = 1.0 + overlap_enhancement * entropy_factor
                        effective_weight = weight * boundary_enhancement
                    else:
                        effective_weight = weight

                    weighted_influence += effective_weight * base_value

                patch_influence[i, j] = weighted_influence

        # Multi-scale smoothing strategy
        smoothed_influence = gaussian_filter(patch_influence, sigma=1.0)
        medium_smooth = gaussian_filter(patch_influence, sigma=2.5)
        heavy_smooth = gaussian_filter(patch_influence, sigma=5.0)

        # Adaptive blending strategy
        local_variance = np.zeros_like(patch_influence)
        window_size = 3

        for i in range(window_size // 2, self.nx - window_size // 2):
            for j in range(window_size // 2, self.ny - window_size // 2):
                local_patch = patch_influence[i - window_size // 2:i + window_size // 2 + 1,
                              j - window_size // 2:j + window_size // 2 + 1]
                local_variance[i, j] = np.var(local_patch)

        max_var = np.max(local_variance)
        if max_var > 1e-10:
            normalized_variance = local_variance / max_var
        else:
            normalized_variance = np.zeros_like(local_variance)

        # Adaptive blending
        final_influence = np.zeros_like(patch_influence)
        for i in range(self.nx):
            for j in range(self.ny):
                var_factor = normalized_variance[i, j]

                w1 = 0.3 + 0.4 * var_factor
                w2 = 0.4 + 0.2 * var_factor
                w3 = 0.2 - 0.3 * var_factor
                w4 = 0.1 - 0.3 * var_factor

                total_weight = w1 + w2 + w3 + w4
                if total_weight > 0:
                    w1, w2, w3, w4 = w1 / total_weight, w2 / total_weight, w3 / total_weight, w4 / total_weight

                final_influence[i, j] = (w1 * patch_influence[i, j] +
                                         w2 * smoothed_influence[i, j] +
                                         w3 * medium_smooth[i, j] +
                                         w4 * heavy_smooth[i, j])

        return base_map + final_influence

    def generate_initial_patches(self, patch_count=8, target_mean=None, target_range=None,
                                 natural_variability=0.4, **kwargs):
        """
        Generate initial vegetation distribution.

        Args:
            patch_count: Number of patches
            target_mean: Target mean value
            target_range: Target range
            natural_variability: Natural variability
            **kwargs: Additional keyword arguments

        Returns:
            ndarray: Vegetation distribution array
        """
        print(f"Generating vegetation distribution, patch count: {patch_count}")

        start_time = time.time()

        self.update_count = 0
        self.current_seed = self.initial_seed

        if self.current_seed is not None:
            np.random.seed(self.current_seed)

        # Get target parameters
        if target_mean is None:
            target_mean = 0.3
        if target_range is None:
            target_range = (0.1, 0.5)

        # 1. Generate fractal noise as base
        base_noise = self._natural_fractal_noise(octaves=4, persistence=0.6, lacunarity=2.0)

        # 2. Add river influence
        with_rivers = self._add_natural_river_influence(base_noise, river_strength=0.3)

        # 3. Add patches using K-means clustering
        with_patches = self._add_kmeans_patches(with_rivers,
                                                num_patches=patch_count,
                                                patch_strength=0.4)

        # 4. Add fine-scale random variation
        fine_noise = np.random.normal(0, natural_variability * 0.15, (self.nx, self.ny))
        final_map = with_patches + fine_noise

        # 5. Light smoothing
        final_map = gaussian_filter(final_map, sigma=1.0)

        # 6. Map to target mean and range
        # First normalize to zero mean, unit variance
        final_map = (final_map - final_map.mean()) / (final_map.std() + 1e-10)

        # Calculate target standard deviation (based on range)
        range_width = target_range[1] - target_range[0]
        target_std = range_width / 4.0  # Use 4 times standard deviation to cover most range

        # Map to target distribution: centered on target_mean with target_std
        final_map = final_map * target_std + target_mean

        # Force constraint within target range
        final_map = np.clip(final_map, target_range[0], target_range[1])

        # Fine-tune to ensure accurate mean
        current_mean = final_map.mean()
        if abs(current_mean - target_mean) > 1e-6:
            adjustment = target_mean - current_mean
            final_map = final_map + adjustment
            # Constrain range again
            final_map = np.clip(final_map, target_range[0], target_range[1])

        self.current_values = final_map

        end_time = time.time()
        print(f"Vegetation distribution generation complete, time: {end_time - start_time:.3f}s")
        print(f"Influence range: [{final_map.min():.3f}, {final_map.max():.3f}], "
              f"mean: {final_map.mean():.3f}, std: {final_map.std():.3f}")

        return final_map.copy()

    def update_patches(self, evolution_factor=0.2, target_mean=None, target_range=None):
        """
        Update vegetation distribution, regenerate each time using incremented seed.

        Args:
            evolution_factor: Evolution factor
            target_mean: Target mean value
            target_range: Target range

        Returns:
            ndarray: Updated vegetation distribution array
        """
        self.update_count += 1
        print(f"Vegetation update #{self.update_count}")
        start_time = time.time()

        seed_increment = 10000
        if self.initial_seed is not None:
            self.current_seed = self.initial_seed + (self.update_count * seed_increment)
        else:
            self.current_seed = None

        if self.current_seed is not None:
            np.random.seed(self.current_seed)

        # Get target parameters
        if target_mean is None:
            target_mean = 0.3
        if target_range is None:
            target_range = (0.1, 0.5)

        # 1. Generate fractal noise as base
        base_noise = self._natural_fractal_noise(octaves=4, persistence=0.6, lacunarity=2.0)

        # 2. Add river influence
        with_rivers = self._add_natural_river_influence(base_noise, river_strength=0.3)

        # 3. Add patches using K-means clustering
        patch_count = max(5, int(8 + evolution_factor * 4))
        with_patches = self._add_kmeans_patches(with_rivers,
                                                num_patches=patch_count,
                                                patch_strength=0.4)

        # 4. Add fine-scale random variation
        natural_variability = 0.4 + evolution_factor * 0.2
        fine_noise = np.random.normal(0, natural_variability * 0.15, (self.nx, self.ny))
        final_map = with_patches + fine_noise

        # 5. Light smoothing
        final_map = gaussian_filter(final_map, sigma=1.0)

        # 6. Map to target mean and range
        # First normalize to zero mean, unit variance
        final_map = (final_map - final_map.mean()) / (final_map.std() + 1e-10)

        # Calculate target standard deviation (based on range)
        range_width = target_range[1] - target_range[0]
        target_std = range_width / 4.0

        # Increase variability with evolution
        evolution_std_multiplier = 1.0 + evolution_factor * 0.3
        target_std *= evolution_std_multiplier

        # Map to target distribution
        final_map = final_map * target_std + target_mean

        # Force constraint within target range
        final_map = np.clip(final_map, target_range[0], target_range[1])

        # Fine-tune to ensure accurate mean
        current_mean = final_map.mean()
        if abs(current_mean - target_mean) > 1e-6:
            adjustment = target_mean - current_mean
            final_map = final_map + adjustment
            final_map = np.clip(final_map, target_range[0], target_range[1])

        self.current_values = final_map

        end_time = time.time()
        print(f"Vegetation update complete, time: {end_time - start_time:.3f}s")
        print(f"Influence range: [{final_map.min():.3f}, {final_map.max():.3f}], "
              f"mean: {final_map.mean():.3f}, patch count: {patch_count}")

        return final_map.copy()

    def get_vegetation_at_point(self, x, y):
        """
        Get vegetation value at specified coordinates.

        Args:
            x, y: Query coordinates

        Returns:
            float: Vegetation value
        """
        if self.current_values is None:
            return 0.5

        # Fast index calculation
        i = int((x - self.xmn) / self.xsiz)
        j = int((y - self.ymn) / self.ysiz)

        # Boundary check
        i = max(0, min(i, self.nx - 1))
        j = max(0, min(j, self.ny - 1))

        return float(self.current_values[i, j])

    def get_vegetation_at_points_batch(self, x_array, y_array):
        """
        Batch query vegetation values for multiple points.

        Args:
            x_array, y_array: Coordinate arrays

        Returns:
            ndarray: Vegetation value array
        """
        if self.current_values is None:
            return np.full_like(x_array, 0.5)

        # Vectorized index calculation
        i_array = ((x_array - self.xmn) / self.xsiz).astype(int)
        j_array = ((y_array - self.ymn) / self.ysiz).astype(int)

        # Boundary check
        i_array = np.clip(i_array, 0, self.nx - 1)
        j_array = np.clip(j_array, 0, self.ny - 1)

        return self.current_values[i_array, j_array]

    def get_vegetation_map(self):
        """
        Get complete copy of current vegetation distribution map.

        Args:
            None

        Returns:
            ndarray: Vegetation distribution map array
        """
        return self.current_values.copy() if self.current_values is not None else None

    def get_vegetation_statistics(self):
        """
        Get vegetation distribution statistics.

        Args:
            None

        Returns:
            dict: Statistics dictionary
        """
        if self.current_values is None:
            return None

        return {
            'min': float(self.current_values.min()),
            'max': float(self.current_values.max()),
            'mean': float(self.current_values.mean()),
            'std': float(self.current_values.std()),
            'median': float(np.median(self.current_values)),
            'q25': float(np.percentile(self.current_values, 25)),
            'q75': float(np.percentile(self.current_values, 75)),
            'update_count': self.update_count,
            'current_seed': self.current_seed,
            'initial_seed': self.initial_seed
        }


def create_vegetation_generator(params):
    """
    Create vegetation generator instance from parameter dictionary.

    Args:
        params: Parameter dictionary

    Returns:
        VegetationPatchGenerator: Vegetation generator instance
    """
    nx = params.get('nx', 250)
    ny = params.get('ny', 250)
    xmn = params.get('xmn', 0)
    ymn = params.get('ymn', 0)
    xsiz = params.get('xsiz', 24)
    ysiz = params.get('ysiz', 24)

    # Prioritize independent vegetation seed, fallback to main seed if not available
    vegetation_seed = params.get('vegetation_seed', params.get('seed', None))

    return VegetationPatchGenerator(nx, ny, xmn, ymn, xsiz, ysiz, vegetation_seed)