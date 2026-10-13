"""Nieder-style dot arrays and visual control conditions."""

import numpy as np

NUMEROSITY_STIMULUS_MODES = [
    "standard",
    "equal_area",
    "equal_circumference",
    "shape",
    "linear",
    "high_density",
    "low_density",
    "variable_features",
]


class DotArrayGenerator:
    def __init__(
        self,
        grid_size=15,
        stimulus_mode="standard",
        min_dot_distance=3.0,
        stimulus_family="nieder",
        position_lattice_size=5,
        aperture_radius=None,
        aperture_value=0.65,
        dot_value=0.05,
        dot_radius_values=None,
        equal_area_total_area=None,
        density_candidate_samples=256,
    ):
        assert stimulus_mode in NUMEROSITY_STIMULUS_MODES

        self.grid_size = grid_size
        self.stimulus_mode = stimulus_mode
        self.min_dot_distance = min_dot_distance
        self.stimulus_family = stimulus_family
        self.position_lattice_size = position_lattice_size
        self.aperture_radius = (
            float(aperture_radius) if aperture_radius is not None else float(grid_size) * 0.42
        )
        self.aperture_value = float(aperture_value)
        self.dot_value = float(dot_value)
        if dot_radius_values is None:
            dot_radius_values = [0.035, 0.045, 0.055, 0.065, 0.075]
        self.dot_radius_values = np.array(dot_radius_values, dtype=float)
        # Validate explicit stimulus parameters instead of inventing radii or sample counts.
        if self.dot_radius_values.size == 0 or np.any(self.dot_radius_values <= 0):
            raise ValueError("dot_radius_values must contain positive radii")
        if int(density_candidate_samples) <= 0:
            raise ValueError("density_candidate_samples must be positive")
        max_radius_value = float(np.max(self.dot_radius_values))
        if max_radius_value <= 1.0:
            self.dot_radius_values = self.dot_radius_values * float(grid_size)
        self.equal_area_total_area = equal_area_total_area
        self.density_candidate_samples = int(density_candidate_samples)
        self.coords = np.arange(grid_size, dtype=float)
        self.xx, self.yy = np.meshgrid(self.coords, self.coords)
        self.center = (float(grid_size) - 1.0) / 2.0
        self.aperture_mask = (self.xx - self.center) ** 2 + (
            self.yy - self.center
        ) ** 2 <= self.aperture_radius**2
        self.lattice_positions = self._build_lattice_positions()

    def render(self, numerosity, linear_angle=None):
        # Flatten the Nieder-style image for the recurrent model input.
        return self._render_nieder(numerosity, linear_angle=linear_angle).reshape(-1)

    def _render_nieder(self, numerosity, linear_angle=None):
        mode = self.stimulus_mode
        positions = self._sample_nieder_positions(
            numerosity,
            mode,
            linear_angle=linear_angle,
        )
        radii = self._sample_nieder_radii(numerosity, mode)
        shapes = self._sample_nieder_shapes(numerosity, mode)
        angles = np.random.uniform(0.0, 2.0 * np.pi, size=numerosity)
        stimulus = np.zeros((self.grid_size, self.grid_size), dtype=float)
        stimulus[self.aperture_mask] = self.aperture_value

        for (cx, cy), radius, shape, angle in zip(positions, radii, shapes, angles):
            dot_mask = self._nieder_item_mask(shape, cx, cy, radius, angle)
            stimulus[dot_mask & self.aperture_mask] = self.dot_value
        return np.clip(stimulus, 0.0, 1.0)

    def _build_lattice_positions(self):
        margin = self.aperture_radius * 0.68
        points = np.linspace(
            self.center - margin,
            self.center + margin,
            int(self.position_lattice_size),
        )
        positions = []
        center_index = int(self.position_lattice_size) // 2
        for y_index, y in enumerate(points):
            for x_index, x in enumerate(points):
                if x_index == center_index and y_index == center_index:
                    continue
                if (x - self.center) ** 2 + (y - self.center) ** 2 <= (
                    self.aperture_radius - self.max_dot_radius
                ) ** 2:
                    positions.append((x, y))
        if len(positions) == 0:
            raise ValueError("nieder lattice has no valid dot positions")
        return np.array(positions, dtype=float)

    @property
    def max_dot_radius(self):
        max_radius = float(np.max(self.dot_radius_values))
        equal_area_one_dot_radius = np.sqrt(float(self._equal_area_total()) / np.pi)
        equal_circumference_one_dot_radius = self._equal_circumference_total() / (2.0 * np.pi)
        return max(
            max_radius,
            float(equal_area_one_dot_radius),
            float(equal_circumference_one_dot_radius),
        )

    def _reference_dot_radius(self):
        return float(np.median(self.dot_radius_values))

    def _linear_dot_radius(self):
        return self._reference_dot_radius()

    def _equal_area_total(self):
        if self.equal_area_total_area is not None:
            return float(self.equal_area_total_area)
        reference_radius = self._reference_dot_radius()
        return 3.0 * np.pi * reference_radius**2

    def _equal_circumference_total(self):
        reference_radius = self._reference_dot_radius()
        return 3.0 * 2.0 * np.pi * reference_radius

    def _sample_nieder_positions(self, numerosity, mode, linear_angle=None):
        if numerosity > len(self.lattice_positions):
            raise ValueError(
                f"numerosity {numerosity} exceeds lattice positions {len(self.lattice_positions)}"
            )
        if mode == "linear":
            return self._sample_linear_positions(numerosity, angle=linear_angle)
        if mode == "high_density":
            return self._sample_density_ranked_positions(numerosity, choose_high=True)
        if mode == "low_density":
            return self._sample_density_ranked_positions(numerosity, choose_high=False)
        indices = np.random.choice(len(self.lattice_positions), size=numerosity, replace=False)
        return self.lattice_positions[indices]

    def _sample_linear_positions(self, numerosity, angle=None):
        radius = self._linear_dot_radius()
        half_span = self.aperture_radius - radius - 2.0
        if half_span <= 0.0:
            raise ValueError("Linear dot arrays require room for distinct dot positions")
        if angle is None:
            angle = np.random.uniform(0.0, np.pi)
        angle = float(angle) % np.pi
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        normal = np.array([-direction[1], direction[0]], dtype=float)
        offset = np.random.uniform(-0.04, 0.04) * half_span
        if numerosity == 1:
            steps = np.array([np.random.uniform(-0.15, 0.15) * half_span])
        else:
            visible_gap = max(2.0, 0.06 * float(self.grid_size))
            spacing = 2.0 * radius + visible_gap
            line_half_span = min(half_span, spacing * (numerosity - 1) / 2.0)
            steps = np.linspace(-line_half_span, line_half_span, numerosity)
        return (
            np.array([self.center, self.center], dtype=float)
            + offset * normal
            + steps[:, None] * direction
        )

    def _sample_density_ranked_positions(self, numerosity, choose_high):
        best_indices = None
        best_score = None
        for _ in range(self.density_candidate_samples):
            indices = np.random.choice(
                len(self.lattice_positions),
                size=numerosity,
                replace=False,
            )
            positions = self.lattice_positions[indices]
            score = self._average_pair_distance(positions)
            if best_score is None:
                best_score = score
                best_indices = indices
                continue
            if choose_high and score < best_score:
                best_score = score
                best_indices = indices
            elif not choose_high and score > best_score:
                best_score = score
                best_indices = indices
        return self.lattice_positions[best_indices]

    def _average_pair_distance(self, positions):
        if len(positions) <= 1:
            return 0.0
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
        return float(np.mean(distances))

    def _sample_nieder_radii(self, numerosity, mode):
        if mode == "equal_area":
            total_area = self._equal_area_total()
            radius = np.sqrt(float(total_area) / (np.pi * numerosity))
            return np.full(numerosity, radius, dtype=float)

        if mode == "equal_circumference":
            radius = self._equal_circumference_total() / (2.0 * np.pi * numerosity)
            return np.full(numerosity, radius, dtype=float)

        if mode == "linear":
            radius = self._linear_dot_radius()
            return np.full(numerosity, radius, dtype=float)

        if mode in {"high_density", "low_density", "shape"}:
            radius = self._reference_dot_radius()
            return np.full(numerosity, radius, dtype=float)

        if mode == "variable_features":
            base = np.random.choice(self.dot_radius_values, size=numerosity, replace=True)
            jitter = np.random.uniform(0.55, 1.05, size=numerosity)
            return base * jitter

        return np.random.choice(self.dot_radius_values, size=numerosity, replace=True)

    def _sample_nieder_shapes(self, numerosity, mode):
        if mode == "shape":
            choices = ["ellipse", "square", "diamond", "triangle"]
            return np.random.choice(choices, size=numerosity, replace=True)
        if mode == "variable_features":
            choices = ["circle", "ellipse", "square", "rectangle", "diamond", "triangle"]
            return np.random.choice(choices, size=numerosity, replace=True)
        return np.full(numerosity, "circle", dtype=object)

    def _nieder_item_mask(self, shape, cx, cy, radius, angle):
        dx = self.xx - float(cx)
        dy = self.yy - float(cy)
        cos_a = np.cos(float(angle))
        sin_a = np.sin(float(angle))
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy

        if shape == "ellipse":
            return (xr / (1.45 * radius)) ** 2 + (yr / (0.68 * radius)) ** 2 <= 1.0
        if shape == "square":
            half_side = radius * np.sqrt(np.pi) / 2.0
            return (np.abs(xr) <= half_side) & (np.abs(yr) <= half_side)
        if shape == "rectangle":
            return (np.abs(xr) <= 1.40 * radius) & (np.abs(yr) <= 0.55 * radius)
        if shape == "diamond":
            half_diagonal = radius * np.sqrt(np.pi / 2.0)
            return np.abs(xr) + np.abs(yr) <= half_diagonal
        if shape == "triangle":
            return self._regular_polygon_mask(cx, cy, radius * 1.35, 3, angle)
        return dx**2 + dy**2 <= radius**2

    def _regular_polygon_mask(self, cx, cy, radius, n_sides, angle):
        vertex_angles = float(angle) + np.linspace(
            0.0,
            2.0 * np.pi,
            int(n_sides),
            endpoint=False,
        )
        vertices = np.stack(
            [
                float(cx) + float(radius) * np.cos(vertex_angles),
                float(cy) + float(radius) * np.sin(vertex_angles),
            ],
            axis=1,
        )
        signed_area = 0.0
        for idx in range(len(vertices)):
            x1, y1 = vertices[idx]
            x2, y2 = vertices[(idx + 1) % len(vertices)]
            signed_area += x1 * y2 - x2 * y1
        sign = 1.0 if signed_area >= 0.0 else -1.0
        inside = np.ones_like(self.xx, dtype=bool)
        for idx in range(len(vertices)):
            x1, y1 = vertices[idx]
            x2, y2 = vertices[(idx + 1) % len(vertices)]
            cross = (x2 - x1) * (self.yy - y1) - (y2 - y1) * (self.xx - x1)
            inside &= sign * cross >= -1e-9
        return inside
