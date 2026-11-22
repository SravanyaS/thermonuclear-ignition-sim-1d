# This file contains all the base parameters, boundary conditions etc required for the simulation model

# Importing necessary libraries
from dataclasses import dataclass
import math
import numpy as np


# Defining input validation helpers
def _validate_finite_positive(value: float, name: str) -> None:
    """Ensures a physical parameter is non-null, a real number, finite, and strictly > 0."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{name}' must be a valid number, got {type(value).__name__}.")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Parameter '{name}' must be finite and strictly positive, got {value}.")


def _validate_finite_non_negative(value: float, name: str) -> None:
    """Ensures a parameter is non-null, a real number, finite, and >= 0."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{name}' must be a valid number, got {type(value).__name__}.")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Parameter '{name}' must be finite and non-negative, got {value}.")


# Setting up the 1D meridional spatial and temporal grid layout for finite difference computation
@dataclass(frozen=True)
class NeutronStarGridConfig:
    """1D meridional spatial and temporal grid from equator (0) to pole (pi/2)."""
    star_radius_km: float = 10.0  # Canonical neutron star radius (~10 km)
    nx: int = 250  # Number of spatial grid points
    total_time_s: float = 8.0  # Total burst duration to simulate (seconds)
    dt: float = 0.0005  # Incremental time step size (seconds)

    def __post_init__(self) -> None:
        """Validates spatial geometry and time-stepping invariants upon initialization."""
        _validate_finite_positive(self.star_radius_km, "star_radius_km")
        _validate_finite_positive(self.total_time_s, "total_time_s")
        _validate_finite_positive(self.dt, "dt")

        if self.nx is None or isinstance(self.nx, bool) or not isinstance(self.nx, int) or self.nx < 20:
            raise ValueError(f"Grid resolution 'nx' must be an integer >= 20, got {self.nx}.")
        if self.dt >= self.total_time_s:
            raise ValueError(f"Time step dt ({self.dt}) must be smaller than total_time_s ({self.total_time_s}).")

    @property
    def domain_length_km(self) -> float:
        """Quarter-circumference arc length L = radius of neutron star * (pi / 2) in km."""
        return self.star_radius_km * (math.pi / 2.0)

    @property
    def dx(self) -> float:
        """Uniform spatial spacing dx = L / (nx - 1) in km."""
        return self.domain_length_km / (self.nx - 1)

    def create_spatial_grid(self) -> np.ndarray:
        """Generates 1D array of positions x in [0, L] km from the equator."""
        return np.linspace(0.0, self.domain_length_km, self.nx, dtype=np.float64)

    def create_latitude_grid_rad(self) -> np.ndarray:
        """Maps positions x in [0, L] to latitude angles theta in [0, pi/2] radians."""
        return (math.pi / 2.0) * (self.create_spatial_grid() / self.domain_length_km)


# Defining the thermal, nuclear burning, and rotational transport parameters on the grid
@dataclass(frozen=True)
class BurstPhysicsConfig:
    """Dimensionless thermal, nuclear burning, and Coriolis transport parameters."""
    base_diffusivity: float = 0.85  # Lateral thermal diffusivity at equator (km^2/s)
    polar_coriolis_drag: float = 2.8  # Rotational confinement factor near poles
    ignition_latitude_deg: float = 5.0  # Initial spark latitude (0 = equator, 90 = pole)
    ignition_width_km: float = 0.6  # Spatial width of initial hotspot (km)
    ignition_temp: float = 1.15  # Peak temperature of initial spark
    quiescent_temp: float = 0.10  # Pre-burst background surface temperature T_0
    initial_fuel_fraction: float = 1.0  # Unburned fuel mass fraction Y_0 in (0, 1]
    fuel_energy_release: float = 3.2  # Nuclear energy released per unit fuel burned (Q)
    reaction_prefactor: float = 18.0  # Arrhenius reaction rate prefactor (A)
    activation_temp: float = 1.4  # Arrhenius activation barrier (T_act)
    ignition_threshold: float = 0.22  # Critical temperature threshold to sustain fusion (T_ign)
    cooling_rate: float = 0.45  # Radiative cooling rate from photosphere (lambda)

    def __post_init__(self) -> None:
        """Validates physical bounds and ignition conditions upon initialization."""
        _validate_finite_positive(self.base_diffusivity, "base_diffusivity")
        _validate_finite_non_negative(self.polar_coriolis_drag, "polar_coriolis_drag")
        _validate_finite_positive(self.ignition_width_km, "ignition_width_km")
        _validate_finite_positive(self.ignition_temp, "ignition_temp")
        _validate_finite_positive(self.quiescent_temp, "quiescent_temp")
        _validate_finite_positive(self.fuel_energy_release, "fuel_energy_release")
        _validate_finite_positive(self.reaction_prefactor, "reaction_prefactor")
        _validate_finite_positive(self.activation_temp, "activation_temp")
        _validate_finite_positive(self.ignition_threshold, "ignition_threshold")
        _validate_finite_non_negative(self.cooling_rate, "cooling_rate")

        if (
                self.ignition_latitude_deg is None
                or isinstance(self.ignition_latitude_deg, bool)
                or not isinstance(self.ignition_latitude_deg, (int, float))
                or not math.isfinite(self.ignition_latitude_deg)
                or not (0.0 <= self.ignition_latitude_deg <= 90.0)
        ):
            raise ValueError(
                f"ignition_latitude_deg must lie between 0 (equator) and 90 (pole), got {self.ignition_latitude_deg}."
            )

        if (
                self.initial_fuel_fraction is None
                or isinstance(self.initial_fuel_fraction, bool)
                or not isinstance(self.initial_fuel_fraction, (int, float))
                or not math.isfinite(self.initial_fuel_fraction)
                or not (0.0 < self.initial_fuel_fraction <= 1.0)
        ):
            raise ValueError(
                f"initial_fuel_fraction must lie in (0, 1], got {self.initial_fuel_fraction}."
            )

        if self.ignition_temp <= self.ignition_threshold:
            raise ValueError(
                f"ignition_temp ({self.ignition_temp}) must exceed ignition_threshold ({self.ignition_threshold}) to ignite."
            )

    def ignition_center_km(self, grid: NeutronStarGridConfig) -> float:
        """Converts ignition latitude in degrees (0-90) to meridional distance x (km)."""
        if grid is None or not isinstance(grid, NeutronStarGridConfig):  # <-- ADDED THIS CHECK
            raise TypeError("Parameter 'grid' must be a valid NeutronStarGridConfig instance.")  # <-- ADDED THIS CHECK
        return grid.domain_length_km * (self.ignition_latitude_deg / 90.0)

# Defining checks for CFL numerical stability condition
def validate_cfl_stability(
        grid: NeutronStarGridConfig,
        physics: BurstPhysicsConfig,
        max_cfl: float = 0.5,
) -> float:
    """
    Enforces the Courant-Friedrichs-Lewy (CFL) numerical stability condition:
    CFL = D_max * dt / dx^2 <= 0.5 for 1D explicit Forward Euler diffusion.
    """
    if grid is None or not isinstance(grid, NeutronStarGridConfig):
        raise TypeError("Parameter 'grid' must be a valid NeutronStarGridConfig instance.")
    if physics is None or not isinstance(physics, BurstPhysicsConfig):
        raise TypeError("Parameter 'physics' must be a valid BurstPhysicsConfig instance.")

    cfl = physics.base_diffusivity * grid.dt / (grid.dx ** 2)
    if cfl > max_cfl:
        safe_dt = max_cfl * (grid.dx ** 2) / physics.base_diffusivity
        raise ValueError(
            f"CFL stability violation: CFL = {cfl:.4f} exceeds limit {max_cfl}. "
            f"Reduce dt below {safe_dt:.6f} s."
        )
    return cfl
