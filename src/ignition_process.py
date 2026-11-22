from abc import ABC, abstractmethod
import numpy as np
from src.config import BurstPhysicsConfig, NeutronStarGridConfig


# --- DEFENSIVE ARRAY VALIDATION HELPER ---

def _validate_1d_state_array(arr: np.ndarray, name: str, expected_len: int | None = None) -> None:
    """Ensures a 1D simulation state array is non-null, finite, and matches grid dimensions."""
    if arr is None or not isinstance(arr, np.ndarray):
        raise TypeError(f"State array '{name}' must be a valid numpy.ndarray, got {type(arr).__name__}.")
    if arr.ndim != 1:
        raise ValueError(f"State array '{name}' must be 1D, got shape {arr.shape}.")
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"State array '{name}' length ({arr.size}) does not match grid nx ({expected_len}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"State array '{name}' contains NaN or Inf values.")


# --- 1. THERMONUCLEAR REACTION KINETICS STRATEGIES ---

# Defines the abstract contract for swappable nuclear burning rate models
class ReactionStrategy(ABC):
    """Defines the contract for thermonuclear reaction rate models R(T, Y)."""

    @abstractmethod
    def compute_rate(self, temperature: np.ndarray, fuel_fraction: np.ndarray) -> np.ndarray:
        """Computes local nuclear fuel burning rate across all spatial grid points."""
        pass


# Models one-step exothermic Arrhenius nuclear fusion above a critical ignition temperature
class ArrheniusReaction(ReactionStrategy):
    """Computes one-step Arrhenius fusion kinetics R(T, Y) = A * Y * exp(-T_act / T) for T > T_ign."""

    def __init__(self, physics: BurstPhysicsConfig):
        if physics is None or not isinstance(physics, BurstPhysicsConfig):
            raise TypeError("Parameter 'physics' must be a valid BurstPhysicsConfig instance.")
        self.prefactor = physics.reaction_prefactor
        self.activation_temp = physics.activation_temp
        self.ignition_threshold = physics.ignition_threshold

    def compute_rate(self, temperature: np.ndarray, fuel_fraction: np.ndarray) -> np.ndarray:
        """Evaluates local nuclear burning rates where surface temperature exceeds ignition_threshold."""
        _validate_1d_state_array(temperature, "temperature")
        _validate_1d_state_array(fuel_fraction, "fuel_fraction", expected_len=temperature.size)

        rate = np.zeros_like(temperature, dtype=np.float64)
        ignited = temperature > self.ignition_threshold
        rate[ignited] = (
            self.prefactor
            * fuel_fraction[ignited]
            * np.exp(-self.activation_temp / temperature[ignited])
        )
        return rate


# --- 2. LATERAL THERMAL TRANSPORT & CORIOLIS CONFINEMENT STRATEGIES ---

# Defines the abstract contract for lateral heat diffusivity profiles across stellar latitudes
class DiffusivityStrategy(ABC):
    """Defines the contract for effective lateral thermal diffusivity D_eff(x) across latitudes."""

    @abstractmethod
    def compute_diffusivity_profile(
        self,
        grid: NeutronStarGridConfig,
        physics: BurstPhysicsConfig,
    ) -> np.ndarray:
        """Computes a 1D array of effective thermal diffusivity D_eff (km^2/s) at every grid node."""
        pass


# Models a non-rotating neutron star with uniform lateral heat conduction at all latitudes
class UniformDiffusivity(DiffusivityStrategy):
    """Provides constant baseline diffusivity D_0 across all latitudes for non-rotating control runs."""

    def compute_diffusivity_profile(
        self,
        grid: NeutronStarGridConfig,
        physics: BurstPhysicsConfig,
    ) -> np.ndarray:
        """Generates a uniform diffusivity profile equal to base_diffusivity across the grid."""
        if grid is None or not isinstance(grid, NeutronStarGridConfig):
            raise TypeError("Parameter 'grid' must be a valid NeutronStarGridConfig instance.")
        if physics is None or not isinstance(physics, BurstPhysicsConfig):
            raise TypeError("Parameter 'physics' must be a valid BurstPhysicsConfig instance.")

        return np.full(grid.nx, physics.base_diffusivity, dtype=np.float64)


# Models Spitkovsky et al. (2002) rotational confinement where Coriolis drag slows high-latitude flames
class LatitudeCoriolisDrag(DiffusivityStrategy):
    """Computes latitude-dependent diffusivity D_eff(theta) = D_0 / (1 + alpha * sin(theta))."""

    def compute_diffusivity_profile(
        self,
        grid: NeutronStarGridConfig,
        physics: BurstPhysicsConfig,
    ) -> np.ndarray:
        """Evaluates Coriolis-impeded diffusivity from equator (theta = 0) to pole (theta = pi/2)."""
        if grid is None or not isinstance(grid, NeutronStarGridConfig):
            raise TypeError("Parameter 'grid' must be a valid NeutronStarGridConfig instance.")
        if physics is None or not isinstance(physics, BurstPhysicsConfig):
            raise TypeError("Parameter 'physics' must be a valid BurstPhysicsConfig instance.")

        latitude_rad = grid.create_latitude_grid_rad()
        return physics.base_diffusivity / (1.0 + physics.polar_coriolis_drag * np.sin(latitude_rad))


# --- 3. SPATIAL BOUNDARY CONDITION STRATEGIES ---

# Defines the abstract contract for enforcing physical boundary conditions at grid endpoints
class BoundaryConditionStrategy(ABC):
    """Defines the contract for enforcing boundary conditions at equator (x=0) and pole (x=L)."""

    @abstractmethod
    def apply(self, field: np.ndarray) -> None:
        """Applies boundary conditions in-place to a 1D state field."""
        pass


# Enforces insulated zero-flux boundaries representing equatorial symmetry and polar convergence
class NeumannZeroFluxBoundary(BoundaryConditionStrategy):
    """Enforces zero spatial gradient (dT/dx = 0) at equator (x=0) and pole (x=L) to prevent heat loss."""

    def apply(self, field: np.ndarray) -> None:
        """Copies interior adjacent node values onto both boundary endpoints in-place."""
        _validate_1d_state_array(field, "field")
        if field.size < 3:
            raise ValueError("Field must contain at least 3 grid nodes to apply boundary conditions.")
        field = field
        field[-1] = field[-2]
