from dataclasses import dataclass
import numpy as np
from src.config import (
    BurstPhysicsConfig,
    NeutronStarGridConfig,
    validate_cfl_stability,
)
from src.ignition_process import (
    ArrheniusReaction,
    BoundaryConditionStrategy,
    DiffusivityStrategy,
    LatitudeCoriolisDrag,
    NeumannZeroFluxBoundary,
    ReactionStrategy,
)


# --- 1. SIMULATION TELEMETRY & OBSERVATIONAL RESULT CONTAINER ---

# Stores time-series X-ray light curves, spatial snapshots, and rise-time diagnostics
@dataclass
class SimulationResult:
    """Stores synthetic X-ray telescope telemetry and 1D space-time flame evolution."""
    x_km: np.ndarray                 # Meridional grid coordinates from equator to pole (km)
    times_s: np.ndarray              # Time stamps for every integration step (s)
    light_curve_raw: np.ndarray      # Unnormalized bolometric X-ray flux F(t) ~ sum(T^4 - T0^4) dx
    light_curve_norm: np.ndarray     # Normalized X-ray flux F(t) / F_peak in [0, 1]
    burned_fraction: np.ndarray      # Fraction of total surface fuel consumed over time
    snapshot_times_s: np.ndarray     # Time stamps of saved spatial profiles (s)
    T_history: np.ndarray            # 2D array (n_snapshots, nx) of temperature evolution
    Y_history: np.ndarray            # 2D array (n_snapshots, nx) of remaining fuel evolution

    @property
    def peak_time_s(self) -> float:
        """Returns the time (s) at which the synthetic X-ray light curve reaches maximum flux."""
        peak_idx = int(np.argmax(self.light_curve_norm))
        return float(self.times_s[peak_idx])

    @property
    def rise_time_10_90_s(self) -> float:
        """
        Computes standard observational X-ray burst rise time (t_90 - t_10 in seconds):
        time elapsed for normalized flux to climb from 10% to 90% of peak brightness.
        """
        peak_idx = int(np.argmax(self.light_curve_norm))
        rising_flux = self.light_curve_norm[: peak_idx + 1]
        rising_times = self.times_s[: peak_idx + 1]

        t_10_indices = np.where(rising_flux >= 0.10)[0]
        t_90_indices = np.where(rising_flux >= 0.90)[0]

        if len(t_10_indices) == 0 or len(t_90_indices) == 0:
            return 0.0
        return float(rising_times[t_90_indices][0] - rising_times[t_10_indices][0])


# --- 2. COUPLED 1D REACTION-DIFFUSION PDE SOLVER ---

# Evolves surface temperature T(x, t) and fuel mass fraction Y(x, t) via explicit finite differences
class ThermonuclearFlameSolver1D:
    """
    Solves the 1D coupled thermonuclear reaction-diffusion equations on a neutron star surface
    using second-order central spatial differences and CFL-validated Forward Euler stepping.
    """

    def __init__(
        self,
        grid: NeutronStarGridConfig,
        physics: BurstPhysicsConfig,
        reaction: ReactionStrategy | None = None,
        diffusivity: DiffusivityStrategy | None = None,
        boundary: BoundaryConditionStrategy | None = None,
    ):
        if grid is None or not isinstance(grid, NeutronStarGridConfig):
            raise TypeError("Parameter 'grid' must be a valid NeutronStarGridConfig instance.")
        if physics is None or not isinstance(physics, BurstPhysicsConfig):
            raise TypeError("Parameter 'physics' must be a valid BurstPhysicsConfig instance.")

        # Enforce Courant-Friedrichs-Lewy stability condition before initializing fields
        self.cfl_number = validate_cfl_stability(grid, physics)

        self.grid = grid
        self.physics = physics

        # Dependency injection with physically realistic defaults
        self.reaction = reaction if reaction is not None else ArrheniusReaction(physics)
        self.diffusivity = diffusivity if diffusivity is not None else LatitudeCoriolisDrag()
        self.boundary = boundary if boundary is not None else NeumannZeroFluxBoundary()

        # Initialize spatial grid and latitude-dependent effective diffusivity profile
        self.x = self.grid.create_spatial_grid()
        self.dx = self.grid.dx
        self.D_profile = self.diffusivity.compute_diffusivity_profile(self.grid, self.physics)

        # Initialize cold pre-burst surface temperature and unburned fuel fraction
        self.T = np.full(self.grid.nx, self.physics.quiescent_temp, dtype=np.float64)
        self.Y = np.full(self.grid.nx, self.physics.initial_fuel_fraction, dtype=np.float64)

        # Trigger localized thermonuclear spark at the configured ignition latitude
        ignition_x = self.physics.ignition_center_km(self.grid)
        half_width = self.physics.ignition_width_km / 2.0
        hotspot_mask = np.abs(self.x - ignition_x) <= half_width

        # Ensure at least the closest grid node ignites even if ignition_width_km < dx
        if not np.any(hotspot_mask):
            closest_idx = int(np.argmin(np.abs(self.x - ignition_x)))
            hotspot_mask[closest_idx] = True

        self.T[hotspot_mask] = self.physics.ignition_temp
        self.boundary.apply(self.T)
        self.boundary.apply(self.Y)

    def run(self, num_snapshots: int = 80) -> SimulationResult:
        """Integrates the PDE system forward in time and synthesizes the X-ray burst light curve."""
        if num_snapshots is None or not isinstance(num_snapshots, int) or num_snapshots < 2:
            raise ValueError(f"'num_snapshots' must be an integer >= 2, got {num_snapshots}.")

        n_steps = int(self.grid.total_time_s / self.grid.dt)
        snapshot_interval = max(1, n_steps // num_snapshots)

        times = np.zeros(n_steps, dtype=np.float64)
        light_curve_raw = np.zeros(n_steps, dtype=np.float64)
        burned_fraction = np.zeros(n_steps, dtype=np.float64)

        T_snapshots = []
        Y_snapshots = []
        snapshot_times = []

        T0_fourth = self.physics.quiescent_temp ** 4
        initial_total_fuel = float(np.sum(self.Y))

        for step in range(n_steps):
            t = step * self.grid.dt
            times[step] = t

            # Integrate excess blackbody X-ray flux across the surface (Stefan-Boltzmann F ~ T^4)
            excess_flux = np.sum(np.maximum(0.0, self.T ** 4 - T0_fourth)) * self.dx
            light_curve_raw[step] = excess_flux
            burned_fraction[step] = 1.0 - (float(np.sum(self.Y)) / initial_total_fuel)

            if step % snapshot_interval == 0:
                T_snapshots.append(self.T.copy())
                Y_snapshots.append(self.Y.copy())
                snapshot_times.append(t)

            # Second-order central difference spatial Laplacian d^2T / dx^2 on interior nodes
            d2T_dx2 = (self.T[:-2] - 2.0 * self.T[1:-1] + self.T[2:]) / (self.dx ** 2)

            # Local thermonuclear reaction rate R(T, Y)
            reaction_rate = self.reaction.compute_rate(self.T, self.Y)

            # Explicit Forward-Euler update for interior temperature and fuel depletion
            dT = (
                self.D_profile[1:-1] * d2T_dx2
                + self.physics.fuel_energy_release * reaction_rate[1:-1]
                - self.physics.cooling_rate * (self.T[1:-1] - self.physics.quiescent_temp)
            )
            self.T[1:-1] = np.maximum(self.physics.quiescent_temp, self.T[1:-1] + self.grid.dt * dT)
            self.Y[1:-1] = np.maximum(0.0, self.Y[1:-1] - self.grid.dt * reaction_rate[1:-1])

            # Enforce physical boundary conditions at equator (x=0) and pole (x=L)
            self.boundary.apply(self.T)
            self.boundary.apply(self.Y)

        # Normalize light curve to peak = 1.0 for direct observational rise-time comparison
        peak_flux = float(np.max(light_curve_raw))
        light_curve_norm = light_curve_raw / peak_flux if peak_flux > 0.0 else light_curve_raw.copy()

        return SimulationResult(
            x_km=self.x,
            times_s=times,
            light_curve_raw=light_curve_raw,
            light_curve_norm=light_curve_norm,
            burned_fraction=burned_fraction,
            snapshot_times_s=np.array(snapshot_times, dtype=np.float64),
            T_history=np.array(T_snapshots, dtype=np.float64),
            Y_history=np.array(Y_snapshots, dtype=np.float64),
        )
