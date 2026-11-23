import pytest
from src.config import (
    BurstPhysicsConfig,
    NeutronStarGridConfig,
    validate_cfl_stability,
)


# --- TESTS FOR CONFIGURATION & CFL STABILITY GUARDS (src/config.py) ---

def test_grid_config_rejects_invalid_inputs() -> None:
    """Verifies NeutronStarGridConfig blocks null, negative, or under-resolved inputs."""
    with pytest.raises(TypeError):
        NeutronStarGridConfig(star_radius_km=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NeutronStarGridConfig(star_radius_km=-10.0)
    with pytest.raises(ValueError):
        NeutronStarGridConfig(nx=10)  # Below minimum threshold of 20 nodes
    with pytest.raises(ValueError):
        NeutronStarGridConfig(total_time_s=1.0, dt=1.5)  # dt >= total_time_s


def test_physics_config_rejects_out_of_bounds_parameters() -> None:
    """Verifies BurstPhysicsConfig blocks out-of-range latitudes, fuel fractions, and cold sparks."""
    with pytest.raises(ValueError):
        BurstPhysicsConfig(ignition_latitude_deg=95.0)  # Latitude cannot exceed 90 deg
    with pytest.raises(ValueError):
        BurstPhysicsConfig(initial_fuel_fraction=1.2)   # Fuel mass fraction must lie in (0, 1]
    with pytest.raises(ValueError):
        BurstPhysicsConfig(ignition_temp=0.15, ignition_threshold=0.22)  # Spark too cold to ignite


def test_cfl_stability_guard_blocks_unstable_timestep() -> None:
    """Verifies validate_cfl_stability raises ValueError when D * dt / dx^2 > 0.5."""
    grid = NeutronStarGridConfig(star_radius_km=10.0, nx=250, total_time_s=2.0, dt=0.05)
    physics = BurstPhysicsConfig(base_diffusivity=0.85)
    with pytest.raises(ValueError, match="CFL stability violation"):
        validate_cfl_stability(grid, physics)
