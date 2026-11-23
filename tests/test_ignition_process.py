import numpy as np
from src.config import BurstPhysicsConfig, NeutronStarGridConfig
from src.ignition_process import (
    ArrheniusReaction,
    LatitudeCoriolisDrag,
    NeumannZeroFluxBoundary,
    UniformDiffusivity,
)


# --- TESTS FOR REACTION, DIFFUSIVITY & BOUNDARY MODELS (src/ignition_process.py) ---

def test_arrhenius_reaction_respects_ignition_threshold() -> None:
    """Verifies nuclear reaction rate is strictly zero below ignition_threshold and positive above."""
    physics = BurstPhysicsConfig(ignition_threshold=0.22)
    reaction = ArrheniusReaction(physics)

    temps = np.array([0.10, 0.22, 0.50, 1.15], dtype=np.float64)
    fuel = np.ones_like(temps)
    rates = reaction.compute_rate(temps, fuel)

    assert rates[0] == 0.0
    assert rates[1] == 0.0
    assert rates[2] > 0.0
    assert rates[3] > rates[2]  # Higher temperature yields faster burning

def test_coriolis_diffusivity_decreases_from_equator_to_pole() -> None:
    """Verifies Spitkovsky et al. (2002) Coriolis drag reduces diffusivity toward the pole."""
    grid = NeutronStarGridConfig(nx=100)
    physics = BurstPhysicsConfig(base_diffusivity=0.85, polar_coriolis_drag=2.8)

    coriolis_profile = LatitudeCoriolisDrag().compute_diffusivity_profile(grid, physics)
    uniform_profile = UniformDiffusivity().compute_diffusivity_profile(grid, physics)

    assert np.isclose(coriolis_profile[0], physics.base_diffusivity)
    assert coriolis_profile[-1] < coriolis_profile[0]
    assert np.all(coriolis_profile <= uniform_profile)


def test_neumann_zero_flux_boundary_symmetry() -> None:
    """Verifies NeumannZeroFluxBoundary mirrors adjacent interior nodes at both ends."""
    field = np.array([0.1, 0.45, 0.6, 0.85, 0.1], dtype=np.float64)
    NeumannZeroFluxBoundary().apply(field)

    assert np.isclose(field.item(0), field.item(1))
    assert np.isclose(field.item(-1), field.item(-2))

