import numpy as np
from src.config import BurstPhysicsConfig, NeutronStarGridConfig
from src.ignition_calculations import ThermonuclearFlameSolver1D


# --- TESTS FOR COUPLED PDE SOLVER & ASTROPHYSICAL INVARIANTS (src/ignition_calculations.py) ---

def test_solver_preserves_physical_bounds() -> None:
    """Verifies temperature stays >= T_0, fuel remains in, and flux stays finite."""
    grid = NeutronStarGridConfig(nx=120, total_time_s=3.0, dt=0.001)
    physics = BurstPhysicsConfig(ignition_latitude_deg=10.0)
    solver = ThermonuclearFlameSolver1D(grid=grid, physics=physics)
    result = solver.run(num_snapshots=20)

    assert np.all(np.isfinite(result.T_history))
    assert np.all(np.isfinite(result.xray_flux_norm))
    assert np.all(result.T_history >= physics.quiescent_temp)
    assert np.all((result.Y_history >= 0.0) & (result.Y_history <= 1.0))
    assert np.isclose(np.max(result.xray_flux_norm), 1.0)


def test_high_latitude_ignition_produces_slower_rise_than_equator() -> None:
    """Verifies high-latitude ignition (65 deg) yields a longer rise time than equatorial ignition (5 deg)."""
    grid = NeutronStarGridConfig(nx=180, total_time_s=7.0, dt=0.0008)

    eq_physics = BurstPhysicsConfig(
        ignition_latitude_deg=15.0,
        ignition_width_km=1.2,
        polar_coriolis_drag=2.5,
    )
    hl_physics = BurstPhysicsConfig(
        ignition_latitude_deg=75.0,
        ignition_width_km=1.2,
        polar_coriolis_drag=2.5,
    )

    eq_result = ThermonuclearFlameSolver1D(grid=grid, physics=eq_physics).run(num_snapshots=30)
    hl_result = ThermonuclearFlameSolver1D(grid=grid, physics=hl_physics).run(num_snapshots=30)

    assert hl_result.rise_time_10_90_s > eq_result.rise_time_10_90_s
