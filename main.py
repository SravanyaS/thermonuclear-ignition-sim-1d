from src.config import BurstPhysicsConfig, NeutronStarGridConfig
from src.ignition_calculations import ThermonuclearFlameSolver1D
from src.ignition_process import (
    ArrheniusReaction,
    LatitudeCoriolisDrag,
    NeumannZeroFluxBoundary,
)
from src.visualisations_builder import BurstVisualizer


# Executes the Bult, Altamirano et al. (2022) equator vs. high-latitude ignition experiment
def run_altamirano_comparison_experiment() -> None:
    """Runs equatorial (5 deg) and high-latitude (65 deg) burst simulations and compares rise times."""
    grid = NeutronStarGridConfig(star_radius_km=10.0, nx=250, total_time_s=8.0, dt=0.0005)

    # Scenario A: Low-accretion equatorial ignition (weak Coriolis drag -> fast burst rise)
    equator_physics = BurstPhysicsConfig(ignition_latitude_deg=5.0, polar_coriolis_drag=2.8)
    equator_solver = ThermonuclearFlameSolver1D(
        grid=grid,
        physics=equator_physics,
        reaction=ArrheniusReaction(equator_physics),
        diffusivity=LatitudeCoriolisDrag(),
        boundary=NeumannZeroFluxBoundary(),
    )
    equator_result = equator_solver.run(num_snapshots=100)

    # Scenario B: High-accretion high-latitude ignition (strong Coriolis drag -> slow burst rise)
    high_lat_physics = BurstPhysicsConfig(ignition_latitude_deg=65.0, polar_coriolis_drag=2.8)
    high_lat_solver = ThermonuclearFlameSolver1D(
        grid=grid,
        physics=high_lat_physics,
        reaction=ArrheniusReaction(high_lat_physics),
        diffusivity=LatitudeCoriolisDrag(),
        boundary=NeumannZeroFluxBoundary(),
    )
    high_lat_result = high_lat_solver.run(num_snapshots=100)

    print("=== 1D Thermonuclear Flame Simulation Summary (cf. Altamirano et al. 2022) ===")
    print(f"CFL Stability Number         : {equator_solver.cfl_number:.4f} (limit <= 0.5000)")
    print(f"Equatorial Burst (5 deg)     : Peak at t = {equator_result.peak_time_s:.2f} s | "
          f"Rise t_10->90 = {equator_result.rise_time_10_90_s:.2f} s")
    print(f"High-Latitude Burst (65 deg) : Peak at t = {high_lat_result.peak_time_s:.2f} s | "
          f"Rise t_10->90 = {high_lat_result.rise_time_10_90_s:.2f} s")

    BurstVisualizer.plot_equator_vs_high_latitude(
        equator_res=equator_result,
        high_lat_res=high_lat_result,
        output_path="outputs/altamirano_2022_comparison.png",
    )


if __name__ == "__main__":
    run_altamirano_comparison_experiment()
