from pathlib import Path
import matplotlib.pyplot as plt
from src.ignition_calculations import SimulationResult


# --- PUBLICATION-GRADE DIAGNOSTIC VISUALIZATION MODULE ---

# Generates space-time flame heatmaps and comparative X-ray light curves for Altamirano et al. (2022)
class BurstVisualizer:
    """Generates multi-panel diagnostic plots comparing equatorial and high-latitude ignition runs."""

    @staticmethod
    def plot_equator_vs_high_latitude(
        equator_res: SimulationResult,
        high_lat_res: SimulationResult,
        output_path: str = "outputs/altamirano_2022_comparison.png",
    ) -> None:
        """
        Produces a 3-panel figure:
          (1) Space-time temperature heatmap for fast equatorial ignition (5 deg),
          (2) Space-time temperature heatmap for slow high-latitude ignition (65 deg),
          (3) Normalized synthetic X-ray telescope light curves comparing rise times (t_10->90).
        """
        if equator_res is None or high_lat_res is None:
            raise ValueError("Both equator_res and high_lat_res SimulationResult objects must be non-null.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # --- Panel 1: Space-Time Flame Propagation (Equatorial Ignition) ---
        ax0 = axes[0]
        im0 = ax0.pcolormesh(
            equator_res.x_km,
            equator_res.snapshot_times_s,
            equator_res.T_history,
            shading="auto",
            cmap="inferno",
            vmin=0.1,
            vmax=1.35,
        )
        ax0.set_title("Equatorial Ignition ($\\theta_{\\mathrm{ign}} = 5^\\circ$)\nWeak Coriolis Confinement")
        ax0.set_xlabel("Meridional Distance from Equator $x$ (km)")
        ax0.set_ylabel("Time $t$ (s)")
        fig.colorbar(im0, ax=ax0, label="Surface Temp $T$ (dim.)")

        # --- Panel 2: Space-Time Flame Propagation (High-Latitude Ignition) ---
        ax1 = axes[1]
        im1 = ax1.pcolormesh(
            high_lat_res.x_km,
            high_lat_res.snapshot_times_s,
            high_lat_res.T_history,
            shading="auto",
            cmap="inferno",
            vmin=0.1,
            vmax=1.35,
        )
        ax1.set_title("High-Latitude Ignition ($\\theta_{\\mathrm{ign}} = 65^\\circ$)\nStrong Coriolis Confinement")
        ax1.set_xlabel("Meridional Distance from Equator $x$ (km)")
        ax1.set_ylabel("Time $t$ (s)")
        fig.colorbar(im1, ax=ax1, label="Surface Temp $T$ (dim.)")

        # --- Panel 3: Synthetic X-Ray Telescope Light Curve Comparison ---
        ax2 = axes[2]
        eq_rise = equator_res.rise_time_10_90_s
        hl_rise = high_lat_res.rise_time_10_90_s

        ax2.plot(
            equator_res.times_s,
            equator_res.light_curve_norm,
            color="#d62728",
            linewidth=2.2,
            label=f"Equator ($5^\\circ$): $t_{{10\\to 90}} = {eq_rise:.2f}\\,\\mathrm{{s}}$",
        )
        ax2.plot(
            high_lat_res.times_s,
            high_lat_res.light_curve_norm,
            color="#1f77b4",
            linewidth=2.2,
            linestyle="--",
            label=f"High-Lat ($65^\\circ$): $t_{{10\\to 90}} = {hl_rise:.2f}\\,\\mathrm{{s}}$",
        )

        ax2.axhline(0.10, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
        ax2.axhline(0.90, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
        ax2.set_title("Synthetic NICER X-Ray Light Curves\n(cf. Bult, Altamirano et al. 2022)")
        ax2.set_xlabel("Time $t$ (s)")
        ax2.set_ylabel("Normalized X-Ray Flux ($F / F_{\\mathrm{peak}}$)")
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(output_path, dpi=250)
        plt.show()
