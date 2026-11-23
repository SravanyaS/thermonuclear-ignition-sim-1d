import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import BurstPhysicsConfig, NeutronStarGridConfig
from src.heasarc_client import HeasarcBurstClient
from src.ignition_calculations import ThermonuclearFlameSolver1D
from src.ignition_process import (
    ArrheniusReaction,
    LatitudeCoriolisDrag,
    NeumannZeroFluxBoundary,
    UniformDiffusivity,
)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1D Neutron Star Thermonuclear Flame Simulator",
    layout="wide",
)

st.title("1D Neutron Star Thermonuclear Flame & X-Ray Burst Explorer")
st.markdown(
    "Explore how **ignition latitude**, **Coriolis rotational confinement**, and **hotspot width** "
    "govern lateral thermonuclear flame spreading and disk-integrated X-ray light-curve rise times."
)

# --- SIDEBAR CONTROLS: PHYSICAL & GRID PARAMETERS ---
st.sidebar.header("Burst & Stellar Parameters")

ignition_lat = st.sidebar.slider(
    "Ignition Latitude (deg)",
    min_value=0.0,
    max_value=85.0,
    value=75.0,
    step=5.0,
    help="0 deg = Equator (weak Coriolis confinement); 85 deg = Near Pole (strong Coriolis confinement).",
)

coriolis_drag = st.sidebar.slider(
    "Polar Coriolis Drag Factor",
    min_value=0.0,
    max_value=4.5,
    value=2.5,
    step=0.1,
    help="Controls lateral diffusivity suppression D_eff(theta) = D_0 / (1 + alpha * sin(theta)).",
)

ignition_width = st.sidebar.slider(
    "Initial Hotspot Width (km)",
    min_value=0.4,
    max_value=2.5,
    value=1.2,
    step=0.1,
    help="Hotspots narrower than the critical ignition radius quench before launching a flame.",
)

use_coriolis = st.sidebar.checkbox(
    "Enable Latitude-Dependent Coriolis Confinement",
    value=True,
    help="Uncheck to compare against a non-rotating star with uniform diffusivity D_0.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Reference Baseline Comparison")
show_equator_baseline = st.sidebar.checkbox(
    "Overlay Fast Equatorial Reference (15 deg)",
    value=True,
)

# --- RUN SIMULATION USING CORE ENGINE ---
grid = NeutronStarGridConfig(star_radius_km=10.0, nx=200, total_time_s=8.0, dt=0.0006)

custom_physics = BurstPhysicsConfig(
    ignition_latitude_deg=ignition_lat,
    ignition_width_km=ignition_width,
    polar_coriolis_drag=coriolis_drag,
)
diffusivity_model = LatitudeCoriolisDrag() if use_coriolis else UniformDiffusivity()

custom_solver = ThermonuclearFlameSolver1D(
    grid=grid,
    physics=custom_physics,
    reaction=ArrheniusReaction(custom_physics),
    diffusivity=diffusivity_model,
    boundary=NeumannZeroFluxBoundary(),
)
custom_res = custom_solver.run(num_snapshots=90)

baseline_res = None
if show_equator_baseline:
    baseline_physics = BurstPhysicsConfig(
        ignition_latitude_deg=15.0,
        ignition_width_km=1.2,
        polar_coriolis_drag=coriolis_drag,
    )
    baseline_solver = ThermonuclearFlameSolver1D(
        grid=grid,
        physics=baseline_physics,
        reaction=ArrheniusReaction(baseline_physics),
        diffusivity=diffusivity_model,
        boundary=NeumannZeroFluxBoundary(),
    )
    baseline_res = baseline_solver.run(num_snapshots=90)

# --- TOP KPI METRICS ROW ---
final_burned_pct = float(custom_res.burned_fraction.item(-1)) * 100.0
flame_quenched = final_burned_pct < 15.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("10% -> 90% Rise Time", f"{custom_res.rise_time_10_90_s:.2f} s")
col2.metric("Peak Brightness Time", f"{custom_res.peak_time_s:.2f} s")
col3.metric("Total Fuel Burned", f"{final_burned_pct:.1f}%")
col4.metric("CFL Stability Number", f"{custom_solver.cfl_number:.3f} (<= 0.50)")

if flame_quenched:
    st.warning(
        "Hotspot Quenched: The initial spark was too narrow or too strongly confined by Coriolis drag "
        "to sustain a runaway flame across the star. Try increasing the hotspot width or lowering the latitude."
    )

# --- TABS: SIMULATION VIEW VS. LIVE NASA HEASARC ARCHIVE ---
tab_sim, tab_obs = st.tabs(["1D Flame Simulation", "Live NASA HEASARC Observational Archive"])

with tab_sim:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Space-Time Temperature Evolution ({ignition_lat:.0f}° Ignition)",
            "Normalized X-Ray Light Curve",
        ),
        horizontal_spacing=0.18,
    )

    fig.add_trace(
        go.Heatmap(
            x=custom_res.x_km,
            y=custom_res.snapshot_times_s,
            z=custom_res.T_history,
            colorscale="Inferno",
            zmin=0.1,
            zmax=1.35,
            colorbar=dict(title="Temp T", x=0.42, thickness=14),
            hovertemplate="Distance x: %{x:.2f} km<br>Time t: %{y:.2f} s<br>Temp T: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=custom_res.times_s,
            y=custom_res.xray_flux_norm,
            mode="lines",
            name=f"Selected ({ignition_lat:.0f}°): t_10→90 = {custom_res.rise_time_10_90_s:.2f} s",
            line=dict(color="#1f77b4", width=3),
        ),
        row=1,
        col=2,
    )

    if baseline_res is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_res.times_s,
                y=baseline_res.xray_flux_norm,
                mode="lines",
                name=f"Equatorial Ref (15°): t_10→90 = {baseline_res.rise_time_10_90_s:.2f} s",
                line=dict(color="#d62728", width=2.5, dash="dash"),
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=0.10, line_dash="dot", line_color="gray", opacity=0.6, row=1, col=2)
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray", opacity=0.6, row=1, col=2)

    fig.update_xaxes(title_text="Meridional Distance x (km)", row=1, col=1)
    fig.update_yaxes(title_text="Time t (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time t (s)", row=1, col=2)
    fig.update_yaxes(title_text="Normalized X-Ray Flux (F / F_peak)", row=1, col=2)

    fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(t=60, b=90, l=50, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.78),
    )

    st.plotly_chart(fig, use_container_width=True)

with tab_obs:
    st.subheader("Compare Simulated Rise Time Against NASA HEASARC MINBAR Bursts")
    target_source = st.selectbox(
        "Select X-Ray Burster Source (MINBAR Archive)",
        options=["4U 1730-22", "GS 1826-24", "4U 1636-536", "SAX J1808.4-3658"],
        index=0,
    )

    @st.cache_data(ttl=3600)
    def load_heasarc_data(source: str):
        return HeasarcBurstClient().fetch_minbar_bursts(source_name=source, limit=60)

    burst_df, data_status = load_heasarc_data(target_source)
    st.caption(f"Data Source: **{data_status}** | Records loaded: **{len(burst_df)}**")

    obs_fig = go.Figure()
    obs_fig.add_trace(
        go.Histogram(
            x=burst_df["trise"],
            nbinsx=15,
            name=f"Observed {target_source} Rise Times (s)",
            marker_color="#636efa",
            opacity=0.75,
        )
    )
    obs_fig.add_vline(
        x=custom_res.rise_time_10_90_s,
        line_width=3,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Simulated ({ignition_lat:.0f}°): {custom_res.rise_time_10_90_s:.2f} s",
        annotation_position="top right",
    )
    obs_fig.update_layout(
        title=f"Observed Rise-Time Distribution vs. Simulated Burst ({target_source})",
        xaxis_title="Burst Rise Time (seconds)",
        yaxis_title="Observed Burst Count",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(obs_fig, use_container_width=True)
    st.dataframe(burst_df.head(15), use_container_width=True)
