import numpy as np
import matplotlib.pyplot as plt


def run_minimal_flame_prototype():
    # --- 1. Spatial & Time Grid Setup ---
    L = 15.7              # Domain length in km (~1/4 circumference of a 10 km neutron star)
    nx = 200              # Number of spatial grid points
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]      # Grid spacing between adjacent points

    dt = 0.0005           # Time step in seconds (must satisfy D * dt / dx^2 <= 0.5)
    t_max = 6.0           # Total simulation duration in seconds
    n_steps = int(t_max / dt)

    # --- 2. Physical Parameters ---
    D = 0.8               # Thermal diffusivity (how fast heat conducts sideways)
    Q = 3.2               # Nuclear energy released per unit fuel burned
    A = 18.0              # Reaction rate prefactor
    T_act = 1.4           # Activation temperature barrier for nuclear fusion
    T_ign = 0.22          # Minimum temperature threshold to ignite fuel
    cooling = 0.45        # Radiative cooling rate into space
    T0 = 0.10             # Quiescent background surface temperature

    # Check numerical stability (CFL condition)
    cfl = D * dt / (dx ** 2)
    assert cfl <= 0.5, f"Unstable time step! CFL = {cfl:.3f} > 0.5"

    # --- 3. Initial Conditions (Cold Star + Localized Ignition Spark) ---
    T = np.full(nx, T0)
    Y = np.ones(nx)       # 100% fresh unburned fuel everywhere

    # Ignite a small hotspot near the equator (x = 0.5 km)
    T[np.abs(x - 0.5) < 0.3] = 1.15

    # Arrays to store outputs
    times = np.zeros(n_steps)
    light_curve = np.zeros(n_steps)
    snapshots = []
    snapshot_times = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]

    # --- 4. Time-Stepping Loop ---
    for step in range(n_steps):
        t = step * dt
        times[step] = t

        # Integrate total X-ray flux across the star (Stefan-Boltzmann F ~ T^4)
        light_curve[step] = np.sum(T**4 - T0**4) * dx

        # Save temperature profile at specific times for plotting
        if any(abs(t - st) < dt / 2 for st in snapshot_times):
            snapshots.append((t, T.copy()))

        # Spatial second derivative (heat conduction between neighboring cells)
        d2T_dx2 = (T[:-2] - 2.0 * T[1:-1] + T[2:]) / (dx ** 2)

        # Nuclear reaction rate (only active where T > T_ign)
        R = np.zeros(nx)
        burning = T > T_ign
        R[burning] = A * Y[burning] * np.exp(-T_act / T[burning])

        # Update interior temperature and deplete fuel
        T[1:-1] += dt * (D * d2T_dx2 + Q * R[1:-1] - cooling * (T[1:-1] - T0))
        Y[1:-1] = np.maximum(0.0, Y[1:-1] - dt * R[1:-1])

        # Insulated (zero-flux) boundaries at the equator (x=0) and pole (x=L)
        T[0] = T[1]
        T[-1] = T[-2]

    # Normalize light curve peak to 1.0
    light_curve /= np.max(light_curve)

    # --- 5. Plot Flame Wave & Synthetic X-Ray Light Curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for t_snap, T_snap in snapshots:
        ax1.plot(x, T_snap, label=f"t = {t_snap:.1f} s")
    ax1.set_title("1D Thermonuclear Flame Propagation")
    ax1.set_xlabel("Distance from Equator x (km)")
    ax1.set_ylabel("Surface Temperature T (dimensionless)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(times, light_curve, color="crimson", linewidth=2)
    ax2.set_title("Synthetic Type-I X-Ray Burst Light Curve")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized X-Ray Flux (F / F_peak)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/minimal_prototype.png", dpi=200)
    plt.show()
    print("Simulation complete! Plot saved to outputs/minimal_prototype.png")


if __name__ == "__main__":
    run_minimal_flame_prototype()
