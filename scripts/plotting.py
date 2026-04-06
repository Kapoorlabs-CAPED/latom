"""Matplotlib visualization functions for TDSE simulation results.

All functions take a matplotlib Axes object and data dict/arrays,
making them reusable from both scripts and the GUI.
"""

from parser import (
    collect_wf_to_hdf5,
    find_wf_snapshots,
    parse_imag_observables,
    parse_real_observables,
    parse_wavefunction,
)
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_imag_convergence(ax, ground_data, excited_data_list=None):
    """Plot imaginary time energy convergence for ground and excited states.

    Args:
        ax: Matplotlib Axes object.
        ground_data: DataFrame from parse_imag_observables for ground state.
        excited_data_list: List of (n, DataFrame) tuples for excited states.
    """
    ax.clear()
    colors = ["b", "r", "g", "m", "c", "orange", "purple", "brown"]

    if ground_data is not None:
        x = (
            ground_data["step"].values
            if "step" in ground_data.columns
            else np.arange(len(ground_data))
        )
        e = ground_data["energy_real"].values
        if len(x) > 0 and len(e) > 0:
            ax.plot(x, e, color=colors[0], linewidth=0.8, label="Ground state")

    if excited_data_list:
        for n, data in excited_data_list:
            if data is None:
                continue
            x = (
                data["step"].values
                if "step" in data.columns
                else np.arange(len(data))
            )
            e = data["energy_real"].values
            if len(x) > 0 and len(e) > 0:
                color = colors[n % len(colors)]
                ax.plot(x, e, color=color, linewidth=0.8, label=f"Excited {n}")

    ax.set_xlabel("Imaginary time step")
    ax.set_ylabel("Energy (a.u.)")
    ax.set_title("Imaginary Time Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_energy_vs_time(ax, data, phase="real"):
    """Plot energy vs time from observable data.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_imag_observables or parse_real_observables.
        phase: 'imag' for imaginary time, 'real' for real time.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    if phase == "imag":
        x = (
            data["step"].values
            if "step" in data.columns
            else np.arange(len(data))
        )
        xlabel = "Imaginary time step"
    else:
        x = data["time"].values
        xlabel = "Time (a.u.)"

    e_real = data["energy_real"].values

    if len(x) > 0 and len(e_real) > 0:
        ax.plot(x, e_real, "b-", linewidth=0.8, label="Re(E)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy (a.u.)")
    title = "Imaginary Time" if phase == "imag" else "Real Time"
    ax.set_title(f"Energy vs {title}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_wavefunction_2d(ax, wf, dx, dy, nx, ny, vmin_orders=7):
    """Plot 2D probability density |psi(x,y)|^2 as log-scale heatmap.

    Translates the IDL showwf.idl visualization: log10 scale with
    30 contour levels spanning 7 orders of magnitude.

    Args:
        ax: Matplotlib Axes object.
        wf: Complex numpy array of shape (nx, ny).
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        vmin_orders: Orders of magnitude below max to display.
    """
    # Remove existing colorbars before clearing (colorbars live on the figure,
    # not the axes, so ax.clear() alone leaves them behind)
    fig = ax.figure
    for cb_ax in [a for a in fig.axes if a is not ax]:
        try:
            cb_ax.remove()
        except Exception:
            pass
    ax.clear()
    if wf is None:
        ax.set_title("No wavefunction data")
        return

    prob_dens = np.abs(wf) ** 2
    prob_dens_max = prob_dens.max()
    if prob_dens_max == 0:
        ax.set_title("Wavefunction is zero")
        return

    prob_dens = prob_dens / prob_dens_max
    # Avoid log(0)
    prob_dens = np.clip(prob_dens, 1e-30, None)
    log_dens = np.log10(prob_dens)

    vmax = 0.0
    vmin = -vmin_orders

    # Build coordinate arrays
    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    y = (np.arange(ny) - ny / 2 + 0.5) * dy

    im = ax.pcolormesh(
        x,
        y,
        log_dens.T,
        vmin=vmin,
        vmax=vmax,
        cmap="hot_r",
        shading="auto",
    )

    ax.set_xlabel("Electron 1 position (a.u.)")
    ax.set_ylabel("Electron 2 position (a.u.)")
    ax.set_title(r"$\log_{10}|\psi(x_1,x_2)|^2$")
    ax.set_aspect("equal")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(|\psi|^2 / \max)$")


def plot_density_1d(ax, wf, dx, dy, nx, ny, axis="x"):
    """Plot 1D marginal density by integrating over one dimension.

    Args:
        ax: Matplotlib Axes object.
        wf: Complex numpy array of shape (nx, ny).
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        axis: 'x' to integrate over y (show x density), or 'y'.
    """
    ax.clear()
    if wf is None:
        ax.set_title("No wavefunction data")
        return

    prob_dens = np.abs(wf) ** 2

    if axis == "x":
        density = np.sum(prob_dens, axis=1) * dy
        coords = (np.arange(nx) - nx / 2 + 0.5) * dx
        label = "Electron 1"
        xlabel = "Position (a.u.)"
    else:
        density = np.sum(prob_dens, axis=0) * dx
        coords = (np.arange(ny) - ny / 2 + 0.5) * dy
        label = "Electron 2"
        xlabel = "Position (a.u.)"

    ax.plot(coords, density, "b-", linewidth=0.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density (a.u.)")
    ax.set_title(f"1D Marginal Density ({label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_ionization(ax, data):
    """Plot ground state population vs time.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_real_observables.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    t = data["time"].values
    pop = data["ground_pop"].values

    if len(t) > 0 and len(pop) > 0:
        ax.plot(
            t,
            pop,
            "b-",
            linewidth=0.8,
            label=r"$|\langle\psi_0|\psi(t)\rangle|^2$",
        )
        ax.plot(
            t,
            1.0 - pop,
            "r-",
            linewidth=0.8,
            alpha=0.7,
            label="Ionized fraction",
        )

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Population")
    ax.set_title("Ground State Population")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_dipole(ax, data):
    """Plot dipole expectation values vs time.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_real_observables.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    t = data["time"].values
    ex = data["expect_x"].values
    ey = data["expect_y"].values

    if len(t) > 0 and len(ex) > 0:
        ax.plot(
            t,
            ex,
            "b-",
            linewidth=0.8,
            alpha=0.8,
            label=r"$\langle x_1\rangle$",
        )
    if len(t) > 0 and len(ey) > 0:
        ax.plot(
            t,
            ey,
            "r-",
            linewidth=0.8,
            alpha=0.8,
            label=r"$\langle x_2\rangle$",
        )

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Position (a.u.)")
    ax.set_title("Dipole Expectation Values")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_vector_potential(ax, data):
    """Plot vector potential A(t) vs time.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_real_observables.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    t = data["time"].values
    vp = data["vecpot_x"].values

    if len(t) > 0 and len(vp) > 0:
        ax.plot(t, vp, "g-", linewidth=0.8)

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("A(t) (a.u.)")
    ax.set_title("Vector Potential")
    ax.grid(True, alpha=0.3)


def plot_spectrum(ax, data, dt=None):
    """Plot HHG spectrum from FFT of dipole acceleration.

    Computes the FFT of d<x>/dt (numerical derivative of expect_x)
    and plots |FFT|^2 vs harmonic order.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_real_observables.
        dt: Time step. If None, computed from data.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    t = data["time"].values
    ex = data["expect_x"].values

    if len(t) < 10 or len(ex) < 10:
        ax.set_title("Insufficient data for spectrum")
        return

    if dt is None:
        dt = t[1] - t[0]
    if dt <= 0:
        ax.set_title("Invalid time step")
        return

    # Numerical acceleration: d^2<x>/dt^2
    accel = np.gradient(np.gradient(ex, dt), dt)

    # Apply Hann window
    window = np.hanning(len(accel))
    accel_windowed = accel * window

    # FFT
    spectrum = np.abs(np.fft.rfft(accel_windowed)) ** 2
    freqs = np.fft.rfftfreq(len(accel_windowed), d=dt)
    omega = freqs * 2.0 * np.pi  # angular frequency in a.u.

    # Plot on log scale
    spectrum = np.clip(spectrum, 1e-30, None)
    ax.semilogy(omega, spectrum, "b-", linewidth=0.5)

    ax.set_xlabel(r"$\omega$ (a.u.)")
    ax.set_ylabel(r"$|\ddot{d}(\omega)|^2$ (arb. units)")
    ax.set_title("Dipole Acceleration Spectrum")
    ax.set_xlim(0, omega[-1])
    ax.grid(True, alpha=0.3)


def plot_linear_response_spectrum(ax, data, ground_energy=None, dt=None):
    """Plot linear response absorption spectrum from kick-mode dipole data.

    Computes the FFT of the total dipole d(t) = <x1>(t) + <x2>(t).
    Peaks appear at omega_n = E_n - E_0 (transition energies from the
    ground state).

    Args:
        ax: Matplotlib Axes object.
        data: DataFrame from parse_real_observables (kick mode run).
        ground_energy: Ground state energy in a.u. If provided, x-axis
            shows absolute energies E_n = E_0 + omega instead of omega.
        dt: Time step. If None, computed from data.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    t = data["time"].values
    ex = data["expect_x"].values
    ey = data["expect_y"].values

    if len(t) < 10:
        ax.set_title("Insufficient data for spectrum")
        return

    if dt is None:
        dt = t[1] - t[0]
    if dt <= 0:
        ax.set_title("Invalid time step")
        return

    # Total dipole
    dipole = ex + ey

    # Subtract mean (DC component)
    dipole = dipole - dipole.mean()

    # Apply Hann window
    window = np.hanning(len(dipole))
    dipole_windowed = dipole * window

    # FFT
    spectrum = np.abs(np.fft.rfft(dipole_windowed)) ** 2
    freqs = np.fft.rfftfreq(len(dipole_windowed), d=dt)
    omega = freqs * 2.0 * np.pi  # angular frequency

    spectrum = np.clip(spectrum, 1e-30, None)

    if ground_energy is not None:
        # Show absolute energy E_n = E_0 + omega
        energies = ground_energy + omega
        ax.semilogy(energies, spectrum, "b-", linewidth=0.5)
        ax.set_xlabel("Energy (a.u.)")
        ax.set_title("Linear Response Spectrum (absolute energies)")
        ax.axvline(
            x=ground_energy,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"$E_0$ = {ground_energy:.3f}",
        )
        ax.legend(fontsize=8)
    else:
        ax.semilogy(omega, spectrum, "b-", linewidth=0.5)
        ax.set_xlabel(r"$\omega$ (a.u.)")
        ax.set_title("Linear Response Absorption Spectrum")

    ax.set_ylabel(r"$|d(\omega)|^2$ (arb. units)")
    ax.grid(True, alpha=0.3)


def plot_norm(ax, data, phase="real"):
    """Plot wavefunction norm vs time.

    Args:
        ax: Matplotlib Axes object.
        data: dict from parse_imag_observables or parse_real_observables.
        phase: 'imag' for imaginary time, 'real' for real time.
    """
    ax.clear()
    if data is None:
        ax.set_title("No data")
        return

    if phase == "imag":
        x = (
            data["step"].values
            if "step" in data.columns
            else np.arange(len(data))
        )
        xlabel = "Imaginary time step"
    else:
        x = data["time"].values
        xlabel = "Time (a.u.)"

    norm = data["norm"].values

    if len(x) > 0 and len(norm) > 0:
        ax.plot(x, norm, "b-", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Norm")
    ax.set_title("Wavefunction Norm")
    ax.grid(True, alpha=0.3)


def create_wf_animation(
    output_dir,
    nx,
    ny,
    dx,
    dy,
    gif_path,
    vmin_orders=7,
    fps=5,
    xlim=None,
    ylim=None,
):
    """Create an animated GIF of real-time wavefunction snapshots.

    Loads all wf_real_NNNNNN.dat files from the output directory and
    creates a GIF showing the time evolution of |psi(x1,x2)|^2.

    Args:
        output_dir: Path to the directory containing wf_real_*.dat files.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        gif_path: Output path for the GIF file.
        vmin_orders: Orders of magnitude below max to display.
        fps: Frames per second in the GIF.
        xlim: Optional (xmin, xmax) tuple for zoom.
        ylim: Optional (ymin, ymax) tuple for zoom.

    Returns:
        Number of frames in the animation, or 0 if no snapshots found.
    """
    output_dir = Path(output_dir)
    snapshots = find_wf_snapshots(output_dir)

    if not snapshots:
        print("No wavefunction snapshots found.")
        return 0

    # Build coordinate arrays
    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    y = (np.arange(ny) - ny / 2 + 0.5) * dy

    fig, ax = plt.subplots(figsize=(8, 7))

    # Load first frame to set up the plot
    ts0, path0 = snapshots[0]
    wf0 = parse_wavefunction(path0, nx, ny)
    if wf0 is None:
        print("Failed to load first snapshot.")
        return 0

    prob0 = np.abs(wf0) ** 2
    prob0 = np.clip(prob0 / max(prob0.max(), 1e-30), 1e-30, None)
    log0 = np.log10(prob0)

    im = ax.pcolormesh(
        x,
        y,
        log0.T,
        vmin=-vmin_orders,
        vmax=0.0,
        cmap="hot_r",
        shading="auto",
    )
    ax.set_xlabel("Electron 1 position (a.u.)")
    ax.set_ylabel("Electron 2 position (a.u.)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(|\psi|^2 / \max)$")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    title = ax.set_title("")

    def update(frame_idx):
        ts_idx, fpath = snapshots[frame_idx]
        wf = parse_wavefunction(fpath, nx, ny)
        if wf is None:
            return [im, title]
        prob = np.abs(wf) ** 2
        pmax = max(prob.max(), 1e-30)
        prob = np.clip(prob / pmax, 1e-30, None)
        log_prob = np.log10(prob)
        im.set_array(log_prob.T.ravel())
        label = "final" if ts_idx == -1 else f"step {ts_idx}"
        title.set_text(rf"$|\psi(x_1,x_2)|^2$ — {label}")
        return [im, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        blit=False,
        interval=1000 // fps,
    )
    anim.save(str(gif_path), writer="pillow", fps=fps)
    anim.event_source = (
        None  # prevent matplotlib trying to start event loop after save
    )
    plt.close(fig)
    print(f"Animation saved to {gif_path} ({len(snapshots)} frames)")
    return len(snapshots)


def load_config(filepath):
    """Parse the simulation.cfg file into a dictionary."""
    conf = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=")
            conf[key.strip()] = val.strip()
    return conf


if __name__ == "__main__":
    workdir = Path(__file__).resolve().parents[1] / "res"

    cfg_path = workdir / "simulation.cfg"

    if cfg_path.exists():
        config = load_config(cfg_path)
        nx, ny = int(config["grid_nx"]), int(config["grid_ny"])
        dx, dy = float(config["grid_dx"]), float(config["grid_dy"])

        # Load all data
        real_data = parse_real_observables(workdir / config["obser_file"])
        imag_data = parse_imag_observables(workdir / config["obser_imag_file"])

        # --- Ground state wavefunction (single converged result) ---
        wf_ground = parse_wavefunction(workdir / "wf_ground.dat", nx, ny)
        if wf_ground is not None:
            fig_gs, ax_gs = plt.subplots(figsize=(8, 8))
            plot_wavefunction_2d(ax_gs, wf_ground, dx, dy, nx, ny)
            ax_gs.set_title(r"Ground State $|\psi_0(x_1,x_2)|^2$")
            fig_gs.savefig(workdir / "ground_state_wf.png")
            print("Ground state wavefunction plot saved.")

        # --- Imaginary time energy convergence ---
        if imag_data is not None:
            fig_imag, ax_imag = plt.subplots(figsize=(8, 6))
            plot_energy_vs_time(ax_imag, imag_data, phase="imag")

            e_vals = imag_data["energy_real"].values
            if len(e_vals) > 0:
                final_e = e_vals[-1]
                ax_imag.text(
                    0.05,
                    0.95,
                    f"Converged E: {final_e:.6f} a.u.",
                    transform=ax_imag.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                )
                print(f"Imaginary time converged to: {final_e:.6f} a.u.")

            fig_imag.savefig(workdir / "imaginary_phase.png")

        # --- Real-time dashboard ---
        if real_data is not None:
            fig_real, axes = plt.subplots(3, 2, figsize=(12, 15))
            fig_real.suptitle("Real-Time Propagation Dashboard", fontsize=16)

            plot_energy_vs_time(axes[0, 0], real_data, phase="real")
            plot_dipole(axes[0, 1], real_data)
            plot_ionization(axes[1, 0], real_data)
            plot_vector_potential(axes[1, 1], real_data)
            plot_spectrum(axes[2, 0], real_data)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig_real.savefig(workdir / "real_time_dashboard.png")

        # --- Animated GIF of real-time wavefunction evolution ---
        n_frames = create_wf_animation(
            workdir,
            nx,
            ny,
            dx,
            dy,
            gif_path=workdir / "wf_evolution.gif",
            fps=5,
        )
        if n_frames > 0:
            print(f"Wavefunction evolution GIF: {n_frames} frames")
        else:
            print("No real-time wavefunction snapshots found for animation.")

        # --- Collect all wavefunctions into HDF5 ---
        h5_path = workdir / "wavefunctions.h5"
        real_dt = float(config.get("real_dt", 0.1))
        n_written = collect_wf_to_hdf5(
            workdir, nx, ny, h5_path, real_dt=real_dt
        )
        print(f"Collected {n_written} wavefunctions into {h5_path}")
