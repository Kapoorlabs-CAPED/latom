"""Matplotlib visualization functions for TDSE simulation results.

All functions take a matplotlib Axes object and data dict/arrays,
making them reusable from both scripts and the GUI.
"""

import numpy as np


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
        x = data.get("step", np.array([]))
        xlabel = "Imaginary time step"
    else:
        x = data.get("time", np.array([]))
        xlabel = "Time (a.u.)"

    e_real = data.get("energy_real", np.array([]))
    e_imag = data.get("energy_imag", np.array([]))

    if len(x) > 0 and len(e_real) > 0:
        ax.plot(x, e_real, "b-", linewidth=0.8, label="Re(E)")
    if len(x) > 0 and len(e_imag) > 0:
        ax.plot(x, e_imag, "r--", linewidth=0.8, alpha=0.7, label="Im(E)")

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

    t = data.get("time", np.array([]))
    pop = data.get("ground_pop", np.array([]))

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

    t = data.get("time", np.array([]))
    ex = data.get("expect_x", np.array([]))
    ey = data.get("expect_y", np.array([]))

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

    t = data.get("time", np.array([]))
    vp = data.get("vecpot_x", np.array([]))

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

    t = data.get("time", np.array([]))
    ex = data.get("expect_x", np.array([]))

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

    # Convert to harmonic order (freq / laser_freq)
    # Use fundamental from the data if possible
    omega0 = 1.556  # default laser frequency in a.u.

    harmonics = freqs * 2.0 * np.pi / omega0

    # Plot on log scale
    spectrum = np.clip(spectrum, 1e-30, None)
    ax.semilogy(harmonics, spectrum, "b-", linewidth=0.5)

    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel(r"$|d(\omega)|^2$ (arb. units)")
    ax.set_title("High Harmonic Generation Spectrum")
    ax.set_xlim(0, min(60, harmonics[-1]))
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
        x = data.get("step", np.array([]))
        xlabel = "Imaginary time step"
    else:
        x = data.get("time", np.array([]))
        xlabel = "Time (a.u.)"

    norm = data.get("norm", np.array([]))

    if len(x) > 0 and len(norm) > 0:
        ax.plot(x, norm, "b-", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Norm")
    ax.set_title("Wavefunction Norm")
    ax.grid(True, alpha=0.3)
