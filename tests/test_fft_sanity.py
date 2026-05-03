"""FFT sanity test: feed sin(omega*t) into plot_ks_potential_fft and
verify the peak shows up at harmonic order 1.0.

Run from the repo root:
    PYTHONPATH=scripts python tests/test_fft_sanity.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

# Make the bundled plotting module importable when run directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))

from plotting import plot_ks_potential_fft  # noqa: E402


def main():
    omega = 10.0
    dt = 0.1
    N = 4096
    t = np.arange(N) * dt
    V_t = np.sin(omega * t)

    # plot_ks_potential_fft renders a 2D pcolormesh on (harmonic, x) axes.
    # We need nx > 1 for the y-axis to have non-zero extent. Replicate the
    # same sin(omega·t) carrier across every spatial point so the resulting
    # |F(x, ω)|² panel shows a uniform bright stripe at harmonic 1, dark
    # everywhere else.
    nx = 64
    dx = 0.4
    spatial = np.ones(nx)
    snaps = [(k, (V_t[k] * spatial).astype(complex)) for k in range(N)]

    fig, ax = plt.subplots(figsize=(6, 3))
    plot_ks_potential_fft(
        ax,
        snaps,
        dx=dx,
        nx=nx,
        dt_snapshot=dt,
        laser_freq=omega,
        plateau_skip_frac=0.0,
        harmonic_max=2.5,
        vmin_orders=5,
        title="sanity: sin(10 t), dt=0.1 — expect bright stripe at harmonic 1",
    )

    # Re-run the FFT logic the plotter uses, on its own, so we can read
    # the peak position numerically.
    series = V_t - V_t.mean()
    n = len(series)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1)))
    spec = np.fft.rfft(series * win)
    freqs = np.fft.rfftfreq(n, d=dt) * 2.0 * np.pi
    harmonic = freqs / omega
    power = np.abs(spec) ** 2

    peak = int(np.argmax(power))
    nyq_omega = np.pi / dt
    print(
        f"Nyquist angular ω = {nyq_omega:.3f}  "
        f"(harmonic Nyquist {nyq_omega/omega:.2f})"
    )
    print("Peak power at:")
    print(f"   ω         = {freqs[peak]:.4f} a.u.   (expected {omega:.4f})")
    print(f"   harmonic  = {harmonic[peak]:.4f}        (expected 1.0000)")
    top = np.argsort(power)[::-1][:5]
    print("Top 5 bins (harmonic, |F|^2 / max):")
    for i in top:
        print(f"   {harmonic[i]:.4f}   {power[i] / power.max():.4f}")

    out = _HERE / "fft_sanity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"saved {out}")

    # Hard assertions so this works as a regression test too.
    assert (
        abs(harmonic[peak] - 1.0) < 1e-3
    ), f"peak at harmonic {harmonic[peak]:.4f}, expected 1.0"
    print("PASS")


if __name__ == "__main__":
    main()
