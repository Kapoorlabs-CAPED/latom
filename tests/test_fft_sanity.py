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

    # Multi-harmonic signal: a fundamental at ω_L plus 2ω_L and 3ω_L
    # with decreasing amplitudes — what V_KS looks like physically when
    # the system radiates integer harmonics of the driving laser. The
    # plot must show *three* sharp peaks at harmonic order 1, 2, 3.
    A1, A2, A3 = 1.0, 0.3, 0.1
    V_t = (
        A1 * np.sin(omega * t)
        + A2 * np.sin(2.0 * omega * t)
        + A3 * np.sin(3.0 * omega * t)
    )

    # Same temporal signal at every spatial point → bright vertical
    # stripes at harmonic 1, 2, 3.
    nx = 64
    dx = 0.4

    # Build the (x, ω) FFT grid the way ExactTDDFT.cc would, but in
    # numpy so this test is independent of the C++ binary. Use the same
    # online-integration formula:
    #   F̂(x, ω_j) = Σ_n V(x, n·dt) · exp(-iω_j·n·dt) · dt
    n_omega = 800
    omega_min = 0.0
    omega_max = 4.0 * omega
    omegas = np.linspace(omega_min, omega_max, n_omega)
    # exp(-i ω t)
    phase = np.exp(-1j * np.outer(omegas, t)) * dt  # (n_omega, N)
    spec = phase @ V_t  # (n_omega,)
    power_1d = np.abs(spec) ** 2
    power = np.tile(power_1d, (nx, 1))  # (nx, n_omega)

    x_axis = (np.arange(nx) - nx / 2 + 0.5) * dx
    fft_data = {
        "x": x_axis,
        "omega": omegas,
        "harmonic": omegas / omega,
        "power": power,
        "omega_L": omega,
    }

    fig, ax = plt.subplots(figsize=(6, 3))
    plot_ks_potential_fft(
        ax,
        fft_data,
        title="sanity: sin(ωt) + 0.3 sin(2ωt) + 0.1 sin(3ωt) — peaks at h=1,2,3",
        harmonic_min=0.0,
        harmonic_max=4.0,
        vmin_orders=5,
    )

    # Numerical readout (use the same integral the C++ accumulator does).
    harmonic = omegas / omega
    power = power_1d

    # Find the three local maxima nearest the expected harmonic positions.
    def peak_near(h_target, half_window=0.05):
        mask = np.abs(harmonic - h_target) < half_window
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None, 0.0
        local = idxs[np.argmax(power[idxs])]
        return harmonic[local], power[local]

    h1, p1 = peak_near(1.0)
    h2, p2 = peak_near(2.0)
    h3, p3 = peak_near(3.0)
    pmax = max(p1, p2, p3)
    print("Detected harmonic peaks:")
    print(
        f"   h=1: harmonic={h1:.4f}  |F|^2/max={p1/pmax:.4f}  (expect 1.000)"
    )
    print(
        f"   h=2: harmonic={h2:.4f}  |F|^2/max={p2/pmax:.4f}  (expect ~0.090)"
    )
    print(
        f"   h=3: harmonic={h3:.4f}  |F|^2/max={p3/pmax:.4f}  (expect ~0.010)"
    )

    out = _HERE / "fft_sanity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"saved {out}")

    # Regression assertions: all three peaks must be in the right place
    # AND the relative amplitudes must roughly track A1:A2:A3 = 1:0.3:0.1.
    assert h1 is not None and abs(h1 - 1.0) < 0.05, f"h=1 missing or off: {h1}"
    assert h2 is not None and abs(h2 - 2.0) < 0.05, f"h=2 missing or off: {h2}"
    assert h3 is not None and abs(h3 - 3.0) < 0.05, f"h=3 missing or off: {h3}"
    # |F|^2 ratio expected ≈ A_n^2 → 1 : 0.09 : 0.01.
    ratio_h2 = p2 / p1
    ratio_h3 = p3 / p1
    assert 0.05 < ratio_h2 < 0.15, f"h=2 amplitude ratio off: {ratio_h2:.3f}"
    assert 0.005 < ratio_h3 < 0.02, f"h=3 amplitude ratio off: {ratio_h3:.4f}"
    print(
        "PASS — all three integer-harmonic peaks present at the right ratios."
    )


if __name__ == "__main__":
    main()
