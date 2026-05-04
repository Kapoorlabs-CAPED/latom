---
title: "latom: a 2D two-electron TDSE solver with exact-TDDFT Kohn–Sham reconstruction"
tags:
  - Python
  - C++
  - Qt
  - quantum mechanics
  - time-dependent Schrödinger equation
  - density functional theory
  - Crank–Nicolson
  - high-harmonic generation
authors:
  - name: Varun Kapoor
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: KapoorLabs, Paris, France
    index: 1
date: 4 May 2026
bibliography: paper.bib
---

# Summary

`latom` is an interactive solver for the two-electron, one-dimensional-each
time-dependent Schrödinger equation (TDSE) of model helium driven by an
intense laser field in the velocity gauge. The same numerical kernel
supports two operating modes: (i) plain TDSE propagation of the
correlated two-electron wavefunction $\psi(x_1, x_2, t)$, and (ii) an
exact time-dependent density-functional-theory (TDDFT) construction in
which the Kohn–Sham (KS) orbital $\varphi_{\mathrm{KS}}(x, t)$ and its
effective potential $v_{\mathrm{KS}}(x, t)$ are reverse-engineered from
the exact two-electron solution at every time step. Five workflow tabs
in the PyQt5 graphical interface — TDSE, Exact-TDDFT, Reproduce
Periodicity Paper, Autoionization Computer, and Kick / Linear Response
— share a common Crank–Nicolson C++ kernel and a single plain-text
configuration file.

This paper introduces the underlying numerical machinery — the
Crank–Nicolson split-operator propagator, imaginary-time relaxation
for ground and excited eigenstates, Gram–Schmidt orthogonalisation for
the excited spectrum, three pulse shapes (sinusoidal, trapezoidal,
kick), an exact-TDDFT KS-orbital reconstruction, and an online (no
aliasing) Fourier transform of the resulting $v_{\mathrm{KS}}(x, t)$ —
and demonstrates its use on the three-case pulse-shape study of
@Kapoor2013Periodicity.

# Statement of need

Single-active-electron and small two-electron model systems remain a
workhorse for understanding strong-field phenomena that are too costly
to attack with full three-dimensional helium codes: high-harmonic
generation, non-sequential double ionisation, autoionising-state
dynamics, and the time-dependent Kohn–Sham potential of TDDFT. The
literature contains several legacy Fortran/C++ codebases for this
purpose, but they typically (i) hard-code their parameters, (ii) ship
without a graphical front-end, and (iii) duplicate code between the
TDSE and the exact-TDDFT modes that share a propagator. `latom`
factors the propagator, the wavefunction operations, the imaginary-time
machinery, and the online spectral diagnostics into a shared library;
provides a single interactive front-end that selects the workflow at
run time; and writes a single, plain-text configuration that is
readable, diffable, and trivially scriptable.

The numerical kernel that `latom` exposes through this graphical
front-end has previously been used to study Floquet structure
extraction from real-time propagated wave functions
[@Kapoor2012Floquet], the periodicity (and breakdown thereof) of the
time-dependent Kohn–Sham equation in the Floquet regime
[@Kapoor2013Periodicity], and autoionising-state dynamics within
time-dependent density-functional theory [@Kapoor2016Autoionization].
Re-packaging this kernel with an interactive front-end and a single
unified configuration lowers the barrier for further work along these
lines.

# Numerical methods

## Time-dependent Schrödinger equation

In the velocity gauge, the two-electron Hamiltonian is

$$
H(t) = \sum_{i=1}^{2} \left[ \tfrac{1}{2}\left(p_i + A(t)\right)^2
  + v_{\mathrm{ext}}(x_i) \right] + v_{\mathrm{ee}}(x_1 - x_2),
$$

with the soft-core nuclear and electron–electron potentials

$$
v_{\mathrm{ext}}(x) = -\frac{2}{\sqrt{x^2 + \varepsilon^2}},
\qquad
v_{\mathrm{ee}}(x_1 - x_2) = \frac{1}{\sqrt{(x_1-x_2)^2 + \varepsilon^2}}.
$$

Atomic units are used throughout. The wavefunction is discretised on
a uniform Cartesian grid of size $N_x \times N_y$ with spacings
$\Delta x, \Delta y$ and stored as a flat `complex<double>` array; the
kinetic energy uses second-order finite differences, and absorbing
boundaries are imposed via an imaginary potential of the form
$V_{\mathrm{abs}}(x) \propto x^{16}$ localised near the box edges.

## Crank–Nicolson propagator

For a state $\psi^n$ at time $t_n$ the Crank–Nicolson update over one
timestep $\Delta t$ reads [@Crank1947]

$$
\left( 1 + \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n+1}
= \left( 1 - \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n}.
$$

The scheme is unitary, second-order accurate in time, and
unconditionally stable. `latom` performs the two-electron update via
Strang splitting: the $x_1$-direction sweep, the $x_2$-direction
sweep, and the electron–electron coupling block are each handled by a
tridiagonal Crank–Nicolson solve along the relevant axis, in a
symmetric ABCBA pattern around the time midpoint. Every sweep reduces
to the inversion of a tridiagonal matrix of dimension $N_x$ or $N_y$,
so the cost per timestep is $\mathcal{O}(N_x N_y)$. The
right-hand-side build loops along disjoint rows/columns are
parallelised across cores with OpenMP — `OMP_NUM_THREADS` at run time
controls the thread count.

The same routine drives both real-time propagation and the
imaginary-time relaxation described next; only the timestep argument
changes from real to imaginary, and the Hamiltonian (with or without
the laser term) is selected accordingly.

## Pulse shapes

The vector potential $A(t)$ supports three shapes, selected via the
`laser_pulse_shape` config key:

* **Sinusoidal** (sin² envelope):
  $A(t) = \tfrac{\alpha}{\omega}\,\sin^2\!\left(\tfrac{\omega t}{2 N_c}\right)
          \sin(\omega t - \varphi)$
  spanning $N_c$ optical cycles.
* **Trapezoidal**: a linear ramp-up of $N_{\text{up}}$ cycles, a
  plateau of $N_{\text{plat}}$ cycles at full amplitude, then a linear
  ramp-down of $N_{\text{down}}$ cycles to zero. Used by the paper-
  reproduction workflow because it is the canonical shape for studying
  Floquet steady-state behaviour during the plateau.
* **Kick**: $A(t) = A_0$ identically — the electric field is a Dirac
  delta at $t=0$, used to extract the linear-response spectrum from
  the post-kick free evolution.

The carrier-envelope phase $\varphi$ is exposed via `laser_phi`. The
config field `laser_alpha` is the electric-field amplitude
$E_0 = \alpha$; the C++ kernel computes the vector-potential amplitude
in velocity gauge as $A_0 = E_0 / \omega$, which is the standard
identity for a sinusoidal carrier $A(t) = A_0 \sin(\omega t)$ giving
$E(t) = -A_0 \omega \cos(\omega t)$. This matches the convention used
in @Kapoor2013Periodicity.

The simulated propagation duration is computed automatically from the
pulse cycle counts so that the laser stops exactly when the pulse
ends; users see and can override the resulting `real_steps` count
in the GUI.

## Eigenvalue equations via imaginary-time propagation

The time-independent Schrödinger equation $H \phi_n = E_n \phi_n$ is
solved without ever forming or diagonalising $H$. Replacing
$t \to -i\tau$ in the TDSE turns the unitary evolution into a
contraction [@Lehtovaara2007]:

$$
\psi(\tau) = e^{-H \tau}\, \psi(0)
           = \sum_{n} c_n e^{-E_n \tau}\, \phi_n,
$$

where $\{\phi_n\}$ is the (unknown) eigenbasis of $H$. As
$\tau \to \infty$ every excited component decays faster than the
ground-state component, so the renormalised state collapses onto the
ground state:

$$
\phi_0 = \lim_{\tau\to\infty}
         \frac{e^{-H\tau} \psi(0)}{\| e^{-H\tau} \psi(0) \|}.
$$

In practice `latom` propagates $\psi$ with the same Crank–Nicolson
scheme using a purely imaginary timestep
$\Delta t = -i\,\Delta\tau$, renormalising $\psi$ after every step.
Convergence is monitored through the Rayleigh quotient
$\langle \psi | H | \psi \rangle$, which decreases monotonically to
$E_0$. The number of steps $N_{\tau}$ and the imaginary timestep
$\Delta\tau$ are exposed as `imag_steps` and `imag_dt`.

### A duality worth highlighting

The above Wick rotation works because of an unusual feature of
quantum mechanics that is easy to take for granted: *the same*
Hermitian operator $H$ both labels the eigenvalue equation that we
want to solve, $H\phi_n = E_n \phi_n$, and generates the unitary
time-evolution that we know how to integrate, $i\partial_t \psi = H
\psi$. Substituting $t \to -i\tau$ analytically continues the
evolution operator $e^{-iHt/\hbar}$ to the contraction $e^{-H\tau}$,
which on a single Hermitian operator is well-defined and damps every
eigenvector exponentially in proportion to its eigenvalue.

This duality is *not* generic. Most eigenvalue problems in scientific
computing — graph Laplacians, structural-mechanics stiffness
matrices, the Helmholtz equation, kernel matrices in machine learning
— do not come paired with a natural time-dependent partner equation
that can be analytically continued in this way; one usually reaches
for Krylov-subspace solvers (Lanczos, Davidson, LOBPCG) or
preconditioned conjugate-gradient iteration on the static problem
itself. It is precisely because the Schrödinger equation supplies
both a static and a dynamic equation built from a single Hermitian
$H$ that the imaginary-time trick reduces the eigenvalue problem to
"run the same propagator with a different timestep."

`latom` exploits this duality directly: a single Crank–Nicolson
routine handles both the laser-driven dynamics and the ground-state
solve. Maintaining one propagator instead of two reduces the surface
area of the code, eliminates a class of bugs that plague separate
imaginary- and real-time implementations, and lets every numerical
improvement (operator splitting, OpenMP parallelism, regridding) flow
automatically to both regimes.

## Excited states by Gram–Schmidt projection

Once the ground state $\phi_0$ is converged, the same imaginary-time
machinery yields excited states one at a time. To prevent the
iterate from collapsing back onto $\phi_0$ (which would always win
the exponential race), the propagated state is re-orthogonalised
against the previously converged eigenstates after every
imaginary-time step:

$$
\tilde\psi^{n+1} = \psi^{n+1}
                 - \sum_{k=0}^{N-1}
                   \langle \phi_k | \psi^{n+1} \rangle\, \phi_k,
\qquad
\psi^{n+1} \leftarrow \frac{\tilde\psi^{n+1}}{\| \tilde\psi^{n+1} \|}.
$$

Higher excited states converge more slowly because their relative
gap to neighbouring states shrinks; `latom` accordingly multiplies
the imaginary-step budget by $N \cdot$ `excited_imag_mult` for the
$N$-th excited state. Converged $\phi_n$ are dumped to
`wf_excited_<N>.dat` and can be reloaded to resume work or to seed a
real-time propagation from a non-ground initial state.

## Exact-TDDFT: reverse-engineering the KS orbital

In Exact-TDDFT mode `latom` simultaneously propagates the exact
two-electron wavefunction and reconstructs the corresponding KS
orbital. For the spin-singlet two-electron system, the KS orbital is
related to the exact density and current by [@Kapoor2013Periodicity,
eq. 29]

$$
\varphi_{\mathrm{KS}}(x, t) = \sqrt{n(x, t)/2}\,e^{i \theta(x, t)},
\qquad
\partial_x[n(x, t)\,\partial_x \theta(x, t)] = -\partial_t n(x, t),
$$

where $n(x, t) = 2 \int \mathrm{d}x'\,|\psi(x, x', t)|^2$ is the
one-body density and $\theta$ is the phase implied by the continuity
equation. The effective KS potential
$v_{\mathrm{KS}}(x, t)$ is then extracted from a forward / backward
half-step of $\varphi_{\mathrm{KS}}$ under a bare-kinetic Hamiltonian
($H_{\mathrm{bare}} = -\tfrac{1}{2}\partial_x^2$) and the
identification

$$
v_{\mathrm{KS}}(x, t) = \frac{i}{\Delta t}
   \log\!\left[\,\varphi_{+}(x, t)/\varphi_{-}(x, t)\,\right],
$$

where $\varphi_{\pm}$ are the half-step images of
$\varphi_{\mathrm{KS}}$. All three building blocks
(`denskohnsham`, `denskohnshamcorrector`, `phasekohnshamorbital`)
live in the shared `wavefunction.cc` library and are reused between
the TDSE and Exact-TDDFT binaries.

## Online $v_{\mathrm{KS}}(x, \omega)$ Fourier transform

The classic post-processing route — dump $v_{\mathrm{KS}}$ snapshots
to disk every $N_{\text{snap}}$ steps and FFT them in Python —
introduces aliasing whenever the snapshot cadence
$\Delta t \cdot N_{\text{snap}}$ exceeds $\pi/\omega_L$. For
short-wavelength laser fields ($\omega_L \sim 2.6$ a.u. in the
paper's high-frequency case), the resulting Nyquist falls below the
laser carrier itself, and harmonic peaks vanish from the spectrum.

`latom` instead accumulates the Fourier integral *online* inside the
real-time loop:

$$
\hat v_{\mathrm{KS}}(x, \omega_k) =
   \sum_{n=0}^{N-1} v_{\mathrm{KS}}(x, n\Delta t)\,
                    e^{-i \omega_k n \Delta t}\, \Delta t,
$$

over a user-chosen frequency grid $\{\omega_k\}$ spanning
`fft_harmonic_min` to `fft_harmonic_max` in units of $\omega_L$. The
inner loop is parallelised across $\omega$ with OpenMP. At simulation
end the normalised power $|\hat v_{\mathrm{KS}}(x, \omega_k)|^2$ is
dumped to `vks_fft.dat` for plotting. Because the integral is
sampled at the full real-time step, there is no aliasing regardless
of how often (or rarely) snapshots are written to disk.

# Implementation

The C++ kernel (`TDSE.cc`, `ExactTDDFT.cc`, `wavefunction.cc`,
`hamop.cc`, `grid.cc`, `fluid.cc`) builds with a single `make` call
under any modern GCC/Clang and OpenMP. Two binaries are produced:
`TDSE` for the basic 2e Schrödinger workflow, and `ExactTDDFT` for
the KS-reconstruction and online-FFT workflow. The eight Crank–
Nicolson sweeps in the 2D propagator are OpenMP-parallel; the online
$v_{\mathrm{KS}}$ FFT loop is too.

The PyQt5 GUI (`gui/main_window.py`) presents five workflow tabs:

1. **TDSE** — propagate $\psi(x_1, x_2, t)$ under a chosen pulse.
2. **Exact-TDDFT** — additionally reconstruct $\varphi_{\mathrm{KS}}$
   and $v_{\mathrm{KS}}$ each step plus the online FFT.
3. **Reproduce Periodicity Paper** — preset cards for the three
   trapezoidal cases of @Kapoor2013Periodicity (low-$\omega$,
   high-$\omega$, resonant), launched in parallel from a single
   shared imaginary-time preflight.
4. **Autoionization Computer** — Feit–Fleck–Steiger spectral
   projection at a target energy [@FleckFeitSteiger1982].
5. **Kick / Linear Response** — constant-$A_0$ pulse for absorption
   spectroscopy.

Each tab is self-contained: only the parameters that the active
workflow actually consumes are exposed, and switching tabs flips a
`mode` field that selects the right binary. A live progress bar tied
to the simulation's elapsed propagation time replaces the legacy
log textbox. When a previous run's output files are present on disk
the GUI offers to load them instead of recomputing — useful for
adjusting plot style or dynamic range without re-running.

When the user changes `grid_nx`/`grid_ny` between runs but still asks
to load a cached ground state, `latom` infers the smaller-grid file
size from line counts and instructs the C++ binary to read at that
small grid and *regrid* onto the current larger grid via the
`wavefunction::regrid` routine, rather than silently corrupting the
initial state.

# Availability and reproducibility

`latom` is distributed under an open-source licence and is available
at the project repository. The C++ kernel builds with a single
`make -j -B all` invocation under any modern GCC/Clang with OpenMP;
the Python front-end requires PyQt5, `numpy`, `matplotlib`, `scipy`,
and `h5py`. Configuration files used to produce every figure in this
paper are shipped under `gui/experiment_*` and are loaded with one
click from the GUI. A regression test
(`tests/test_fft_sanity.py`) verifies that the FFT pipeline correctly
resolves a synthetic three-harmonic signal at $\omega/\omega_L = 1,
2, 3$ with the expected $1:0.09:0.01$ amplitude ratios.

# References
