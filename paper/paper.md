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
authors:
  - name: Varun Kapoor
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: KapoorLabs, Paris, France
    index: 1
date: 3 May 2026
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
the exact two-electron solution at every time step. A PyQt5 graphical
interface drives the C++ Crank–Nicolson core through a flat
configuration file and exposes ground-state, excited-state, real-time,
kick-mode, and spectral-projection (Feit–Fleck–Steiger) workflows from
one window.

This paper introduces the underlying numerical machinery — the
Crank–Nicolson split-operator propagator, imaginary-time relaxation for
ground and excited eigenstates, and Gram–Schmidt orthogonalisation for
the excited-state spectrum — and demonstrates its use on a pulse-shape
study (sinusoidal, trapezoidal, and impulsive "kick" fields) that
reproduces the periodicity-violation results of the exact-KS-potential
literature.

# Statement of need

Single-active-electron and small two-electron model systems remain a
workhorse for understanding strong-field phenomena that are too costly
to study with full three-dimensional helium codes: high-harmonic
generation, non-sequential double ionisation, autoionising-state
dynamics, and the time-dependent Kohn–Sham potential of TDDFT. The
literature contains several legacy Fortran/C++ codebases for this
purpose, but they typically (i) hard-code their parameters, (ii) ship
without a graphical front-end, and (iii) duplicate code between the TDSE
and the exact-TDDFT modes that share a propagator. `latom` factors the
propagator, the wavefunction operations, and the imaginary-time
machinery into a shared library; provides a single interactive
front-end that selects the operating mode at run time; and writes a
single, plain-text configuration that is readable and diffable.

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

Atomic units are used throughout. The wavefunction is discretised on a
uniform Cartesian grid of size $N_x \times N_y$ with spacings
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
unconditionally stable. `latom` performs the two-electron update by
operator splitting: the $x_1$-direction sweep, the $x_2$-direction
sweep, and the electron–electron coupling block are each handled by a
tridiagonal Crank–Nicolson solve along the relevant axis. Because every
sweep reduces to the inversion of a tridiagonal matrix of dimension
$N_x$ or $N_y$, the cost per timestep is $\mathcal{O}(N_x N_y)$.

The same routine drives both real-time propagation and the
imaginary-time relaxation described below; only the timestep argument
changes from real to imaginary, and the Hamiltonian (with or without
the laser term) is selected accordingly.

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
scheme using a purely imaginary timestep $\Delta t = -i\,\Delta\tau$,
renormalising $\psi$ after every step. Convergence is monitored through
the Rayleigh quotient $\langle \psi | H | \psi \rangle$, which decreases
monotonically to $E_0$. The number of steps $N_{\tau}$ and the
imaginary timestep $\Delta\tau$ are exposed as `imag_steps` and
`imag_dt` in the configuration file.

## Excited states by Gram–Schmidt projection

Once the ground state $\phi_0$ is converged, the same imaginary-time
machinery yields excited states one at a time. To prevent the iterate
from collapsing back onto $\phi_0$ (which would always win the
exponential race) the propagated state is re-orthogonalised against the
previously converged eigenstates after every imaginary-time step:

$$
\tilde\psi^{n+1} = \psi^{n+1}
                 - \sum_{k=0}^{N-1}
                   \langle \phi_k | \psi^{n+1} \rangle\, \phi_k,
\qquad
\psi^{n+1} \leftarrow \frac{\tilde\psi^{n+1}}{\| \tilde\psi^{n+1} \|}.
$$

This is the classical Gram–Schmidt procedure applied at every step of
the imaginary-time relaxation. Because the orthogonalisation is
enforced continuously rather than at the end, the iterate spends its
entire trajectory inside the orthogonal complement of
$\{\phi_0, \dots, \phi_{N-1}\}$, and the lowest eigenvalue of $H$ *in
that subspace* — i.e. $E_N$ — is returned.

Higher excited states converge more slowly because their relative gap
to neighbouring states shrinks; `latom` accordingly multiplies the
imaginary-step budget by $N \times$ `excited_imag_mult` for the
$N$-th excited state. The sequence of converged $\phi_n$ is dumped to
`wf_excited_<N>.dat` and can be reloaded to resume work or to seed a
real-time propagation from a non-ground initial state.

# Availability and reproducibility

`latom` is distributed under an open-source licence and is available at
the project repository. The C++ kernel builds with a single `make` call
under any modern GCC/Clang; the Python front-end requires PyQt5,
`numpy`, `matplotlib`, and `h5py`. Configuration files used to produce
every figure in this paper are shipped under `gui/experiment_*` and are
loaded with one click from the GUI.

# References
