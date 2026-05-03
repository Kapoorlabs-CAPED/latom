..
   JOSS canonically expects ``paper.md`` (Markdown with a YAML
   frontmatter), not reStructuredText. This RST draft is kept here
   while the manuscript is in flight; the YAML metadata below can be
   lifted verbatim into a Markdown wrapper at submission time.

   ---
   title: "latom: a 2D two-electron TDSE solver with exact-TDDFT
           Kohn–Sham reconstruction"
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

==========================================================================
latom: a 2D two-electron TDSE solver with exact-TDDFT KS reconstruction
==========================================================================

:Authors: Varun Kapoor (KapoorLabs, Paris, France)
:Date: 2026-05-03


Summary
=======

``latom`` is an interactive solver for the two-electron, one-dimensional-each
time-dependent Schrödinger equation (TDSE) of model helium driven by an
intense laser field in the velocity gauge. The same numerical kernel
supports two operating modes: (i) plain TDSE propagation of the
correlated two-electron wavefunction :math:`\psi(x_1, x_2, t)`, and
(ii) an exact time-dependent density-functional-theory
(TDDFT) construction in which the Kohn–Sham (KS) orbital
:math:`\varphi_{\rm KS}(x, t)` and its effective potential
:math:`v_{\rm KS}(x, t)` are reverse-engineered from the exact
two-electron solution at every time step. A PyQt5 graphical interface
drives the C++ Crank–Nicolson core through a flat configuration file
and exposes ground-state, excited-state, real-time, kick-mode, and
spectral-projection (Feit–Fleck–Steiger) workflows from one window.

This paper introduces the underlying numerical machinery — the
Crank–Nicolson split-operator propagator, imaginary-time relaxation
for ground and excited eigenstates, and Gram–Schmidt orthogonalisation
for the excited-state spectrum — and demonstrates its use on a
pulse-shape study (sinusoidal, trapezoidal, and impulsive "kick"
fields) that reproduces the periodicity-violation results of the
exact-KS-potential literature.


Statement of need
=================

Single-active-electron and small two-electron model systems remain a
workhorse for understanding strong-field phenomena that are too costly
to study with full three-dimensional helium codes: high-harmonic
generation, non-sequential double ionisation, autoionising-state
dynamics, and the time-dependent Kohn–Sham potential of TDDFT. The
literature contains several legacy Fortran/C++ codebases for this
purpose, but they typically (i) hard-code their parameters, (ii) ship
without a graphical front-end, and (iii) duplicate code between the
TDSE and the exact-TDDFT modes that share a propagator. ``latom``
factors the propagator, the wavefunction operations, and the
imaginary-time machinery into a shared library; provides a single
interactive front-end that selects the operating mode at run time;
and writes a single, plain-text configuration that is readable and
diffable.

The numerical kernel that ``latom`` exposes through this graphical
front-end has previously been used to study Floquet structure
extraction from real-time propagated wave functions [Kapoor2012Floquet]_,
the periodicity (and breakdown thereof) of the time-dependent
Kohn–Sham equation in the Floquet regime [Kapoor2013Periodicity]_, and
autoionising-state dynamics within time-dependent density-functional
theory [Kapoor2016Autoionization]_. Re-packaging this kernel with an
interactive front-end and a single unified configuration lowers the
barrier for further work along these lines.


Numerical methods
=================

Time-dependent Schrödinger equation
-----------------------------------

In the velocity gauge, the two-electron Hamiltonian is

.. math::

   H(t) = \sum_{i=1}^{2} \left[ \frac{1}{2}\left(p_i + A(t)\right)^2
   + v_{\rm ext}(x_i) \right] + v_{\rm ee}(x_1 - x_2),

with the soft-core nuclear and electron–electron potentials

.. math::

   v_{\rm ext}(x) = -\frac{2}{\sqrt{x^2 + \varepsilon^2}}, \qquad
   v_{\rm ee}(x_1 - x_2) = \frac{1}{\sqrt{(x_1-x_2)^2 + \varepsilon^2}}.

Atomic units are used throughout. The wavefunction is discretised on
a uniform Cartesian grid of size :math:`N_x \times N_y` with spacings
:math:`\Delta x, \Delta y` and stored as a flat ``complex<double>``
array; the kinetic energy uses second-order finite differences, and
absorbing boundaries are imposed via an imaginary potential of the
form :math:`V_{\rm abs}(x) \propto x^{16}` localised near the box
edges.


Crank–Nicolson propagator
-------------------------

For a state :math:`\psi^n` at time :math:`t_n` the Crank–Nicolson
update for one timestep :math:`\Delta t` reads

.. math::

   \left( 1 + \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n+1}
   = \left( 1 - \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n}.

The scheme is unitary, second-order accurate in time, and
unconditionally stable. ``latom`` performs the two-electron update by
operator splitting: the :math:`x_1`-direction sweep, the
:math:`x_2`-direction sweep, and the electron–electron coupling block
are each handled by a tridiagonal Crank–Nicolson solve along the
relevant axis. Because every sweep reduces to the inversion of a
tridiagonal matrix of dimension :math:`N_x` or :math:`N_y`, the cost
per timestep is :math:`O(N_x N_y)`.

The same routine drives both real-time propagation and the
imaginary-time relaxation described below; only the timestep argument
changes from real to imaginary, and the Hamiltonian (with or without
the laser term) is selected accordingly.


Eigenvalue equations via imaginary-time propagation
---------------------------------------------------

The time-independent Schrödinger equation
:math:`H \phi_n = E_n \phi_n` is solved without ever forming or
diagonalising :math:`H`. Replacing :math:`t \to -i\tau` in the TDSE
turns the unitary evolution into a contraction:

.. math::

   \psi(\tau) = e^{-H \tau}\, \psi(0)
              = \sum_{n} c_n e^{-E_n \tau}\, \phi_n,

where :math:`\{\phi_n\}` is the (unknown) eigenbasis of :math:`H`. As
:math:`\tau \to \infty` every excited component decays faster than the
ground-state component, so the renormalised state collapses onto the
ground state:

.. math::

   \phi_0 = \lim_{\tau\to\infty}
            \frac{e^{-H\tau} \psi(0)}{\| e^{-H\tau} \psi(0) \|}.

In practice ``latom`` propagates :math:`\psi` with the same
Crank–Nicolson scheme using a purely imaginary timestep
:math:`\Delta t = -i\,\Delta\tau`, renormalising :math:`\psi` after
every step. Convergence is monitored through the Rayleigh quotient
:math:`\langle \psi | H | \psi \rangle`, which decreases monotonically
to :math:`E_0`. The number of steps :math:`N_{\tau}` and the imaginary
timestep :math:`\Delta \tau` are exposed as ``imag_steps`` and
``imag_dt`` in the configuration file.


Excited states by Gram–Schmidt projection
-----------------------------------------

Once the ground state :math:`\phi_0` is converged, the same
imaginary-time machinery yields excited states one at a time. To
prevent the iterate from collapsing back onto :math:`\phi_0` (which
would always win the exponential race) the propagated state is
re-orthogonalised against the previously converged eigenstates after
every imaginary-time step:

.. math::

   \tilde\psi^{n+1} = \psi^{n+1}
                    - \sum_{k=0}^{N-1}
                      \langle \phi_k | \psi^{n+1} \rangle\, \phi_k,
   \qquad
   \psi^{n+1} \leftarrow \frac{\tilde\psi^{n+1}}
                              {\| \tilde\psi^{n+1} \|}.

This is the classical Gram–Schmidt procedure applied at every step of
the imaginary-time relaxation. Because the orthogonalisation is
enforced continuously rather than at the end, the iterate spends its
entire trajectory inside the orthogonal complement of
:math:`\{\phi_0, \dots, \phi_{N-1}\}`, and the lowest eigenvalue of
:math:`H` *in that subspace* — i.e. :math:`E_N` — is returned.

Higher excited states converge more slowly because their relative
gap to neighbouring states shrinks; ``latom`` accordingly multiplies
the imaginary-step budget by :math:`N \times` ``excited_imag_mult``
for the :math:`N`-th excited state. The sequence of converged
:math:`\phi_n` is dumped to ``wf_excited_<N>.dat`` and can be reloaded
to resume work or to seed a real-time propagation from a non-ground
initial state.


Availability and reproducibility
================================

``latom`` is distributed under [SPDX licence] and is available at
[repository URL]. The C++ kernel builds with a single ``make`` call
under any modern GCC/Clang; the Python front-end requires PyQt5,
``numpy``, ``matplotlib``, and ``h5py``. Configuration files used to
produce every figure in this paper are shipped under ``gui/experiment_*``
and are loaded with one click from the GUI.


References
==========

.. [Crank1947] J. Crank and P. Nicolson,
   "A practical method for numerical evaluation of solutions of
   partial differential equations of the heat-conduction type",
   *Math. Proc. Camb. Philos. Soc.* **43**, 50 (1947).

.. [FleckFeitSteiger1976] M. D. Feit, J. A. Fleck Jr., and A. Steiger,
   "Solution of the Schrödinger equation by a spectral method",
   *J. Comput. Phys.* **47**, 412 (1982).

.. [Lehtovaara2007] L. Lehtovaara, J. Toivanen, and J. Eloranta,
   "Solution of time-independent Schrödinger equation by the
   imaginary time propagation method",
   *J. Comput. Phys.* **221**, 148 (2007).

.. [Kapoor2012Floquet] V. Kapoor and D. Bauer,
   "Floquet analysis of real-time wave functions without solving the
   Floquet equation",
   *Phys. Rev. A* **85**, 023407 (2012).
   doi:`10.1103/PhysRevA.85.023407
   <https://doi.org/10.1103/PhysRevA.85.023407>`_.

.. [Kapoor2013Periodicity] V. Kapoor, M. Ruggenthaler, and D. Bauer,
   "Periodicity of the time-dependent Kohn–Sham equation and the
   Floquet theorem",
   *Phys. Rev. A* **87**, 042521 (2013).
   doi:`10.1103/PhysRevA.87.042521
   <https://doi.org/10.1103/PhysRevA.87.042521>`_.

.. [Kapoor2016Autoionization] V. Kapoor,
   "Autoionization in time-dependent density-functional theory",
   *Phys. Rev. A* **93**, 063408 (2016).
   doi:`10.1103/PhysRevA.93.063408
   <https://doi.org/10.1103/PhysRevA.93.063408>`_.
