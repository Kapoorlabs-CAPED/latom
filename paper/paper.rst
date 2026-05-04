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

==========================================================================
latom: a 2D two-electron TDSE solver with exact-TDDFT KS reconstruction
==========================================================================

:Authors: Varun Kapoor (KapoorLabs, Paris, France)
:Date: 2026-05-04


Summary
=======

``latom`` is an interactive solver for the two-electron,
one-dimensional-each time-dependent Schrödinger equation (TDSE) of
model helium driven by an intense laser field in the velocity gauge.
The same numerical kernel supports two operating modes: (i) plain
TDSE propagation of the correlated two-electron wavefunction
:math:`\psi(x_1, x_2, t)`, and (ii) an exact time-dependent
density-functional-theory (TDDFT) construction in which the
Kohn–Sham (KS) orbital :math:`\varphi_{\rm KS}(x, t)` and its
effective potential :math:`v_{\rm KS}(x, t)` are reverse-engineered
from the exact two-electron solution at every time step. Five
workflow tabs in the PyQt5 graphical interface — TDSE, Exact-TDDFT,
Reproduce Periodicity Paper, Autoionization Computer, and Kick /
Linear Response — share a common Crank–Nicolson C++ kernel and a
single plain-text configuration file.

This paper introduces the underlying numerical machinery — the
Crank–Nicolson split-operator propagator, imaginary-time relaxation
for ground and excited eigenstates, Gram–Schmidt orthogonalisation
for the excited spectrum, three pulse shapes (sinusoidal,
trapezoidal, kick), an exact-TDDFT KS-orbital reconstruction, and an
online (no aliasing) Fourier transform of the resulting
:math:`v_{\rm KS}(x, t)` — and demonstrates its use on the
three-case pulse-shape study of [Kapoor2013Periodicity]_.


Statement of need
=================

Single-active-electron and small two-electron model systems remain a
workhorse for understanding strong-field phenomena that are too costly
to attack with full three-dimensional helium codes: high-harmonic
generation, non-sequential double ionisation, autoionising-state
dynamics, and the time-dependent Kohn–Sham potential of TDDFT. The
literature contains several legacy Fortran/C++ codebases for this
purpose, but they typically (i) hard-code their parameters, (ii) ship
without a graphical front-end, and (iii) duplicate code between the
TDSE and the exact-TDDFT modes that share a propagator. ``latom``
factors the propagator, the wavefunction operations, the
imaginary-time machinery, and the online spectral diagnostics into a
shared library; provides a single interactive front-end that selects
the workflow at run time; and writes a single, plain-text
configuration that is readable, diffable, and trivially scriptable.

The numerical kernel that ``latom`` exposes through this graphical
front-end has previously been used to study Floquet structure
extraction from real-time propagated wave functions
[Kapoor2012Floquet]_, the periodicity (and breakdown thereof) of
the time-dependent Kohn–Sham equation in the Floquet regime
[Kapoor2013Periodicity]_, and autoionising-state dynamics within
time-dependent density-functional theory
[Kapoor2016Autoionization]_. Re-packaging this kernel with an
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

Atomic units are used throughout. The wavefunction is discretised
on a uniform Cartesian grid of size :math:`N_x \times N_y` with
spacings :math:`\Delta x, \Delta y` and stored as a flat
``complex<double>`` array; the kinetic energy uses second-order
finite differences, and absorbing boundaries are imposed via an
imaginary potential of the form
:math:`V_{\rm abs}(x) \propto x^{16}` localised near the box edges.


Crank–Nicolson propagator
-------------------------

For a state :math:`\psi^n` at time :math:`t_n` the Crank–Nicolson
update for one timestep :math:`\Delta t` reads [Crank1947]_

.. math::

   \left( 1 + \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n+1}
   = \left( 1 - \tfrac{i \Delta t}{2} H(t_{n+1/2}) \right) \psi^{n}.

The scheme is unitary, second-order accurate in time, and
unconditionally stable. ``latom`` performs the two-electron update
via Strang splitting: the :math:`x_1`-direction sweep, the
:math:`x_2`-direction sweep, and the electron–electron coupling
block are each handled by a tridiagonal Crank–Nicolson solve along
the relevant axis, in a symmetric ABCBA pattern around the
time midpoint. Every sweep reduces to the inversion of a tridiagonal
matrix of dimension :math:`N_x` or :math:`N_y`, so the cost per
timestep is :math:`O(N_x N_y)`. The right-hand-side build loops
along disjoint rows/columns are parallelised across cores with
OpenMP — ``OMP_NUM_THREADS`` at run time controls the thread count.

The same routine drives both real-time propagation and the
imaginary-time relaxation described below; only the timestep
argument changes from real to imaginary, and the Hamiltonian (with
or without the laser term) is selected accordingly.


Pulse shapes
------------

The vector potential :math:`A(t)` supports three shapes, selected
via the ``laser_pulse_shape`` config key:

* **Sinusoidal** (sin² envelope):
  :math:`A(t) = \frac{\alpha}{\omega}\,\sin^2(\omega t/2N_c)
  \sin(\omega t - \varphi)` spanning :math:`N_c` optical cycles.
* **Trapezoidal**: a linear ramp-up of :math:`N_{\rm up}` cycles, a
  plateau of :math:`N_{\rm plat}` cycles at full amplitude, then a
  linear ramp-down of :math:`N_{\rm down}` cycles to zero. Used by
  the paper-reproduction workflow because it is the canonical shape
  for studying Floquet steady-state behaviour during the plateau.
* **Kick**: :math:`A(t) = A_0` identically — the electric field is
  a Dirac delta at :math:`t=0`, used to extract the linear-response
  spectrum from the post-kick free evolution.

The carrier-envelope phase :math:`\varphi` is exposed via
``laser_phi``. The config field ``laser_alpha`` is the
electric-field amplitude :math:`E_0 = \alpha`; the C++ kernel
computes the vector-potential amplitude in velocity gauge as
:math:`A_0 = E_0 / \omega`, the standard identity for a sinusoidal
carrier :math:`A(t) = A_0 \sin(\omega t)` giving
:math:`E(t) = -A_0 \omega \cos(\omega t)`. This matches the
convention of [Kapoor2013Periodicity]_.

The simulated propagation duration is computed automatically from
the pulse cycle counts so that the laser stops exactly when the
pulse ends; users see and can override the resulting ``real_steps``
count in the GUI.


Eigenvalue equations via imaginary-time propagation
---------------------------------------------------

The time-independent Schrödinger equation
:math:`H \phi_n = E_n \phi_n` is solved without ever forming or
diagonalising :math:`H`. Replacing :math:`t \to -i\tau` in the TDSE
turns the unitary evolution into a contraction [Lehtovaara2007]_:

.. math::

   \psi(\tau) = e^{-H \tau}\, \psi(0)
              = \sum_{n} c_n e^{-E_n \tau}\, \phi_n,

where :math:`\{\phi_n\}` is the (unknown) eigenbasis of :math:`H`.
As :math:`\tau \to \infty` every excited component decays faster
than the ground-state component, so the renormalised state
collapses onto the ground state:

.. math::

   \phi_0 = \lim_{\tau\to\infty}
            \frac{e^{-H\tau} \psi(0)}{\| e^{-H\tau} \psi(0) \|}.

In practice ``latom`` propagates :math:`\psi` with the same
Crank–Nicolson scheme using a purely imaginary timestep
:math:`\Delta t = -i\,\Delta\tau`, renormalising :math:`\psi` after
every step. Convergence is monitored through the Rayleigh quotient
:math:`\langle \psi | H | \psi \rangle`, which decreases
monotonically to :math:`E_0`. The number of steps :math:`N_{\tau}`
and the imaginary timestep :math:`\Delta \tau` are exposed as
``imag_steps`` and ``imag_dt`` in the configuration file.


A duality worth highlighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above Wick rotation works because of an unusual feature of
quantum mechanics that is easy to take for granted: *the same*
Hermitian operator :math:`H` both labels the eigenvalue equation
that we want to solve, :math:`H\phi_n = E_n \phi_n`, and generates
the unitary time-evolution that we know how to integrate,
:math:`i\partial_t \psi = H \psi`. Substituting :math:`t \to -i\tau`
analytically continues the evolution operator
:math:`e^{-iHt/\hbar}` to the contraction :math:`e^{-H\tau}`, which
on a single Hermitian operator is well-defined and damps every
eigenvector exponentially in proportion to its eigenvalue.

This duality is *not* generic. Most eigenvalue problems in
scientific computing — graph Laplacians, structural-mechanics
stiffness matrices, the Helmholtz equation, kernel matrices in
machine learning — do not come paired with a natural
time-dependent partner equation that can be analytically continued
in this way; one usually reaches for Krylov-subspace solvers
(Lanczos, Davidson, LOBPCG) or preconditioned conjugate-gradient
iteration on the static problem itself. It is precisely because
the Schrödinger equation supplies *both* a static and a dynamic
equation built from a single Hermitian :math:`H` that the
imaginary-time trick reduces the eigenvalue problem to "run the
same propagator with a different timestep."

``latom`` exploits this duality directly: a single Crank–Nicolson
routine handles both the laser-driven dynamics and the ground-state
solve. Maintaining one propagator instead of two reduces the
surface area of the code, eliminates a class of bugs that plague
separate imaginary- and real-time implementations, and lets every
numerical improvement (operator splitting, OpenMP parallelism,
regridding) flow automatically to both regimes.


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

Higher excited states converge more slowly because their relative
gap to neighbouring states shrinks; ``latom`` accordingly multiplies
the imaginary-step budget by :math:`N \cdot` ``excited_imag_mult``
for the :math:`N`-th excited state. Converged :math:`\phi_n` are
dumped to ``wf_excited_<N>.dat`` and can be reloaded to resume work
or to seed a real-time propagation from a non-ground initial state.


Exact-TDDFT: reverse-engineering the KS orbital
-----------------------------------------------

In Exact-TDDFT mode ``latom`` simultaneously propagates the exact
two-electron wavefunction and reconstructs the corresponding KS
orbital. For the spin-singlet two-electron system, the KS orbital
is related to the exact density and current by
[Kapoor2013Periodicity]_

.. math::

   \varphi_{\rm KS}(x, t) = \sqrt{n(x, t)/2}\,e^{i \theta(x, t)},
   \qquad
   \partial_x \!\left[n(x, t)\,\partial_x \theta(x, t)\right]
   = -\partial_t n(x, t),

where :math:`n(x, t) = 2 \int dx'\,|\psi(x, x', t)|^2` is the
one-body density and :math:`\theta` is the phase implied by the
continuity equation. The effective KS potential
:math:`v_{\rm KS}(x, t)` is then extracted from a forward / backward
half-step of :math:`\varphi_{\rm KS}` under a bare-kinetic
Hamiltonian (:math:`H_{\rm bare} = -\tfrac{1}{2}\partial_x^2`) and
the identification

.. math::

   v_{\rm KS}(x, t) = \frac{i}{\Delta t}
      \log\!\left[\,\varphi_{+}(x, t)/\varphi_{-}(x, t)\,\right],

where :math:`\varphi_{\pm}` are the half-step images of
:math:`\varphi_{\rm KS}`. All three building blocks
(``denskohnsham``, ``denskohnshamcorrector``,
``phasekohnshamorbital``) live in the shared ``wavefunction.cc``
library and are reused between the TDSE and Exact-TDDFT binaries.


Online :math:`v_{\rm KS}(x, \omega)` Fourier transform
------------------------------------------------------

The classic post-processing route — dump :math:`v_{\rm KS}`
snapshots to disk every :math:`N_{\rm snap}` steps and FFT them in
Python — introduces aliasing whenever the snapshot cadence
:math:`\Delta t \cdot N_{\rm snap}` exceeds :math:`\pi/\omega_L`.
For short-wavelength laser fields (:math:`\omega_L \sim 2.6` a.u.
in the paper's high-frequency case), the resulting Nyquist falls
below the laser carrier itself and harmonic peaks vanish from the
spectrum.

``latom`` instead accumulates the Fourier integral *online* inside
the real-time loop. For each spatial grid point :math:`x` and each
frequency :math:`\omega_j` in a user-chosen grid we evaluate the
discrete sum

.. math::

   \hat v_{\rm KS}(x, \omega_j)
   = \int_0^T v_{\rm KS}(x, t)\, w(t)\, e^{-i\omega_j t}\, dt
   \;\approx\; \sum_{n=0}^{N-1} v_{\rm KS}(x, n\Delta t)\,
              w(n\Delta t)\, e^{-i\omega_j\, n\Delta t}\, \Delta t,

for :math:`j = 0, 1, \dots, N_\omega - 1`, where :math:`\Delta t` is
the *simulation* timestep (``real_dt``), :math:`N \cdot \Delta t = T`
is the total propagation duration, and
:math:`w(t) = \tfrac{1}{2}(1 - \cos(2\pi t / T))` is a Hann window
applied to the integrand inside the C++ accumulator so the dumped
spectrum is leakage-cleaned at source.

**Streamed loop ordering.** The double sum can be evaluated with
either index outermost. The standard offline DFT ordering would
store the full time series and then sweep frequencies::

    for j in 0..N_ω:                       # outer = frequency
        Fhat[j] = 0
        for n in 0..N:                     # inner = time
            Fhat[j] += V(x, n·Δt) · w(n·Δt) · exp(-i ω_j n Δt) · Δt

The same arithmetic with the loops swapped — what ``latom`` does — is::

    for n in 0..N:                         # outer = the sim's time loop
        for j in 0..N_ω:                   # inner = frequency
            Fhat[j] += V(x, n·Δt) · w(n·Δt) · exp(-i ω_j n Δt) · Δt

Because the simulation already iterates time step-by-step (Crank–
Nicolson is intrinsically a time loop), hooking the FFT accumulator
into that loop is free — at every step, before ``realpot`` is
discarded, its contribution is added to *every* frequency bin of
:math:`\hat v_{\rm KS}`. The ``j``-loop is parallelised across
cores with OpenMP. The total floating-point cost,
:math:`\mathcal{O}(N \cdot N_\omega \cdot N_x)`, is identical to the
offline ordering.

The streamed ordering wins on three practical counts:

* **Memory is constant in** :math:`N`. Only the accumulator
  :math:`\hat V[N_x \times N_\omega]` lives in RAM —
  :math:`N_x \cdot N_\omega \cdot 16` bytes (≈ 12 MB for the
  defaults) — *regardless of how many timesteps the simulation runs*.
  The full time series is never materialised; in the offline
  ordering it would be :math:`N \cdot N_x` complex doubles (e.g.
  270 MB for the paper's case 1).
* **The frequency grid is arbitrary.** ``np.fft.rfft`` produces
  frequencies :math:`\omega_k = 2\pi k/(N\Delta t)` fixed by the
  record length. Our :math:`\{\omega_j\}` is whatever we like —
  uniform between user-set bounds, log-spaced, or densely packed
  around the harmonics of interest.
* **No snapshot-cadence aliasing.** The integrand is sampled at the
  simulation's own timestep :math:`\Delta t`, whose Nyquist
  :math:`\pi / \Delta t` sits far above any laser frequency of
  interest (:math:`\sim 31.4` a.u. for :math:`\Delta t = 0.1`).
  Whether the user asks for periodic disk snapshots or not is
  immaterial to the FFT.

**Choosing** :math:`N_\omega`. The grid size is a *sampling*
choice, not a discovery — it does not require knowing where the
peaks lie. The intrinsic linewidth of any spectral peak is set by
the simulation duration:
:math:`\Delta \omega_{\rm Fourier} = 2\pi/T`. Picking the bin
spacing :math:`d\omega = (\omega_{\max} - \omega_{\min})/
(N_\omega - 1)` small enough to resolve that linewidth — typically
:math:`N_\omega \gtrsim k \cdot (\omega_{\max} - \omega_{\min})\,
T/(2\pi)` with :math:`k \sim 3`–:math:`5` — guarantees every peak
is sampled by several bins. Increasing :math:`N_\omega` further is
harmless (just more memory in :math:`\hat V`); reducing it broadens
the apparent peak shape but does *not* introduce false frequencies
(the time-domain Nyquist :math:`\pi/\Delta t` is set by the
simulation step, not by :math:`N_\omega`).

At simulation end the raw power
:math:`|\hat v_{\rm KS}(x, \omega_j)|^2` is dumped to ``vks_fft.dat``
for plotting. The Python plotter does *no further FFT* — it just
reads the file and renders.


Spectral projection: the same trick, applied to :math:`\psi(t)`
---------------------------------------------------------------

The streaming Fourier-projection accumulator is not unique to the
KS-potential FFT — it is the *same* trick that underlies the
Feit–Fleck–Steiger spectral method [FleckFeitSteiger1982]_ used by
the **Autoionization Computer** workflow. The two are dual flavours
of a single operation, hooked into the same Crank–Nicolson
real-time loop:

============================ ==================================== =====================================
Aspect                       FFT of :math:`v_{\rm KS}`            Feit–Fleck–Steiger
============================ ==================================== =====================================
Sampled at each step         scalar :math:`v_{\rm KS}(x, t_n)`    full wavefunction :math:`\psi(t_n)`
Frequencies probed           grid :math:`\{\omega_j\}` of         one target energy
                             :math:`N_\omega` values              :math:`E_{\rm target}`
Sign in the kernel           :math:`e^{-i \omega_j t_n}`          :math:`e^{+i E_{\rm target} t_n}`
Window :math:`w(t)`          Hann                                 Hann
Output                       spectrum                              normalised eigenstate
                             :math:`|\hat v_{\rm KS}(x,\omega)|^2` :math:`\phi_{k^*}`
Cost per timestep            :math:`O(N_x \cdot N_\omega)`         :math:`O(N_x \cdot N_y)`
============================ ==================================== =====================================

For the Feit–Fleck–Steiger projection, expanding
:math:`\psi(t) = \sum_k c_k\,e^{-i E_k t}\,\phi_k` in the field-free
eigenbasis and substituting into

.. math::

   \psi_{\rm proj}(T) = \int_0^T w(t)\, e^{+i E_{\rm target} t}\,
                              \psi(t)\, dt

selects the term :math:`E_k = E_{\rm target}` in the sum:
:math:`\psi_{\rm proj}/\|\psi_{\rm proj}\| \to \phi_{k^*}` as
:math:`T \to \infty`. The Hann window damps neighbouring-eigenstate
leakage. So extracting an eigenstate at energy :math:`E` and
dumping a Fourier spectrum at frequencies :math:`\omega` are *the
same operation* — only what gets weighted (a wavefunction vs a
scalar observable) and which :math:`\omega`'s are scanned (one vs
many) differ. ``latom`` implements both as a single pattern
wrapped around the time loop: multiply the integrand by
:math:`\exp(\pm i\omega t)\, w(t)\, \Delta t` and add to a running
accumulator.

Combined with the imaginary-time eigenvalue extraction discussed
earlier, ``latom`` therefore reuses the *same* Crank–Nicolson
kernel to do three apparently different things — propagate under a
laser field, find the lowest eigenstate by contraction, and project
onto an arbitrary eigenstate by streaming Fourier — each obtained
by choosing what the time loop multiplies its iterate by.


Implementation
==============

The C++ kernel (``TDSE.cc``, ``ExactTDDFT.cc``, ``wavefunction.cc``,
``hamop.cc``, ``grid.cc``, ``fluid.cc``) builds with a single
``make`` call under any modern GCC/Clang and OpenMP. Two binaries
are produced: ``TDSE`` for the basic 2e Schrödinger workflow, and
``ExactTDDFT`` for the KS-reconstruction and online-FFT workflow.
The eight Crank–Nicolson sweeps in the 2D propagator are
OpenMP-parallel; the online :math:`v_{\rm KS}` FFT loop is too.

The PyQt5 GUI (``gui/main_window.py``) presents five workflow tabs:

1. **TDSE** — propagate :math:`\psi(x_1, x_2, t)` under a chosen
   pulse.
2. **Exact-TDDFT** — additionally reconstruct
   :math:`\varphi_{\rm KS}` and :math:`v_{\rm KS}` each step plus
   the online FFT.
3. **Reproduce Periodicity Paper** — preset cards for the three
   trapezoidal cases of [Kapoor2013Periodicity]_ (low-:math:`\omega`,
   high-:math:`\omega`, resonant), launched in parallel from a
   single shared imaginary-time preflight.
4. **Autoionization Computer** — Feit–Fleck–Steiger spectral
   projection at a target energy [FleckFeitSteiger1982]_.
5. **Kick / Linear Response** — constant-:math:`A_0` pulse for
   absorption spectroscopy.

Each tab is self-contained: only the parameters that the active
workflow actually consumes are exposed, and switching tabs flips a
``mode`` field that selects the right binary. A live progress bar
tied to the simulation's elapsed propagation time replaces the
legacy log textbox. When a previous run's output files are present
on disk the GUI offers to load them instead of recomputing — useful
for adjusting plot style or dynamic range without re-running.

When the user changes ``grid_nx``/``grid_ny`` between runs but
still asks to load a cached ground state, ``latom`` infers the
smaller-grid file size from line counts and instructs the C++
binary to read at that small grid and *regrid* onto the current
larger grid via the ``wavefunction::regrid`` routine, rather than
silently corrupting the initial state.


Availability and reproducibility
================================

``latom`` is distributed under an open-source licence and is
available at the project repository. The C++ kernel builds with a
single ``make -j -B all`` invocation under any modern GCC/Clang
with OpenMP; the Python front-end requires PyQt5, ``numpy``,
``matplotlib``, ``scipy``, and ``h5py``. Configuration files used
to produce every figure in this paper are shipped under
``gui/experiment_*`` and are loaded with one click from the GUI. A
regression test (``tests/test_fft_sanity.py``) verifies that the
FFT pipeline correctly resolves a synthetic three-harmonic signal
at :math:`\omega/\omega_L = 1, 2, 3` with the expected
:math:`1:0.09:0.01` amplitude ratios.


References
==========

.. [Crank1947] J. Crank and P. Nicolson,
   "A practical method for numerical evaluation of solutions of
   partial differential equations of the heat-conduction type",
   *Math. Proc. Camb. Philos. Soc.* **43**, 50 (1947).

.. [FleckFeitSteiger1982] M. D. Feit, J. A. Fleck Jr., and A. Steiger,
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
