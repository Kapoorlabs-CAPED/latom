# latom

[![License BSD-3](https://img.shields.io/pypi/l/latom.svg?color=green)](https://github.com/Kapoorlabs-CAPED/latom/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/latom.svg?color=green)](https://pypi.org/project/latom)
[![Python Version](https://img.shields.io/pypi/pyversions/latom.svg?color=green)](https://python.org)

A quantum mechanical solver for low-dimensional atomic systems. Solves the time-dependent Schrodinger equation (TDSE) for a 1D model of Helium (two electrons, each in one dimension) using the Crank-Nicolson propagator. Includes a PyQt5 GUI for parameter configuration and matplotlib-based visualization.

## What the code does

### Ground state via imaginary time propagation

Propagates an initial random wavefunction in imaginary time (t -> -i*tau). Higher-energy components decay exponentially faster, so after enough steps the wavefunction converges to the ground state. The wavefunction is renormalized after each step.

### Excited states via Gram-Schmidt orthogonalization

After computing the ground state, higher excited states are obtained one at a time. For the n-th excited state:
1. Start with a new random wavefunction
2. Propagate in imaginary time
3. After each step, project out all lower converged states (Gram-Schmidt)
4. Renormalize

This gives the first N spin-singlet eigenstates of the Hamiltonian.

### Real time propagation with laser field

The converged ground state is propagated in real time under an external laser field in the **velocity gauge**. The vector potential A(t) couples to the electrons via the A*p term in the kinetic energy. The code records observables at each timestep and dumps wavefunction snapshots at configurable intervals.

### Autoionizing state extraction (Feit-Fleck-Steiger method)

Extracts doubly-excited autoionizing states embedded in the continuum using the spectral method of Feit, Fleck, and Steiger. The idea:

1. Start from a converged bound state (ground or excited)
2. Propagate in real time with **zero laser field** (field-free evolution)
3. At each timestep, accumulate the Fourier projection at the target energy E:
   ```
   psi_proj(T) = integral_0^T  W(t) * exp(i*E*t) * psi(t) dt
   ```
4. The Hann window W(t) = 0.5*(1 - cos(2*pi*t/T)) suppresses spectral leakage
5. After propagation, normalize psi_proj to obtain the eigenstate at energy E

This method can resolve autoionizing resonances that are inaccessible via imaginary time propagation because they lie above the single-ionization threshold. It also works for any excited state whose energy is known (e.g., from a linear response spectrum — see below).

Enable with `auto_mode = 1` and set `auto_target_energy` to the desired energy in atomic units.

### Linear response spectrum (kick mode)

Computes the absorption spectrum of the system by applying a velocity-gauge impulse ("kick") and analyzing the subsequent field-free dynamics:

1. Converge the ground state via imaginary time propagation
2. Apply a constant vector potential A(t) = A_0 (equivalent to a Dirac delta E-field kick)
3. Propagate in real time — the constant A couples to both electrons via A*p
4. Record the dipole d(t) = <x1>(t) + <x2>(t) at each timestep
5. Fourier transform d(t) to obtain the absorption spectrum |d(omega)|^2

Peaks in the spectrum appear at transition frequencies omega_n = E_n - E_0. Adding the ground state energy E_0 to each peak gives the absolute energy of the excited state. These energies can then be used as `auto_target_energy` to extract the corresponding wavefunctions via the Feit-Fleck-Steiger method.

Enable with `kick_mode = 1` and set `kick_strength` to a small value (default 0.01 a.u.) to stay in the linear response regime.

### 1D Helium energy levels (soft-Coulomb, eps=1.0)

Reference values for the 1D model Helium atom with soft-Coulomb potentials (from [Kapoor, Phys. Rev. A 93, 063408, 2016](https://doi.org/10.1103/PhysRevA.93.063408)):

| State | Energy (a.u.) | Energy (eV) | Notes |
|-------|--------------|-------------|-------|
| Ground state (1s^2) | -2.238 | -60.90 | Spin-singlet, obtainable via imaginary time |
| First excited singlet | -1.705 | -46.40 | Via Gram-Schmidt or auto mode |
| He+ threshold | -0.884 | -24.06 | Single ionization threshold |
| AI_1 (autoionizing) | -0.884 | -24.06 | Lowest doubly-excited, just above He+ threshold |
| AI_2 (autoionizing) | -0.816 | -22.20 | Second doubly-excited state |
| AI_3 (autoionizing) | -0.538 | -14.64 | Above second ionization threshold |

**Workflow for autoionizing states:**
1. Run imaginary time to get ground state at E_0 = -2.238 a.u.
2. Run kick mode to get absorption spectrum
3. Identify peak at omega_n, compute E_n = E_0 + omega_n
4. Run auto mode with `auto_target_energy = E_n`

### Crank-Nicolson split-operator scheme

The 2D Hamiltonian is split into x- and y-direction kinetic operators, single-particle potentials, and the electron-electron interaction. Each direction is solved implicitly via tridiagonal matrix inversion (Thomas algorithm). The interaction potential exp(-i*dt*V_ee) is applied as a multiplicative phase.

### Observables

- Total energy (real and imaginary parts)
- Dipole expectation values (electron 1 and electron 2 positions)
- Wavefunction norm
- Ground state population |<psi_0|psi(t)>|^2
- Vector potential A(t)
- HHG spectrum (FFT of dipole acceleration)
- Single and double ionization probabilities

### Visualization

- 2D wavefunction |psi(x1,x2)|^2 as log-scale heatmap (replaces legacy IDL scripts)
- Energy convergence during imaginary time
- Real-time dashboard: energy, dipole, ionization, vector potential, spectrum
- Animated GIF of wavefunction time evolution
- 1D marginal density profiles

## The Hamiltonian

```
H = T_x + T_y + V_x + V_y + V_ee

T_mu = -1/2 * d^2/dx_mu^2                    (kinetic energy)
V_x  = -Z / sqrt(x^2 + eps^2)                (electron-nucleus Coulomb, Z=2)
V_y  = -Z / sqrt(y^2 + eps^2)                (electron-nucleus Coulomb, Z=2)
V_ee = 1 / sqrt((x-y)^2 + eps^2)             (electron-electron repulsion)
```

In the velocity gauge with laser field:
```
T_mu -> T_mu + A(t)*p_mu + A(t)^2/2
```

The softening parameter eps (default 1.0 a.u.) regularizes the Coulomb singularity in 1D.

## Atomic units conversion table

All quantities in the code are in **atomic units** (a.u.). Here are the key conversions:

### Fundamental constants in atomic units

| Quantity | a.u. value | SI value |
|----------|-----------|----------|
| Electron mass m_e | 1 | 9.109 x 10^-31 kg |
| Elementary charge e | 1 | 1.602 x 10^-19 C |
| Reduced Planck constant hbar | 1 | 1.055 x 10^-34 J*s |
| Bohr radius a_0 | 1 | 5.292 x 10^-11 m = 0.5292 A |
| Hartree energy E_h | 1 | 4.360 x 10^-18 J = 27.211 eV |
| Atomic time unit | 1 | 2.419 x 10^-17 s = 24.19 as |

### Wavelength / Frequency

| Wavelength (nm) | Frequency (a.u.) | Notes |
|-----------------|-------------------|-------|
| 800 | 0.0570 | Ti:Sapphire laser |
| 400 | 0.1140 | Second harmonic of 800 nm |
| 200 | 0.2279 | Deep UV |
| 1064 | 0.0428 | Nd:YAG |
| 10.6 um | 0.00428 | CO2 laser |

**Conversion formula:**
```
omega (a.u.) = 45.5634 / wavelength (nm)
wavelength (nm) = 45.5634 / omega (a.u.)
```

The code default frequency is 1.556 a.u., corresponding to ~29.3 nm (XUV range).

### Electric field strength / Intensity

| Intensity (W/cm^2) | E-field (a.u.) | A-field amplitude (a.u.) at 800 nm |
|--------------------|----------------|-------------------------------------|
| 10^12 | 0.00534 | 0.0937 |
| 10^13 | 0.01688 | 0.2962 |
| 10^14 | 0.05338 | 0.9366 |
| 10^15 | 0.16882 | 2.9618 |
| 10^16 | 0.53381 | 9.3658 |

**Conversion formulas:**
```
E_0 (a.u.) = sqrt(I (W/cm^2) / 3.51 x 10^16)
I (W/cm^2) = 3.51 x 10^16 * E_0^2 (a.u.)

A_0 (a.u.) = E_0 (a.u.) / omega (a.u.)
```

The atomic unit of electric field is E_h / (e * a_0) = 5.142 x 10^11 V/m.

### Energy

| Energy (a.u.) | eV | Notes |
|--------------|-----|-------|
| -2.904 | -79.01 | Helium ground state (exact) |
| -2.146 | -58.38 | He+ ground state (exact: -Z^2/2 = -2) |
| 0.0570 | 1.55 | 800 nm photon |
| 1.0 | 27.21 | 1 Hartree |

### Time

| Time (a.u.) | fs | Optical cycles at 800 nm |
|------------|-----|--------------------------|
| 1 | 0.02419 | 0.0218 |
| 41.34 | 1.0 | 0.902 |
| 110.3 | 2.669 | 1.0 (one cycle at 800 nm) |
| 1000 | 24.19 | 9.02 |

**Conversion:**
```
t (a.u.) = t (fs) / 0.02419
t (fs) = t (a.u.) * 0.02419
```

### Vector potential in velocity gauge

In the velocity gauge the laser-electron coupling is through A(t), not E(t). The relation:
```
E(t) = -dA/dt
```

For a monochromatic field E(t) = E_0 sin(omega*t):
```
A(t) = (E_0/omega) cos(omega*t) = A_0 cos(omega*t)
```

The code uses A(t) = A_0 * sin^2(pi*t/T) * sin(omega*t), where T is the total pulse duration and sin^2 provides a smooth envelope.

The `laser_alpha` parameter in the config is A_0/omega (the "alpha" or quiver amplitude), so:
```
A_0 = laser_alpha * laser_freq
E_0 = laser_alpha * laser_freq^2
I = 3.51e16 * (laser_alpha * laser_freq^2)^2  W/cm^2
```

## Installation

```bash
pip install latom
```

Development install:
```bash
git clone https://github.com/Kapoorlabs-CAPED/latom.git
cd latom
pip install -e .
```

Build the C++ solver:
```bash
cd src/latom/solver
make
```

## Usage

### GUI

```bash
python gui/main_window.py
```

### Command line (scripts)

Run a simulation:
```bash
cd scripts
python runner.py
```

Plot results:
```bash
python plotting.py
```

### Configuration

All parameters can be set via the GUI or a JSON config file. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| grid_nx, grid_ny | 1500 | Grid points per dimension |
| grid_dx, grid_dy | 0.2 | Grid spacing (a.u.) |
| imag_dt | 0.25 | Imaginary time step |
| imag_steps | 100 | Imaginary time steps for ground state |
| real_dt | 0.1 | Real time step |
| real_steps | 2000 | Real time propagation steps |
| laser_freq | 1.556 | Laser frequency (a.u.) |
| laser_alpha | 0.1 | Quiver amplitude E_0/omega^2 |
| laser_cycles | 40 | Number of optical cycles in pulse |
| n_excited | 0 | Number of excited states to compute |
| load_ground | 0 | Load ground state from file (1) or compute (0) |
| coulomb_eps | 1.0 | Coulomb softening parameter |
| absorb_ampl | 50.0 | Absorbing boundary strength |
| auto_mode | 0 | Enable autoionizing mode (Feit-Fleck-Steiger) |
| auto_target_energy | 0.0 | Target energy for spectral projection (a.u.) |
| auto_input_wf | "" | Input wavefunction file for auto mode |
| kick_mode | 0 | Enable kick mode (linear response spectrum) |
| kick_strength | 0.01 | Kick amplitude A_0 (a.u.) |

## Project structure

```
latom/
  src/latom/
    solver/          C++ source (TDSE.cc, wavefunction.cc, etc.)
  gui/               PyQt5 GUI (main_window.py, workers.py, etc.)
  scripts/           Standalone scripts (runner.py, plotting.py, etc.)
```

## License

BSD-3-Clause
