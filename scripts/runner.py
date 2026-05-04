"""Subprocess manager for building and running the C++ TDSE solver."""

import subprocess
from parser import collect_wf_to_hdf5
from pathlib import Path

from config import SimulationConfig

import latom


def get_solver_dir():
    """Return the path to the C++ solver source directory.

    Returns:
        Path to the solver directory containing Makefile and sources.
    """
    package_root = Path(latom.__file__).resolve().parent
    return package_root / "solver"


_BINARY_NAMES = {"tdse": "TDSE", "exact_tddft": "ExactTDDFT"}

# Per-binary source files. Headers + shared sources affect both binaries;
# the driver .cc only affects its own binary.
_SHARED_SOURCES = ("wavefunction.cc", "hamop.cc", "grid.cc", "fluid.cc")
_DRIVER_SOURCES = {"tdse": "TDSE.cc", "exact_tddft": "ExactTDDFT.cc"}


def _binary_name(mode):
    """Return the executable name for a solver mode."""
    try:
        return _BINARY_NAMES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown solver mode '{mode}'. Expected one of {list(_BINARY_NAMES)}."
        )


def is_solver_built(solver_dir=None, mode="tdse"):
    """Check if the solver binary for the given mode exists."""
    if solver_dir is None:
        solver_dir = get_solver_dir()
    return (Path(solver_dir) / _binary_name(mode)).is_file()


def is_solver_stale(solver_dir=None, mode="tdse"):
    """Check if any *relevant* source is newer than the binary for this mode.

    Only the sources that go into this binary count: the driver .cc, the
    shared .cc files, and any header. Editing the *other* mode's driver
    file does not make this binary stale.
    """
    if solver_dir is None:
        solver_dir = get_solver_dir()
    solver_dir = Path(solver_dir)
    binary = solver_dir / _binary_name(mode)
    if not binary.is_file():
        return True
    binary_mtime = binary.stat().st_mtime

    relevant = {_DRIVER_SOURCES[mode], *_SHARED_SOURCES}
    for name in relevant:
        p = solver_dir / name
        if p.is_file() and p.stat().st_mtime > binary_mtime:
            return True
    for h in solver_dir.glob("*.h"):
        if h.stat().st_mtime > binary_mtime:
            return True
    return False


def build_solver(solver_dir=None, on_output=None):
    """Compile the C++ solver using make.

    Args:
        solver_dir: Path to solver directory. Uses default if None.
        on_output: Optional callback(str) for each line of build output.

    Returns:
        True if build succeeded.

    Raises:
        RuntimeError: If compilation fails.
    """
    if solver_dir is None:
        solver_dir = get_solver_dir()
    solver_dir = Path(solver_dir)

    # `-B` forces every target to be remade unconditionally; one Build click
    # rebuilds both binaries from scratch, no staleness games. Don't use
    # `clean all` with -j: those targets race and can leave you with nothing.
    proc = subprocess.Popen(
        ["make", "-j4", "-B", "all"],
        cwd=str(solver_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if on_output:
            on_output(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Build failed (exit code {proc.returncode}):\n"
            + "\n".join(output_lines)
        )
    return True


def run_simulation(config, work_dir, solver_dir=None, on_output=None):
    """Run the TDSE solver with the given configuration.

    Args:
        config: SimulationConfig instance.
        work_dir: Working directory for output files.
        solver_dir: Path to solver directory. Uses default if None.
        on_output: Optional callback(str) for each line of stdout.

    Returns:
        subprocess.Popen handle (process may still be running).
    """
    if solver_dir is None:
        solver_dir = get_solver_dir()
    solver_dir = Path(solver_dir)
    work_dir = Path(work_dir)

    # Ensure work directory exists — solver writes all files here
    work_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale per-step snapshots from any prior run. The C++ writes
    # files indexed by timestep, but the cadence (wf_every / ks_every)
    # can differ between runs — without this cleanup, files from the
    # previous run at non-overlapping step numbers persist and pollute
    # the next FFT / live-snapshot view. Ground-state files (wf_ground,
    # wf_heliumplus, ks_ground) and observable logs are preserved.
    for pattern in (
        "wf_real_*.dat",
        "ks_real_*.dat",
        "realpot_real_*.dat",
    ):
        for old in work_dir.glob(pattern):
            try:
                old.unlink()
            except OSError:
                pass

    # Validate cached ground-state files against the requested grid. If
    # the user changed grid_nx/ny since the last run, ``wf_ground.dat``
    # (etc.) has the wrong number of lines for direct loading. Instead
    # of recomputing, we infer the *small* grid the file was built at
    # (assumed square for the 2D ground state, direct for the 1D files)
    # and tell the C++ binary to read at that small grid and regrid onto
    # the current grid via ``wavefunction::regrid``.
    nx = int(config.grid_nx)
    ny = int(config.grid_ny)
    config.regrid_from_nx_2e = 0
    config.regrid_from_nx_1e = 0

    p2e = work_dir / "wf_ground.dat"
    if p2e.is_file():
        try:
            with open(p2e, "rb") as f:
                lines = sum(1 for _ in f)
        except OSError:
            lines = nx * ny
        if lines != nx * ny:
            small = int(round(lines**0.5))
            if small * small == lines:
                config.regrid_from_nx_2e = small
                if on_output:
                    on_output(
                        f"wf_ground.dat is {small}×{small} (was computed "
                        f"on a smaller grid). Will regrid onto {nx}×{ny} "
                        f"via wavefunction::regrid at startup."
                    )
            else:
                msg = (
                    f"WARNING: wf_ground.dat has {lines} lines, not a "
                    f"square shape; cannot infer a regrid source. "
                    f"Renaming to wf_ground.dat.stale and recomputing."
                )
                if on_output:
                    on_output(msg)
                try:
                    p2e.rename(p2e.with_suffix(p2e.suffix + ".stale"))
                except OSError:
                    pass
                config.load_ground = 0

    # 1D ground orbitals (He+ and KS): line count = small_nx directly.
    for fname in ("wf_heliumplus.dat", "ks_ground.dat"):
        path = work_dir / fname
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                lines = sum(1 for _ in f)
        except OSError:
            continue
        if lines != nx:
            # All 1D auxiliary files share the same small grid — use the
            # *first* one we see to set regrid_from_nx_1e.
            if config.regrid_from_nx_1e == 0:
                config.regrid_from_nx_1e = lines
                if on_output:
                    on_output(
                        f"{fname} has length {lines} (smaller grid). "
                        f"Will regrid onto {nx} via wavefunction::regrid."
                    )

    # Excited-state files are not regridded (they're per-state, generally
    # not produced at multiple grids). Force-disable load_excited if the
    # current grid_nx/ny doesn't match any present excited file.
    for path in work_dir.glob("wf_excited_*.dat"):
        try:
            with open(path, "rb") as f:
                lines = sum(1 for _ in f)
        except OSError:
            continue
        if lines != nx * ny:
            if getattr(config, "load_excited", 0):
                if on_output:
                    on_output(
                        f"{path.name} grid mismatch ({lines} vs {nx*ny}); "
                        "disabling load_excited for this run."
                    )
                config.load_excited = 0
            try:
                path.rename(path.with_suffix(path.suffix + ".stale"))
            except OSError:
                pass

    # Write config file, overriding output_dir to "." since cwd is work_dir
    config_path = work_dir / "simulation.cfg"
    config.output_dir = "."
    config.write_ini(config_path)

    # Pick binary based on solver mode
    mode = getattr(config, "mode", "tdse")
    binary = solver_dir / _binary_name(mode)
    if not binary.is_file():
        raise FileNotFoundError(
            f"Solver binary not found at {binary}. Run build_solver() first."
        )

    proc = subprocess.Popen(
        [str(binary), str(config_path)],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return proc


if __name__ == "__main__":

    workdir = Path(__file__).resolve().parents[1] / "res"

    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Work directory set to {workdir}")
    config = SimulationConfig()
    config.write_ini(workdir / "test_config")
    build_solver()
    print("Starting simulation...")
    # Capture the process handle
    proc = run_simulation(config, workdir)

    # WAIT for it to finish and print output
    for line in proc.stdout:
        print(line.rstrip())

    proc.wait()
    print(f"Simulation finished with exit code {proc.returncode}")

    # Collect wavefunctions into HDF5
    h5_path = workdir / "wavefunctions.h5"
    n_written = collect_wf_to_hdf5(
        workdir,
        config.grid_nx,
        config.grid_ny,
        h5_path,
        real_dt=config.real_dt,
    )
    print(f"Collected {n_written} wavefunctions into {h5_path}")
