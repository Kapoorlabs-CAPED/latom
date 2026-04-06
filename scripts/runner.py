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


def is_solver_built(solver_dir=None):
    """Check if the TDSE binary exists.

    Args:
        solver_dir: Path to solver directory. Uses default if None.

    Returns:
        True if the binary exists.
    """
    if solver_dir is None:
        solver_dir = get_solver_dir()
    return (Path(solver_dir) / "TDSE").is_file()


def is_solver_stale(solver_dir=None):
    """Check if any source file is newer than the TDSE binary.

    Args:
        solver_dir: Path to solver directory. Uses default if None.

    Returns:
        True if the binary is out of date and needs rebuilding.
    """
    if solver_dir is None:
        solver_dir = get_solver_dir()
    solver_dir = Path(solver_dir)
    binary = solver_dir / "TDSE"
    if not binary.is_file():
        return True
    binary_mtime = binary.stat().st_mtime
    for ext in ("*.cc", "*.h"):
        for src in solver_dir.glob(ext):
            if src.stat().st_mtime > binary_mtime:
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

    proc = subprocess.Popen(
        ["make", "-j4"],
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

    # Write config file, overriding output_dir to "." since cwd is work_dir
    config_path = work_dir / "simulation.cfg"
    config.output_dir = "."
    config.write_ini(config_path)

    # Find the TDSE binary
    tdse_bin = solver_dir / "TDSE"
    if not tdse_bin.is_file():
        raise FileNotFoundError(
            f"TDSE binary not found at {tdse_bin}. Run build_solver() first."
        )

    proc = subprocess.Popen(
        [str(tdse_bin), str(config_path)],
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
