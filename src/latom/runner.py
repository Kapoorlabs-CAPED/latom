"""Subprocess manager for building and running the C++ TDSE solver."""

import subprocess
from pathlib import Path


def get_solver_dir():
    """Return the path to the C++ solver source directory.

    Returns:
        Path to the solver directory containing Makefile and sources.
    """
    return Path(__file__).parent / "solver"


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

    # Create output directory
    output_path = work_dir / config.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Write config file
    config_path = work_dir / "simulation.cfg"
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
