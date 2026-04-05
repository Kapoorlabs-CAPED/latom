"""Parsers for C++ solver output files."""

import re
from pathlib import Path

import numpy as np

# Column names matching the fprintf format strings in TDSE.cc

IMAG_COLUMNS = [
    "step",  # %li
    "energy_real",  # %.14le
    "energy_imag",  # %.14le
    "energy_real2",  # %.14le (duplicate)
    "norm",  # %.14le
    "expect_x",  # %.14le
    "expect_y",  # %.14le
    "doub_ionized",  # %.14le
    "expect_x2",  # %.14le (duplicate)
]

REAL_COLUMNS = [
    "time",  # %.14le
    "energy_real",  # %.14le
    "energy_imag",  # %.14le
    "norm",  # %.14le
    "expect_x",  # %.14le
    "expect_y",  # %.14le
    "vecpot_x",  # %.14le
    "ground_pop",  # %.14le  |<ground|psi>|^2
]


def parse_imag_observables(filepath):
    """Parse imaginary-time observable output file.

    Args:
        filepath: Path to the obserimag_laser.dat file.

    Returns:
        dict with keys from IMAG_COLUMNS, values are numpy arrays.
        Returns None if file doesn't exist or is empty.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    try:
        data = np.loadtxt(str(filepath))
    except (ValueError, OSError):
        return None

    if data.ndim == 1:
        if data.size == 0:
            return None
        data = data.reshape(1, -1)

    ncols = min(data.shape[1], len(IMAG_COLUMNS))
    result = {}
    for i in range(ncols):
        result[IMAG_COLUMNS[i]] = data[:, i]
    return result


def parse_real_observables(filepath):
    """Parse real-time observable output file.

    Args:
        filepath: Path to the obser_laser.dat file.

    Returns:
        dict with keys from REAL_COLUMNS, values are numpy arrays.
        Returns None if file doesn't exist or is empty.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    try:
        data = np.loadtxt(str(filepath))
    except (ValueError, OSError):
        return None

    if data.ndim == 1:
        if data.size == 0:
            return None
        data = data.reshape(1, -1)

    ncols = min(data.shape[1], len(REAL_COLUMNS))
    result = {}
    for i in range(ncols):
        result[REAL_COLUMNS[i]] = data[:, i]
    return result


def parse_wavefunction(filepath, nx, ny):
    """Parse wavefunction output file into 2D complex array.

    The C++ solver writes pairs of (real, imag) values in row-major
    order: outer loop over x (ngpsx), inner loop over y (ngpsy).

    Args:
        filepath: Path to the wf_laser.dat file.
        nx: Number of grid points in x.
        ny: Number of grid points in y.

    Returns:
        Complex numpy array of shape (nx, ny).
        Returns None if file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    try:
        data = np.loadtxt(str(filepath))
    except (ValueError, OSError):
        return None

    if data.ndim == 1:
        data = data.reshape(-1, 2)

    wf = data[:, 0] + 1j * data[:, 1]
    expected = nx * ny
    if wf.size < expected:
        return None

    return wf[:expected].reshape(nx, ny)


def find_wf_snapshots(output_dir):
    """Find all real-time wavefunction snapshot files in output directory.

    Looks for files matching wf_real_NNNNNN.dat (periodic dumps)
    and wf_real_final.dat.

    Args:
        output_dir: Path to the output directory.

    Returns:
        Sorted list of (timestep_index, filepath) tuples.
    """
    output_dir = Path(output_dir)
    snapshots = []

    for f in sorted(output_dir.glob("wf_real_*.dat")):
        if f.name == "wf_real_final.dat":
            continue
        m = re.match(r"wf_real_(\d+)\.dat", f.name)
        if m:
            snapshots.append((int(m.group(1)), f))
    # Append final if it exists
    final = output_dir / "wf_real_final.dat"
    if final.is_file():
        snapshots.append((-1, final))
    return snapshots


def tail_observables(filepath, last_n=100):
    """Read the last N lines of an observable file (for live updates).

    Args:
        filepath: Path to the observable file.
        last_n: Number of trailing lines to read.

    Returns:
        numpy array of shape (last_n, ncols) or None.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        return None

    try:
        lines = filepath.read_text().strip().split("\n")
    except OSError:
        return None

    if not lines or lines == [""]:
        return None

    lines = lines[-last_n:]
    rows = []
    for line in lines:
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue

    if not rows:
        return None

    return np.array(rows)
