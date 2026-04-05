"""Parsers for C++ solver output files."""

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

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
        pandas DataFrame with columns from IMAG_COLUMNS.
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
    return pd.DataFrame(data[:, :ncols], columns=IMAG_COLUMNS[:ncols])


def parse_real_observables(filepath):
    """Parse real-time observable output file.

    Args:
        filepath: Path to the obser_laser.dat file.

    Returns:
        pandas DataFrame with columns from REAL_COLUMNS.
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
    return pd.DataFrame(data[:, :ncols], columns=REAL_COLUMNS[:ncols])


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


def collect_wf_to_hdf5(output_dir, nx, ny, h5_path, real_dt=None):
    """Collect all wavefunction .dat files into a single HDF5 file.

    Reads the ground state, excited states, and all real-time snapshots
    from the output directory and stores them in a structured HDF5 file.

    Args:
        output_dir: Path to the directory containing wavefunction .dat files.
        nx: Number of grid points in x.
        ny: Number of grid points in y.
        h5_path: Output path for the HDF5 file.
        real_dt: Real time step (a.u.) for computing snapshot times.
            If None, only step indices are stored.

    Returns:
        Number of datasets written.
    """
    output_dir = Path(output_dir)
    h5_path = Path(h5_path)
    count = 0

    with h5py.File(str(h5_path), "w") as hf:
        # Store grid metadata
        hf.attrs["nx"] = nx
        hf.attrs["ny"] = ny
        if real_dt is not None:
            hf.attrs["real_dt"] = real_dt

        # Ground state
        wf_ground = parse_wavefunction(output_dir / "wf_ground.dat", nx, ny)
        if wf_ground is not None:
            grp = hf.create_group("ground_state")
            grp.create_dataset("real", data=wf_ground.real)
            grp.create_dataset("imag", data=wf_ground.imag)
            count += 1

        # Excited states
        exc_idx = 1
        while True:
            exc_path = output_dir / f"wf_excited_{exc_idx}.dat"
            wf_exc = parse_wavefunction(exc_path, nx, ny)
            if wf_exc is None:
                break
            grp = hf.create_group(f"excited_state_{exc_idx}")
            grp.create_dataset("real", data=wf_exc.real)
            grp.create_dataset("imag", data=wf_exc.imag)
            count += 1
            exc_idx += 1

        # Real-time snapshots
        snapshots = find_wf_snapshots(output_dir)
        if snapshots:
            rt_grp = hf.create_group("real_time")
            step_indices = []
            for ts_idx, fpath in snapshots:
                wf = parse_wavefunction(fpath, nx, ny)
                if wf is None:
                    continue
                label = "final" if ts_idx == -1 else f"step_{ts_idx:06d}"
                ds_grp = rt_grp.create_group(label)
                ds_grp.create_dataset("real", data=wf.real)
                ds_grp.create_dataset("imag", data=wf.imag)
                ds_grp.attrs["step_index"] = ts_idx
                if real_dt is not None and ts_idx >= 0:
                    ds_grp.attrs["time_au"] = ts_idx * real_dt
                step_indices.append(ts_idx)
                count += 1
            rt_grp.attrs["n_snapshots"] = len(step_indices)

    return count


def load_wf_from_hdf5(h5_path, group="ground_state"):
    """Load a wavefunction from an HDF5 file.

    Args:
        h5_path: Path to the HDF5 file created by collect_wf_to_hdf5.
        group: HDF5 group path, e.g. 'ground_state', 'excited_state_1',
            or 'real_time/step_000200'.

    Returns:
        Complex numpy array of shape (nx, ny), or None if not found.
    """
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        return None

    with h5py.File(str(h5_path), "r") as hf:
        if group not in hf:
            return None
        grp = hf[group]
        return np.array(grp["real"]) + 1j * np.array(grp["imag"])


def list_hdf5_snapshots(h5_path):
    """List all real-time snapshot keys and their metadata from an HDF5 file.

    Args:
        h5_path: Path to the HDF5 file.

    Returns:
        List of dicts with keys 'group', 'step_index', 'time_au' (if available).
        Returns empty list if file doesn't exist or has no snapshots.
    """
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        return []

    entries = []
    with h5py.File(str(h5_path), "r") as hf:
        # Ground state
        if "ground_state" in hf:
            entries.append(
                {
                    "group": "ground_state",
                    "step_index": -2,
                    "label": "Ground State",
                }
            )

        # Excited states
        exc_idx = 1
        while f"excited_state_{exc_idx}" in hf:
            entries.append(
                {
                    "group": f"excited_state_{exc_idx}",
                    "step_index": -2,
                    "label": f"Excited State {exc_idx}",
                }
            )
            exc_idx += 1

        # Real-time snapshots
        if "real_time" in hf:
            rt_grp = hf["real_time"]
            for key in sorted(rt_grp.keys()):
                sub = rt_grp[key]
                entry = {
                    "group": f"real_time/{key}",
                    "step_index": int(sub.attrs.get("step_index", -1)),
                    "label": key,
                }
                if "time_au" in sub.attrs:
                    entry["time_au"] = float(sub.attrs["time_au"])
                entries.append(entry)

    return entries
