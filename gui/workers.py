"""Background QThread workers for build and simulation tasks."""

import sys
from pathlib import Path

# Add scripts directory to path for sibling imports — must run before local imports
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from PyQt5.QtCore import QThread, pyqtSignal  # noqa: E402
from runner import build_solver, run_simulation  # noqa: E402


class BuildWorker(QThread):
    """Worker thread that compiles the C++ solver.

    Signals:
        output(str): Emitted for each line of build output.
        finished(bool, str): Emitted when done (success, message).
    """

    output = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, solver_dir=None):
        super().__init__()
        self.solver_dir = solver_dir

    def run(self):
        try:
            build_solver(
                solver_dir=self.solver_dir,
                on_output=lambda line: self.output.emit(line),
            )
            self.finished.emit(True, "Build succeeded.")
        except RuntimeError as e:
            self.finished.emit(False, str(e))


class SimulationWorker(QThread):
    """Worker thread that runs the TDSE solver subprocess.

    Signals:
        output(str): Emitted for each line of solver stdout.
        finished(bool, str): Emitted when done (success, message).
    """

    output = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config, work_dir, solver_dir=None):
        super().__init__()
        self.config = config
        self.work_dir = work_dir
        self.solver_dir = solver_dir
        self._process = None

    def run(self):
        try:
            self._process = run_simulation(
                self.config,
                self.work_dir,
                solver_dir=self.solver_dir,
            )

            for line in self._process.stdout:
                line = line.rstrip()
                self.output.emit(line)

            self._process.wait()

            if self._process.returncode == 0:
                self.finished.emit(True, "Simulation completed.")
            else:
                self.finished.emit(
                    False,
                    f"Simulation exited with code {self._process.returncode}",
                )
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        """Terminate the running simulation subprocess."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
