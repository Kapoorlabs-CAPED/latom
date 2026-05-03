"""Background QThread workers for build and simulation tasks."""

import sys
from pathlib import Path

# Add scripts directory to path for sibling imports — must run before local imports
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import threading  # noqa: E402

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


class BatchSimulationWorker(QThread):
    """Run an optional preflight job, then several solver subprocesses
    concurrently.

    The ``preflight`` job (if given) runs by itself first — used to
    compute shared ground-state files once. Once it completes (and a
    user-supplied ``after_preflight`` callable has finished its
    bookkeeping, e.g. copying GS files into each case dir), all
    ``parallel_jobs`` are launched together. Each entry in
    ``parallel_jobs`` is a (name, config, work_dir) triple.

    Signals:
        output(str): One log line, prefixed with the job name.
        job_finished(str, bool, str): once per job (preflight included).
        finished(bool, str): summary, once after the last job exits.
    """

    output = pyqtSignal(str)
    job_finished = pyqtSignal(str, bool, str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        parallel_jobs,
        preflight=None,
        after_preflight=None,
        solver_dir=None,
    ):
        super().__init__()
        self.parallel_jobs = list(parallel_jobs)
        self.preflight = preflight  # (name, config, work_dir) or None
        self.after_preflight = after_preflight  # callable() -> None
        self.solver_dir = solver_dir
        self._processes = {}

    def _consume(self, name, proc):
        for line in proc.stdout:
            self.output.emit(f"[{name}] {line.rstrip()}")

    def _run_one(self, name, config, work_dir):
        proc = run_simulation(
            config,
            work_dir,
            solver_dir=self.solver_dir,
        )
        self._processes[name] = proc
        t = threading.Thread(
            target=self._consume,
            args=(name, proc),
            daemon=True,
        )
        t.start()
        return proc, t

    def run(self):
        results = []
        if self.preflight is not None:
            name, config, work_dir = self.preflight
            self.output.emit(
                f"[{name}] preflight: computing shared ground states…"
            )
            try:
                proc, t = self._run_one(name, config, work_dir)
                proc.wait()
                t.join(timeout=5.0)
                ok = proc.returncode == 0
                self.job_finished.emit(
                    name,
                    ok,
                    (
                        f"completed (rc={proc.returncode})"
                        if ok
                        else f"failed (rc={proc.returncode})"
                    ),
                )
                results.append((name, ok))
                if not ok:
                    self.finished.emit(
                        False,
                        f"Preflight {name} failed; aborting batch.",
                    )
                    return
                if self.after_preflight is not None:
                    try:
                        self.after_preflight()
                    except Exception as e:
                        self.output.emit(
                            f"[{name}] after_preflight failed: {e}"
                        )
                        self.finished.emit(False, str(e))
                        return
            except Exception as e:
                self.output.emit(f"[{name}] preflight launch failed: {e}")
                self.job_finished.emit(name, False, str(e))
                self.finished.emit(False, str(e))
                return

        readers = []
        for name, config, work_dir in self.parallel_jobs:
            try:
                proc, t = self._run_one(name, config, work_dir)
                readers.append((name, proc, t))
            except Exception as e:
                self.output.emit(f"[{name}] launch failed: {e}")
                self.job_finished.emit(name, False, str(e))
                results.append((name, False))

        for name, proc, t in readers:
            proc.wait()
            t.join(timeout=5.0)
            ok = proc.returncode == 0
            self.job_finished.emit(
                name,
                ok,
                (
                    f"completed (rc={proc.returncode})"
                    if ok
                    else f"failed (rc={proc.returncode})"
                ),
            )
            results.append((name, ok))

        all_ok = all(ok for _, ok in results)
        summary = ", ".join(
            f"{name}={'OK' if ok else 'FAIL'}" for name, ok in results
        )
        self.finished.emit(all_ok, f"Batch finished: {summary}")

    def stop(self):
        for proc in self._processes.values():
            if proc.poll() is None:
                proc.terminate()
