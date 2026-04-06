"""Main PyQt5 window for the latom TDSE solver GUI."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Add scripts directory for sibling imports — must run before local imports
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from parser import (  # noqa: E402
    find_wf_snapshots,
    parse_imag_observables,
    parse_real_observables,
    parse_wavefunction,
)

from config import SimulationConfig  # noqa: E402
from plot_canvas import PlotCanvas  # noqa: E402
from plotting import (  # noqa: E402
    create_wf_animation,
    plot_density_1d,
    plot_dipole,
    plot_energy_vs_time,
    plot_ionization,
    plot_spectrum,
    plot_vector_potential,
    plot_wavefunction_2d,
)
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from runner import is_solver_built  # noqa: E402
from workers import BuildWorker, SimulationWorker  # noqa: E402


class MainWindow(QMainWindow):
    """Main application window for the latom TDSE solver."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("latom - Helium TDSE Solver")
        self.setMinimumSize(1200, 800)

        self.config = SimulationConfig()
        self.work_dir = Path.cwd() / "latom_work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._build_worker = None
        self._sim_worker = None
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_outputs)

        self._setup_ui()
        self._update_build_status()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel: parameters + controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        scroll_layout.addWidget(self._make_grid_group())
        scroll_layout.addWidget(self._make_time_group())
        scroll_layout.addWidget(self._make_laser_group())
        scroll_layout.addWidget(self._make_physics_group())
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll, stretch=1)
        left_layout.addWidget(self._make_controls_group())

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        left_layout.addWidget(self.log_text)

        left.setMaximumWidth(360)
        splitter.addWidget(left)

        # Right panel: tabbed plots
        self.tabs = QTabWidget()
        self._canvases = {}
        tab_names = [
            "Energy (Imag)",
            "Ground State WF",
            "Energy (Real)",
            "Dipole",
            "Ionization",
            "Vector Potential",
            "Spectrum",
            "1D Density",
        ]
        for name in tab_names:
            canvas = PlotCanvas()
            self._canvases[name] = canvas
            self.tabs.addTab(canvas, name)

        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 860])

    # ---- Parameter groups ----

    def _make_spin(self, label, vmin, vmax, value, is_float=False, decimals=4):
        """Create a labeled spin box and return (widget_row, spinbox)."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        row_layout.addWidget(lbl)

        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(vmin, vmax)
            spin.setValue(value)
        else:
            spin = QSpinBox()
            spin.setRange(int(vmin), int(vmax))
            spin.setValue(int(value))

        row_layout.addWidget(spin)
        return row, spin

    def _make_grid_group(self):
        grp = QGroupBox("Grid")
        lay = QVBoxLayout(grp)
        r, self.spin_nx = self._make_spin(
            "N_x", 10, 10000, self.config.grid_nx
        )
        lay.addWidget(r)
        r, self.spin_ny = self._make_spin(
            "N_y", 10, 10000, self.config.grid_ny
        )
        lay.addWidget(r)
        r, self.spin_dx = self._make_spin(
            "dx", 0.01, 10.0, self.config.grid_dx, True
        )
        lay.addWidget(r)
        r, self.spin_dy = self._make_spin(
            "dy", 0.01, 10.0, self.config.grid_dy, True
        )
        lay.addWidget(r)
        return grp

    def _make_time_group(self):
        grp = QGroupBox("Time Propagation")
        lay = QVBoxLayout(grp)
        r, self.spin_imag_dt = self._make_spin(
            "Imag dt", 0.001, 10.0, self.config.imag_dt, True
        )
        lay.addWidget(r)
        r, self.spin_imag_steps = self._make_spin(
            "Imag steps", 1, 1000000, self.config.imag_steps
        )
        lay.addWidget(r)
        r, self.spin_real_dt = self._make_spin(
            "Real dt", 0.001, 10.0, self.config.real_dt, True
        )
        lay.addWidget(r)
        r, self.spin_real_steps = self._make_spin(
            "Real steps", 1, 1000000, self.config.real_steps
        )
        lay.addWidget(r)
        return grp

    def _make_laser_group(self):
        grp = QGroupBox("Laser (Velocity Gauge)")
        lay = QVBoxLayout(grp)
        r, self.spin_freq = self._make_spin(
            "Frequency", 0.01, 100.0, self.config.laser_freq, True
        )
        lay.addWidget(r)
        r, self.spin_alpha = self._make_spin(
            "Alpha", 0.001, 100.0, self.config.laser_alpha, True
        )
        lay.addWidget(r)
        r, self.spin_cycles = self._make_spin(
            "Cycles", 1.0, 10000.0, self.config.laser_cycles, True, decimals=1
        )
        lay.addWidget(r)
        return grp

    def _make_physics_group(self):
        grp = QGroupBox("Physics")
        lay = QVBoxLayout(grp)
        r, self.spin_eps = self._make_spin(
            "Coulomb eps", 0.01, 10.0, self.config.coulomb_eps, True
        )
        lay.addWidget(r)
        r, self.spin_absorb = self._make_spin(
            "Absorb ampl",
            0.0,
            1000.0,
            self.config.absorb_ampl,
            True,
            decimals=1,
        )
        lay.addWidget(r)
        r, self.spin_box = self._make_spin(
            "Ionization box", 1, 10000, self.config.ionization_box
        )
        lay.addWidget(r)
        r, self.spin_n_excited = self._make_spin(
            "N excited states", 0, 20, self.config.n_excited
        )
        lay.addWidget(r)
        self.chk_load_ground = QCheckBox("Load ground state from file")
        self.chk_load_ground.setChecked(bool(self.config.load_ground))
        lay.addWidget(self.chk_load_ground)

        # Autoionizing mode (Feit-Fleck-Steiger)
        self.chk_auto_mode = QCheckBox(
            "Autoionizing mode (Feit-Fleck-Steiger)"
        )
        self.chk_auto_mode.setChecked(bool(self.config.auto_mode))
        lay.addWidget(self.chk_auto_mode)
        r, self.spin_auto_target_energy = self._make_spin(
            "Target energy (a.u.)",
            -10.0,
            10.0,
            self.config.auto_target_energy,
            is_float=True,
            decimals=4,
        )
        lay.addWidget(r)

        # Kick mode (linear response)
        self.chk_kick_mode = QCheckBox("Kick mode (linear response spectrum)")
        self.chk_kick_mode.setChecked(bool(self.config.kick_mode))
        lay.addWidget(self.chk_kick_mode)
        r, self.spin_kick_strength = self._make_spin(
            "Kick strength A_0",
            0.0,
            1.0,
            self.config.kick_strength,
            is_float=True,
            decimals=4,
        )
        lay.addWidget(r)
        return grp

    def _make_controls_group(self):
        grp = QGroupBox("Controls")
        lay = QVBoxLayout(grp)

        self.build_status_label = QLabel("Solver: checking...")
        lay.addWidget(self.build_status_label)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_build = QPushButton("Build Solver")
        self.btn_build.clicked.connect(self._on_build)
        btn_layout.addWidget(self.btn_build)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        lay.addWidget(btn_row)

        btn_row2 = QWidget()
        btn_layout2 = QHBoxLayout(btn_row2)
        btn_layout2.setContentsMargins(0, 0, 0, 0)

        btn_save = QPushButton("Save Config")
        btn_save.clicked.connect(self._on_save_config)
        btn_layout2.addWidget(btn_save)

        btn_load = QPushButton("Load Config")
        btn_load.clicked.connect(self._on_load_config)
        btn_layout2.addWidget(btn_load)

        btn_refresh = QPushButton("Refresh Plots")
        btn_refresh.clicked.connect(self._refresh_all_plots)
        btn_layout2.addWidget(btn_refresh)

        lay.addWidget(btn_row2)

        btn_row3 = QWidget()
        btn_layout3 = QHBoxLayout(btn_row3)
        btn_layout3.setContentsMargins(0, 0, 0, 0)

        btn_gif = QPushButton("Create WF Animation (GIF)")
        btn_gif.clicked.connect(self._on_create_gif)
        btn_layout3.addWidget(btn_gif)

        lay.addWidget(btn_row3)
        return grp

    # ------------------------------------------------------------ Actions

    def _log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _update_build_status(self):
        if is_solver_built():
            self.build_status_label.setText("Solver: built")
            self.build_status_label.setStyleSheet("color: green;")
        else:
            self.build_status_label.setText("Solver: not built")
            self.build_status_label.setStyleSheet("color: red;")

    def _collect_config(self):
        """Read GUI spin boxes into a SimulationConfig."""
        self.config = SimulationConfig(
            grid_nx=self.spin_nx.value(),
            grid_ny=self.spin_ny.value(),
            grid_dx=self.spin_dx.value(),
            grid_dy=self.spin_dy.value(),
            imag_dt=self.spin_imag_dt.value(),
            imag_steps=self.spin_imag_steps.value(),
            real_dt=self.spin_real_dt.value(),
            real_steps=self.spin_real_steps.value(),
            laser_freq=self.spin_freq.value(),
            laser_alpha=self.spin_alpha.value(),
            laser_cycles=self.spin_cycles.value(),
            coulomb_eps=self.spin_eps.value(),
            absorb_ampl=self.spin_absorb.value(),
            ionization_box=self.spin_box.value(),
            n_excited=self.spin_n_excited.value(),
            load_ground=1 if self.chk_load_ground.isChecked() else 0,
            auto_mode=1 if self.chk_auto_mode.isChecked() else 0,
            auto_target_energy=self.spin_auto_target_energy.value(),
            kick_mode=1 if self.chk_kick_mode.isChecked() else 0,
            kick_strength=self.spin_kick_strength.value(),
        )

    def _populate_spins(self):
        """Write config values into GUI spin boxes."""
        self.spin_nx.setValue(self.config.grid_nx)
        self.spin_ny.setValue(self.config.grid_ny)
        self.spin_dx.setValue(self.config.grid_dx)
        self.spin_dy.setValue(self.config.grid_dy)
        self.spin_imag_dt.setValue(self.config.imag_dt)
        self.spin_imag_steps.setValue(self.config.imag_steps)
        self.spin_real_dt.setValue(self.config.real_dt)
        self.spin_real_steps.setValue(self.config.real_steps)
        self.spin_freq.setValue(self.config.laser_freq)
        self.spin_alpha.setValue(self.config.laser_alpha)
        self.spin_cycles.setValue(self.config.laser_cycles)
        self.spin_eps.setValue(self.config.coulomb_eps)
        self.spin_absorb.setValue(self.config.absorb_ampl)
        self.spin_box.setValue(self.config.ionization_box)
        self.spin_n_excited.setValue(self.config.n_excited)
        self.chk_load_ground.setChecked(bool(self.config.load_ground))
        self.chk_auto_mode.setChecked(bool(self.config.auto_mode))
        self.spin_auto_target_energy.setValue(self.config.auto_target_energy)
        self.chk_kick_mode.setChecked(bool(self.config.kick_mode))
        self.spin_kick_strength.setValue(self.config.kick_strength)

    # ---- Build ----

    def _on_build(self):
        self.btn_build.setEnabled(False)
        self._log("Building solver...")
        self._build_worker = BuildWorker()
        self._build_worker.output.connect(self._log)
        self._build_worker.finished.connect(self._on_build_done)
        self._build_worker.start()

    def _on_build_done(self, success, msg):
        self._log(msg)
        self.btn_build.setEnabled(True)
        self._update_build_status()
        if not success:
            QMessageBox.warning(self, "Build Failed", msg)

    # ---- Run ----

    def _on_run(self):
        if not is_solver_built():
            QMessageBox.warning(self, "Not Built", "Build the solver first.")
            return

        self._collect_config()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._log("Starting simulation...")

        self._sim_worker = SimulationWorker(self.config, str(self.work_dir))
        self._sim_worker.output.connect(self._log)
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.start()

        # Start polling output files for live plot updates
        self._poll_timer.start(2000)

    def _on_sim_done(self, success, msg):
        self._poll_timer.stop()
        self._log(msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_all_plots()
        if not success:
            QMessageBox.warning(self, "Simulation Error", msg)

    def _on_stop(self):
        if self._sim_worker:
            self._sim_worker.stop()
            self._log("Stopping simulation...")

    # ---- Config I/O ----

    def _on_save_config(self):
        self._collect_config()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "", "JSON (*.json)"
        )
        if path:
            self.config.save_json(Path(path))
            self._log(f"Config saved to {path}")

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "JSON (*.json)"
        )
        if path:
            self.config = SimulationConfig.load_json(Path(path))
            self._populate_spins()
            self._log(f"Config loaded from {path}")

    # ---- Plotting ----

    def _get_output_dir(self):
        return self.work_dir / self.config.output_dir

    def _poll_outputs(self):
        """Poll observable files and update live plots."""
        out = self._get_output_dir()

        imag_data = parse_imag_observables(out / self.config.obser_imag_file)
        real_data = parse_real_observables(out / self.config.obser_file)

        if imag_data is not None:
            c = self._canvases["Energy (Imag)"]
            plot_energy_vs_time(c.axes, imag_data, phase="imag")
            c.clear_and_draw()

        if real_data is not None:
            for name, func in [
                (
                    "Energy (Real)",
                    lambda ax: plot_energy_vs_time(
                        ax, real_data, phase="real"
                    ),
                ),
                ("Dipole", lambda ax: plot_dipole(ax, real_data)),
                ("Ionization", lambda ax: plot_ionization(ax, real_data)),
                (
                    "Vector Potential",
                    lambda ax: plot_vector_potential(ax, real_data),
                ),
            ]:
                c = self._canvases[name]
                func(c.axes)
                c.clear_and_draw()

    def _refresh_all_plots(self):
        """Refresh all plots from output files."""
        out = self._get_output_dir()
        nx = self.config.grid_nx
        ny = self.config.grid_ny
        dx = self.config.grid_dx
        dy = self.config.grid_dy

        imag_data = parse_imag_observables(out / self.config.obser_imag_file)
        real_data = parse_real_observables(out / self.config.obser_file)

        # Imag energy
        c = self._canvases["Energy (Imag)"]
        plot_energy_vs_time(c.axes, imag_data, phase="imag")
        c.clear_and_draw()

        # Ground state wavefunction (single converged result from imag prop)
        wf_ground = parse_wavefunction(out / "wf_ground.dat", nx, ny)
        c = self._canvases["Ground State WF"]
        plot_wavefunction_2d(c.axes, wf_ground, dx, dy, nx, ny)
        if wf_ground is not None:
            c.axes.set_title(r"Ground State $|\psi_0(x_1,x_2)|^2$")
        c.clear_and_draw()

        # Real-time plots
        c = self._canvases["Energy (Real)"]
        plot_energy_vs_time(c.axes, real_data, phase="real")
        c.clear_and_draw()

        c = self._canvases["Dipole"]
        plot_dipole(c.axes, real_data)
        c.clear_and_draw()

        c = self._canvases["Ionization"]
        plot_ionization(c.axes, real_data)
        c.clear_and_draw()

        c = self._canvases["Vector Potential"]
        plot_vector_potential(c.axes, real_data)
        c.clear_and_draw()

        # Spectrum (post-simulation)
        c = self._canvases["Spectrum"]
        plot_spectrum(c.axes, real_data)
        c.clear_and_draw()

        # 1D Density from latest real-time wf snapshot or final
        snapshots = find_wf_snapshots(out)
        wf_latest = None
        if snapshots:
            _, last_path = snapshots[-1]
            wf_latest = parse_wavefunction(last_path, nx, ny)

        c = self._canvases["1D Density"]
        plot_density_1d(c.axes, wf_latest, dx, dy, nx, ny)
        c.clear_and_draw()

        # Excited state wavefunctions — add/update tabs dynamically
        for n in range(1, 21):
            tab_name = f"Excited {n} WF"
            exc_file = out / f"wf_excited_{n}.dat"
            if exc_file.is_file():
                wf_exc = parse_wavefunction(exc_file, nx, ny)
                if tab_name not in self._canvases:
                    canvas = PlotCanvas()
                    self._canvases[tab_name] = canvas
                    self.tabs.addTab(canvas, tab_name)
                c = self._canvases[tab_name]
                plot_wavefunction_2d(c.axes, wf_exc, dx, dy, nx, ny)
                if wf_exc is not None:
                    c.axes.set_title(
                        rf"Excited State {n} $|\psi_{n}(x_1,x_2)|^2$"
                    )
                c.clear_and_draw()
            else:
                break

    def _on_create_gif(self):
        """Create animated GIF from real-time wavefunction snapshots."""
        out = self._get_output_dir()
        gif_path = out / "wf_evolution.gif"
        self._log("Creating wavefunction animation...")

        try:
            n_frames = create_wf_animation(
                out,
                self.config.grid_nx,
                self.config.grid_ny,
                self.config.grid_dx,
                self.config.grid_dy,
                gif_path=gif_path,
                fps=5,
            )
            if n_frames > 0:
                self._log(f"GIF saved: {gif_path} ({n_frames} frames)")
            else:
                self._log("No wavefunction snapshots found for animation.")
        except Exception as e:
            self._log(f"GIF creation failed: {e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("latom")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
