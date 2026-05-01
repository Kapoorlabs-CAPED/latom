"""Main PyQt5 window for the latom TDSE solver GUI."""

import re  # noqa: E402
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
    plot_imag_convergence,
    plot_ionization,
    plot_spectrum,
    plot_vector_potential,
    plot_wavefunction_2d,
)
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
from runner import is_solver_built, is_solver_stale  # noqa: E402
from workers import BuildWorker, SimulationWorker  # noqa: E402


class MainWindow(QMainWindow):
    """Main application window for the latom TDSE solver."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("latom - Helium TDSE Solver")
        self.setMinimumSize(1200, 800)

        self.config = SimulationConfig()
        self._base_dir = Path.cwd()
        self._experiment_name = "latom_work"

        self._build_worker = None
        self._sim_worker = None
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_outputs)

        self._wf_snapshots = []  # list of (timestep, path) tuples found so far
        self._wf_snap_idx = -1  # index into _wf_snapshots currently displayed
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

        left_layout.addWidget(self._make_experiment_group())

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
            "Excited States",
            "Energy (Real)",
            "Dipole",
            "Ionization",
            "Vector Potential",
            "Spectrum",
            "1D Density",
            "Live WF",
        ]
        for name in tab_names:
            if name == "Live WF":
                # Live WF tab: fixed-size square canvas + prev/next scroll buttons
                container = QWidget()
                vbox = QVBoxLayout(container)
                vbox.setContentsMargins(0, 0, 0, 0)
                canvas = PlotCanvas(
                    width=7, height=7, dpi=100, constrained=True
                )
                canvas.setFixedSize(700, 700)
                self._canvases[name] = canvas
                vbox.addWidget(canvas, alignment=Qt.AlignCenter)

                nav_row = QWidget()
                nav_layout = QHBoxLayout(nav_row)
                nav_layout.setContentsMargins(4, 2, 4, 2)
                self._btn_wf_prev = QPushButton("◀ Prev")
                self._btn_wf_prev.clicked.connect(self._on_wf_prev)
                self._btn_wf_next = QPushButton("Next ▶")
                self._btn_wf_next.clicked.connect(self._on_wf_next)
                self._lbl_wf_snap = QLabel("No snapshots yet")
                self._lbl_wf_snap.setAlignment(Qt.AlignCenter)
                nav_layout.addWidget(self._btn_wf_prev)
                nav_layout.addWidget(self._lbl_wf_snap, stretch=1)
                nav_layout.addWidget(self._btn_wf_next)
                vbox.addWidget(nav_row)

                self.tabs.addTab(container, name)
            else:
                # "Excited States" manages its own subplot grid — no default axes
                canvas = PlotCanvas(single_axes=(name != "Excited States"))
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

    # ---- Experiment directory ----

    @property
    def work_dir(self):
        return self._base_dir / self._experiment_name

    def _make_experiment_group(self):
        grp = QGroupBox("Experiment")
        lay = QVBoxLayout(grp)

        base_row = QWidget()
        base_layout = QHBoxLayout(base_row)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.addWidget(QLabel("Base dir:"))
        self._edit_base_dir = QLabel(str(self._base_dir))
        self._edit_base_dir.setStyleSheet("font-size: 10px; color: #555;")
        self._edit_base_dir.setWordWrap(True)
        base_layout.addWidget(self._edit_base_dir, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.setMaximumWidth(60)
        btn_browse.clicked.connect(self._on_browse_base_dir)
        base_layout.addWidget(btn_browse)
        lay.addWidget(base_row)

        name_row = QWidget()
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.addWidget(QLabel("Experiment:"))
        self._edit_exp_name = QLabel(self._experiment_name)
        self._edit_exp_name.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self._edit_exp_name, stretch=1)
        btn_name = QPushButton("Rename")
        btn_name.setMaximumWidth(60)
        btn_name.clicked.connect(self._on_rename_experiment)
        name_layout.addWidget(btn_name)
        lay.addWidget(name_row)

        self._lbl_work_dir = QLabel(str(self.work_dir))
        self._lbl_work_dir.setStyleSheet("font-size: 9px; color: #777;")
        self._lbl_work_dir.setWordWrap(True)
        lay.addWidget(self._lbl_work_dir)

        return grp

    def _on_browse_base_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select base directory", str(self._base_dir)
        )
        if directory:
            self._base_dir = Path(directory)
            self._edit_base_dir.setText(str(self._base_dir))
            self._lbl_work_dir.setText(str(self.work_dir))

    def _on_rename_experiment(self):
        name, ok = QInputDialog.getText(
            self,
            "Experiment name",
            "Enter experiment name:",
            text=self._experiment_name,
        )
        if ok and name.strip():
            self._experiment_name = name.strip()
            self._edit_exp_name.setText(self._experiment_name)
            self._lbl_work_dir.setText(str(self.work_dir))

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

        # Solver mode: TDSE (default) vs Exact-TDDFT (KS reconstruction).
        mode_row = QWidget()
        mode_lay = QHBoxLayout(mode_row)
        mode_lay.setContentsMargins(0, 0, 0, 0)
        mode_lay.addWidget(QLabel("Solver mode"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("TDSE (2e Schrodinger)", "tdse")
        self.cmb_mode.addItem(
            "Exact-TDDFT (reconstruct KS orbital)", "exact_tddft"
        )
        idx = self.cmb_mode.findData(getattr(self.config, "mode", "tdse"))
        if idx >= 0:
            self.cmb_mode.setCurrentIndex(idx)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_lay.addWidget(self.cmb_mode, 1)
        lay.addWidget(mode_row)

        # Exact-TDDFT specific caching toggles
        self.chk_load_heplus = QCheckBox("Load He+ ground state from file")
        self.chk_load_heplus.setChecked(
            bool(getattr(self.config, "load_heplus", 0))
        )
        self.chk_load_heplus.setToolTip(
            "Exact-TDDFT only: load wf_heliumplus.dat if present, else compute via 1D imag-time."
        )
        lay.addWidget(self.chk_load_heplus)
        self.chk_load_ks_ground = QCheckBox("Load KS ground orbital from file")
        self.chk_load_ks_ground.setChecked(
            bool(getattr(self.config, "load_ks_ground", 0))
        )
        self.chk_load_ks_ground.setToolTip(
            "Exact-TDDFT only: load ks_ground.dat if present, else build from 2e GS."
        )
        lay.addWidget(self.chk_load_ks_ground)

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
        r, self.spin_excited_imag_mult = self._make_spin(
            "Excited steps multiplier", 1, 100, self.config.excited_imag_mult
        )
        r.setToolTip(
            "Multiplier for excited state imaginary time steps.\n"
            "State N gets N × multiplier × imag_steps.\n"
            "Default 1: state 1 gets imag_steps, state 2 gets 2×imag_steps, etc."
        )
        lay.addWidget(r)
        self.chk_load_ground = QCheckBox("Load ground state from file")
        self.chk_load_ground.setChecked(bool(self.config.load_ground))
        lay.addWidget(self.chk_load_ground)
        self.chk_load_excited = QCheckBox("Load excited states from file")
        self.chk_load_excited.setChecked(bool(self.config.load_excited))
        self.chk_load_excited.setToolTip(
            "When checked, loads wf_excited_N.dat if it exists instead of recomputing."
        )
        lay.addWidget(self.chk_load_excited)
        r, self.spin_laser_init_state = self._make_spin(
            "Laser initial state", 0, 20, self.config.laser_init_state
        )
        r.setToolTip("0 = ground state, N = Nth excited state")
        lay.addWidget(r)

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

        btn_row4 = QWidget()
        btn_layout4 = QHBoxLayout(btn_row4)
        btn_layout4.setContentsMargins(0, 0, 0, 0)

        btn_save_current = QPushButton("Save Current Plot...")
        btn_save_current.clicked.connect(self._on_save_current_plot)
        btn_layout4.addWidget(btn_save_current)

        btn_save_all = QPushButton("Save All Plots...")
        btn_save_all.clicked.connect(self._on_save_all_plots)
        btn_layout4.addWidget(btn_save_all)

        lay.addWidget(btn_row4)

        btn_row5 = QWidget()
        btn_layout5 = QHBoxLayout(btn_row5)
        btn_layout5.setContentsMargins(0, 0, 0, 0)

        btn_load_exc = QPushButton("Load Excited States from File")
        btn_load_exc.clicked.connect(self._on_load_excited_states)
        btn_load_exc.setToolTip(
            "Scan output directory for wf_excited_N.dat files and display them."
        )
        btn_layout5.addWidget(btn_load_exc)

        btn_load_exc_pick = QPushButton("Load Single Excited State...")
        btn_load_exc_pick.clicked.connect(self._on_load_excited_state_pick)
        btn_load_exc_pick.setToolTip(
            "Pick a specific wf_excited_N.dat file to load and display."
        )
        btn_layout5.addWidget(btn_load_exc_pick)

        lay.addWidget(btn_row5)
        return grp

    # ------------------------------------------------------------ Actions

    def _log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _update_build_status(self):
        mode = (
            self.cmb_mode.currentData()
            if hasattr(self, "cmb_mode")
            else "tdse"
        )
        label = {"tdse": "TDSE", "exact_tddft": "ExactTDDFT"}.get(mode, mode)
        if not is_solver_built(mode=mode):
            self.build_status_label.setText(f"{label}: not built")
            self.build_status_label.setStyleSheet("color: red;")
        elif is_solver_stale(mode=mode):
            self.build_status_label.setText(
                f"{label}: REBUILD NEEDED (source changed)"
            )
            self.build_status_label.setStyleSheet("color: orange;")
        else:
            self.build_status_label.setText(f"{label}: built (up to date)")
            self.build_status_label.setStyleSheet("color: green;")

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
            excited_imag_mult=self.spin_excited_imag_mult.value(),
            load_ground=1 if self.chk_load_ground.isChecked() else 0,
            load_excited=1 if self.chk_load_excited.isChecked() else 0,
            laser_init_state=self.spin_laser_init_state.value(),
            auto_mode=1 if self.chk_auto_mode.isChecked() else 0,
            auto_target_energy=self.spin_auto_target_energy.value(),
            kick_mode=1 if self.chk_kick_mode.isChecked() else 0,
            kick_strength=self.spin_kick_strength.value(),
            mode=self.cmb_mode.currentData(),
            load_heplus=1 if self.chk_load_heplus.isChecked() else 0,
            load_ks_ground=1 if self.chk_load_ks_ground.isChecked() else 0,
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
        self.spin_excited_imag_mult.setValue(self.config.excited_imag_mult)
        self.chk_load_ground.setChecked(bool(self.config.load_ground))
        self.chk_load_excited.setChecked(bool(self.config.load_excited))
        self.spin_laser_init_state.setValue(self.config.laser_init_state)
        self.chk_auto_mode.setChecked(bool(self.config.auto_mode))
        self.spin_auto_target_energy.setValue(self.config.auto_target_energy)
        self.chk_kick_mode.setChecked(bool(self.config.kick_mode))
        self.spin_kick_strength.setValue(self.config.kick_strength)
        idx = self.cmb_mode.findData(getattr(self.config, "mode", "tdse"))
        if idx >= 0:
            self.cmb_mode.setCurrentIndex(idx)
        self.chk_load_heplus.setChecked(
            bool(getattr(self.config, "load_heplus", 0))
        )
        self.chk_load_ks_ground.setChecked(
            bool(getattr(self.config, "load_ks_ground", 0))
        )
        self._on_mode_changed()

    def _on_mode_changed(self):
        """Enable/disable Exact-TDDFT-only widgets based on selected mode."""
        is_tddft = self.cmb_mode.currentData() == "exact_tddft"
        self.chk_load_heplus.setEnabled(is_tddft)
        self.chk_load_ks_ground.setEnabled(is_tddft)
        if hasattr(self, "build_status_label"):
            self._update_build_status()

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
        if is_solver_stale():
            QMessageBox.warning(
                self,
                "Rebuild Required",
                "Source files have changed since the last build.\n"
                "Click 'Build Solver' before running.",
            )
            return

        self._collect_config()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._wf_snapshots = []
        self._wf_snap_idx = -1
        self._lbl_wf_snap.setText("Waiting for snapshots...")
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

    def _load_excited_imag_data(self, out):
        """Load imaginary time convergence data for all excited states."""
        excited_imag = []
        for n in range(1, 21):
            f = out / f"obserimag_excited_{n}.dat"
            if f.is_file():
                data = parse_imag_observables(f)
                if data is not None:
                    excited_imag.append((n, data))
            else:
                break
        return excited_imag

    def _poll_outputs(self):
        """Poll observable files and update live plots."""
        out = self._get_output_dir()

        imag_data = parse_imag_observables(out / self.config.obser_imag_file)
        real_data = parse_real_observables(out / self.config.obser_file)
        excited_imag = self._load_excited_imag_data(out)

        if imag_data is not None or excited_imag:
            c = self._canvases["Energy (Imag)"]
            plot_imag_convergence(c.axes, imag_data, excited_imag)
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

        # Live WF: check for new snapshots, auto-advance to latest
        snapshots = find_wf_snapshots(out)
        if len(snapshots) > len(self._wf_snapshots):
            self._wf_snapshots = snapshots
            self._wf_snap_idx = len(snapshots) - 1  # jump to latest
            self._show_wf_snapshot(self._wf_snap_idx)

    def _show_wf_snapshot(self, idx):
        """Render snapshot at index idx into the Live WF canvas."""
        if not self._wf_snapshots:
            return
        idx = max(0, min(idx, len(self._wf_snapshots) - 1))
        self._wf_snap_idx = idx
        ts, path = self._wf_snapshots[idx]
        nx, ny = self.config.grid_nx, self.config.grid_ny
        dx, dy = self.config.grid_dx, self.config.grid_dy
        wf = parse_wavefunction(path, nx, ny)
        c = self._canvases["Live WF"]
        plot_wavefunction_2d(c.axes, wf, dx, dy, nx, ny)
        label = "final" if ts == -1 else f"step {ts}"
        c.axes.set_title(rf"$|\psi(x_1,x_2)|^2$ — {label}")
        c.clear_and_draw()
        self._lbl_wf_snap.setText(
            f"Snapshot {idx + 1} / {len(self._wf_snapshots)}  ({label})"
        )

    def _on_wf_prev(self):
        self._show_wf_snapshot(self._wf_snap_idx - 1)

    def _on_wf_next(self):
        self._show_wf_snapshot(self._wf_snap_idx + 1)

    def _refresh_all_plots(self):
        """Refresh all plots from output files."""
        out = self._get_output_dir()
        nx = self.config.grid_nx
        ny = self.config.grid_ny
        dx = self.config.grid_dx
        dy = self.config.grid_dy

        imag_data = parse_imag_observables(out / self.config.obser_imag_file)
        real_data = parse_real_observables(out / self.config.obser_file)
        excited_imag = self._load_excited_imag_data(out)

        # Imag energy — ground state + all excited states
        c = self._canvases["Energy (Imag)"]
        plot_imag_convergence(c.axes, imag_data, excited_imag)
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

        # 1D Density and Live WF from real-time snapshots
        snapshots = find_wf_snapshots(out)
        wf_latest = None
        if snapshots:
            _, last_path = snapshots[-1]
            wf_latest = parse_wavefunction(last_path, nx, ny)
            # Populate snapshot browser if not already done
            if len(snapshots) != len(self._wf_snapshots):
                self._wf_snapshots = snapshots
                self._wf_snap_idx = len(snapshots) - 1
            self._show_wf_snapshot(self._wf_snap_idx)

        c = self._canvases["1D Density"]
        plot_density_1d(c.axes, wf_latest, dx, dy, nx, ny)
        c.clear_and_draw()

        # Excited state wavefunctions — all in a single tab with grid layout
        excited_wfs = []
        for n in range(1, 21):
            exc_file = out / f"wf_excited_{n}.dat"
            if exc_file.is_file():
                wf_exc = parse_wavefunction(exc_file, nx, ny)
                if wf_exc is not None:
                    excited_wfs.append((n, wf_exc))
            else:
                break

        if excited_wfs:
            self._draw_excited_states(excited_wfs, dx, dy, nx, ny)
        else:
            c = self._canvases["Excited States"]
            c.fig.clear()
            ax = c.fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No excited states computed yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.axis("off")
            c.fig.tight_layout(pad=2.0)
            c.draw()

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

    def _on_save_current_plot(self):
        """Save the currently visible tab's figure to file."""
        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        canvas = self._canvases.get(tab_name)
        if canvas is None:
            self._log("No plot to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            f"{tab_name.replace(' ', '_')}.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Saved: {path}")

    def _on_save_all_plots(self):
        """Save all plot figures to a chosen directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select output directory"
        )
        if not directory:
            return
        fmt, _ = QFileDialog.getSaveFileName(
            self,
            "Choose format (enter any filename with .png/.svg/.pdf)",
            "plots.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        ext = Path(fmt).suffix if fmt else ".png"
        saved = 0
        for name, canvas in self._canvases.items():
            fname = Path(directory) / (
                name.replace(" ", "_").replace("/", "-") + ext
            )
            try:
                canvas.fig.savefig(str(fname), dpi=150, bbox_inches="tight")
                saved += 1
            except Exception as e:
                self._log(f"Could not save {name}: {e}")
        self._log(f"Saved {saved} plots to {directory}")

    def _on_load_excited_states(self):
        """Scan output directory for wf_excited_N.dat files and display them."""
        out = self._get_output_dir()
        nx = self.config.grid_nx
        ny = self.config.grid_ny
        dx = self.config.grid_dx
        dy = self.config.grid_dy

        excited_wfs = []
        for n in range(1, 21):
            exc_file = out / f"wf_excited_{n}.dat"
            if exc_file.is_file():
                wf_exc = parse_wavefunction(exc_file, nx, ny)
                if wf_exc is not None:
                    excited_wfs.append((n, wf_exc))
            else:
                break

        if not excited_wfs:
            self._log("No wf_excited_N.dat files found in output directory.")
            return

        self._log(f"Loaded {len(excited_wfs)} excited state(s) from file.")
        self._draw_excited_states(excited_wfs, dx, dy, nx, ny)
        # Switch to the Excited States tab
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Excited States":
                self.tabs.setCurrentIndex(i)
                break

    def _on_load_excited_state_pick(self):
        """Open a file dialog to pick a specific excited state file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Excited State Wavefunction",
            str(self._get_output_dir()),
            "DAT files (*.dat)",
        )
        if not path:
            return

        nx = self.config.grid_nx
        ny = self.config.grid_ny
        dx = self.config.grid_dx
        dy = self.config.grid_dy

        wf_exc = parse_wavefunction(Path(path), nx, ny)
        if wf_exc is None:
            self._log(f"Failed to load wavefunction from {path}")
            return

        m = re.search(r"wf_excited_(\d+)", Path(path).name)
        n = int(m.group(1)) if m else 0
        label = f"Excited {n}" if n else Path(path).stem

        self._log(f"Loaded excited state from {path}")
        self._draw_excited_states(
            [(n or 1, wf_exc)], dx, dy, nx, ny, titles=[label]
        )
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Excited States":
                self.tabs.setCurrentIndex(i)
                break

    def _draw_excited_states(self, excited_wfs, dx, dy, nx, ny, titles=None):
        """Render excited state wavefunctions into the Excited States tab."""
        c = self._canvases["Excited States"]
        c.fig.clear()

        n_states = len(excited_wfs)
        n_cols = min(n_states, 3)
        n_rows = (n_states + n_cols - 1) // n_cols
        axes = c.fig.subplots(n_rows, n_cols, squeeze=False)

        for idx, (n, wf_exc) in enumerate(excited_wfs):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            plot_wavefunction_2d(ax, wf_exc, dx, dy, nx, ny)
            title = (
                titles[idx]
                if titles
                else rf"Excited {n}: $|\psi_{n}(x_1,x_2)|^2$"
            )
            ax.set_title(title, fontsize=9)

        for idx in range(n_states, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].axis("off")

        c.fig.tight_layout(pad=2.0)
        c.draw()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("latom")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
