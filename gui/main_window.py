"""Main PyQt5 window for the latom TDSE solver GUI."""

import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Add scripts directory for sibling imports — must run before local imports
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from parser import (  # noqa: E402
    find_ks_snapshots,
    find_wf_snapshots,
    parse_imag_observables,
    parse_orbital_1d,
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
    plot_ks_orbital,
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
        self._ks_snapshots = []
        self._ks_snap_idx = -1
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

        # Mode tabs are the only thing in the parameter panel: each tab is a
        # self-contained parameter set for one solver mode.
        scroll_layout.addWidget(self._make_mode_tabs_group())
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
            "Ground State KS Orbital",
            "Live KS Orbital",
            "Live WF",
        ]
        # Tabs that are only meaningful in one mode. The others are shown
        # in both. _apply_tab_visibility() toggles these on mode change.
        self._tdse_only_tabs = ("1D Density",)
        self._tddft_only_tabs = ("Ground State KS Orbital", "Live KS Orbital")
        for name in tab_names:
            if name == "Live WF":
                container = self._make_live_tab(
                    name,
                    square=True,
                    on_prev=self._on_wf_prev,
                    on_next=self._on_wf_next,
                    label_attr="_lbl_wf_snap",
                )
                self.tabs.addTab(container, name)
            elif name == "Live KS Orbital":
                container = self._make_live_tab(
                    name,
                    square=False,
                    on_prev=self._on_ks_prev,
                    on_next=self._on_ks_next,
                    label_attr="_lbl_ks_snap",
                )
                self.tabs.addTab(container, name)
            else:
                # "Excited States" manages its own subplot grid — no default axes
                canvas = PlotCanvas(single_axes=(name != "Excited States"))
                self._canvases[name] = canvas
                self.tabs.addTab(canvas, name)
        self._apply_tab_visibility()

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

    # ------------------------------------------------------------ Mode tabs
    #
    # Each tab is a complete, self-contained parameter set for one solver
    # mode. Widgets are stored in per-tab dicts (self._tab_widgets[mode]) so
    # the two tabs do not share state — switching tabs picks the mode and
    # uses *that tab's* values. Most parameters are duplicated by design:
    # Exact-TDDFT runs the same TDSE machinery internally, so every TDSE
    # control (excited states, autoionization, kick mode, ...) is also
    # available in the TDDFT tab.

    def _add_spin(
        self,
        layout,
        target,
        key,
        label,
        vmin,
        vmax,
        value,
        is_float=False,
        decimals=4,
        tooltip=None,
    ):
        row, spin = self._make_spin(
            label, vmin, vmax, value, is_float, decimals
        )
        if tooltip:
            row.setToolTip(tooltip)
        target[key] = spin
        layout.addWidget(row)
        return spin

    def _add_check(self, layout, target, key, label, checked, tooltip=None):
        chk = QCheckBox(label)
        chk.setChecked(bool(checked))
        if tooltip:
            chk.setToolTip(tooltip)
        target[key] = chk
        layout.addWidget(chk)
        return chk

    def _build_param_tab(self, mode):
        """Return a QWidget containing the full parameter UI for one mode.

        The widget references are stored in self._tab_widgets[mode] keyed by
        the matching SimulationConfig field name.
        """
        widgets = {}
        self._tab_widgets[mode] = widgets

        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(4, 4, 4, 4)

        if mode == "exact_tddft":
            outer.addWidget(
                QLabel(
                    "Solves the 2e TDSE and reconstructs the Kohn-Sham orbital\n"
                    "and effective potential each time step (eq. 29)."
                )
            )

        cfg = self.config

        # ----- Grid -----
        grp = QGroupBox("Grid")
        g = QVBoxLayout(grp)
        self._add_spin(g, widgets, "grid_nx", "N_x", 10, 10000, cfg.grid_nx)
        self._add_spin(g, widgets, "grid_ny", "N_y", 10, 10000, cfg.grid_ny)
        self._add_spin(
            g, widgets, "grid_dx", "dx", 0.01, 10.0, cfg.grid_dx, True
        )
        self._add_spin(
            g, widgets, "grid_dy", "dy", 0.01, 10.0, cfg.grid_dy, True
        )
        outer.addWidget(grp)

        # ----- Time -----
        grp = QGroupBox("Time Propagation")
        g = QVBoxLayout(grp)
        self._add_spin(
            g, widgets, "imag_dt", "Imag dt", 0.001, 10.0, cfg.imag_dt, True
        )
        self._add_spin(
            g, widgets, "imag_steps", "Imag steps", 1, 1000000, cfg.imag_steps
        )
        self._add_spin(
            g, widgets, "real_dt", "Real dt", 0.001, 10.0, cfg.real_dt, True
        )
        self._add_spin(
            g, widgets, "real_steps", "Real steps", 1, 1000000, cfg.real_steps
        )
        outer.addWidget(grp)

        # ----- Laser -----
        grp = QGroupBox("Laser (Velocity Gauge)")
        g = QVBoxLayout(grp)
        self._add_spin(
            g,
            widgets,
            "laser_freq",
            "Frequency",
            0.01,
            100.0,
            cfg.laser_freq,
            True,
        )
        self._add_spin(
            g,
            widgets,
            "laser_alpha",
            "Alpha",
            0.001,
            100.0,
            cfg.laser_alpha,
            True,
        )
        self._add_spin(
            g,
            widgets,
            "laser_cycles",
            "Cycles",
            1.0,
            10000.0,
            cfg.laser_cycles,
            True,
            decimals=1,
        )
        outer.addWidget(grp)

        # ----- Physics -----
        grp = QGroupBox("Physics")
        g = QVBoxLayout(grp)
        self._add_spin(
            g,
            widgets,
            "coulomb_eps",
            "Coulomb eps",
            0.01,
            10.0,
            cfg.coulomb_eps,
            True,
        )
        self._add_spin(
            g,
            widgets,
            "absorb_ampl",
            "Absorb ampl",
            0.0,
            1000.0,
            cfg.absorb_ampl,
            True,
            decimals=1,
        )
        self._add_spin(
            g,
            widgets,
            "ionization_box",
            "Ionization box",
            1,
            10000,
            cfg.ionization_box,
        )
        self._add_check(
            g,
            widgets,
            "load_ground",
            "Load 2e ground state from file",
            cfg.load_ground,
            "Load wf_ground.dat if present, else compute via imaginary-time.",
        )
        outer.addWidget(grp)

        # ----- States (TDSE-style features, available in both modes) -----
        grp = QGroupBox("States")
        g = QVBoxLayout(grp)
        self._add_spin(
            g, widgets, "n_excited", "N excited states", 0, 20, cfg.n_excited
        )
        self._add_spin(
            g,
            widgets,
            "excited_imag_mult",
            "Excited steps multiplier",
            1,
            100,
            cfg.excited_imag_mult,
            tooltip=(
                "Multiplier for excited state imaginary time steps.\n"
                "State N gets N * multiplier * imag_steps."
            ),
        )
        self._add_check(
            g,
            widgets,
            "load_excited",
            "Load excited states from file",
            cfg.load_excited,
            "When checked, loads wf_excited_N.dat if it exists instead of recomputing.",
        )
        self._add_spin(
            g,
            widgets,
            "laser_init_state",
            "Laser initial state",
            0,
            20,
            cfg.laser_init_state,
            tooltip="0 = ground state, N = Nth excited state",
        )
        outer.addWidget(grp)

        # ----- Autoionizing -----
        grp = QGroupBox("Autoionizing (Feit-Fleck-Steiger)")
        g = QVBoxLayout(grp)
        self._add_check(
            g,
            widgets,
            "auto_mode",
            "Enable autoionizing mode",
            cfg.auto_mode,
        )
        self._add_spin(
            g,
            widgets,
            "auto_target_energy",
            "Target energy (a.u.)",
            -10.0,
            10.0,
            cfg.auto_target_energy,
            is_float=True,
            decimals=4,
        )
        outer.addWidget(grp)

        # ----- Kick (linear response) -----
        grp = QGroupBox("Kick (linear response)")
        g = QVBoxLayout(grp)
        self._add_check(
            g,
            widgets,
            "kick_mode",
            "Enable kick mode",
            cfg.kick_mode,
        )
        self._add_spin(
            g,
            widgets,
            "kick_strength",
            "Kick strength A_0",
            0.0,
            1.0,
            cfg.kick_strength,
            is_float=True,
            decimals=4,
        )
        outer.addWidget(grp)

        # The He+ and KS ground orbitals are derived artefacts: the
        # Exact-TDDFT binary auto-loads them from disk if cached, else
        # recomputes — no user toggle needed.

        outer.addStretch()
        return tab

    def _make_mode_tabs_group(self):
        """Top-level tab widget: each tab is a self-contained mode."""
        self._tab_widgets = {"tdse": {}, "exact_tddft": {}}
        grp = QGroupBox("Solver Mode")
        lay = QVBoxLayout(grp)
        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_param_tab("tdse"), "TDSE")
        self.mode_tabs.addTab(
            self._build_param_tab("exact_tddft"), "Exact-TDDFT"
        )
        idx = 1 if getattr(self.config, "mode", "tdse") == "exact_tddft" else 0
        self.mode_tabs.setCurrentIndex(idx)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        lay.addWidget(self.mode_tabs)
        return grp

    def _current_mode(self):
        """Return the solver mode string for the active tab."""
        return "exact_tddft" if self.mode_tabs.currentIndex() == 1 else "tdse"

    def _active_widgets(self):
        """Widget dict for the currently active mode tab."""
        return self._tab_widgets[self._current_mode()]

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
        mode = self._current_mode() if hasattr(self, "mode_tabs") else "tdse"
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

    # Field names matching SimulationConfig that are in every tab.
    _COMMON_FIELDS = (
        "grid_nx",
        "grid_ny",
        "grid_dx",
        "grid_dy",
        "imag_dt",
        "imag_steps",
        "real_dt",
        "real_steps",
        "laser_freq",
        "laser_alpha",
        "laser_cycles",
        "coulomb_eps",
        "absorb_ampl",
        "ionization_box",
        "load_ground",
        "load_excited",
        "n_excited",
        "excited_imag_mult",
        "laser_init_state",
        "auto_mode",
        "auto_target_energy",
        "kick_mode",
        "kick_strength",
    )
    _TDDFT_ONLY_FIELDS = ()

    @staticmethod
    def _read_widget(w):
        from PyQt5.QtWidgets import QCheckBox

        if isinstance(w, QCheckBox):
            return 1 if w.isChecked() else 0
        return w.value()

    @staticmethod
    def _write_widget(w, value):
        from PyQt5.QtWidgets import QCheckBox

        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)

    def _collect_config(self):
        """Read the active tab's widgets into a SimulationConfig."""
        widgets = self._active_widgets()
        mode = self._current_mode()
        kwargs = {"mode": mode}
        for k in self._COMMON_FIELDS:
            kwargs[k] = self._read_widget(widgets[k])
        for k in self._TDDFT_ONLY_FIELDS:
            if k in widgets:
                kwargs[k] = self._read_widget(widgets[k])
        self.config = SimulationConfig(**kwargs)

    def _populate_spins(self):
        """Write self.config into both tabs so each tab reflects the load.

        The mode tab is then switched to match config.mode.
        """
        for mode, widgets in self._tab_widgets.items():
            for k in self._COMMON_FIELDS:
                if k in widgets:
                    self._write_widget(widgets[k], getattr(self.config, k))
            for k in self._TDDFT_ONLY_FIELDS:
                if k in widgets:
                    self._write_widget(widgets[k], getattr(self.config, k, 0))
        idx = 1 if getattr(self.config, "mode", "tdse") == "exact_tddft" else 0
        self.mode_tabs.setCurrentIndex(idx)
        self._on_mode_changed()

    def _on_mode_changed(self, *_):
        """Refresh build status and plot-tab visibility when mode changes."""
        if hasattr(self, "build_status_label"):
            self._update_build_status()
        self._apply_tab_visibility()

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
        mode = self._current_mode()
        if not is_solver_built(mode=mode):
            QMessageBox.warning(self, "Not Built", "Build the solver first.")
            return
        if is_solver_stale(mode=mode):
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

        # Live KS Orbital (Exact-TDDFT only)
        if self._current_mode() == "exact_tddft":
            ks_snaps = find_ks_snapshots(out)
            if len(ks_snaps) > len(self._ks_snapshots):
                self._ks_snapshots = ks_snaps
                self._ks_snap_idx = len(ks_snaps) - 1
                self._show_ks_snapshot(self._ks_snap_idx)

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

    # ---- Live KS orbital snapshot tab ----

    def _make_live_tab(self, name, square, on_prev, on_next, label_attr):
        """Build a live snapshot tab (canvas + prev/next nav).

        Used by Live WF (square 700x700) and Live KS Orbital (default size).
        """
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        if square:
            canvas = PlotCanvas(width=7, height=7, dpi=100, constrained=True)
            canvas.setFixedSize(700, 700)
            vbox.addWidget(canvas, alignment=Qt.AlignCenter)
        else:
            canvas = PlotCanvas()
            vbox.addWidget(canvas)
        self._canvases[name] = canvas

        nav_row = QWidget()
        nav_layout = QHBoxLayout(nav_row)
        nav_layout.setContentsMargins(4, 2, 4, 2)
        btn_prev = QPushButton("◀ Prev")
        btn_prev.clicked.connect(on_prev)
        btn_next = QPushButton("Next ▶")
        btn_next.clicked.connect(on_next)
        lbl = QLabel("No snapshots yet")
        lbl.setAlignment(Qt.AlignCenter)
        setattr(self, label_attr, lbl)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(lbl, stretch=1)
        nav_layout.addWidget(btn_next)
        vbox.addWidget(nav_row)
        return container

    def _show_ks_snapshot(self, idx):
        if not self._ks_snapshots:
            return
        idx = max(0, min(idx, len(self._ks_snapshots) - 1))
        self._ks_snap_idx = idx
        ts, path = self._ks_snapshots[idx]
        nx, dx = self.config.grid_nx, self.config.grid_dx
        orbital = parse_orbital_1d(path, nx)
        c = self._canvases["Live KS Orbital"]
        label = "final" if ts == -1 else f"step {ts}"
        plot_ks_orbital(
            c.axes, orbital, dx, nx, title=rf"KS orbital — {label}"
        )
        c.clear_and_draw()
        self._lbl_ks_snap.setText(
            f"Snapshot {idx + 1} / {len(self._ks_snapshots)}  ({label})"
        )

    def _on_ks_prev(self):
        self._show_ks_snapshot(self._ks_snap_idx - 1)

    def _on_ks_next(self):
        self._show_ks_snapshot(self._ks_snap_idx + 1)

    def _apply_tab_visibility(self):
        """Show/hide plot tabs based on the current solver mode."""
        if not hasattr(self, "tabs"):
            return
        mode = self._current_mode() if hasattr(self, "mode_tabs") else "tdse"
        is_tddft = mode == "exact_tddft"
        for i in range(self.tabs.count()):
            name = self.tabs.tabText(i)
            if name in self._tdse_only_tabs:
                self.tabs.setTabVisible(i, not is_tddft)
            elif name in self._tddft_only_tabs:
                self.tabs.setTabVisible(i, is_tddft)
            else:
                self.tabs.setTabVisible(i, True)

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

        # Exact-TDDFT: ground-state KS orbital + live KS orbital snapshots
        if self._current_mode() == "exact_tddft":
            ks_gs = parse_orbital_1d(out / self.config.ks_ground_file, nx)
            c = self._canvases["Ground State KS Orbital"]
            plot_ks_orbital(
                c.axes, ks_gs, dx, nx, title="Ground-state KS orbital"
            )
            c.clear_and_draw()

            ks_snaps = find_ks_snapshots(out)
            if ks_snaps:
                if len(ks_snaps) != len(self._ks_snapshots):
                    self._ks_snapshots = ks_snaps
                    self._ks_snap_idx = len(ks_snaps) - 1
                self._show_ks_snapshot(self._ks_snap_idx)

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
