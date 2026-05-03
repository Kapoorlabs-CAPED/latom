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
    parse_vks_fft,
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
    plot_ks_potential_fft,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from runner import is_solver_built, is_solver_stale  # noqa: E402
from workers import (  # noqa: E402
    BatchSimulationWorker,
    BuildWorker,
    SimulationWorker,
)


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
        # When non-None, plot tabs read from this dir + use this config
        # (set by the View dropdown after a paper-batch run).
        self._active_case_dir = None
        self._active_case_cfg = None
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

        # Progress strip — replaces the previous (large) log textbox so the
        # parameter panel above gets that vertical space. The latest one-line
        # status message is shown above a percentage bar.
        prog_row = QWidget()
        prog_lay = QVBoxLayout(prog_row)
        prog_lay.setContentsMargins(0, 0, 0, 0)
        prog_lay.setSpacing(2)
        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet(
            "font-size: 11px; color: #444; font-family: monospace;"
        )
        self.status_label.setWordWrap(True)
        prog_lay.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(16)
        prog_lay.addWidget(self.progress_bar)
        left_layout.addWidget(prog_row)

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
            "KS Potential FFT",
            "Paper FFT Comparison",
            "Live WF",
        ]
        # Tabs that are only meaningful in one mode. The others are shown
        # in both. _apply_tab_visibility() toggles these on mode change.
        self._tdse_only_tabs = ("1D Density",)
        self._tddft_only_tabs = (
            "Ground State KS Orbital",
            "Live KS Orbital",
            "KS Potential FFT",
            "Paper FFT Comparison",
        )
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
                # "Excited States" and "Paper FFT Comparison" manage their
                # own subplot grids — no default axes for those.
                canvas = PlotCanvas(
                    single_axes=name
                    not in ("Excited States", "Paper FFT Comparison")
                )
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

        # Pulse shape selector — single source of truth for what A(t) is.
        # All three shapes share the same Frequency / Alpha controls; the
        # extra inputs that follow apply only to specific shapes (the C++
        # ignores them otherwise).
        shape_row = QWidget()
        shape_lay = QHBoxLayout(shape_row)
        shape_lay.setContentsMargins(0, 0, 0, 0)
        shape_lay.addWidget(QLabel("Pulse shape"))
        cmb = QComboBox()
        cmb.addItem("Sinusoidal (sin² × sin)", "sinusoidal")
        cmb.addItem("Trapezoidal (ramp + plateau)", "trapezoidal")
        cmb.addItem("Kick (constant A₀)", "kick")
        idx = cmb.findData(getattr(cfg, "laser_pulse_shape", "sinusoidal"))
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        widgets["laser_pulse_shape"] = cmb
        shape_lay.addWidget(cmb, 1)
        g.addWidget(shape_row)

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
            "Alpha (E)",
            0.001,
            100.0,
            cfg.laser_alpha,
            True,
            tooltip="Electric-field amplitude. A_max = alpha × ω.",
        )
        self._add_spin(
            g,
            widgets,
            "laser_cycles",
            "Cycles (sinusoidal)",
            1.0,
            10000.0,
            cfg.laser_cycles,
            True,
            decimals=1,
            tooltip="Sinusoidal sin² envelope only.",
        )
        self._add_spin(
            g,
            widgets,
            "laser_ramp_cycles",
            "Ramp-up cycles",
            0.0,
            1000.0,
            getattr(cfg, "laser_ramp_cycles", 2.0),
            True,
            decimals=2,
            tooltip="Trapezoidal only.",
        )
        self._add_spin(
            g,
            widgets,
            "laser_plateau_cycles",
            "Plateau cycles",
            0.0,
            10000.0,
            getattr(cfg, "laser_plateau_cycles", 16.0),
            True,
            decimals=2,
            tooltip="Trapezoidal only.",
        )
        self._add_spin(
            g,
            widgets,
            "laser_rampdown_cycles",
            "Ramp-down cycles",
            0.0,
            1000.0,
            getattr(cfg, "laser_rampdown_cycles", 0.0),
            True,
            decimals=2,
            tooltip="Trapezoidal only. 0 = sharp cutoff after plateau.",
        )
        self._add_spin(
            g,
            widgets,
            "kick_strength",
            "Kick strength A₀",
            0.0,
            1.0,
            cfg.kick_strength,
            is_float=True,
            decimals=4,
            tooltip="Kick only — constant value of A(t).",
        )
        self._add_spin(
            g,
            widgets,
            "laser_phi",
            "Carrier phase φ (rad)",
            -100.0,
            100.0,
            getattr(cfg, "laser_phi", 0.0),
            is_float=True,
            decimals=4,
            tooltip="Carrier-envelope phase. Carrier is sin(ωt − φ). "
            "Ignored for kick.",
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
            "Load wf_ground.dat if present, else compute via imaginary-time. "
            "Synced across all tabs and used by the Reproduce-Paper batch.",
        )
        widgets["load_ground"].toggled.connect(
            lambda checked: self._sync_load_ground(checked)
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

        # Kick is now selected via the Pulse-shape combo (kick_strength
        # spin lives in the Laser group), so no separate "Kick" panel.

        # The He+ and KS ground orbitals are derived artefacts: the
        # Exact-TDDFT binary auto-loads them from disk if cached, else
        # recomputes — no user toggle needed.

        outer.addStretch()
        return tab

    # Paper-reproduction cases. ω = laser_freq, E = laser_alpha (latom's
    # convention sets A_amp = alpha · omega, so passing E in laser_alpha
    # gives the paper's stated A_max = ω·E directly).
    @staticmethod
    def _pulse_total_time(shape, freq, cycles, ramp, plateau, rampdown=0.0):
        """Return the total propagation time (a.u.) implied by the pulse.

        - sinusoidal: ``cycles * (2π/ω)``
        - trapezoidal: ``(ramp + plateau + rampdown) * (2π/ω)``
        - kick: 0.0 — caller falls back to its own real_steps.
        """
        if freq <= 0:
            return 0.0
        period = 2.0 * 3.141592653589793 / freq
        if shape == "trapezoidal":
            return (float(ramp) + float(plateau) + float(rampdown)) * period
        if shape == "sinusoidal":
            return float(cycles) * period
        return 0.0

    @classmethod
    def _real_steps_for_pulse(
        cls, shape, freq, cycles, ramp, plateau, real_dt, rampdown=0.0
    ):
        """Number of real-time steps needed to cover the full pulse."""
        if real_dt <= 0:
            return 0
        T = cls._pulse_total_time(shape, freq, cycles, ramp, plateau, rampdown)
        if T <= 0:
            return 0
        return int(round(T / real_dt + 0.5))

    # Paper cases use symmetric trapezoid by default: ramp-down = ramp-up.
    # User can override per-card or in the laser group.
    _PAPER_CASES = {
        "case_1": dict(
            label="Case 1: low-ω (ω=0.056, E=0.063)",
            laser_pulse_shape="trapezoidal",
            laser_freq=0.056,
            laser_alpha=0.063,
            laser_ramp_cycles=2.0,
            laser_plateau_cycles=16.0,
            laser_rampdown_cycles=2.0,
        ),
        "case_2": dict(
            label="Case 2: high-ω (ω=2.6, E=0.34)",
            laser_pulse_shape="trapezoidal",
            laser_freq=2.6,
            laser_alpha=0.34,
            laser_ramp_cycles=4.0,
            laser_plateau_cycles=172.0,
            laser_rampdown_cycles=4.0,
        ),
        "case_3": dict(
            label="Case 3: resonant (ω=0.533, E=0.016)",
            laser_pulse_shape="trapezoidal",
            laser_freq=0.533,
            laser_alpha=0.016,
            laser_ramp_cycles=2.0,
            laser_plateau_cycles=148.0,
            laser_rampdown_cycles=2.0,
        ),
    }

    def _make_mode_tabs_group(self):
        """Top-level tab widget: each tab is a self-contained mode.

        Tabs 0/1 are the actual solver modes. Tab 2 (Reproduce Periodicity Paper)
        contains preset buttons that configure tab 1 (Exact-TDDFT) with
        paper parameters and switch to it.
        """
        self._tab_widgets = {"tdse": {}, "exact_tddft": {}}
        grp = QGroupBox("Solver Mode")
        lay = QVBoxLayout(grp)
        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_param_tab("tdse"), "TDSE")
        self.mode_tabs.addTab(
            self._build_param_tab("exact_tddft"), "Exact-TDDFT"
        )
        self.mode_tabs.addTab(
            self._build_reproduce_tab(), "Reproduce Periodicity Paper"
        )
        idx = 1 if getattr(self.config, "mode", "tdse") == "exact_tddft" else 0
        self.mode_tabs.setCurrentIndex(idx)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        lay.addWidget(self.mode_tabs)

        # Live refresh of the paper-case cards when real_dt changes in
        # either parameter tab.
        for mode in ("tdse", "exact_tddft"):
            w = self._tab_widgets.get(mode, {}).get("real_dt")
            if w is not None:
                w.valueChanged.connect(
                    lambda _v: self._refresh_paper_case_cards()
                )
        return grp

    def _build_reproduce_tab(self):
        """Tab with one card per paper case.

        Each card shows the case's pulse parameters and the derived
        propagation time / real_steps for the *current* real_dt. Cards
        update live when real_dt is edited in the Exact-TDDFT tab.
        """
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        header = QLabel(
            "Reproduce three pulse-shape cases from <br>"
            '<a href="https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.042521">'
            "Phys. Rev. A 87, 042521 (2013)</a>."
        )
        header.setOpenExternalLinks(True)
        header.setStyleSheet("font-weight: 600;")
        outer.addWidget(header)

        sub = QLabel(
            "Each case writes into latom_work/&lt;case&gt;/. real_steps "
            "is auto-computed from each case's pulse duration so the "
            "simulation stops exactly when the laser ends."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #555; font-size: 11px;")
        outer.addWidget(sub)

        # Case cards. Each card has:
        #  - a description label (refreshed by _refresh_paper_case_cards)
        #  - per-case real_dt and real_steps override spins. Default to
        #    the values in the Exact-TDDFT tab and the auto-computed
        #    real_steps respectively; user can edit either to override.
        self._paper_case_labels = {}
        self._paper_case_overrides = {}
        for key, case in self._PAPER_CASES.items():
            grp = QGroupBox(case["label"])
            grp.setStyleSheet(
                "QGroupBox { font-weight: 600; margin-top: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin;"
                " left: 8px; padding: 0 4px; }"
            )
            v = QVBoxLayout(grp)
            v.setContentsMargins(8, 6, 8, 6)
            v.setSpacing(4)

            info = QLabel("")
            info.setTextFormat(Qt.RichText)
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 11px;")
            v.addWidget(info)
            self._paper_case_labels[key] = info

            # Override spins inline.
            ovr_row = QWidget()
            ovr_lay = QHBoxLayout(ovr_row)
            ovr_lay.setContentsMargins(0, 0, 0, 0)
            ovr_lay.addWidget(QLabel("real_dt:"))
            spin_dt = QDoubleSpinBox()
            spin_dt.setDecimals(4)
            spin_dt.setRange(0.0001, 10.0)
            spin_dt.setValue(self.config.real_dt)
            spin_dt.setMaximumWidth(80)
            ovr_lay.addWidget(spin_dt)
            ovr_lay.addSpacing(8)
            ovr_lay.addWidget(QLabel("real_steps:"))
            spin_steps = QSpinBox()
            spin_steps.setRange(1, 100_000_000)
            spin_steps.setMaximumWidth(110)
            ovr_lay.addWidget(spin_steps)
            ovr_lay.addStretch()
            v.addWidget(ovr_row)

            self._paper_case_overrides[key] = {
                "real_dt": spin_dt,
                "real_steps": spin_steps,
                "real_steps_user_set": False,
            }

            # When the user touches real_dt, the auto real_steps changes.
            spin_dt.valueChanged.connect(
                lambda _v, k=key: self._refresh_paper_case_cards()
            )
            # Mark real_steps as user-overridden so we stop auto-refilling.
            spin_steps.valueChanged.connect(
                lambda _v, k=key: self._mark_paper_steps_overridden(k)
            )

            btn = QPushButton("Load this case into Exact-TDDFT tab")
            btn.clicked.connect(
                lambda _checked=False, k=key: self._apply_paper_case(k)
            )
            v.addWidget(btn)

            outer.addWidget(grp)

        # Global "Load 2e ground state from file" toggle, mirrored across
        # all three tabs.
        chk = QCheckBox("Load 2e ground state from file (skip imag-time)")
        chk.setChecked(bool(self.config.load_ground))
        chk.setToolTip(
            "If checked AND latom_work/wf_ground.dat (and the He+ / KS\n"
            "ground orbitals) exist, the batch skips imag-time entirely\n"
            "and copies those files into each case dir."
        )
        chk.toggled.connect(lambda checked: self._sync_load_ground(checked))
        self._reproduce_load_ground_chk = chk
        outer.addWidget(chk)

        btn_all = QPushButton("Run all three cases (parallel)")
        btn_all.setToolTip(
            "Spawns three ExactTDDFT subprocesses concurrently. The Paper "
            "FFT Comparison tab populates as cases complete; switch the "
            "View dropdown to inspect any single case's full plot set."
        )
        btn_all.setStyleSheet(
            "QPushButton { background-color: #2e7dd7; color: white;"
            " font-weight: 600; padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #245fb0; }"
        )
        btn_all.clicked.connect(self._on_run_paper_batch)
        outer.addWidget(btn_all)

        outer.addStretch()
        # Initial population.
        self._refresh_paper_case_cards()
        return tab

    def _refresh_paper_case_cards(self):
        """Recompute the displayed total time / real_steps in each case
        card using the current ``real_dt`` from the Exact-TDDFT tab."""
        if not getattr(self, "_paper_case_labels", None):
            return
        # Pull real_dt live from the Exact-TDDFT tab; fall back to the
        # config default if widgets aren't built yet.
        try:
            real_dt = float(
                self._tab_widgets["exact_tddft"]["real_dt"].value()
            )
        except Exception:
            real_dt = float(getattr(self.config, "real_dt", 0.1))

        for key, case in self._PAPER_CASES.items():
            shape = case.get("laser_pulse_shape", "trapezoidal")
            freq = case["laser_freq"]
            ramp = case.get("laser_ramp_cycles", 0.0)
            plateau = case.get("laser_plateau_cycles", 0.0)
            rampdown = case.get("laser_rampdown_cycles", 0.0)
            E = case["laser_alpha"]
            # Per-card override: read from the card's own real_dt spin
            # if present, otherwise the global default.
            card_real_dt = real_dt
            if key in getattr(self, "_paper_case_overrides", {}):
                card_real_dt = float(
                    self._paper_case_overrides[key]["real_dt"].value()
                )
            T = self._pulse_total_time(
                shape, freq, 0.0, ramp, plateau, rampdown
            )
            steps = self._real_steps_for_pulse(
                shape,
                freq,
                0.0,
                ramp,
                plateau,
                card_real_dt,
                rampdown,
            )
            period = 2 * 3.141592653589793 / freq if freq > 0 else 0.0
            text = (
                f"<b>ω</b> = {freq:g} a.u. &nbsp;&nbsp; "
                f"<b>E</b> = {E:g} a.u. &nbsp;&nbsp; "
                f"<b>shape</b> = {shape}<br>"
                f"period 2π/ω = {period:.3f} a.u.<br>"
                f"ramp / plateau / rampdown = "
                f"{ramp:g} / {plateau:g} / {rampdown:g} cycles<br>"
                f"<b>total T</b> = {T:.2f} a.u.  "
                f"(auto real_steps @ dt={card_real_dt:g} = <b>{steps}</b>)"
            )
            self._paper_case_labels[key].setText(text)
            # If user has overridden real_steps in the card, also reflect
            # the auto value as a placeholder for clarity.
            if key in getattr(self, "_paper_case_overrides", {}):
                ovr = self._paper_case_overrides[key]
                if not ovr["real_steps_user_set"]:
                    ovr["real_steps"].blockSignals(True)
                    ovr["real_steps"].setValue(steps)
                    ovr["real_steps"].blockSignals(False)

    def _mark_paper_steps_overridden(self, key):
        """User edited real_steps for this case → stop auto-refilling it."""
        ovr = self._paper_case_overrides.get(key)
        if ovr is not None:
            ovr["real_steps_user_set"] = True

    def _sync_load_ground(self, checked):
        """Mirror the load_ground toggle to every tab + the Reproduce-Paper
        page so it behaves like a single global setting."""
        targets = []
        for mode in ("tdse", "exact_tddft"):
            w = self._tab_widgets.get(mode, {}).get("load_ground")
            if w is not None:
                targets.append(w)
        repro = getattr(self, "_reproduce_load_ground_chk", None)
        if repro is not None:
            targets.append(repro)
        for w in targets:
            if w.isChecked() != bool(checked):
                w.blockSignals(True)
                w.setChecked(bool(checked))
                w.blockSignals(False)

    def _apply_paper_case(self, key):
        """Write a paper-case preset into BOTH parameter tabs and switch
        to the Exact-TDDFT tab.

        Every widget in every parameter tab that the case touches
        (pulse_shape, ω, E, ramp/plateau/rampdown, real_dt, real_steps)
        is updated to match. ``real_steps`` is auto-derived from the
        case's pulse duration and the tab's current ``real_dt`` — the
        same value the batch will run with, so the GUI never lies about
        what the simulation will do.
        """
        case = self._PAPER_CASES[key]

        # Reference real_dt is taken from the Exact-TDDFT tab; use it
        # for both tabs so they end up showing the same auto real_steps.
        ref_widgets = self._tab_widgets["exact_tddft"]
        if "real_dt" in ref_widgets:
            real_dt = float(self._read_widget(ref_widgets["real_dt"]))
        else:
            real_dt = float(self.config.real_dt)

        steps = self._real_steps_for_pulse(
            case.get("laser_pulse_shape", "sinusoidal"),
            case["laser_freq"],
            case.get("laser_cycles", 0.0),
            case.get("laser_ramp_cycles", 0.0),
            case.get("laser_plateau_cycles", 0.0),
            real_dt,
            case.get("laser_rampdown_cycles", 0.0),
        )

        # Apply to BOTH tabs so whichever one the user looks at next
        # shows the case-driven values.
        for mode in ("tdse", "exact_tddft"):
            widgets = self._tab_widgets.get(mode, {})
            for field, value in case.items():
                if field == "label" or field not in widgets:
                    continue
                self._write_widget(widgets[field], value)
            if steps > 0 and "real_steps" in widgets:
                self._write_widget(widgets["real_steps"], steps)
            if "real_dt" in widgets:
                self._write_widget(widgets["real_dt"], real_dt)

        # Mirror to the per-card override spins; reset the user-set flag
        # so further real_dt changes can keep auto-tracking the new case.
        ovr = self._paper_case_overrides.get(key)
        if ovr is not None and steps > 0:
            ovr["real_steps_user_set"] = False
            ovr["real_steps"].blockSignals(True)
            ovr["real_steps"].setValue(steps)
            ovr["real_steps"].blockSignals(False)
            ovr["real_dt"].blockSignals(True)
            ovr["real_dt"].setValue(real_dt)
            ovr["real_dt"].blockSignals(False)

        # Card text labels reflect the new state.
        self._refresh_paper_case_cards()

        self.mode_tabs.setCurrentIndex(1)
        self._log(
            f"Loaded {case['label']}: real_steps auto-set to {steps} "
            f"(real_dt={real_dt:g}). Click Run to launch."
        )

    def _current_mode(self):
        """Return the solver mode string for the active tab.

        Tab index 2 (Reproduce Periodicity Paper) is a *preset* tab, not a solver mode
        — when active, run as Exact-TDDFT.
        """
        idx = self.mode_tabs.currentIndex()
        return "exact_tddft" if idx >= 1 else "tdse"

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

        # Per-case view selector — picks which output directory the plot
        # tabs render from. Default "Single run" = self.work_dir; after a
        # paper-batch run, each completed case_N appears here.
        view_row = QWidget()
        view_lay = QHBoxLayout(view_row)
        view_lay.setContentsMargins(0, 0, 0, 0)
        view_lay.addWidget(QLabel("View:"))
        self.cmb_view_case = QComboBox()
        self.cmb_view_case.addItem("Single run", None)
        self.cmb_view_case.currentIndexChanged.connect(
            self._on_view_case_changed
        )
        view_lay.addWidget(self.cmb_view_case, 1)
        lay.addWidget(view_row)

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
        """Show the latest message in the status strip.

        The full historical log went away when the QTextEdit was
        replaced by the progress bar. If the user wants persistent log
        output, point them at the per-job stdout files in work_dir.
        """
        # Strip leading prefixes like "[case_1] " so the status label
        # stays readable.
        text = msg.rstrip()
        if text:
            self.status_label.setText(text[-200:])

    def _set_progress(self, percent, status=None):
        self.progress_bar.setValue(int(max(0, min(100, percent))))
        if status:
            self.status_label.setText(status[-200:])

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
        "laser_pulse_shape",
        "laser_freq",
        "laser_alpha",
        "laser_cycles",
        "laser_ramp_cycles",
        "laser_plateau_cycles",
        "laser_rampdown_cycles",
        "laser_phi",
        "kick_strength",
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
    )
    _TDDFT_ONLY_FIELDS = ()

    @staticmethod
    def _read_widget(w):
        from PyQt5.QtWidgets import QCheckBox, QComboBox

        if isinstance(w, QCheckBox):
            return 1 if w.isChecked() else 0
        if isinstance(w, QComboBox):
            return w.currentData()
        return w.value()

    @staticmethod
    def _write_widget(w, value):
        from PyQt5.QtWidgets import QCheckBox, QComboBox

        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            idx = w.findData(value)
            if idx >= 0:
                w.setCurrentIndex(idx)
        else:
            w.setValue(value)

    def _collect_config(self):
        """Read the active tab's widgets into a SimulationConfig."""
        widgets = self._active_widgets()
        mode = self._current_mode()
        kwargs = {"mode": mode}
        for k in self._COMMON_FIELDS:
            if k in widgets:
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

    # Fields that are *case-determined* when the Reproduce-Paper tab is
    # active — disable them in the TDSE / Exact-TDDFT tabs so the user
    # can't accidentally edit values that the batch will overwrite from
    # the case cards.
    _PAPER_DRIVEN_FIELDS = (
        "laser_pulse_shape",
        "laser_freq",
        "laser_alpha",
        "laser_ramp_cycles",
        "laser_plateau_cycles",
        "laser_rampdown_cycles",
        "real_dt",
        "real_steps",
    )

    def _on_mode_changed(self, *_):
        """Refresh build status, plot-tab visibility, and freeze
        case-determined fields when the Reproduce-Paper tab is active."""
        if hasattr(self, "build_status_label"):
            self._update_build_status()
        self._apply_tab_visibility()
        on_repro = self.mode_tabs.currentIndex() == 2
        for mode in ("tdse", "exact_tddft"):
            for field in self._PAPER_DRIVEN_FIELDS:
                w = self._tab_widgets.get(mode, {}).get(field)
                if w is not None:
                    w.setEnabled(not on_repro)

    # ---- Build ----

    def _on_build(self):
        self.btn_build.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # indeterminate (busy) bar
        self._log("Building solver...")
        self._build_worker = BuildWorker()
        self._build_worker.output.connect(self._log)
        self._build_worker.finished.connect(self._on_build_done)
        self._build_worker.start()

    def _on_build_done(self, success, msg):
        self._log(msg)
        self.btn_build.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self._set_progress(100 if success else 0, msg)
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
        # Determinate bar — _poll_outputs computes percent = t_now/T_total.
        self.progress_bar.setRange(0, 100)
        self._set_progress(0, "Starting simulation…")
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
        self.progress_bar.setRange(0, 100)
        self._set_progress(100 if success else 0, msg)
        self._refresh_all_plots()
        if not success:
            QMessageBox.warning(self, "Simulation Error", msg)

    def _on_stop(self):
        if self._sim_worker:
            self._sim_worker.stop()
            self._log("Stopping simulation...")

    # ---- Reproduce Paper: parallel batch over the three cases ----

    def _on_run_paper_batch(self):
        """Spawn ExactTDDFT for all three paper cases concurrently.

        Each case runs in latom_work/<case_name>/ with a config built
        from the Exact-TDDFT tab's *current* widget values (grid, time,
        physics) overridden with that case's pulse-shape / ω / E /
        ramp / plateau. The Paper FFT Comparison tab refreshes when the
        last case finishes.
        """
        if not is_solver_built(mode="exact_tddft"):
            QMessageBox.warning(
                self,
                "Not Built",
                "Build the solver first (Exact-TDDFT binary needed).",
            )
            return
        if is_solver_stale(mode="exact_tddft"):
            QMessageBox.warning(
                self,
                "Rebuild Required",
                "Source files have changed since the last build. "
                "Click 'Build Solver' before running.",
            )
            return

        # Snapshot the current Exact-TDDFT-tab values, then overlay the
        # paper-case fields per job.
        widgets = self._tab_widgets["exact_tddft"]
        base_kwargs = {"mode": "exact_tddft"}
        for k in self._COMMON_FIELDS:
            if k in widgets:
                base_kwargs[k] = self._read_widget(widgets[k])
        for k in self._TDDFT_ONLY_FIELDS:
            if k in widgets:
                base_kwargs[k] = self._read_widget(widgets[k])

        # The 2e GS, He+ GS and seed KS orbital are field-free — same
        # for all three cases. We want to compute them at most ONCE.
        # Three sub-cases:
        #   (a) load_ground=1 AND latom_work/wf_ground.dat exists →
        #       skip the preflight entirely; use the user's existing
        #       file as the shared source.
        #   (b) load_ground=1 but the file is missing → error out
        #       (the user explicitly asked to load and there's nothing
        #       to load).
        #   (c) load_ground=0 → run a preflight in latom_work/shared_ground/
        #       with real_steps=0 so the binary returns straight after
        #       imag-time; that dir becomes the shared source.
        #
        # Either way, in all three case dirs we copy the GS files in
        # and pass load_ground=1 to the per-case binary so it doesn't
        # redo imag-time per case.
        gs_files = ("wf_ground.dat", "wf_heliumplus.dat", "ks_ground.dat")
        load_ground = bool(base_kwargs.get("load_ground", 0))
        single_run_dir = self.work_dir
        shared_dir = self.work_dir / "shared_ground"

        preflight = None
        if load_ground:
            existing = single_run_dir / "wf_ground.dat"
            if not existing.is_file():
                QMessageBox.warning(
                    self,
                    "Ground state file missing",
                    f"'Load 2e ground state from file' is checked but "
                    f"{existing} doesn't exist. Either uncheck the option "
                    f"to compute, or run a single TDSE/Exact-TDDFT job "
                    f"first to produce the GS file.",
                )
                return
            shared_dir = single_run_dir
            self._log(
                f"Reusing existing ground state from {single_run_dir}; "
                "no preflight needed."
            )
        else:
            shared_dir.mkdir(parents=True, exist_ok=True)
            preflight_kwargs = dict(base_kwargs)
            preflight_kwargs["real_steps"] = 0  # imag-time only
            preflight_cfg = SimulationConfig(**preflight_kwargs)
            preflight = ("shared_ground", preflight_cfg, str(shared_dir))

        parallel_jobs = []
        for case_key, case in self._PAPER_CASES.items():
            kwargs = dict(base_kwargs)
            for field, value in case.items():
                if field == "label":
                    continue
                kwargs[field] = value
            kwargs["load_ground"] = 1  # always use the shared 2e GS

            # Per-card overrides: real_dt and real_steps from the
            # Reproduce-Paper card win over both the case defaults and
            # the global Exact-TDDFT-tab values. real_steps is
            # auto-computed from the pulse duration unless the user has
            # explicitly edited it (real_steps_user_set).
            ovr = self._paper_case_overrides.get(case_key, {})
            if "real_dt" in ovr:
                kwargs["real_dt"] = float(ovr["real_dt"].value())
            auto_steps = self._real_steps_for_pulse(
                kwargs.get("laser_pulse_shape", "sinusoidal"),
                kwargs["laser_freq"],
                kwargs.get("laser_cycles", 0.0),
                kwargs.get("laser_ramp_cycles", 0.0),
                kwargs.get("laser_plateau_cycles", 0.0),
                kwargs["real_dt"],
                kwargs.get("laser_rampdown_cycles", 0.0),
            )
            if ovr.get("real_steps_user_set"):
                kwargs["real_steps"] = int(ovr["real_steps"].value())
            elif auto_steps > 0:
                kwargs["real_steps"] = auto_steps

            # Auto-pick ks_every so the V_KS(x,ω) FFT can resolve up to
            # ~harmonic 2.5 of this case's ω_L. Nyquist condition:
            #   π / (real_dt × ks_every) ≥ ~3·ω_L  ⇒
            #   ks_every ≤ π / (3·ω_L·real_dt)
            wL = float(kwargs["laser_freq"])
            dt = float(kwargs["real_dt"])
            ks_every_auto = max(1, int(3.141592653589793 / (3.0 * wL * dt)))
            kwargs["ks_every"] = ks_every_auto

            cfg = SimulationConfig(**kwargs)
            case_dir = self.work_dir / case_key
            case_dir.mkdir(parents=True, exist_ok=True)
            parallel_jobs.append((case_key, cfg, str(case_dir)))
            ks_e = kwargs["ks_every"]
            nyq_h = 3.141592653589793 / (kwargs["real_dt"] * ks_e * wL)
            self._log(
                f"[{case_key}] ω={kwargs['laser_freq']:g}, "
                f"E={kwargs['laser_alpha']:g}, "
                f"ramp/plateau/rampdown="
                f"{kwargs.get('laser_ramp_cycles',0)}/"
                f"{kwargs.get('laser_plateau_cycles',0)}/"
                f"{kwargs.get('laser_rampdown_cycles',0)} cycles → "
                f"T={kwargs['real_steps'] * kwargs['real_dt']:.2f} a.u., "
                f"real_steps={kwargs['real_steps']} "
                f"(real_dt={kwargs['real_dt']:g}, "
                f"ks_every={ks_e}, Nyquist harmonic≈{nyq_h:.1f})"
            )

        self._batch_pending = {name for name, _, _ in parallel_jobs}
        self._batch_dirs = {name: Path(d) for name, _, d in parallel_jobs}
        self._batch_cfgs = {name: cfg for name, cfg, _ in parallel_jobs}

        def _materialise_shared_gs():
            import shutil

            for fname in gs_files:
                src = shared_dir / fname
                if not src.is_file():
                    self._log(
                        f"[shared_ground] WARNING: {fname} not in "
                        f"{shared_dir}; cases will recompute it themselves."
                    )
                    continue
                for case_key in self._PAPER_CASES.keys():
                    dst = self.work_dir / case_key / fname
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(src, dst)
            self._log(
                f"Copied GS files from {shared_dir} into each case dir; "
                "launching 3 cases in parallel…"
            )

        if preflight is None:
            # No preflight subprocess — materialise immediately, then
            # treat all jobs as parallel.
            _materialise_shared_gs()
            after_preflight = None
        else:
            self._log(
                f"Preflight (shared GS) → then {len(parallel_jobs)} parallel "
                f"cases ({', '.join(self._batch_pending)})…"
            )
            after_preflight = _materialise_shared_gs

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self._set_progress(0, f"Launching batch ({len(parallel_jobs)} cases)…")
        # Start the batch poller so the progress bar is driven by real
        # propagation time, not just by case-completion count.
        if not hasattr(self, "_batch_poll_timer"):
            self._batch_poll_timer = QTimer(self)
            self._batch_poll_timer.timeout.connect(self._poll_batch_progress)
        self._batch_poll_timer.start(2000)
        self._batch_worker = BatchSimulationWorker(
            parallel_jobs,
            preflight=preflight,
            after_preflight=after_preflight,
        )
        self._batch_worker.output.connect(self._log)
        self._batch_worker.job_finished.connect(self._on_batch_job_done)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.start()

    def _on_batch_job_done(self, name, success, msg):
        self._log(f"[{name}] {msg}")
        self._batch_pending.discard(name)
        # Progress = fraction of cases finished.
        total = max(len(self._batch_dirs), 1)
        done = total - len(self._batch_pending)
        self._set_progress(
            int(100 * done / total),
            f"{done}/{total} cases done — {name}: {msg}",
        )
        # Refresh the 3-up comparison + the dropdown.
        self._refresh_paper_fft_comparison()
        self._populate_view_dropdown_from_batch()

    def _on_batch_done(self, success, msg):
        self._log(msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "_batch_poll_timer"):
            self._batch_poll_timer.stop()
        self.progress_bar.setRange(0, 100)
        self._set_progress(100 if success else 0, msg)
        self._refresh_paper_fft_comparison()
        self._populate_view_dropdown_from_batch()
        if not success:
            QMessageBox.warning(self, "Batch incomplete", msg)

    def _poll_batch_progress(self):
        """Aggregate progress across all running batch cases.

        For each case dir, read the latest simulation time from
        obser_laser.dat and divide by that case's total propagation time.
        Overall percentage = average of the per-case fractions, weighted
        by total time so a long case carries more weight.
        """
        if not getattr(self, "_batch_dirs", None):
            return
        total_T = 0.0
        elapsed_T = 0.0
        per_case = []
        for case_key, work_dir in self._batch_dirs.items():
            cfg = self._batch_cfgs.get(case_key)
            if cfg is None:
                continue
            T_total = float(cfg.real_dt) * float(cfg.real_steps)
            if T_total <= 0:
                continue
            t_now = self._last_sim_time(work_dir, cfg.obser_file)
            t_now = max(0.0, min(t_now, T_total))
            total_T += T_total
            elapsed_T += t_now
            per_case.append((case_key, t_now, T_total))
        if total_T <= 0:
            return
        pct = 100.0 * elapsed_T / total_T
        # Build a compact per-case status string.
        bits = ", ".join(f"{k}={tn:.0f}/{tT:.0f}" for k, tn, tT in per_case)
        self._set_progress(pct, f"batch {pct:.1f}% — {bits} a.u.")

    def _refresh_paper_fft_comparison(self):
        """Render |V_KS(x,ω)|² for each finished case into the 1×3 panel."""
        if not hasattr(self, "_batch_dirs"):
            return
        c = self._canvases["Paper FFT Comparison"]
        fig = c.figure
        fig.clear()
        case_keys = list(self._PAPER_CASES.keys())
        axes = fig.subplots(1, len(case_keys))
        if len(case_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, case_keys):
            cfg = self._batch_cfgs.get(key)
            out_dir = self._batch_dirs.get(key)
            if cfg is None or out_dir is None or not out_dir.exists():
                ax.set_title(self._PAPER_CASES[key]["label"])
                ax.text(
                    0.5,
                    0.5,
                    "(not run)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            # FFT data was accumulated online by ExactTDDFT.cc into
            # vks_fft.dat; just read and render it. No Python FFT.
            fft_data = parse_vks_fft(
                out_dir / getattr(cfg, "vks_fft_file", "vks_fft.dat")
            )
            plot_ks_potential_fft(
                ax,
                fft_data,
                title=self._PAPER_CASES[key]["label"],
                harmonic_min=0.0,
                harmonic_max=2.5,
                vmin_orders=4,
            )
        c.draw_idle()

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
        if self._active_case_dir is not None:
            return self._active_case_dir
        return self.work_dir / self.config.output_dir

    def _effective_config(self):
        """Config the plotters should use — either the live one (single
        run) or the snapshotted config of the View-selected case."""
        return (
            self._active_case_cfg
            if self._active_case_cfg is not None
            else self.config
        )

    def _on_view_case_changed(self, *_):
        case_key = self.cmb_view_case.currentData()
        if case_key is None:
            self._active_case_dir = None
            self._active_case_cfg = None
            self._log("View: single run")
        else:
            self._active_case_dir = self._batch_dirs.get(case_key)
            self._active_case_cfg = self._batch_cfgs.get(case_key)
            label = self._PAPER_CASES.get(case_key, {}).get("label", case_key)
            self._log(f"View: {label} → {self._active_case_dir}")
        # Reset live-WF and KS snapshot state so the new dir's snapshots
        # populate from scratch.
        self._wf_snapshots = []
        self._wf_snap_idx = -1
        self._ks_snapshots = []
        self._ks_snap_idx = -1
        self._refresh_all_plots()

    def _populate_view_dropdown_from_batch(self):
        """Add (or refresh) entries for completed batch cases."""
        if not hasattr(self, "_batch_dirs"):
            return
        # Remember current selection; rebuild keeping it where possible.
        current = self.cmb_view_case.currentData()
        self.cmb_view_case.blockSignals(True)
        self.cmb_view_case.clear()
        self.cmb_view_case.addItem("Single run", None)
        for case_key in self._PAPER_CASES.keys():
            d = self._batch_dirs.get(case_key)
            if d is None or not d.exists():
                continue
            label = self._PAPER_CASES[case_key]["label"]
            self.cmb_view_case.addItem(label, case_key)
        idx = self.cmb_view_case.findData(current)
        if idx < 0:
            idx = 0
        self.cmb_view_case.setCurrentIndex(idx)
        self.cmb_view_case.blockSignals(False)

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

    @staticmethod
    def _last_sim_time(work_dir, obser_file="obser_laser.dat"):
        """Cheap tail-read of the simulation time from the obser log.

        Returns 0.0 if the file doesn't exist yet or is unreadable. The
        first column of obser_laser.dat is the propagation time in a.u.
        Reading just the last 4 KiB is enough — no need to parse the
        whole file every poll tick.
        """
        path = Path(work_dir) / obser_file
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096), 0)
                tail = f.read().decode(errors="ignore")
            for line in reversed(tail.strip().splitlines()):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tok = line.split()
                if tok:
                    return float(tok[0])
        except (OSError, ValueError):
            pass
        return 0.0

    def _poll_outputs(self):
        """Poll observable files and update live plots + progress bar."""
        out = self._get_output_dir()

        # Live progress for the active single run, if any.
        if self._sim_worker is not None and self._sim_worker.isRunning():
            t_now = self._last_sim_time(out, self.config.obser_file)
            t_total = float(self.config.real_dt) * float(
                self.config.real_steps
            )
            if t_total > 0:
                pct = 100.0 * t_now / t_total
                self._set_progress(
                    pct,
                    f"running… t = {t_now:.2f} / {t_total:.2f} a.u. "
                    f"({pct:.1f}%)",
                )

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
        cfg = self._effective_config()
        nx, ny = cfg.grid_nx, cfg.grid_ny
        dx, dy = cfg.grid_dx, cfg.grid_dy
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
        cfg = self._effective_config()
        nx, dx = cfg.grid_nx, cfg.grid_dx
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
        """Refresh all plots from output files.

        Uses _effective_config() so that the View dropdown can re-route
        the whole plot panel to a paper-batch case's directory + config
        without touching the user's edit state.
        """
        cfg = self._effective_config()
        out = self._get_output_dir()
        nx = cfg.grid_nx
        ny = cfg.grid_ny
        dx = cfg.grid_dx
        dy = cfg.grid_dy

        imag_data = parse_imag_observables(out / cfg.obser_imag_file)
        real_data = parse_real_observables(out / cfg.obser_file)
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
        if cfg.mode == "exact_tddft":
            ks_gs = parse_orbital_1d(out / cfg.ks_ground_file, nx)
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

            # KS potential FFT — read the file the C++ wrote online.
            fft_data = parse_vks_fft(
                out / getattr(cfg, "vks_fft_file", "vks_fft.dat")
            )
            c = self._canvases["KS Potential FFT"]
            plot_ks_potential_fft(
                c.axes,
                fft_data,
                title=(
                    rf"$\log_{{10}}|\hat V_{{\mathrm{{KS}}}}(x,\omega)|^2$  "
                    rf"$\omega_L$={cfg.laser_freq:g}, "
                    rf"E={cfg.laser_alpha:g}"
                ),
                harmonic_min=0.0,
                harmonic_max=2.5,
                vmin_orders=4,
            )
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

        cfg = self._effective_config()
        try:
            n_frames = create_wf_animation(
                out,
                cfg.grid_nx,
                cfg.grid_ny,
                cfg.grid_dx,
                cfg.grid_dy,
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
