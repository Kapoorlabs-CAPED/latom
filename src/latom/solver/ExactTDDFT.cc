// ExactTDDFT driver: solve the 2-electron 1D-each TDSE for helium, then
// reconstruct the exact Kohn-Sham orbital each step via eq. 29 of the
// reference (phi_KS(x) = sqrt(n(x)/2) * exp(i theta(x)), with theta from
// the integrated current). The KS orbital is also propagated forward and
// backward in time so the effective KS potential can be extracted.
//
// Standalone binary; shares wavefunction.cc / grid.cc / hamop.cc / fluid.cc
// with the TDSE binary, but defines its own potential symbols.

#include "TDSE.h"
#include "wavefunction.h"
#include "fluid.h"
#include "grid.h"
#include "hamop.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ============= Config (superset of TDSE.cc; unknown keys ignored) =============
static long   cfg_ngpsx = 1500;
static long   cfg_ngpsy = 1500;
static double cfg_deltx = 0.2;
static double cfg_delty = 0.2;
static double cfg_imag_timestep = 0.25;
static double cfg_real_timestep = 0.1;
static long   cfg_no_of_imag_timesteps = 1000;
static long   cfg_no_of_real_timesteps = 20000;
static int    cfg_obs_output_every = 1;
static long   cfg_wf_output_every = 20000;
// Cadence for the 1D KS-orbital and effective-potential files. They're
// tiny compared to the 2D wavefunction, so we dump them more frequently
// to satisfy Nyquist for the V_KS(x,ω) FFT. 0 means "use wf_every".
static long   cfg_ks_output_every = 0;
static int    cfg_box = 50;
static int    cfg_init_type = 3;
static double cfg_laser_freq = 1.556;
static double cfg_laser_alpha = 0.1;
static double cfg_laser_cycles = 400.0;
// Pulse shape: "sinusoidal" / "trapezoidal" / "kick". Mirrors TDSE.cc.
static char   cfg_laser_pulse_shape[64] = "sinusoidal";
static double cfg_laser_ramp_cycles = 2.0;
static double cfg_laser_plateau_cycles = 16.0;
static double cfg_laser_rampdown_cycles = 0.0;
// Carrier-envelope phase φ in radians. Carrier is sin(ωt − φ).
static double cfg_laser_phi = 0.0;
static double cfg_coulomb_eps = 1.0;
static double cfg_absorb_ampl = 50.0;

// Caching: only the 2e ground state has a user toggle; the He+ orbital
// and the initial KS orbital are derived artefacts and are auto-loaded
// from disk if their files exist, else recomputed.
static int    cfg_load_ground = 0;
static long   cfg_regrid_from_nx_2e = 0;
static long   cfg_regrid_from_nx_1e = 0;
static long   cfg_heplus_imag_steps = 0;  // 0 -> reuse cfg_no_of_imag_timesteps

// Constant A_0 used when laser_pulse_shape == "kick".
static double cfg_kick_strength = 0.01;

// Online FFT of V_KS(x, t) — accumulated at full real_dt resolution
// during the real-time loop, dumped once at the end. No aliasing from
// the wf_every / ks_every snapshot cadence.
static long   cfg_fft_n_omega = 500;
static double cfg_fft_harmonic_min = 0.0;
static double cfg_fft_harmonic_max = 5.0;
static char   cfg_vks_fft_file[512] = "vks_fft.dat";

static char   cfg_output_dir[512]   = "res";
static char   cfg_obser_file[512]   = "obser_laser.dat";
static char   cfg_obser_imag_file[512] = "obserimag_laser.dat";
static char   cfg_wf_file[512]      = "wf_laser.dat";
static char   cfg_ground_file[512]  = "wf_ground.dat";
static char   cfg_heplus_file[512]  = "wf_heliumplus.dat";
static char   cfg_ks_ground_file[512] = "ks_ground.dat";
static char   cfg_ks_orbital_file[512] = "kohnshamorbital_laser.dat";
static char   cfg_realpot_file[512] = "realpot_laser.dat";

static void read_config(const char* filename)
{
  FILE* f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Warning: cannot open config file '%s', using defaults.\n", filename);
    return;
  }
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    char key[256], value[256];
    if (sscanf(line, " %255[^= ] = %255[^\n\r]", key, value) == 2) {
      if      (strcmp(key, "grid_nx") == 0)         cfg_ngpsx = atol(value);
      else if (strcmp(key, "grid_ny") == 0)         cfg_ngpsy = atol(value);
      else if (strcmp(key, "grid_dx") == 0)         cfg_deltx = atof(value);
      else if (strcmp(key, "grid_dy") == 0)         cfg_delty = atof(value);
      else if (strcmp(key, "imag_dt") == 0)         cfg_imag_timestep = atof(value);
      else if (strcmp(key, "real_dt") == 0)         cfg_real_timestep = atof(value);
      else if (strcmp(key, "imag_steps") == 0)      cfg_no_of_imag_timesteps = atol(value);
      else if (strcmp(key, "real_steps") == 0)      cfg_no_of_real_timesteps = atol(value);
      else if (strcmp(key, "obs_every") == 0)       cfg_obs_output_every = atoi(value);
      else if (strcmp(key, "wf_every") == 0)        cfg_wf_output_every = atol(value);
      else if (strcmp(key, "ks_every") == 0)        cfg_ks_output_every = atol(value);
      else if (strcmp(key, "ionization_box") == 0)  cfg_box = atoi(value);
      else if (strcmp(key, "init_type") == 0)       cfg_init_type = atoi(value);
      else if (strcmp(key, "laser_freq") == 0)      cfg_laser_freq = atof(value);
      else if (strcmp(key, "laser_alpha") == 0)     cfg_laser_alpha = atof(value);
      else if (strcmp(key, "laser_cycles") == 0)    cfg_laser_cycles = atof(value);
      else if (strcmp(key, "laser_pulse_shape") == 0) snprintf(cfg_laser_pulse_shape, sizeof(cfg_laser_pulse_shape), "%s", value);
      else if (strcmp(key, "laser_ramp_cycles") == 0) cfg_laser_ramp_cycles = atof(value);
      else if (strcmp(key, "laser_plateau_cycles") == 0) cfg_laser_plateau_cycles = atof(value);
      else if (strcmp(key, "laser_rampdown_cycles") == 0) cfg_laser_rampdown_cycles = atof(value);
      else if (strcmp(key, "laser_phi") == 0) cfg_laser_phi = atof(value);
      else if (strcmp(key, "coulomb_eps") == 0)     cfg_coulomb_eps = atof(value);
      else if (strcmp(key, "absorb_ampl") == 0)     cfg_absorb_ampl = atof(value);
      else if (strcmp(key, "load_ground") == 0)     cfg_load_ground = atoi(value);
      else if (strcmp(key, "regrid_from_nx_2e") == 0) cfg_regrid_from_nx_2e = atol(value);
      else if (strcmp(key, "regrid_from_nx_1e") == 0) cfg_regrid_from_nx_1e = atol(value);
      else if (strcmp(key, "heplus_imag_steps") == 0) cfg_heplus_imag_steps = atol(value);
      else if (strcmp(key, "kick_strength") == 0)   cfg_kick_strength = atof(value);
      else if (strcmp(key, "fft_n_omega") == 0)     cfg_fft_n_omega = atol(value);
      else if (strcmp(key, "fft_harmonic_min") == 0) cfg_fft_harmonic_min = atof(value);
      else if (strcmp(key, "fft_harmonic_max") == 0) cfg_fft_harmonic_max = atof(value);
      else if (strcmp(key, "vks_fft_file") == 0)    snprintf(cfg_vks_fft_file, sizeof(cfg_vks_fft_file), "%s", value);
      else if (strcmp(key, "output_dir") == 0)      snprintf(cfg_output_dir, sizeof(cfg_output_dir), "%s", value);
      else if (strcmp(key, "obser_file") == 0)      snprintf(cfg_obser_file, sizeof(cfg_obser_file), "%s", value);
      else if (strcmp(key, "obser_imag_file") == 0) snprintf(cfg_obser_imag_file, sizeof(cfg_obser_imag_file), "%s", value);
      else if (strcmp(key, "wf_file") == 0)         snprintf(cfg_wf_file, sizeof(cfg_wf_file), "%s", value);
      else if (strcmp(key, "ground_file") == 0)     snprintf(cfg_ground_file, sizeof(cfg_ground_file), "%s", value);
      else if (strcmp(key, "heplus_file") == 0)     snprintf(cfg_heplus_file, sizeof(cfg_heplus_file), "%s", value);
      else if (strcmp(key, "ks_ground_file") == 0)  snprintf(cfg_ks_ground_file, sizeof(cfg_ks_ground_file), "%s", value);
      else if (strcmp(key, "ks_orbital_file") == 0) snprintf(cfg_ks_orbital_file, sizeof(cfg_ks_orbital_file), "%s", value);
      else if (strcmp(key, "realpot_file") == 0)    snprintf(cfg_realpot_file, sizeof(cfg_realpot_file), "%s", value);
      // Unknown keys are silently ignored so a TDSE config stays usable.
    }
  }
  fclose(f);
  fprintf(stdout, "ExactTDDFT config loaded from '%s'\n", filename);
}

// ============= Forward decls for 1-electron (He+) potentials =============
static double scalarpotx_1e(double x, double y, double z, double time, int me);
static double scalarpot_zero(double x, double y, double z, double time, int me);
static double interactionpot_zero(double x, double y, double z, double time, int me);
static double imagpotx_1e(long xindex, long yindex, long zindex, double time, grid g);
static double imagpot_zero_l(long xindex, long yindex, long zindex, double time, grid g);

// ============= Helpers =============
static bool file_exists(const char* path)
{
  FILE* f = fopen(path, "r");
  if (f) { fclose(f); return true; }
  return false;
}

int main(int argc, char **argv)
{
  int me = 0;
  if (argc > 1) read_config(argv[1]);

  // Build paths
  char p_obser[1024], p_obser_imag[1024], p_wf[1024], p_ground[1024];
  char p_heplus[1024], p_ks_ground[1024], p_ks_orbital[1024], p_realpot[1024];
  snprintf(p_obser,      sizeof(p_obser),      "%s/%s", cfg_output_dir, cfg_obser_file);
  snprintf(p_obser_imag, sizeof(p_obser_imag), "%s/%s", cfg_output_dir, cfg_obser_imag_file);
  snprintf(p_wf,         sizeof(p_wf),         "%s/%s", cfg_output_dir, cfg_wf_file);
  snprintf(p_ground,     sizeof(p_ground),     "%s/%s", cfg_output_dir, cfg_ground_file);
  snprintf(p_heplus,     sizeof(p_heplus),     "%s/%s", cfg_output_dir, cfg_heplus_file);
  snprintf(p_ks_ground,  sizeof(p_ks_ground),  "%s/%s", cfg_output_dir, cfg_ks_ground_file);
  snprintf(p_ks_orbital, sizeof(p_ks_orbital), "%s/%s", cfg_output_dir, cfg_ks_orbital_file);
  snprintf(p_realpot,    sizeof(p_realpot),    "%s/%s", cfg_output_dir, cfg_realpot_file);

  FILE* file_obser = fopen(p_obser, "w");
  FILE* file_obser_imag = fopen(p_obser_imag, "w");

  // ============= Grids =============
  long ngpsx = cfg_ngpsx, ngpsy = cfg_ngpsy, ngpsz = 1;
  double deltx = cfg_deltx, delty = cfg_delty, deltz = 1.0;

  grid g;                                 // 2D xy
  g.set_dim(16);
  g.set_ngps(ngpsx, ngpsy, ngpsz);
  g.set_delt(deltx, delty, deltz);
  g.set_offs(ngpsx/2, ngpsy/2, 0);

  grid gone;                              // 1D x (KS orbital, He+)
  gone.set_dim(15);
  gone.set_ngps(ngpsx, 1, 1);
  gone.set_delt(deltx, 1.0, 1.0);
  gone.set_offs(ngpsx/2, 0, 0);

  // ============= Hamiltonians =============
  int    vecpotflag = 1;
  double charge = 0.0;
  double masses[] = {1.0, 1.0};
  complex<double> imagi(0.0, 1.0);

  // 2-electron Hamiltonian (same as TDSE.cc)
  hamop hamilton(g, vecpot_x, vecpot_y, vecpot_z,
                 scalarpotx, scalarpoty, scalarpotz,
                 interactionpotxy, imagpotx, imagpoty, field, dftpot);

  // 1-electron He+ Hamiltonian (1D, no e-e interaction)
  hamop hamilton1e(gone, vecpot_x, vecpot_y, vecpot_z,
                   scalarpotx_1e, scalarpot_zero, scalarpot_zero,
                   interactionpot_zero, imagpotx_1e, imagpot_zero_l, field, dftpot);

  // For KS orbital propagation: bare kinetic + vector potential, no static potential
  // (the KS effective potential is what we extract a posteriori).
  hamop hamiltonKS(gone, vecpot_x, vecpot_y, vecpot_z,
                   scalarpot_zero, scalarpot_zero, scalarpot_zero,
                   interactionpot_zero, imagpotx_1e, imagpot_zero_l, field, dftpot);

  // ============= Wavefunctions =============
  long size2d = g.ngps_x() * g.ngps_y() * g.ngps_z();
  long size1d = gone.ngps_x();

  wavefunction wf(size2d), wfini(size2d);
  wavefunction wfheliumplus(size1d);
  wavefunction kohnshamorbital(size1d), kohnshamdensity(size1d), kohnshamrealdensity(size1d);
  wavefunction kohnshamorbital_prev(size1d);
  wavefunction upper(size1d), lower(size1d), realpot(size1d);
  wavefunction zero1d(size1d); zero1d.nullify();

  wavefunction staticpot_x(g.ngps_x());
  wavefunction staticpot_y(g.ngps_y());
  wavefunction staticpot_xy(size2d);
  staticpot_x.calculate_fixed_potential_array_x(g, hamilton, 0.0, me);
  staticpot_y.calculate_fixed_potential_array_y(g, hamilton, 0.0, me);
  staticpot_xy.calculate_fixed_potential_array_xy(g, hamilton, 0.0, me);

  wavefunction staticpot_x_1e(gone.ngps_x());
  wavefunction staticpot_y_1e(gone.ngps_y());
  wavefunction staticpot_xy_1e(size1d);
  staticpot_x_1e.calculate_fixed_potential_array_x(gone, hamilton1e, 0.0, me);
  staticpot_y_1e.calculate_fixed_potential_array_y(gone, hamilton1e, 0.0, me);
  staticpot_xy_1e.calculate_fixed_potential_array_xy(gone, hamilton1e, 0.0, me);

  complex<double> complenerg(0.0, 0.0);

  // ============= 2e ground state: load or imag-time =============
  int dumpingstepwidth = 1;
  double imag_timestep = cfg_imag_timestep;
  double real_timestep = cfg_real_timestep;

  if (cfg_load_ground && file_exists(p_ground)) {
    cout << "Loading 2e ground state from " << p_ground << endl;
    FILE* f = fopen(p_ground, "r");
    if (cfg_regrid_from_nx_2e > 0 && cfg_regrid_from_nx_2e != ngpsx) {
      long sn = cfg_regrid_from_nx_2e;
      cout << "  cached grid is " << sn << "x" << sn
           << "; regridding onto " << ngpsx << "x" << ngpsy << endl;
      grid g_small;
      g_small.set_dim(g.dimens());
      g_small.set_ngps(sn, sn, ngpsz);
      g_small.set_delt(deltx, delty, deltz);
      g_small.set_offs(sn / 2, sn / 2, 0);
      wavefunction wfread(sn * sn);
      wfread.init(g_small, 99, 0.1, 0.0, 0.0, f, 0);
      wf.nullify();
      wf.regrid(g, g_small, wfread);
    } else {
      wf.init(g, 99, 0.1, 0.0, 0.0, f, 0);
    }
    fclose(f);
    wf *= 1.0 / sqrt(wf.norm(g));
    complenerg = wf.energy(0.0, g, hamilton, me, masses,
                           staticpot_x, staticpot_y, staticpot_xy, charge);
    cout << "  energy = " << real(complenerg) << endl;
  } else {
    cout << "Computing 2e ground state via imaginary-time propagation" << endl;
    wf.init(g, cfg_init_type, 2.0, 0.0, 0.0);
    wf *= 1.0 / sqrt(wf.norm(g));
    complex<double> ts_imag(0.0, -imag_timestep);
    int counter = 0;
    for (long ts = 0; ts < cfg_no_of_imag_timesteps; ts++) {
      wf.propagate(ts_imag, 0.0, g, hamilton, me, vecpotflag,
                   staticpot_x, staticpot_y, staticpot_xy, charge);
      wf *= 1.0 / sqrt(wf.norm(g));
      if (++counter == cfg_obs_output_every) {
        complenerg = wf.energy(0.0, g, hamilton, me, masses,
                               staticpot_x, staticpot_y, staticpot_xy, charge);
        fprintf(file_obser_imag, "%li %.14le %.14le %.14le\n",
                ts, real(complenerg), imag(complenerg), wf.norm(g));
        fflush(file_obser_imag);
        cout << "Imag(2e): " << ts << "  E = " << real(complenerg) << endl;
        counter = 0;
      }
    }
    FILE* f = fopen(p_ground, "w");
    if (f) { wf.dump_to_file(g, f, dumpingstepwidth); fclose(f); }
    cout << "2e ground state written to " << p_ground << endl;
  }
  fclose(file_obser_imag);

  // ============= He+ 1e ground state: auto-load if cached, else imag-time =============
  if (file_exists(p_heplus)) {
    cout << "Loading He+ ground state from " << p_heplus << endl;
    FILE* f = fopen(p_heplus, "r");
    if (cfg_regrid_from_nx_1e > 0 && cfg_regrid_from_nx_1e != gone.ngps_x()) {
      long sn = cfg_regrid_from_nx_1e;
      cout << "  cached length is " << sn
           << "; regridding onto " << gone.ngps_x() << endl;
      grid gone_small;
      gone_small.set_dim(gone.dimens());
      gone_small.set_ngps(sn, 1, 1);
      gone_small.set_delt(deltx, 1.0, 1.0);
      gone_small.set_offs(sn / 2, 0, 0);
      wavefunction wfread(sn);
      wfread.init(gone_small, 99, 0.1, 0.0, 0.0, f, 0);
      wfheliumplus.nullify();
      wfheliumplus.regrid(gone, gone_small, wfread);
    } else {
      wfheliumplus.init(gone, 99, 0.1, 0.0, 0.0, f, 0);
    }
    fclose(f);
    wfheliumplus *= 1.0 / sqrt(wfheliumplus.norm(gone));
  } else {
    cout << "Computing He+ ground state via 1D imaginary-time propagation" << endl;
    wfheliumplus.init(gone, cfg_init_type, 2.0, 0.0, 0.0);
    wfheliumplus *= 1.0 / sqrt(wfheliumplus.norm(gone));
    long heplus_steps = (cfg_heplus_imag_steps > 0) ? cfg_heplus_imag_steps
                                                    : cfg_no_of_imag_timesteps;
    complex<double> ts_imag(0.0, -imag_timestep);
    for (long ts = 0; ts < heplus_steps; ts++) {
      wfheliumplus.propagate(ts_imag, 0.0, gone, hamilton1e, me, vecpotflag,
                             staticpot_x_1e, staticpot_y_1e, staticpot_xy_1e, charge);
      wfheliumplus *= 1.0 / sqrt(wfheliumplus.norm(gone));
      if (ts % 50 == 0) {
        complex<double> e1 = wfheliumplus.energy(0.0, gone, hamilton1e, me, masses,
                                                 staticpot_x_1e, staticpot_y_1e,
                                                 staticpot_xy_1e, charge);
        cout << "Imag(He+): " << ts << "  E = " << real(e1) << endl;
      }
    }
    FILE* f = fopen(p_heplus, "w");
    if (f) { wfheliumplus.dump_to_file(gone, f, dumpingstepwidth); fclose(f); }
    cout << "He+ ground state written to " << p_heplus << endl;
  }

  // ============= Initial KS orbital: auto-load if cached, else build from 2e GS =============
  if (file_exists(p_ks_ground)) {
    cout << "Loading KS ground orbital from " << p_ks_ground << endl;
    FILE* f = fopen(p_ks_ground, "r");
    if (cfg_regrid_from_nx_1e > 0 && cfg_regrid_from_nx_1e != gone.ngps_x()) {
      long sn = cfg_regrid_from_nx_1e;
      cout << "  cached length is " << sn
           << "; regridding onto " << gone.ngps_x() << endl;
      grid gone_small;
      gone_small.set_dim(gone.dimens());
      gone_small.set_ngps(sn, 1, 1);
      gone_small.set_delt(deltx, 1.0, 1.0);
      gone_small.set_offs(sn / 2, 0, 0);
      wavefunction wfread(sn);
      wfread.init(gone_small, 99, 0.1, 0.0, 0.0, f, 0);
      kohnshamorbital.nullify();
      kohnshamorbital.regrid(gone, gone_small, wfread);
    } else {
      kohnshamorbital.init(gone, 99, 0.1, 0.0, 0.0, f, 0);
    }
    fclose(f);
  } else {
    cout << "Building KS ground orbital from 2e GS density (1D marginal)" << endl;
    kohnshamorbital = wf.dens1d(g);
    FILE* f = fopen(p_ks_ground, "w");
    if (f) { kohnshamorbital.dump_to_file(gone, f, dumpingstepwidth); fclose(f); }
    cout << "KS ground orbital written to " << p_ks_ground << endl;
  }

  wfini = wf;

  // ============= Real-time propagation with KS reconstruction =============
  cout << "\n===== Real-time propagation =====" << endl;

  // Per-snapshot files mirror the TDSE convention so the GUI's
  // find_wf_snapshots / find_ks_snapshots logic can pick them up.
  char snap_path[1024];
  complex<double> timestep(real_timestep, 0.0);
  long no_of_real = cfg_no_of_real_timesteps;
  long counter_obs = 0, counter_wf = 0, counter_ks = 0;
  // Effective KS dump cadence — falls back to wf_every when the user
  // hasn't set ks_every. The 1D KS / realpot files are tiny, so it's
  // worth dumping them frequently (Nyquist for the V_KS(x,ω) FFT scales
  // as 1/(real_dt * ks_every)).
  long ks_every = (cfg_ks_output_every > 0) ? cfg_ks_output_every : cfg_wf_output_every;

  // ============= Online FFT of V_KS(x, t) =============
  //
  // Accumulator: F_hat[ix * n_omega + j] += V_KS(x_ix, t) * exp(-iω_j t) * dt
  //
  // ω_j = (harmonic_min + j * (harmonic_max - harmonic_min)/(n_omega-1)) * ω_L
  //
  // Cost per step: O(N_x · N_ω). With 1500 × 500 ≈ 7.5e5 ops/step, this
  // is negligible next to the Crank–Nicolson 2D solve. Total memory:
  // 16 · N_x · N_ω bytes (≈ 12 MB for the defaults).
  long n_omega = (cfg_fft_n_omega > 1) ? cfg_fft_n_omega : 1;
  double omega_L_fft = cfg_laser_freq;
  double omega_min_fft = cfg_fft_harmonic_min * omega_L_fft;
  double omega_max_fft = cfg_fft_harmonic_max * omega_L_fft;
  double dom_fft = (n_omega > 1)
      ? (omega_max_fft - omega_min_fft) / (double)(n_omega - 1)
      : 0.0;
  std::vector<double> omegas_fft(n_omega);
  for (long j = 0; j < n_omega; j++) {
    omegas_fft[j] = omega_min_fft + j * dom_fft;
  }
  std::vector<complex<double>> Fhat(size1d * (size_t)n_omega,
                                    complex<double>(0.0, 0.0));
  // Hann window: V_KS(x, t) is multiplied by w(t) = 0.5(1 - cos(2π t / T))
  // before each accumulation. Suppresses spectral leakage from the
  // finite-duration record without any post-processing.
  double T_total_fft = (double)no_of_real * real_timestep;
  cout << "Online V_KS FFT: n_omega=" << n_omega
       << "  ω range = [" << omega_min_fft << ", " << omega_max_fft
       << "] a.u.  (harmonic 0..." << cfg_fft_harmonic_max << ")"
       << "  (Hann window over T=" << T_total_fft << " a.u.)" << endl;

  for (long ts = 0; ts < no_of_real; ts++) {
    double time = real_timestep * (double)ts;

    // 1) Propagate exact 2e wavefunction
    wf.propagate(timestep, time + 0.5*real(timestep), g, hamilton, me, vecpotflag,
                 staticpot_x, staticpot_y, staticpot_xy, charge);

    // 2) Reconstruct KS orbital via eq. 29
    kohnshamorbital_prev = kohnshamorbital;
    kohnshamdensity      = wf.denskohnsham(g);
    kohnshamrealdensity  = kohnshamdensity.denskohnshamcorrector(gone, g, wfheliumplus, wf);
    kohnshamorbital      = kohnshamrealdensity.phasekohnshamorbital(gone, g, wf, cfg_box);

    // 3) Forward (-dt/2) and backward (+dt/2) propagate to extract the
    //    effective KS potential a la Lein/Kreibich-style:
    //       V_eff = (i/dt) ln(phi_forward / phi_backward)
    upper = kohnshamorbital;
    lower = kohnshamorbital_prev;
    upper.propagatedft(-0.5*timestep, time + 0.5*real(timestep), gone, hamiltonKS, me,
                       vecpotflag, zero1d, zero1d, charge);
    lower.propagatedft( 0.5*timestep, time + 0.5*real(timestep), gone, hamiltonKS, me,
                       vecpotflag, zero1d, zero1d, charge);
    realpot = (upper / lower).alog(gone, real(timestep));

    // 4) Accumulate the running FFT of V_KS(x, t) at full real_dt
    //    resolution. dt is real(timestep), time is the current step time.
    {
      double dt_step = real(timestep);
      double cur_time = time;
      // Hann window factor — applied to the signal V_KS(x, t) before the
      // FFT integral, so the spectrum we dump is already cleaned of
      // rectangular-window sinc lobes. No post-processing needed.
      double hann_w = 0.5 * (1.0 - cos(2.0 * M_PI * cur_time
                                       / std::max(T_total_fft, 1e-30)));
      #pragma omp parallel for schedule(static)
      for (long j = 0; j < n_omega; j++) {
        double phase_arg = -omegas_fft[j] * cur_time;
        complex<double> phase_dt = complex<double>(cos(phase_arg),
                                                   sin(phase_arg)) * dt_step;
        for (long ix = 0; ix < size1d; ix++) {
          double v = real(realpot[ix]) * hann_w;
          Fhat[(size_t)ix * (size_t)n_omega + (size_t)j] += v * phase_dt;
        }
      }
    }

    counter_obs++;
    counter_wf++;
    counter_ks++;

    if (counter_obs == cfg_obs_output_every) {
      complenerg = wf.energy(0.0, g, hamilton, me, masses,
                             staticpot_x, staticpot_y, staticpot_xy, charge);
      complex<double> gspop = wf * wfini * g.delt_x() * g.delt_y();
      // Columns 1..8 must match TDSE.cc exactly so the GUI's parser
      // (REAL_COLUMNS in scripts/parser.py) lines up:
      //   time, E_re, E_im, norm, <x>, <y>, vecpot_x, |<gs|psi>|^2
      // KS-specific extras go AFTER that (column 9+); the parser
      // truncates trailing columns it doesn't know about.
      fprintf(file_obser,
              " %.14le %.14le %.14le %.14le %.14le %.14le %.14le %.14le %.14le\n",
              time + real(timestep), real(complenerg), imag(complenerg),
              wf.norm(g),
              wf.expect_x(g), wf.expect_y(g),
              hamilton.vecpot_x(time, me),
              real(conj(gspop) * gspop),
              kohnshamorbital.norm(gone));
      fflush(file_obser);
      cout << "Real: " << ts << "  <x>=" << wf.expect_x(g)
           << "  KS norm=" << kohnshamorbital.norm(gone) << endl;
      counter_obs = 0;
    }

    // 2D wavefunction dump (large file → coarser cadence).
    if (counter_wf == cfg_wf_output_every) {
      snprintf(snap_path, sizeof(snap_path), "%s/wf_real_%06ld.dat", cfg_output_dir, ts);
      FILE* f = fopen(snap_path, "w");
      if (f) { wf.dump_to_file(g, f, dumpingstepwidth); fclose(f); }
      counter_wf = 0;
    }

    // 1D KS-orbital and effective-potential dumps (small files →
    // dump as often as needed to satisfy Nyquist for the V_KS FFT).
    if (counter_ks == ks_every) {
      snprintf(snap_path, sizeof(snap_path), "%s/ks_real_%06ld.dat", cfg_output_dir, ts);
      FILE* f = fopen(snap_path, "w");
      if (f) { kohnshamorbital.dump_to_file(gone, f, dumpingstepwidth); fclose(f); }

      snprintf(snap_path, sizeof(snap_path), "%s/realpot_real_%06ld.dat", cfg_output_dir, ts);
      f = fopen(snap_path, "w");
      if (f) { realpot.dump_to_file(gone, f, dumpingstepwidth); fclose(f); }
      counter_ks = 0;
    }
  }

  // Final snapshot
  snprintf(snap_path, sizeof(snap_path), "%s/wf_real_final.dat", cfg_output_dir);
  { FILE* f = fopen(snap_path, "w");
    if (f) { wf.dump_to_file(g, f, dumpingstepwidth); fclose(f); } }
  snprintf(snap_path, sizeof(snap_path), "%s/ks_real_final.dat", cfg_output_dir);
  { FILE* f = fopen(snap_path, "w");
    if (f) { kohnshamorbital.dump_to_file(gone, f, dumpingstepwidth); fclose(f); } }
  snprintf(snap_path, sizeof(snap_path), "%s/realpot_real_final.dat", cfg_output_dir);
  { FILE* f = fopen(snap_path, "w");
    if (f) { realpot.dump_to_file(gone, f, dumpingstepwidth); fclose(f); } }

  // Dump the V_KS(x, ω) spectrum accumulated online. File format:
  //   header line:  # nx dx x_offset n_omega omega_min omega_max omega_L
  //   then nx lines, each with n_omega doubles = |F[ix, j]|^2.
  {
    char fft_path[1024];
    snprintf(fft_path, sizeof(fft_path), "%s/%s",
             cfg_output_dir, cfg_vks_fft_file);
    FILE* f = fopen(fft_path, "w");
    if (f) {
      double x_offset = (-(double)size1d / 2.0 + 0.5) * deltx;
      fprintf(f,
              "# nx=%ld dx=%.14e x_offset=%.14e n_omega=%ld "
              "omega_min=%.14e omega_max=%.14e omega_L=%.14e\n",
              size1d, deltx, x_offset, n_omega,
              omega_min_fft, omega_max_fft, omega_L_fft);
      for (long ix = 0; ix < size1d; ix++) {
        for (long j = 0; j < n_omega; j++) {
          complex<double> z = Fhat[(size_t)ix * (size_t)n_omega + (size_t)j];
          double power = real(z) * real(z) + imag(z) * imag(z);
          fprintf(f, "%.8e ", power);
        }
        fputc('\n', f);
      }
      fclose(f);
      cout << "V_KS(x, ω) FFT written to " << fft_path
           << " (" << size1d << "×" << n_omega << ")" << endl;
    } else {
      cerr << "WARNING: could not open " << fft_path
           << " for writing." << endl;
    }
  }

  fclose(file_obser);

  cout << me << ": ExactTDDFT done." << endl;
  return 0;
}

// ============= Potentials (laser, 2e Coulomb, absorber) =============
//
// Mirrors TDSE.cc — duplicated because each binary is a separate translation
// unit; we cannot share these symbols across the two executables.

double vecpot_x(double time, int me)
{
  if (strcmp(cfg_laser_pulse_shape, "kick") == 0) return cfg_kick_strength;
  if (time <= 0.0) return 0.0;

  double frequ    = cfg_laser_freq;
  double alphahat = cfg_laser_alpha;
  // alphahat is the *electric-field* amplitude E_0 (config: laser_alpha).
  // A_0 = E_0 / omega in velocity gauge.
  double ampl     = alphahat / frequ;

  double phi = cfg_laser_phi;

  if (strcmp(cfg_laser_pulse_shape, "trapezoidal") == 0) {
    double T_period = 2.0 * M_PI / frequ;
    double T_up     = cfg_laser_ramp_cycles     * T_period;
    double T_const  = cfg_laser_plateau_cycles  * T_period;
    double T_down   = cfg_laser_rampdown_cycles * T_period;
    double env = 0.0;
    // Phase 1: ramp-up.   0 <= t < T_up
    // Phase 2: plateau.   T_up <= t < T_up + T_const
    // Phase 3: ramp-down. T_up + T_const <= t < T_up + T_const + T_down
    // Phase 4: laser off. t >= T_up + T_const + T_down
    if (time < T_up) {
      env = (T_up > 0.0) ? time / T_up : 1.0;
    } else if (time < T_up + T_const) {
      env = 1.0;
    } else if (time >= T_up + T_const &&
               time <  T_up + T_const + T_down &&
               T_down > 0.0) {
      env = 1.0 - (time - (T_up + T_const)) / T_down;
    } else {
      env = 0.0;
    }
    return ampl * env * sin(frequ * time - phi);
  }

  // Default "sinusoidal": sin^2 envelope over laser_cycles total cycles.
  double n  = cfg_laser_cycles;
  double ww = 0.5 * frequ / n;
  return ampl * sin(ww*time) * sin(ww*time) * sin(frequ*time - phi);
}

double alpha_y(double, int) { return 0.0; }
double alpha_x(double, int) { return 0.0; }
double alpha_z(double time, int me) { return alpha_y(time, me); }
double vecpot_y(double time, int me) { return vecpot_x(time, me); }
double vecpot_z(double, int) { return 0.0; }

double scalarpotx(double x, double, double, double, int)
{
  double eps = cfg_coulomb_eps;
  return -2.0 / sqrt(x*x + eps*eps);
}
double scalarpoty(double, double y, double, double, int)
{
  double eps = cfg_coulomb_eps;
  return -2.0 / sqrt(y*y + eps*eps);
}
double scalarpotz(double, double, double, double, int) { return 0.0; }

double interactionpotxy(double x, double y, double, double, int)
{
  double eps = cfg_coulomb_eps;
  return 1.0 / sqrt((x-y)*(x-y) + eps*eps);
}

double field(double, int) { return 0.0; }

double imagpotx(long xindex, long, long, double, grid g)
{
  double ampl = cfg_absorb_ampl;
  if (ampl <= 1.0) return 0.0;
  double x = ((double)xindex + 0.5 - 0.5*g.ngps_x()) / (0.5*g.ngps_x());
  x *= x;
  return ampl * x*x*x*x*x*x*x*x;
}
double imagpoty(long, long yindex, long, double, grid g)
{
  double ampl = cfg_absorb_ampl;
  if (ampl <= 1.0) return 0.0;
  double y = ((double)yindex + 0.5 - 0.5*g.ngps_y()) / (0.5*g.ngps_y());
  y *= y;
  return ampl * y*y*y*y*y*y*y*y;
}

double dftpot(grid, double, double, double, double, int,
              const fluid &, const wavefunction &) { return 0.0; }
double dfthartree(grid, double, double, double, double, int,
                  const wavefunction &, double, long) { return 0.0; }

// ============= 1-electron He+ potentials =============
static double scalarpotx_1e(double x, double, double, double, int)
{
  double eps = cfg_coulomb_eps;
  return -2.0 / sqrt(x*x + eps*eps);
}
static double scalarpot_zero(double, double, double, double, int) { return 0.0; }
static double interactionpot_zero(double, double, double, double, int) { return 0.0; }

static double imagpotx_1e(long xindex, long, long, double, grid g)
{
  double ampl = cfg_absorb_ampl;
  if (ampl <= 1.0) return 0.0;
  double x = ((double)xindex + 0.5 - 0.5*g.ngps_x()) / (0.5*g.ngps_x());
  x *= x;
  return ampl * x*x*x*x*x*x*x*x;
}
static double imagpot_zero_l(long, long, long, double, grid) { return 0.0; }
