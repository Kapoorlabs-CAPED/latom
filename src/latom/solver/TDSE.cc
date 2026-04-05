#include "TDSE.h"
#include "wavefunction.h"
#include "fluid.h"
#include "grid.h"
#include "hamop.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

// Global simulation parameters (read from config file or defaults)
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
static int    cfg_box = 50;
static int    cfg_init_type = 3;
static double cfg_laser_freq = 1.556;
static double cfg_laser_alpha = 0.1;
static double cfg_laser_cycles = 400.0;
static double cfg_coulomb_eps = 1.0;
static double cfg_absorb_ampl = 50.0;
static char   cfg_output_dir[512] = "res";
static char   cfg_wf_file[512] = "wf_laser.dat";
static char   cfg_obser_file[512] = "obser_laser.dat";
static char   cfg_obser_imag_file[512] = "obserimag_laser.dat";
static char   cfg_reading_file[512] = "wf_ground.dat";

// Simple config file reader (key = value format)
static void read_config(const char* filename)
{
  FILE* f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Warning: cannot open config file '%s', using defaults.\n", filename);
    return;
  }

  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

    char key[256], value[256];
    if (sscanf(line, " %255[^= ] = %255[^\n\r]", key, value) == 2) {
      if (strcmp(key, "grid_nx") == 0)         cfg_ngpsx = atol(value);
      else if (strcmp(key, "grid_ny") == 0)    cfg_ngpsy = atol(value);
      else if (strcmp(key, "grid_dx") == 0)    cfg_deltx = atof(value);
      else if (strcmp(key, "grid_dy") == 0)    cfg_delty = atof(value);
      else if (strcmp(key, "imag_dt") == 0)    cfg_imag_timestep = atof(value);
      else if (strcmp(key, "real_dt") == 0)    cfg_real_timestep = atof(value);
      else if (strcmp(key, "imag_steps") == 0) cfg_no_of_imag_timesteps = atol(value);
      else if (strcmp(key, "real_steps") == 0) cfg_no_of_real_timesteps = atol(value);
      else if (strcmp(key, "obs_every") == 0)  cfg_obs_output_every = atoi(value);
      else if (strcmp(key, "wf_every") == 0)   cfg_wf_output_every = atol(value);
      else if (strcmp(key, "ionization_box") == 0) cfg_box = atoi(value);
      else if (strcmp(key, "init_type") == 0)  cfg_init_type = atoi(value);
      else if (strcmp(key, "laser_freq") == 0) cfg_laser_freq = atof(value);
      else if (strcmp(key, "laser_alpha") == 0) cfg_laser_alpha = atof(value);
      else if (strcmp(key, "laser_cycles") == 0) cfg_laser_cycles = atof(value);
      else if (strcmp(key, "coulomb_eps") == 0) cfg_coulomb_eps = atof(value);
      else if (strcmp(key, "absorb_ampl") == 0) cfg_absorb_ampl = atof(value);
      else if (strcmp(key, "output_dir") == 0) snprintf(cfg_output_dir, sizeof(cfg_output_dir), "%s", value);
      else if (strcmp(key, "wf_file") == 0) snprintf(cfg_wf_file, sizeof(cfg_wf_file), "%s", value);
      else if (strcmp(key, "obser_file") == 0) snprintf(cfg_obser_file, sizeof(cfg_obser_file), "%s", value);
      else if (strcmp(key, "obser_imag_file") == 0) snprintf(cfg_obser_imag_file, sizeof(cfg_obser_imag_file), "%s", value);
      else if (strcmp(key, "reading_file") == 0) snprintf(cfg_reading_file, sizeof(cfg_reading_file), "%s", value);
      else fprintf(stderr, "Warning: unknown config key '%s'\n", key);
    }
  }
  fclose(f);
  fprintf(stdout, "Config loaded from '%s'\n", filename);
}


int main(int argc, char **argv)
{
  int me=0;

  // Read config file if provided as first argument
  if (argc > 1) {
    read_config(argv[1]);
  }

  FILE *file_wfdat, *file_wf_exciteddat, *file_reading;
  FILE *file_obser, *file_obser_imag;

  // Build output file paths
  char string_wfdat[1024];
  char string_obser[1024];
  char string_obser_imag[1024];
  char string_reading[1024];
  snprintf(string_wfdat, sizeof(string_wfdat), "%s/%s", cfg_output_dir, cfg_wf_file);
  snprintf(string_obser, sizeof(string_obser), "%s/%s", cfg_output_dir, cfg_obser_file);
  snprintf(string_obser_imag, sizeof(string_obser_imag), "%s/%s", cfg_output_dir, cfg_obser_imag_file);
  snprintf(string_reading, sizeof(string_reading), "%s/%s", cfg_output_dir, cfg_reading_file);

  file_wfdat = fopen(string_wfdat, "w");
  file_obser = fopen(string_obser, "w");
  file_obser_imag = fopen(string_obser_imag, "w");
  file_reading = fopen(string_reading, "r");

  long index, rrindex, xindex, yindex, zindex, index2;
  complex<double> imagi(0.0, 1.0);
  double deltx = cfg_deltx;
  double delty = cfg_delty;
  double deltz = 1;
  long ngpsx = cfg_ngpsx;
  long ngpsy = cfg_ngpsy;
  long ngpsz = 1;

  // Declare grid
  grid g;
  g.set_dim(16);
  g.set_ngps(ngpsx, ngpsy, ngpsz);
  g.set_delt(deltx, delty, deltz);
  g.set_offs(ngpsx/2, ngpsy/2, 0);

  // Declare smaller grid for reading
  grid g_small;
  g_small.set_dim(16);
  g_small.set_ngps(ngpsx/2, ngpsy/2, ngpsz/2);
  g_small.set_delt(deltx, delty, deltz);
  g_small.set_offs(ngpsx/4, ngpsy/4, 0);

  // Declare rest of variables
  double imag_timestep = cfg_imag_timestep;
  double real_timestep = cfg_real_timestep;
  long   no_of_imag_timesteps = cfg_no_of_imag_timesteps;
  long   no_of_real_timesteps = cfg_no_of_real_timesteps;
  int    obs_output_every = cfg_obs_output_every;
  long   wf_output_every = cfg_wf_output_every;
  int    dumpingstepwidth = 1;
  int    vecpotflag = 1;
  int    box = cfg_box;
  double masses[] = {1.0, 1.0};
  double charge = 0.0;
  double epsilon = 0.00001;
  hamop interactionhamil(g, vecpot_x, vecpot_y, vecpot_z, interactionpotxy, scalarpoty, scalarpotz, interactionpotxy, imagpotx, imagpoty, field, dftpot);
  hamop hamilton(g, vecpot_x, vecpot_y, vecpot_z, scalarpotx, scalarpoty, scalarpotz, interactionpotxy, imagpotx, imagpoty, field, dftpot);

  wavefunction wf(g.ngps_x()*g.ngps_y()*g.ngps_z());
  wavefunction wf_excited(g.ngps_x()*g.ngps_y()*g.ngps_z());
  wavefunction wfread(g.ngps_x()*g.ngps_y()*g.ngps_z());

  wavefunction wfini(g.ngps_x()*g.ngps_y()*g.ngps_z());
  wavefunction wfeverythingeven(g.ngps_x()*g.ngps_y()*g.ngps_z());
  wavefunction wfeverythingodd(g.ngps_x()*g.ngps_y()*g.ngps_z());

  complex<double> timestep;
  double time=0.0;

  long counter_i=0;
  long counter_ii=0;

  // Initialization
  long outputofinterest=0;

  wf.init(g, cfg_init_type, 2.0, 0.0, 0.0);
  wf*=1.0/sqrt(wf.norm(g));

  if (file_reading) fclose(file_reading);

  cout << "norm wf    : " << wf.norm(g) << "\n";

  wavefunction staticpot_x(g.ngps_x());
  staticpot_x.calculate_fixed_potential_array_x(g, hamilton, 0.0, me);

  wavefunction staticpot_y(g.ngps_y());
  staticpot_y.calculate_fixed_potential_array_y(g, hamilton, 0.0, me);

  wavefunction staticpot(g.ngps_x()*g.ngps_y()*g.ngps_z());
  staticpot.calculate_fixed_potential_array(g, hamilton, 0.0, me);

  wavefunction staticpot_xy(g.ngps_z()*g.ngps_y()*g.ngps_x());
  staticpot_xy.calculate_fixed_potential_array_xy(g, hamilton, 0.0, me);

  complex<double> dftapproxenerg;
  complex<double> complenerg;
  complex<double> groundstatepop, excitedstatepop;

  // ============= Imaginary time propagation =============

  long ts;
  long no_of_timesteps = no_of_imag_timesteps;
  for (ts=0; ts<no_of_timesteps; ts++)
    {
      cout << "Imag: " << ts << "  " << " energy  " << real(complenerg) << "  " << endl;

      counter_i++;
      counter_ii++;
      timestep=complex<double>(0.0*real_timestep, -1.0*imag_timestep);
      time=-imag(timestep*(complex<double>)(ts));

      // Propagation
      wf.propagate(timestep, 0.0, g, hamilton, me, vecpotflag, staticpot_x, staticpot_y, staticpot_xy, charge);
      wf*=1.0/sqrt(wf.norm(g));

      if (counter_ii==obs_output_every)
        {
          complenerg=wf.energy(0.0, g, hamilton, me, masses, staticpot_x, staticpot_y, staticpot_xy, charge);

          fprintf(file_obser_imag,"%li %.14le %.14le %.14le %.14le %.14le %.14le %.14le %.14le\n",
                  ts, real(complenerg), imag(complenerg), real(complenerg), wf.norm(g), wf.expect_x(g), wf.expect_y(g), wf.doub_ionized(g,box), wf.expect_x(g));
          fflush(file_obser_imag);
          counter_ii=0;
        };
    };

  fclose(file_obser_imag);

  wfini=wf;

  counter_i=0;
  counter_ii=0;

  wf=wfini;

  // ============= Real time propagation =============

  timestep=complex<double>(real_timestep, 0.0);
  no_of_timesteps=no_of_real_timesteps;

  for (ts=0; ts<no_of_timesteps; ts++)
    {
      counter_i++;
      counter_ii++;
      time=real(timestep*(complex<double>)(ts));
      cout << "Real: " << ts << "  " << " Expectation value x : " << wf.expect_x(g) << " " << " Energy " << real(complenerg) << " " << endl;

      wf.propagate(timestep, time, g, hamilton, me, vecpotflag, staticpot_x, staticpot_y, staticpot_xy, charge);

      if (counter_ii==obs_output_every)
        {
          complenerg=wf.energy(0.0, g, hamilton, me, masses, staticpot_x, staticpot_y, staticpot_xy, charge);

          groundstatepop=wf*wfini*g.delt_x()*g.delt_y();

          fprintf(file_obser," %.14le %.14le %.14le %.14le %.14le  %.14le  %.14le %.14le\n ",
                  (time+real(timestep)), real(complenerg), imag(complenerg), wf.norm(g), wf.expect_x(g), wf.expect_y(g), hamilton.vecpot_x(time,me), real(conj(groundstatepop)*groundstatepop));
          fflush(file_obser);

          counter_ii=0;
        };
    };

  wf.dump_to_file(g, file_wfdat, dumpingstepwidth);

  fclose(file_obser);
  fclose(file_wfdat);

  cout << me << ": Hasta la vista, ... " << endl;

}

// ========================== Potentials ==========================

double vecpot_x(double time, int me)
{
  double result=0.0;

  double frequ = cfg_laser_freq;
  double alphahat = cfg_laser_alpha;
  double n = cfg_laser_cycles;
  double ampl = alphahat*frequ;
  double dur = n*2.0*M_PI/frequ;
  double ww = 0.5*frequ/n;

  if ((time>0.0))
    {
      result=ampl*sin(ww*time)*sin(ww*time)*sin(frequ*time);
    };

  return result;
}

double alpha_y(double time, int me)
{
  double result=0.0;
  return result;
}

double alpha_x(double time, int me)
{
  double result=0.0;
  return result;
}

double alpha_z(double time, int me)
{
  return alpha_y(time,me);
}

double vecpot_y(double time, int me)
{
  return vecpot_x(time,me);
}

double vecpot_z(double time, int me)
{
  return 0.0;
}

double scalarpotx(double x, double y, double z, double time, int me)
{
  double eps = cfg_coulomb_eps;
  double result;
  result = -2.0/sqrt(x*x+eps*eps); // Coulomb
  return result;
}

double scalarpoty(double x, double y, double z, double time, int me)
{
  double eps = cfg_coulomb_eps;
  double result;
  result = -2.0/sqrt(y*y+eps*eps); // Coulomb
  return result;
}

double scalarpotz(double x, double y, double z, double time, int me)
{
  return 0.0;
}

double interactionpotxy(double x, double y, double z, double time, int me)
{
  double eps = cfg_coulomb_eps;
  return 1.0/sqrt((x-y)*(x-y)+eps*eps); // Electron-electron Coulomb repulsion
}

double interactionpotyz(double x, double y, double z, double time, int me)
{
  return 0.0;
}

double interactionpotxz(double x, double y, double z, double time, int me)
{
  return 0.0;
}

double field(double time, int me)
{
  double result=0.0;
  return result;
}

double imagpotx(long xindex, long yindex, long zindex, double time, grid g)
{
  double x;
  double ampl = cfg_absorb_ampl;

  if (ampl>1.0)
    {
      x=((double) xindex + 0.5 - 0.5*g.ngps_x())/(0.5*g.ngps_x())
        *((double) xindex + 0.5 - 0.5*g.ngps_x())/(0.5*g.ngps_x());
      return ampl*x*x*x*x*x*x*x*x;
    }
  else
    {
      return 0.0;
    };
}

double imagpoty(long xindex, long yindex, long zindex, double time, grid g)
{
  double y;
  double ampl = cfg_absorb_ampl;

  if (ampl>1.0)
    {
      y=((double) yindex + 0.5 - 0.5*g.ngps_y())/(0.5*g.ngps_y())
        *((double) yindex + 0.5 - 0.5*g.ngps_y())/(0.5*g.ngps_y());
      return ampl*y*y*y*y*y*y*y*y;
    }
  else
    {
      return 0.0;
    };
}

double imagpotz(long xindex, long yindex, long zindex, double time, grid g)
{
  double z;
  double ampl = 0.0;

  if (ampl>1.0)
    {
      z=((double) zindex + 0.5 - 0.5*g.ngps_z())/(0.5*g.ngps_z())
        *((double) zindex + 0.5 - 0.5*g.ngps_z())/(0.5*g.ngps_z());
      return ampl*z*z*z*z*z*z*z*z;
    }
  else
    {
      return 0.0;
    };
}

// DFT is not used
double dftpot(grid g, double x, double y, double z, double time, int me,
              const fluid &v_null, const wavefunction &v_eins)
{
  double result;
  result=0.0;
  return result;
}

double dfthartree(grid g, double x, double y, double z, double time, int me,
                  const wavefunction & wf, double eps, long box)
{
  double result=0.0;
  return result;
}
