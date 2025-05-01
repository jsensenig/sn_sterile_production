import sys
sys.path.append("/Users/jonsensenig/work/grams/analysis/sterile_prod_rate/sn_sterile_production/sterile_prod")
from integration import RadiusIntegral
from interaction import Units
units = Units()

interaction_type = 'nu_mu'
grid_type = 'linear'
# collision_kind = 'f1'
collision_kind = 'f1'
num_samples = 100
max_momentum = 600 * units.MeV
core_cutoff = 20. # km

radius_integral = RadiusIntegral(interaction_type=interaction_type,
                                 grid_type=grid_type,
                                 collision_kind=collision_kind,
                                 num_samples=num_samples,
                                 max_momentum=max_momentum,
                                 core_cutoff=core_cutoff,
                                 include_lapse=True)

sn_sim_file = '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.0001996.txt'
# sn_sim_file = '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=3.2505693.txt'

radius_integral.load_supernova_sim(filename=sn_sim_file)
radius_integral.update_sterile_properties(sterile_mass=200*units.MeV, sterile_mixing=1.e-8)

dlde, ps = radius_integral.sterile_differential_luminosity()