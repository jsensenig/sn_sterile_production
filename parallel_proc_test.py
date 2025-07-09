import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.integrate import simpson, trapezoid
from joblib import Parallel, delayed

sys.path.append("/Users/jonsensenig/work/grams/analysis/sterile_prod_rate/sn_sterile_production/sterile_prod")
from sterile_prod.integration import RadiusIntegral

file_list = ['/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.024225788.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.049145483.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.075832883.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.10083760.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.12585405.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.15087403.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.17589356.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.20090931.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.22592784.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.25020160.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.27520034.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.30021245.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.32521253.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.35011666.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.37464308.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.39965120.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.42465913.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.44967201.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.47469086.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.49971696.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=0.75048905.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.0001996.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.2498081.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.4989883.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.7496603.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=1.9996169.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=2.2495575.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=2.5009761.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=2.7512632.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=2.9995875.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=3.2505693.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=3.4998576.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=3.7502533.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=3.9995287.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=4.2511226.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=4.4993564.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=4.7490470.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=4.9990196.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=5.2508404.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=5.4990227.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=5.7501167.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=6.0010090.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=6.9989081.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=7.9996694.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=9.0008587.txt',
 '/Users/jonsensenig/work/grams/analysis/sterile_nu/sn_sim/hydro-SFHo-s18.80-MUONS-T=9.9990568.txt']


def delta_time_func():
    timestep_list = []
    for filename in file_list:
        if filename[-3:] != 'txt':
            continue
        timestep = float(filename.rsplit('=')[-1].removesuffix('.txt'))
        timestep_list.append(timestep)

    timestep_list = np.asarray(timestep_list)
    delta_time = np.diff(timestep_list)
    delta_time = np.concatenate([delta_time, [1]])

    return delta_time


def init_radius_integral(prod_sterile_mass, prod_sterile_mixing, out_file_name):
    interaction_type = 'nu_tau'
    grid_type = 'linear'
    collision_kind = 'f1'
    num_samples = 100
    max_momentum = 500 * 1e-3
    core_cutoff = 20.  # km
    # prod_sterile_mass = 50 # MeV
    # prod_sterile_mixing = 1e-4

    # out_file_name = "diff_number_numu_v0.pkl"
    # out_file_name = "diff_luminosity_...pkl"

    radius_integral = RadiusIntegral(interaction_type=interaction_type,
                                     grid_type=grid_type,
                                     collision_kind=collision_kind,
                                     num_samples=num_samples,
                                     max_momentum=max_momentum,
                                     core_cutoff=core_cutoff,
                                     delta_time=delta_time_func(),
                                     include_absorption=True,
                                     include_lapse=True,
                                     save_to_file=True,
                                     output_file=out_file_name,
                                     radius_sample_step=2)

    radius_integral.load_supernova_sim(filename_list=file_list)
    radius_integral.update_sterile_properties(sterile_mass=prod_sterile_mass * 1e-3, sterile_mixing=prod_sterile_mixing)

    return radius_integral

if __name__ == "__main__":

    index_list = list(np.arange(0,2,2))
    print("index_list:", index_list)
    local_inst = init_radius_integral(prod_sterile_mass=290, prod_sterile_mixing=1e-13,
                                      out_file_name="abs_diff_luminosity_u13_m200.pkl")
    num_processes = 2
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(radius_integral.multiproc_sterile_differential_number, index_list)

    results = Parallel(n_jobs=4)(delayed(local_inst.multiproc_sterile_differential_number)(local_inst, idx) for idx in index_list)
    print(results)
    # local_inst.multiproc_sterile_differential_number(local_inst, 0)

    out_dict = {'time_index': index_list,
                'result': results}
    output_file = "abs_diff_luminosity_u13_m50.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(out_dict, f)
    print("Saved Result to file", output_file)
