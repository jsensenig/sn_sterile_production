import numpy as np
import pickle
from tqdm import tqdm
from sterile_prod.interaction import Interaction, LinearSpacedGrid, LogSpacedGrid
from extern.boltzmann_integration.integral import integration, CollisionIntegralKind
from scipy.integrate import trapezoid


class CollisionIntegral:
    def __init__(self, interaction_type, grid_type, num_samples, max_momentum, min_momentum, step_size):
        self.interaction_type = interaction_type
        self.num_samples = num_samples
        self.max_momentum = max_momentum
        self.interactions = Interaction(num_samples=num_samples, max_momentum=max_momentum, min_momentum=min_momentum)
        self.interaction_list = None

        self.sterile_mass = None
        self.sterile_mixing = None

        self.stepsize = step_size #1.0e-2
        self.grid = None
        if grid_type == 'linear':
            self.grid = LinearSpacedGrid(MOMENTUM_SAMPLES=self.num_samples, MAX_MOMENTUM=self.max_momentum, MIN_MOMENTUM=min_momentum)
        elif grid_type == 'log':
            self.grid = LogSpacedGrid(MOMENTUM_SAMPLES=self.num_samples, MAX_MOMENTUM=self.max_momentum)
        else:
            raise ValueError('Unknown grid type')


    def update_sterile_properties(self, sterile_mass, sterile_mixing):
        self.sterile_mass = sterile_mass
        self.sterile_mixing = sterile_mixing
        self.set_interaction_list(sterile_mass=sterile_mass, sterile_mixing=sterile_mixing)

    def set_interaction_list(self, sterile_mass, sterile_mixing):
        self.interaction_list = self.interactions.construct_interactions(sterile_mass=sterile_mass,
                                                                          sterile_mixing=sterile_mixing,
                                                                          interaction_list=self.interaction_type)
        print("Loaded {} interactions".format(len(self.interaction_list)))

    def integrate_interactions(self, sn_sim, index, collision_kind, is_absorption):
        ps = self.get_ps(grid=self.grid)
        collision_integral = 0
        for interaction in self.interaction_list.values():
            reaction_list = self.interactions.get_reaction(interaction=interaction, grid=self.grid, supernova_sim=sn_sim, index=index, is_absorption=is_absorption)
            reaction = self.interactions.make_reaction(reaction=reaction_list)
            mtx_elm = self.interactions.create_matrix_element(matrix_element=interaction['element'], sterile_nu_mixing=self.sterile_mixing)
            integral_result = self.integrate(momentum=ps.tolist(), reaction_list=reaction_list, creaction=reaction,
                                             mtx_element=[mtx_elm], collision_kind=collision_kind)
            collision_integral += np.array(integral_result)

        return collision_integral

    def integrate(self, momentum, reaction_list, creaction, mtx_element, collision_kind):
        return integration(momentum,
                           reaction_list[1]['grid'].MIN_MOMENTUM+0.,
                           reaction_list[1]['grid'].MAX_MOMENTUM+10.,
                           reaction_list[2]['grid'].MIN_MOMENTUM+0.,
                           reaction_list[2]['grid'].MAX_MOMENTUM+10.,
                           reaction_list[3]['grid'].MAX_MOMENTUM+10.,
                           creaction, mtx_element, self.stepsize, collision_kind)
    @staticmethod
    def get_ps(grid):
        max_mom = grid.MAX_MOMENTUM
        template = grid.TEMPLATE
        upper_element_grid = min(len(template) - 1, np.searchsorted(template, max_mom))
        return template[:upper_element_grid + 1]


class RadiusIntegral:
    def __init__(self, interaction_type, grid_type, collision_kind, num_samples, max_momentum, min_momentum, core_cutoff,
                 abs_cutoff, delta_time, include_absorption=True, include_lapse=True, save_to_file=True, output_file=None,
                 radius_sample_step=1, step_size=1e-2):

        self.save_to_file = save_to_file
        self.output_file = output_file

        self.include_absorption = include_absorption
        self.include_lapse = include_lapse

        self.delta_time = delta_time
        self.core_cutoff = core_cutoff * 1.e5 # convert km to cm
        self.abs_cutoff = abs_cutoff * 1.e5  # convert km to cm
        # self.luminosity_unit_conversion = 3.62e59 #3.e54 #3.e67
        # self.luminosity_unit_conversion = 8.20e35 # why? using 1cm=8.1*10^12GeV ??
        # Unit conversion: https://www.seas.upenn.edu/~amyers/NaturalUnits.pdf
        self.number_density_conversion = 1.31e53 * (2. * np.pi)**3 # cm^-3 (using 1GeV=5.08*10^17cm^-1)
        self.luminosity_unit_conversion = self.number_density_conversion #* 1.602e-3  # using 1GeV=1.602*10^-3ergs
        self.supernova_sim = None
        self.collision_integral = CollisionIntegral(interaction_type=interaction_type, grid_type=grid_type,
                                                    num_samples=num_samples,max_momentum=max_momentum,
                                                    min_momentum=min_momentum, step_size=step_size)

        self.collision_kind = None
        if collision_kind == 'f1':
            self.collision_kind = CollisionIntegralKind.F_1
        elif collision_kind == 'decay':
            self.collision_kind = CollisionIntegralKind.F_decay
        # The type of collision integral to calculate the absorption effect within the SN core
        self.absorption_collision_kind = CollisionIntegralKind.F_decay

        # Set the radius steps, eg 1 is every simulated radius and 2 is every other
        self.radius_sample_step = radius_sample_step

    @staticmethod
    def energy(momentum, mass):
        return np.sqrt(momentum ** 2 + mass ** 2)

    def update_sterile_properties(self, sterile_mass, sterile_mixing):
        self.collision_integral.update_sterile_properties(sterile_mass=sterile_mass, sterile_mixing=sterile_mixing)

    def load_supernova_sim(self, filename_list):
        """
        Load the supernova sim from files. Each file is a timestep in the simulation.
        """
        self.supernova_sim = []
        for filename in filename_list:
            with open(filename, 'r') as f:
                sn_data = np.loadtxt(f, skiprows=2)

            supernova_sim_dict = {'radius': sn_data[:, 0], 'density': sn_data[:, 2],'temp': sn_data[:, 4],
                                  'grav_lapse': sn_data[:, 7], 'mu_nu': sn_data[:, 8], 'mu_e': sn_data[:, 9],
                                  'mu_n': sn_data[:, 10], 'mu_p': sn_data[:, 11], 'mu_mu': sn_data[:, 12],
                                  'mu_nu_mu': sn_data[:, 13], 'Un': sn_data[:, 21], 'Up': sn_data[:, 22],
                                  'mstar_n': sn_data[:, 23], 'mstar_p': sn_data[:, 24]}
            self.supernova_sim.append(supernova_sim_dict)
        print("Loaded", len(self.supernova_sim), "supernova time steps")

    def get_supernova_sim(self):
        return self.supernova_sim

    def gravitational_lapse(self, ps, grav_lapse):
        """
        The sterile neutrinos with energy less than the gravitational pull get trapped in the
        SN core. This is implemented as a cutoff in the energy spectrum. Even the sterile neutrinos
        which escape get lose energy to the gravitational pull (redshifted) which is accounted for
        by grav_lapse**2 (eta^2)
        """
        if not self.include_lapse:
            return 1.0
        sterile_energy = self.energy(momentum=ps, mass=self.collision_integral.sterile_mass)
        grav_lapse_calc = np.ones_like(sterile_energy) * grav_lapse**2
        grav_lapse_calc[sterile_energy < (self.collision_integral.sterile_mass / grav_lapse)] = 0.0
        return grav_lapse_calc

    def sterile_absorption(self, start_idx, supernova_sim, debug=False, expfactor=1, num_line=35, num_costheta=5):
        """
        Within the dense SN core sterile neutrinos can be absorbed through interactions. This integrates over
        the possible processes and acts as an average suppression to the sterile neutrino production.
        Reference: https://arxiv.org/pdf/2503.13607
        Average suppression: Eq.(7)
        Associated collision integral: Eq.(8)
        For integral ds the limits are 0 <= s <= 2Renv
        """
        radius = supernova_sim['radius'][start_idx]
        if not self.include_absorption or radius > self.abs_cutoff:
            return 1.0

        ## Integration steps
        # The line integral
        # upper_line_limit = 2. * self.core_cutoff
        upper_line_limit = 2. * self.abs_cutoff
        sline = np.linspace(0., upper_line_limit, num=num_line)
        delta_s = np.diff(sline)[0]
        # The cos(theta) integral
        cos_theta = np.linspace(-1., 1., num=num_costheta)
        delta_cos = np.abs(np.diff(cos_theta))[0]
        length_to_energy_conv =  expfactor * 5.08e17 * (2. * np.pi) #8.06e12

        # Calculate the decay contribution
        decay_dict = self.collision_integral.interactions.sterile_decays(sterile_mass=self.collision_integral.sterile_mass,
                                                            sterile_mixing=self.collision_integral.sterile_mixing,
                                                            nu_flavor=self.collision_integral.interaction_type)

        decay_integral = 0
        abs_integral = 0
        abs_integral_list = []
        abs_decay_list = []
        for cos_t in cos_theta: # integral across all paths/lines
            integral = 0
            r_tilda = np.sqrt(radius ** 2 + sline ** 2 + 2. * radius * sline * cos_t)
            for i, rt in enumerate(r_tilda): # line integral
                # if rt > self.core_cutoff:
                if rt > self.abs_cutoff:
                    decay_integral = 0
                    for key in decay_dict:
                        decay_integral += sline[i] * decay_dict[key]
                    break
                radius_idx = np.argmin(np.abs(supernova_sim['radius'] - rt))
                ###############
                # temp = supernova_sim['temp'][radius_idx] * 1.0e-3
                ps = self.collision_integral.get_ps(grid=self.collision_integral.grid)
                es = self.energy(momentum=ps, mass=self.collision_integral.sterile_mass)
                # factor = (2. * np.pi ** 2 / ps) * np.exp(es / temp)
                factor = es / ps
                ###############
                # integral += delta_s * np.abs(self.collision_integral.integrate_interactions(sn_sim=supernova_sim, index=radius_idx,
                #                                                           collision_kind=self.absorption_collision_kind, is_absorption=True))
                integral += delta_s * factor * np.abs(self.collision_integral.integrate_interactions(sn_sim=supernova_sim, index=radius_idx,
                                                                          collision_kind=self.absorption_collision_kind, is_absorption=True))
            abs_integral += np.exp(-(integral + decay_integral) * length_to_energy_conv) * delta_cos
            if debug:
                abs_integral_list.append(integral)
                abs_decay_list.append(decay_integral)

        # decay_integral = 0
        # for smax, cos_t in zip(slist, cos_theta):
        #     for key in decay_dict:
        #          decay_integral += smax * decay_dict[key] * delta_cos

        # abs_integral += np.exp(-decay_integral)

        if debug:
            print("ABS:", 0.5 * abs_integral, "-", integral, "-", decay_integral)
            return abs_integral_list, decay_integral

        return 0.5 * abs_integral

    def radius_integral_generator(self, ps, supernova_sim):
        """
        Perform the integration over the SN core volume.
        The core is assumed symmetric so this amounts to an integration over `r` and the angular
        variables only contribute constant factors. The radius of the core is configurable but a
        suggested radius is 20km, reference: https://arxiv.org/pdf/2503.13607
        """
        # delta_radius = np.diff(supernova_sim['radius'])#[::self.radius_sample_step])
        grav_lapse = supernova_sim['grav_lapse']
        total_radius = supernova_sim['radius']
        # index_range = range(0, self.radius_sample_step * len(total_radius[::self.radius_sample_step])-1, self.radius_sample_step)
        # index_range = range(0, len(total_radius[::self.radius_sample_step]))
        index_range = np.arange(0, len(total_radius))
        print("Radius idx range: ", index_range[0], "/", len(total_radius[::self.radius_sample_step]))

        prev_radius = 0
        for idx, radius in zip(index_range[::self.radius_sample_step], total_radius[::self.radius_sample_step]):
            if (idx % 25) == 0 and idx != 0: print("Core Radius:", round(radius * 1.e-5, 2), "[km]")
            if radius > self.core_cutoff:
                break
            integral = self.collision_integral.integrate_interactions(sn_sim=supernova_sim, index=idx,
                                                                      collision_kind=self.collision_kind, is_absorption=False)
            # rint = radius * radius * integral * delta_radius[idx] * self.gravitational_lapse(ps=ps, grav_lapse=grav_lapse[idx])
            mid_radius = (radius + prev_radius) / 2.
            # rint = radius * radius * integral * (radius - prev_radius) * self.gravitational_lapse(ps=ps, grav_lapse=grav_lapse[idx])
            rint = mid_radius * mid_radius * integral * (radius - prev_radius) * self.gravitational_lapse(ps=ps,grav_lapse=grav_lapse[idx])
            prev_radius = radius
            yield rint, idx

    def sterile_differential_number(self):
        """
        This is the time and energy profile of the sterile neutrinos d^2 N / dEdt
        This is similar to dN/dE in Eq(6) of reference: https://arxiv.org/pdf/2503.13607
        but does not integrate over time.
        """
        # FIXME should be midpoint in radius integral
        ps = self.collision_integral.get_ps(grid=self.collision_integral.grid)
        sterile_energy = self.energy(momentum=ps, mass=self.collision_integral.sterile_mass)
        max_num_radius_steps = 10000

        radius_integral = 0
        for i, (dt, sn_sim) in enumerate(zip(self.delta_time, self.supernova_sim)):
            print("Timestep:", i, "/", len(self.supernova_sim))
            radius_gen = self.radius_integral_generator(ps=ps, supernova_sim=sn_sim)
            for _ in range(max_num_radius_steps):
                try:
                    integral, idx = next(radius_gen)
                except StopIteration:
                    break
                absorption = self.sterile_absorption(start_idx=idx, supernova_sim=sn_sim)
                radius_integral += integral * ps * sterile_energy * absorption
            radius_integral *= dt
        # The d^3r -> 4pi r^2 dr so prefactor is 4pi / 2pi^2 = 2/pi
        differential_sterile_number = (2. / np.pi) * radius_integral * self.number_density_conversion

        if self.save_to_file:
            out_dict = {'sterile_mass': self.collision_integral.sterile_mass,
                        'sterile_mixing': self.collision_integral.sterile_mixing,
                        'sterile_energy': sterile_energy, 'differential_sterile_number': differential_sterile_number}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Differential Luminosity to file", self.output_file)

        return differential_sterile_number, ps

    @staticmethod
    def multiproc_sterile_differential_number(instance, input_idx):
        """
        This is the time and energy profile of the sterile neutrinos d^2 N / dEdt
        This is similar to dN/dE in Eq(6) of reference: https://arxiv.org/pdf/2503.13607
        but does not integrate over time.
        """
        print("input_idx", input_idx)
        # delta_time, supernova_sim = input_args
        time_step = 4 if len(instance.supernova_sim) > (input_idx + 4) else len(instance.supernova_sim) - input_idx
        delta_time, supernova_sim = instance.delta_time[input_idx:input_idx+time_step], instance.supernova_sim[input_idx:input_idx+time_step]
        print("input_idx:input_idx+time_step", input_idx, ":", input_idx+time_step)
        # FIXME should be midpoint in radius integral
        ps = instance.collision_integral.get_ps(grid=instance.collision_integral.grid)
        sterile_energy = instance.energy(momentum=ps, mass=instance.collision_integral.sterile_mass)
        max_num_radius_steps = 10000

        radius_integral = 0
        # for i, (dt, sn_sim) in enumerate(input_args):
        for i, (dt, sn_sim) in enumerate(zip(delta_time, supernova_sim)):
            print("Timestep:", i, "/", len(supernova_sim))
            radius_gen = instance.radius_integral_generator(ps=ps, supernova_sim=sn_sim)
            for _ in range(max_num_radius_steps):
                try:
                    integral, idx = next(radius_gen)
                except StopIteration:
                    break
                absorption = instance.sterile_absorption(start_idx=idx, supernova_sim=sn_sim)
                radius_integral += integral * ps * sterile_energy * absorption
            radius_integral *= dt

        return radius_integral

    def sterile_differential_luminosity(self, supernova_sim, save_to_file=True):
        """
        This is the differential sterile neutrino luminosity dL/dE.
        It is not used but is useful for a cross-check with the plots shown in reference: https://arxiv.org/pdf/2309.05860
        Specifically Eq.(3.3) and can be compared to Fig.(3)
        """
        # FIXME should be midpoint in radius integral
        ps = self.collision_integral.get_ps(grid=self.collision_integral.grid)
        sterile_energy = self.energy(momentum=ps, mass=self.collision_integral.sterile_mass)

        radius_integral = 0
        with tqdm(total=len(supernova_sim['radius'][supernova_sim['radius'] < self.core_cutoff]) / self.radius_sample_step) as pbar:
            for integral, idx in self.radius_integral_generator(ps, supernova_sim=supernova_sim):
                absorption = self.sterile_absorption(start_idx=idx, supernova_sim=supernova_sim)
                radius_integral += integral * ps * sterile_energy * absorption
                pbar.update(1)

        differential_luminosity = ((2. / np.pi) * self.luminosity_unit_conversion * sterile_energy * radius_integral)

        if self.save_to_file and save_to_file:
            out_dict = {'sterile_mass': self.collision_integral.sterile_mass,
                        'sterile_mixing': self.collision_integral.sterile_mixing,
                        'sterile_energy': sterile_energy, 'differential_luminosity': differential_luminosity}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Differential Luminosity to file", self.output_file)

        return differential_luminosity, sterile_energy

    def sterile_luminosity(self):
        """
        Integrate the differential sterile neutrino luminosity dL/dE to get the luminosity L.
        """
        # FIXME needs to be dE
        luminosity = []
        time = []
        accum_time = 0
        for i, (dt, sn_sim) in enumerate(zip(self.delta_time, self.supernova_sim)):
            print("Timestep:", i, "/", len(self.supernova_sim))
            accum_time += dt
            tmp_dl, sterile_energy = self.sterile_differential_luminosity(supernova_sim=sn_sim, save_to_file=False)
            energy_integrated = 0
            delta_energy = np.diff(sterile_energy)[0]
            for dl in tmp_dl:
                energy_integrated += dl * delta_energy
            luminosity.append(energy_integrated)
            time.append(accum_time)

        if self.save_to_file:
            out_dict = {'sterile_mass': self.collision_integral.sterile_mass,
                        'sterile_mixing': self.collision_integral.sterile_mixing,
                        'sterile_energy': sterile_energy, 'luminosity': luminosity}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Luminosity to file", self.output_file)

        return luminosity, time