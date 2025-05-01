import numpy as np
import pickle
from interaction import Interaction, LinearSpacedGrid, LogSpacedGrid
from boltzmann_integration.integral import integration, CollisionIntegralKind


class CollisionIntegral:
    def __init__(self, interaction_type, grid_type, num_samples, max_momentum):
        self.interaction_type = interaction_type
        self.num_samples = num_samples
        self.max_momentum = max_momentum
        self.interactions = Interaction(num_samples=num_samples, max_momentum=max_momentum)
        self.interaction_list = None

        self.sterile_mass = None
        self.sterile_mixing = None

        self.stepsize = 1.0e-2
        self.grid = None
        if grid_type == 'linear':
            self.grid = LinearSpacedGrid(MOMENTUM_SAMPLES=self.num_samples, MAX_MOMENTUM=self.max_momentum)
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

    def integrate_interactions(self, sn_sim, index, collision_kind):
        ps = self.get_ps(grid=self.grid)
        collision_integral = 0
        for interaction in self.interaction_list.values():
            reaction_list = self.interactions.get_reaction(interaction=interaction, grid=self.grid, supernova_sim=sn_sim, index=index)
            reaction = self.interactions.make_reaction(reaction=reaction_list)
            mtx_elm = self.interactions.create_matrix_element(matrix_element=interaction['element'], sterile_nu_mixing=self.sterile_mixing)
            integral_result = self.integrate(momentum=ps.tolist(), reaction_list=reaction_list, creaction=reaction,
                                             mtx_element=[mtx_elm], collision_kind=collision_kind)
            collision_integral += np.array(integral_result)

        return collision_integral

    def integrate(self, momentum, reaction_list, creaction, mtx_element, collision_kind):
        return integration(momentum,
                           reaction_list[1]['grid'].MIN_MOMENTUM,
                           reaction_list[1]['grid'].MAX_MOMENTUM,
                           reaction_list[2]['grid'].MIN_MOMENTUM,
                           reaction_list[2]['grid'].MAX_MOMENTUM,
                           reaction_list[3]['grid'].MAX_MOMENTUM,
                           creaction, mtx_element, self.stepsize, collision_kind)
    @staticmethod
    def get_ps(grid):
        max_mom = grid.MAX_MOMENTUM
        template = grid.TEMPLATE
        upper_element_grid = min(len(template) - 1, np.searchsorted(template, max_mom))
        return template[:upper_element_grid + 1]


class RadiusIntegral:
    def __init__(self, interaction_type, grid_type, collision_kind, num_samples, max_momentum, core_cutoff,
                 delta_time, include_absorption=True, include_lapse=True, save_to_file=True, output_file=None):

        self.save_to_file = save_to_file
        self.output_file = output_file

        self.include_absorption = include_absorption
        self.include_lapse = include_lapse

        self.delta_time = delta_time
        self.core_cutoff = core_cutoff * 1.e5 # convert km to cm
        self.luminosity_unit_conversion = 3.62e59 #3.e54 #3.e67
        self.supernova_sim = None
        self.collision_integral = CollisionIntegral(interaction_type=interaction_type, grid_type=grid_type,
                                                    num_samples=num_samples,max_momentum=max_momentum)

        self.collision_kind = None
        if collision_kind == 'f1':
            self.collision_kind = CollisionIntegralKind.F_1
        elif collision_kind == 'decay':
            self.collision_kind = CollisionIntegralKind.F_decay
        # The type of collision integral to calculate the absorption effect within the SN core
        self.absorption_collision_kind = CollisionIntegralKind.F_decay

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
                                  'mu_nu_mu': sn_data[:, 13]}
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

    def sterile_absorption(self, start_idx, supernova_sim):
        """
        Within the dense SN core sterile neutrinos can be absorbed through interactions. This integrates over
        the possible processes and acts as an average suppression to the sterile neutrino production.
        Reference: https://arxiv.org/pdf/2503.13607
        Average suppression: Eq.(7)
        Associated collision integral: Eq.(8)
        """
        if not self.include_absorption:
            return 1.0

        delta_radius = np.diff(supernova_sim['radius'])
        # The integral goes from r to R_core, it is a line integral so 1d
        radius_integral = 0
        for idx, radius in enumerate(supernova_sim['radius'][start_idx:]):
            # if (idx % 10) == 0 and idx != 0: print("Core Radius:", round(radius * 1.e-5, 2), "[km]")
            if radius > self.core_cutoff:
                break
            integral = self.collision_integral.integrate_interactions(sn_sim=supernova_sim, index=idx,
                                                                      collision_kind=self.absorption_collision_kind)
            radius_integral += integral * delta_radius[idx]

        return np.exp(-radius_integral)

    def radius_integral_generator(self, ps, supernova_sim):
        """
        Perform the integration over the SN core volume.
        The core is assumed symmetric so this amounts to an integration over `r` and the angular
        variables only contribute constant factors. The radius of the core is configurable but a
        suggested radius is 20km, reference: https://arxiv.org/pdf/2503.13607
        """
        delta_radius = np.diff(supernova_sim['radius'])
        grav_lapse = supernova_sim['grav_lapse']

        for idx, radius in enumerate(supernova_sim['radius'][1:]):
            if (idx % 50) == 0 and idx != 0: print("Core Radius:", round(radius * 1.e-5, 2), "[km]")
            if radius > self.core_cutoff:
                break
            integral = self.collision_integral.integrate_interactions(sn_sim=supernova_sim, index=idx + 1,
                                                                      collision_kind=self.collision_kind)
            rint = radius * radius * integral * delta_radius[idx] * self.gravitational_lapse(ps=ps, grav_lapse=grav_lapse[idx + 1])
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
                radius_integral += integral * ps * sterile_energy
            radius_integral *= dt

        differential_sterile_number = (2. / np.pi) * radius_integral * self.luminosity_unit_conversion

        if self.save_to_file:
            out_dict = {'sterile_energy': sterile_energy, 'differential_sterile_number': differential_sterile_number}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Differential Luminosity to file", self.output_file)

        return differential_sterile_number, ps

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
        for integral, idx in self.radius_integral_generator(ps, supernova_sim=supernova_sim):
            absorption = self.sterile_absorption(start_idx=idx, supernova_sim=supernova_sim)
            radius_integral += integral * ps * sterile_energy * absorption

        differential_luminosity = ((2. / np.pi) * self.luminosity_unit_conversion * sterile_energy * radius_integral)

        if self.save_to_file and save_to_file:
            out_dict = {'sterile_energy': sterile_energy, 'differential_luminosity': differential_luminosity}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Differential Luminosity to file", self.output_file)

        return differential_luminosity, sterile_energy

    def sterile_luminosity(self):
        """
        Integrate the differential sterile neutrino luminosity dL/dE to get the luminosity L.
        """
        # FIXME needs to be dE
        diff_luminosity = 0
        for i, dt, sn_sim in enumerate(zip(self.delta_time, self.supernova_sim)):
            print("Timestep:", i, "/", len(self.supernova_sim))
            tmp_dl, sterile_energy = self.sterile_differential_luminosity(supernova_sim=self.supernova_sim, save_to_file=False)
            diff_luminosity += (tmp_dl *dt)

        if self.save_to_file:
            out_dict = {'sterile_energy': sterile_energy, 'differential_luminosity': diff_luminosity}
            with open(self.output_file, 'wb') as f:
                pickle.dump(out_dict, f)
            print("Saved Luminosity to file", self.output_file)
