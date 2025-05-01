import numpy as np
from boltzmann_integration.integral import (M_t, grid_t, particle_t, reaction_t)


class Particle:
    def __init__(self, **kwargs):

        self.kwargs = kwargs

        # Initialize the particles
        self.particle_mass()
        self.particle_species = {}
        self.define_particle_species()
        self.consts = PhysicalConstants()

    @classmethod
    def particle_mass(cls):
        """
        Particle mass values take from:
        https://pdg.lbl.gov/2022/listings/contents_listings.html
        """
        cls.proton_mass = 938.27 * Units.MeV
        cls.neutron_mass = 939.57 * Units.MeV
        cls.tau_mass = 1776.86 * Units.MeV
        cls.muon_mass = 105.66 * Units.MeV
        cls.electron_mass = 0.511 * Units.MeV
        cls.neutrino_mass = 0.0 * Units.MeV

    @staticmethod
    def energy(momentum, mass):
        return np.sqrt(momentum ** 2 + mass ** 2)

    @staticmethod
    def make_particle(mass, eta, in_equilibrium, temp, grid, distribution):
        """Fill struct `particle_t` defined in C++ """
        return particle_t(
            eta=eta,
            m=mass,
            grid=Interaction.make_grid(grid=grid, distribution=distribution),
            in_equilibrium=in_equilibrium,
            T=temp
        )

    def map_dict_to_particle(self, particle):
        """Must return variables in order `make_particle()` expects. """
        return (particle['mass'],
                particle['eta'],
                particle['in_equil'],
                particle['T'],
                particle['grid'].TEMPLATE,
                particle['distribution'](energy=self.energy(particle['grid'].TEMPLATE, particle['mass']),
                                         temperature=particle['T'],
                                         chem_potential=particle['mu'])
                )

    def make_particle_specie(self, mass, chemical_potential, statistic):
        if statistic not in ['fermi', 'sterile']:
            raise ValueError

        return {'mass': mass,
                'chemical_potential': chemical_potential,
                'statistic': statistic}

    def define_particle_species(self):
        self.particle_species['nu_e'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi')
        self.particle_species['nu_m'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu_mu',
                                                                  statistic='fermi')
        self.particle_species['nu_t'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi')
        self.particle_species['nu_s'] = self.make_particle_specie(mass=None,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi')

        self.particle_species['muon'] = self.make_particle_specie(mass=self.muon_mass,
                                                                  chemical_potential='mu_mu',
                                                                  statistic='fermi')
        self.particle_species['electron'] = self.make_particle_specie(mass=self.electron_mass,
                                                                      chemical_potential='mu_e',
                                                                      statistic='fermi')

        self.particle_species['neutron'] = self.make_particle_specie(mass=self.neutron_mass,
                                                                     chemical_potential='mu_n',
                                                                     statistic='fermi')
        self.particle_species['proton'] =self.make_particle_specie(mass=self.proton_mass,
                                                                   chemical_potential='mu_p',
                                                                   statistic='fermi')

    def fermi_statistics(self, energy, temperature, chem_potential=0):
        """
         Fermi-Dirac: 1 / e^((E - mu)/T) + 1}
         These should all be in units of MeV
         chem_potential: mu
        """
        return 1. / (np.exp((energy - chem_potential) / temperature) + 1.)

    def bose_statistics(self, energy, temperature, chem_potential=0):
        """
         Fermi-Dirac: 1 / e^((E - mu)/T) + 1}
         These should all be in units of MeV
         chem_potential: mu
        """
        return 1. / (np.exp((energy - chem_potential) / temperature) - 1.)

    def sterile_statistics(self, energy, temperature, chem_potential=0):
        """
         The sterile neutrino is assumed to be thermalized so f_s = 0
        """
        return np.zeros_like(energy)


class Interaction(Particle):
    def __init__(self, num_samples, max_momentum, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.num_samples = num_samples
        self.max_momentum = max_momentum
        self.matrix_elements = {}
        self.current_interaction = None
        self.physical_const = PhysicalConstants()


    def construct_interactions(self, sterile_mass, sterile_mixing, interaction_list):
        if interaction_list == 'nu_mu':
            return self.nu_mu_interactions(sterile_mass=sterile_mass, sterile_mixing=sterile_mixing)
        elif interaction_list == 'nu_tau':
            return self.nu_tau_interactions(sterile_mass=sterile_mass, sterile_mixing=sterile_mixing)

    def nu_mu_interactions(self, sterile_mass, sterile_mixing):
        alpha_flavor, beta_flavor = 'nu_m', 'nu_t'
        print("Active ν_α, ν_β =", alpha_flavor, ",", beta_flavor)

        self.particle_species['nu_s']['mass'] = sterile_mass

        self.matrix_elements[0] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 8 * sterile_mixing, 'K2': 0, 'order': (3, 2, 1, 0)}
        }

        self.matrix_elements[1] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 4 * sterile_mixing, 'K2': 0, 'order': (3, 1, 2, 0)}
        }

        self.matrix_elements[2] = {
            'interaction' : [self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[3] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[4] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,1,2,0)}
        }

        k1_element = 8 * sterile_mixing * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * sterile_mixing * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[5] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}
        }

        k1_element = 8 * sterile_mixing * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * sterile_mixing * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[6] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * sterile_mixing * (G1 + G2)
        k2_element = -G3 * sterile_mixing * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[7] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * sterile_mixing * (G1 + G2)
        k2_element = -G3 * sterile_mixing * (self.particle_species['proton']['mass'] ** 2)
        self.matrix_elements[8] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * sterile_mixing * (G1 + G2)
        k2_element = -G3 * sterile_mixing * (self.particle_species['proton']['mass'] * self.particle_species['neutron']['mass'])
        self.matrix_elements[9] = {
            'interaction': [self.particle_species['muon'], self.particle_species['proton'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        self.matrix_elements[10] = {
            'interaction': [self.particle_species['muon'], self.particle_species['nu_e'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 8 * sterile_mixing, 'K2': 0, 'order': (3, 1, 2, 0)}
        }

        self.current_interaction = 'nu_mu'

        return self.matrix_elements

    def nu_tau_interactions(self, sterile_mass, sterile_mixing):
        alpha_flavor, beta_flavor = 'nu_t', 'nu_m'
        print("Active ν_α, ν_β =", alpha_flavor, ",", beta_flavor)

        self.particle_species['nu_s']['mass'] = sterile_mass

        self.matrix_elements[0] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 8 * sterile_mixing, 'K2': 0, 'order': (3, 2, 1, 0)}
        }

        self.matrix_elements[1] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 4 * sterile_mixing, 'K2': 0, 'order': (3, 1, 2, 0)}
        }

        self.matrix_elements[2] = {
            'interaction' : [self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[3] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[4] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2 * sterile_mixing, 'K2' : 0, 'order' : (3,1,2,0)}
        }

        k1_element = 8 * sterile_mixing * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * sterile_mixing * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[5] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}
        }

        k1_element = 8 * sterile_mixing * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * sterile_mixing * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[6] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * sterile_mixing * (G1 + G2)
        k2_element = -G3 * sterile_mixing * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[7] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * sterile_mixing * (G1 + G2)
        k2_element = -G3 * sterile_mixing * (self.particle_species['proton']['mass'] ** 2)
        self.matrix_elements[8] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        # G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 1. * (1.27 - 1) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        # k1_element = 1. * sterile_mixing * (G1 + G2)
        # k2_element = -G3 * sterile_mixing * (self.particle_species['proton']['mass'] * self.particle_species['neutron']['mass'])
        # self.matrix_elements[9] = {
        #     'interaction': [self.particle_species['muon'], self.particle_species['proton'], self.particle_species['neutron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        # }
        #
        # self.matrix_elements[10] = {
        #     'interaction': [self.particle_species['muon'], self.particle_species['nu_e'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': 8 * sterile_mixing, 'K2': 0, 'order': (3, 1, 2, 0)}
        # }


        self.current_interaction = 'nu_tau'

        return self.matrix_elements

    @staticmethod
    def energy(momentum, mass):
        return np.sqrt(momentum ** 2 + mass ** 2)

    @staticmethod
    def make_grid(grid, distribution):
        """Fill struct `grid_t` defined in C++ """
        return grid_t(
            grid=grid,
            distribution=distribution
        )

    def make_reaction(self, reaction):
        """
        1st element in the reaction list is the particle for which we are
        calculating the collision integral.
        """
        creaction = [
            reaction_t(
                specie=self.make_particle(*self.map_dict_to_particle(particle)),
                side=particle['side']
            )
            for particle in reaction
        ]
        return creaction

    def get_reaction(self, interaction, grid, supernova_sim, index):

        # Update the radius dependent variables
        temp = supernova_sim['temp'][index] * Units.MeV

        reaction_list = []
        for i, part in enumerate(interaction['interaction']):
            side = -1 if i < 2 else 1  # 0+1 -> 2+3 LHS is -1 and RHS is +1
            stats = self.fermi_statistics if part['statistic'] == 'fermi' else self.sterile_statistics
            chem_potential = supernova_sim[part['chemical_potential']][index] * Units.MeV
            p = self.set_particle(mass=part['mass'], grid=grid, distribution=stats, side=side, temp=temp, mu=chem_potential)
            reaction_list.append(p)

        return reaction_list

    def create_matrix_element(self, matrix_element, sterile_nu_mixing):
        const = 8 * self.physical_const.G_F ** 2  # MeV
        k1 = matrix_element['K1'] * sterile_nu_mixing * const
        k2 = matrix_element['K2'] * sterile_nu_mixing * const
        return M_t(list(matrix_element['order']), k1, k2, 0)

    def set_particle(self, mass, grid, distribution, side=1, temp=15, mu=0, eta=1, in_equil=0):
        '''
        1 + 2  --> 3 + 4
        side: -1 = LHS and 1 = RHS
        eta: 1
        in_equil: False
            This is the distribution function we want,
             1. / (exp(energy(p, m) - μ / T) + 1);
        T: dependent on SN sim as function of distance from center
        '''
        return {
            'mass': mass,
            'grid': grid,
            'distribution': distribution,
            'eta': eta,
            'in_equil': in_equil,
            'T': temp,
            'mu': mu,
            'side': side
        }


class LinearSpacedGrid(object):

    def __init__(self, MOMENTUM_SAMPLES=None, MAX_MOMENTUM=None):
        if not MAX_MOMENTUM:
            MAX_MOMENTUM = 100* Units.MeV
        if not MOMENTUM_SAMPLES:
            MOMENTUM_SAMPLES = 100

        self.MIN_MOMENTUM = 0
        self.MAX_MOMENTUM = MAX_MOMENTUM
        self.BOUNDS = (self.MIN_MOMENTUM, self.MAX_MOMENTUM)
        self.MOMENTUM_SAMPLES = MOMENTUM_SAMPLES

        """
        Grid template can be copied when defining a new distribution function and is convenient to\
        calculate any _vectorized_ function over the grid. For example,

        ```python
            particle.conformal_energy(GRID.TEMPLATE)
        ```

        yields an array of particle conformal energy mapped over the `GRID`
        """
        self.TEMPLATE = np.linspace(self.MIN_MOMENTUM, self.MAX_MOMENTUM,
                                       num=self.MOMENTUM_SAMPLES, endpoint=True)


class LogSpacedGrid(object):

    def __init__(self, MOMENTUM_SAMPLES=None, MAX_MOMENTUM=None):
        if not MAX_MOMENTUM:
            MAX_MOMENTUM = 100.* Units.MeV
        if not MOMENTUM_SAMPLES:
            MOMENTUM_SAMPLES = 100

        self.MIN_MOMENTUM = 0
        self.MAX_MOMENTUM = self.MIN_MOMENTUM + MAX_MOMENTUM
        self.BOUNDS = (self.MIN_MOMENTUM, self.MAX_MOMENTUM)
        self.MOMENTUM_SAMPLES = MOMENTUM_SAMPLES

        self.TEMPLATE = self.generate_template()

    def generate_template(self):
        base = 1.2
        return (
            self.MIN_MOMENTUM
            + (self.MAX_MOMENTUM - self.MIN_MOMENTUM)
            * (base ** np.arange(0, self.MOMENTUM_SAMPLES, 1) - 1.)
            / (base ** (self.MOMENTUM_SAMPLES - 1.) - 1.)
        )

class Units:
    def __init__(self, **kwargs):
        self.init_units()

    @classmethod
    def init_units(cls):
        cls.eV = 1.e-9
        cls.keV = cls.eV * 1.e3
        cls.MeV = cls.keV * 1.e3
        cls.GeV = cls.MeV * 1.e3
        cls.TeV = cls.GeV * 1.e3
        cls.sec = 1e22 / 6.582119 / cls.MeV
        cls.kg = 1e27 / 1.782662 * cls.GeV
        cls.m = 1e15 / 1.239842 / cls.GeV
        cls.N = 1e-5 / 8.19 * cls.GeV**2

        # Temperature: $10^9 K$
        cls.K9 = cls.MeV / 11.6045
        cls.g_cm3 = cls.MeV**4 / 2.32011575e5

class PhysicalConstants:
    def __init__(self, **kwargs):
        self.init_physical_constants()

    @classmethod
    def init_physical_constants(cls):
        """ ### Physical constants """

        # Planck mass
        cls.M_p = 1.2209 * 1e22 * Units.MeV
        # Gravitational constant
        cls.G = 1 / cls.M_p ** 2
        # Fermi constant
        cls.G_F = (1.166 * 1e-5 / Units.MeV ** 2)
        # Reduced Planck constant
        cls.hbar = 6.582e-22  # MeV * s
        # Speed of light
        cls.c_cm = 299792458 * 100  # cm /s
        # Hubble constant
        cls.H = 1. / (4.55e17 * Units.sec)
        # Weinberg angle
        cls.sin_theta_w_2 = 0.2312
        cls.gR = cls.sin_theta_w_2
        cls.gL = cls.sin_theta_w_2 - 0.5



