import numpy as np
from astropy import physical_constants
from extern.boltzmann_integration.integral import (M_t, grid_t, particle_t, reaction_t)
from sterile_prod.utils import Units, PhysicalConstants


class Particle:
    def __init__(self, **kwargs):

        self.kwargs = kwargs

        # Initialize the particles
        Units.init_units()
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
        cls.neutrino_mass = 0.0 * Units.eV

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
                particle['distribution'](energy=particle['energy'],
                                         temperature=particle['T'],
                                         chem_potential=particle['mu'])
                )

    def make_particle_specie(self, mass, chemical_potential, statistic, particle_type):
        if statistic not in ['fermi', 'sterile']:
            raise ValueError

        return {'mass': mass,
                'chemical_potential': chemical_potential,
                'statistic': statistic,
                'particle_type': particle_type}

    def define_particle_species(self):
        self.particle_species['nu_e'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi',
                                                                  particle_type='nu_e')
        self.particle_species['nu_m'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu_mu',
                                                                  statistic='fermi',
                                                                  particle_type='nu_m')
        self.particle_species['nu_t'] = self.make_particle_specie(mass=self.neutrino_mass,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi',
                                                                  particle_type='nu_t')
        self.particle_species['nu_s'] = self.make_particle_specie(mass=None,
                                                                  chemical_potential='mu_nu',
                                                                  statistic='fermi',
                                                                  particle_type='nu_s')

        self.particle_species['muon'] = self.make_particle_specie(mass=self.muon_mass,
                                                                  chemical_potential='mu_mu',
                                                                  statistic='fermi',
                                                                  particle_type='muon')
        self.particle_species['electron'] = self.make_particle_specie(mass=self.electron_mass,
                                                                      chemical_potential='mu_e',
                                                                      statistic='fermi',
                                                                      particle_type='electron')

        self.particle_species['neutron'] = self.make_particle_specie(mass=self.neutron_mass,
                                                                     chemical_potential='mu_n',
                                                                     statistic='fermi',
                                                                     particle_type='neutron')
        self.particle_species['proton'] =self.make_particle_specie(mass=self.proton_mass,
                                                                   chemical_potential='mu_p',
                                                                   statistic='fermi',
                                                                   particle_type='proton')

    def fermi_statistics(self, energy, temperature, chem_potential=0):
        """
         Fermi-Dirac: 1 / e^((E - mu)/T) + 1}
         These should all be in units of MeV
         chem_potential: mu
        """
        # print(energy[0], "/", energy[-1])
        # dist = 1. / (np.exp((energy - chem_potential) / temperature) + 1.)
        # print(np.count_nonzero(dist < 0))
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
    def __init__(self, num_samples, max_momentum, min_momentum, **kwargs):

        super().__init__(**kwargs)
        Units.init_units()
        self.mev_units = Units.MeV
        self.gf_units = PhysicalConstants.G_F
        self.kwargs = kwargs
        self.num_samples = num_samples
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
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
            'element': {'K1': 8, 'K2': 0, 'order': (3, 2, 1, 0)}
        }

        self.matrix_elements[1] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 4, 'K2': 0, 'order': (3, 1, 2, 0)}
        }

        self.matrix_elements[2] = {
            'interaction' : [self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[3] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)}
        }

        self.matrix_elements[4] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,1,2,0)}
        }

        k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[5] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}
        }

        # nu_alpha + e- or e+ to e- or e+ and nu_s
        # multiply by 2 to account for both elements
        k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -1. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[6] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'],
                            self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 2. * k1_element, 'K2': 2. * k2_element, 'order': (3, 2, 1, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * (G1 + G2)
        k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[7] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * (G1 + G2)
        k2_element = -G3 * (self.particle_species['proton']['mass'] ** 2)
        self.matrix_elements[8] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        G1 = 1. * (1.27 + 1) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        G2 = 1. * (1.27 - 1) ** 2
        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k1_element = 1. * (G1 + G2)
        k2_element = -G3 * (self.particle_species['proton']['mass'] * self.particle_species['neutron']['mass'])
        self.matrix_elements[9] = {
            'interaction': [self.particle_species['muon'], self.particle_species['proton'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 1, 2, 0)}
        }

        self.matrix_elements[10] = {
            'interaction': [self.particle_species['muon'], self.particle_species['nu_e'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 8, 'K2': 0, 'order': (3, 1, 2, 0)}
        }

        self.current_interaction = 'nu_mu'

        return self.matrix_elements

    def nu_tau_interactions(self, sterile_mass, sterile_mixing):
        alpha_flavor, beta_flavor = 'nu_t', 'nu_m'
        print("Active ν_α, ν_β =", alpha_flavor, ",", beta_flavor)

        self.particle_species['nu_s']['mass'] = sterile_mass
        #
        # self.matrix_elements[0] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
        #                     self.particle_species[alpha_flavor], self.particle_species['nu_s']],
        #     'element': {'K1': 8, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        # }
        #
        # self.matrix_elements[1] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
        #                     self.particle_species[alpha_flavor], self.particle_species['nu_s']],
        #     'element': {'K1': 4, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }
        #
        # self.matrix_elements[2] = {
        #     'interaction' : [self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species[alpha_flavor], self.particle_species['nu_s']],
        #     'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,1,2,0)} # (3, 2, 1, 0)
        # }
        #
        # self.matrix_elements[3] = {
        #     'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
        #     'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,1,2,0)} # (3, 2, 1, 0)
        # }
        #
        # self.matrix_elements[4] = {
        #     'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor], self.particle_species[beta_flavor], self.particle_species['nu_s']],
        #     'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)} # (3, 1, 2, 0)
        # }
        #
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[5] = {
        #     'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 0, 2, 1)} # (3, 2, 1, 0)
        # }
        # # nu_alpha + e- or e+ to e- or e+ and nu_s
        # # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[6] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }
        #
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[7] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }
        #
        # # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5*1.27 + 0.5) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5*1.27 - 0.5) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        # self.matrix_elements[8] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }
        #
        # # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5 * 1.27 + 0.5 - 2.*self.physical_const.sin_theta_w_2) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5 * 1.27 - 0.5 + 2.*self.physical_const.sin_theta_w_2) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2.*self.physical_const.sin_theta_w_2)**2)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['proton']['mass'] ** 2)
        # self.matrix_elements[9] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }

        # Only relevant for muon neutrino mixing!
        # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # mu + p --> n +vs (0 + 1 --> 2 + 3)
        # G1 = 2. * (0.5 * 1.27 + 0.5 - 2.*self.physical_const.sin_theta_w_2) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5 * 1.27 - 0.5 + 2.*self.physical_const.sin_theta_w_2) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2.*self.physical_const.sin_theta_w_2)**2)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['proton']['mass'] * self.particle_species['neutron']['mass'])
        # self.matrix_elements[10] = {
        #     'interaction': [self.particle_species['muon'], self.particle_species['proton'], self.particle_species['neutron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }

        # Only relevant for muon neutrino mixing!
        # self.matrix_elements[11] = {
        #     'interaction': [self.particle_species['muon'], self.particle_species['nu_e'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': 8, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }

        # 07/2025
        # Explicitly defining the charge conjugated processes taken from Table 1 (https://arxiv.org/pdf/2309.05860)

        ############
        # alpha flavor scatter,annihl
        self.matrix_elements[0] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 8, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[1] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 8, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[2] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 4, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[3] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[alpha_flavor], self.particle_species['nu_s']],
            'element': {'K1': 4, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        ############
        # beta flavor scatter,annihl
        self.matrix_elements[4] = {
            'interaction': [self.particle_species[beta_flavor], self.particle_species[beta_flavor],
                            self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element': {'K1': 2, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[5] = {
            'interaction': [self.particle_species[beta_flavor], self.particle_species[beta_flavor],
                            self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element': {'K1': 2, 'K2': 0, 'order': (3, 1, 2, 0)}  # (3, 2, 1, 0)
        }

        ############
        # alpha, beta flavor scatter,annihl

        self.matrix_elements[6] = {
            'interaction' : [self.particle_species[beta_flavor], self.particle_species[alpha_flavor],
                             self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[7] = {
            'interaction': [self.particle_species[beta_flavor], self.particle_species[alpha_flavor],
                            self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element': {'K1': 2, 'K2': 0, 'order': (3, 2, 1, 0)}  # (3, 2, 1, 0)
        }

        self.matrix_elements[8] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor],
                             self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)} # (3, 2, 1, 0)
        }

        self.matrix_elements[9] = {
            'interaction' : [self.particle_species[alpha_flavor], self.particle_species[beta_flavor],
                             self.particle_species[beta_flavor], self.particle_species['nu_s']],
            'element' : {'K1' : 2, 'K2' : 0, 'order' : (3,2,1,0)} # (3, 2, 1, 0)
        }

        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[10] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[10] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[11] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[12] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[11] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[13] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[14] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[15] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }


        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[12] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[16] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[17] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[18] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[13] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[19] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[20] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[21] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[14] = {
        #     'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species[alpha_flavor],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 0, 2, 1)} # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[22] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[23] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[24] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (2, 1, 3, 0)} # (3, 2, 1, 0)
        }

        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        # self.matrix_elements[15] = {
        #     'interaction': [self.particle_species['electron'], self.particle_species['electron'],
        #                     self.particle_species[alpha_flavor],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 0, 2, 1)}  # (3, 2, 1, 0)
        # }

        k1_element = 8 * self.consts.gL ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[25] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        k1_element = 8 * self.consts.gR ** 2
        # k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[26] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        # multiply by 2 to account for both elements
        # k1_element = 8 * (self.consts.gL ** 2 + self.consts.gR ** 2)
        k2_element = -8. * self.consts.gL * self.consts.gR * (self.particle_species['electron']['mass'] ** 2)
        self.matrix_elements[27] = {
            'interaction': [self.particle_species['electron'], self.particle_species['electron'], self.particle_species['electron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (2, 1, 3, 0)} # (3, 2, 1, 0)
        }

        # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5*1.27 + 0.5) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5*1.27 - 0.5) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        # self.matrix_elements[16] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        # }

        G1 = 2. * (0.5*1.27 + 0.5) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        k1_element = G1
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[28] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        G2 = 2. * (0.5 * 1.27 - 0.5) ** 2
        k1_element = G2
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[29] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k2_element = G3
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[30] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }


        # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5*1.27 + 0.5) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5*1.27 - 0.5) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        # self.matrix_elements[17] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }

        G1 = 2. * (0.5*1.27 + 0.5) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        k1_element = G1
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[31] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        G2 = 2. * (0.5 * 1.27 - 0.5) ** 2
        k1_element = G2
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[32] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        G3 = 2. * ((0.5 * 1.27) ** 2 - 0.25)
        k2_element = G3
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[33] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['neutron'], self.particle_species['neutron'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5 * 1.27 + 0.5 - 2.*self.physical_const.sin_theta_w_2) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5 * 1.27 - 0.5 + 2.*self.physical_const.sin_theta_w_2) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2.*self.physical_const.sin_theta_w_2)**2)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['proton']['mass'] ** 2)
        # self.matrix_elements[18] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)} # (3, 1, 2, 0)
        # }

        G1 = 2. * (0.5 * 1.27 + 0.5 - 2.*self.physical_const.sin_theta_w_2) ** 2
        k1_element = G1
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[34] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)} # (3, 2, 1, 0)
        }

        G2 = 2. * (0.5 * 1.27 - 0.5 + 2.*self.physical_const.sin_theta_w_2) ** 2
        k1_element = G2
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[35] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)} # (3, 2, 1, 0)
        }

        G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2.*self.physical_const.sin_theta_w_2)**2)
        k2_element = G3
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[36] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'], self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        # # G1(p1 · p2)(p3 · p4) + G2(p2 · p4)(p1 · p3) − G3m2m3(p1 · p4)
        # G1 = 2. * (0.5 * 1.27 + 0.5 - 2. * self.physical_const.sin_theta_w_2) ** 2  # 2(GA + GV)^2 = 2(0.5*1.27 + 0.5)^2
        # G2 = 2. * (0.5 * 1.27 - 0.5 + 2. * self.physical_const.sin_theta_w_2) ** 2
        # G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2. * self.physical_const.sin_theta_w_2) ** 2)
        # k1_element = 1. * (G1 + G2)
        # k2_element = -G3 * (self.particle_species['proton']['mass'] ** 2)
        # self.matrix_elements[19] = {
        #     'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'],
        #                     self.particle_species['proton'],
        #                     self.particle_species['nu_s']],
        #     'element': {'K1': k1_element, 'K2': k2_element, 'order': (3, 2, 1, 0)}  # (3, 1, 2, 0)
        # }

        G1 = 2. * (0.5 * 1.27 + 0.5 - 2. * self.physical_const.sin_theta_w_2) ** 2
        k1_element = G1
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[37] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'],
                            self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 2, 1, 0)}  # (3, 2, 1, 0)
        }

        G2 = 2. * (0.5 * 1.27 - 0.5 + 2. * self.physical_const.sin_theta_w_2) ** 2
        k1_element = G2
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[38] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'],
                            self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': k1_element, 'K2': 0, 'order': (3, 1, 2, 0)}  # (3, 2, 1, 0)
        }

        G3 = 2. * ((0.5 * 1.27) ** 2 - (0.5 - 2. * self.physical_const.sin_theta_w_2) ** 2)
        k2_element = G3
        # k2_element = -G3 * (self.particle_species['neutron']['mass'] ** 2)
        self.matrix_elements[39] = {
            'interaction': [self.particle_species[alpha_flavor], self.particle_species['proton'],
                            self.particle_species['proton'],
                            self.particle_species['nu_s']],
            'element': {'K1': 0, 'K2': k2_element, 'order': (0, 1, 2, 3)} # (3, 2, 1, 0)
        }

        self.current_interaction = 'nu_tau'

        return self.matrix_elements

    def sterile_decays(self, sterile_mass, sterile_mixing, nu_flavor):
        """
        All units are in GeV of natural units.
        Note: The decay widths are multiplied by a factor of 2 to account for the
              charge conjugated process (in `units_factor`).
        """
        if nu_flavor == 'nu_mu':
            lepton_mass = 0.1057
        elif nu_flavor == 'nu_tau':
            lepton_mass = 1.1
        else:
            print('Unknown neutrino flavor:', nu_flavor)
            raise ValueError

        pi0_mass = 0.135
        pip_mass = 0.140
        weak_mixing_sin_theta = 0.23
        ckm_element_ud = 0.974

        alpha_em = 1. / 137.
        f_pi0 = pi0_mass
        x_pi0 = pi0_mass / sterile_mass
        x_mu = lepton_mass / sterile_mass
        x_pip = pip_mass / sterile_mass
        units_factor = ((1.166e-5) ** 2) * sterile_mixing * (sterile_mass ** 3) * 2.

        def kallen_func(a, b, c):
            return a ** 2 + b ** 2 + c ** 2 - 2. * a * b - 2. * b * c - 2. * c * a

        decay_rate_dict = {}

        # Decay rate of sterile nu to a single gamma
        decay_rate_dict['to_gamma'] = (9. * alpha_em * sterile_mass ** 2) / (2048 * np.pi ** 4)
        decay_rate_dict['to_gamma'] *= units_factor

        # Decay rate of sterile nu to a single pi0, note the kinematic cutoff at < pi0 mass
        if sterile_mass >= f_pi0:
            decay_rate_dict['to_pi0'] = (f_pi0 ** 2) * ((1 - x_pi0 ** 2) ** 2) / (32. * np.pi)
            decay_rate_dict['to_pi0'] *= units_factor

        # Decay rate of sterile nu to a neutrino - anti-neutrino pair
        decay_rate_dict['to_two_nu'] = (sterile_mass ** 2) / (192 * np.pi ** 3)
        decay_rate_dict['to_two_nu'] *= units_factor

        # Decay rate of sterile nu to electron - positron pair
        decay_rate_dict['to_electron_positron'] = (0.25 - weak_mixing_sin_theta ** 2 + 2. * weak_mixing_sin_theta ** 4) * (
                                                              sterile_mass ** 2) / (192 * np.pi ** 3)
        decay_rate_dict['to_electron_positron'] *= units_factor

        # Decay rate of sterile nu to an electron neutrino, positron and muon
        if sterile_mass >= lepton_mass:
            decay_rate_dict['to_nue_positron_muon'] = ((sterile_mass ** 2) / (384 * np.pi ** 3)) * (
                        2. * (1 - x_mu ** 2) * (2. + 9. * x_mu ** 2) + (2. * x_mu ** 2) * (1 - x_mu ** 2) * (
                        -6. - 6. * x_mu ** 2 + x_mu ** 4 + 6. * np.log(x_mu ** 2)))
            decay_rate_dict['to_nue_positron_muon'] *= units_factor

        # Decay rate of sterile nu to a muon and pi+
        if sterile_mass >= (lepton_mass + 0.140):
            decay_rate_dict['to_muon_pi+'] = ((0.135 * ckm_element_ud) ** 2 / (16. * np.pi)) * np.sqrt(
                kallen_func(a=1, b=x_pip ** 2, c=x_mu ** 2)) * (
                                                         1. - x_pip ** 2 - (x_mu ** 2) * (2. + x_pip ** 2 - x_mu ** 2))
            decay_rate_dict['to_muon_pi+'] *= units_factor

        return decay_rate_dict

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

    def get_chemical_potential(self, sn_sim, index, chem_potential, eff_mass, particle_type):
        """
        The Garching supernova simulation includes the mass in the chemical potential for all
        but the nucleons. The nucleon mass to include is the effective mass.
        """
        if particle_type == 'proton':
            return eff_mass + (sn_sim[chem_potential][index] - sn_sim['Up'][index]) * self.mev_units
        elif particle_type == 'neutron':
            return eff_mass + (sn_sim[chem_potential][index] - sn_sim['Un'][index]) * self.mev_units
        else:
            return sn_sim[chem_potential][index] * self.mev_units

    def get_mass(self, sn_sim, index, mass, particle_type):
        if particle_type == 'proton':
            return sn_sim['mstar_p'][index] * self.mev_units
        elif particle_type == 'neutron':
            return sn_sim['mstar_n'][index] * self.mev_units
        else:
            return mass

    def get_reaction(self, interaction, grid, supernova_sim, index):

        # Update the radius dependent variables
        temp = supernova_sim['temp'][index] * self.mev_units

        reaction_list = []
        for i, part in enumerate(interaction['interaction']):
            side = -1 if i < 2 else 1  # 0+1 -> 2+3 LHS is -1 and RHS is +1
            stats = self.fermi_statistics if part['statistic'] == 'fermi' else self.sterile_statistics
            # chem_potential = supernova_sim[part['chemical_potential']][index] * self.mev_units
            part_mass = self.get_mass(sn_sim=supernova_sim, index=index, mass=part['mass'],
                                      particle_type=part['particle_type'])
            chem_potential = self.get_chemical_potential(sn_sim=supernova_sim, index=index,
                                                         chem_potential=part['chemical_potential'],
                                                         eff_mass=part_mass,
                                                         particle_type=part['particle_type'])
            part_energy = self.energy(grid.TEMPLATE, part_mass)
            # p = self.set_particle(energy=part_energy, mass=part['mass'], grid=grid, distribution=stats, side=side, temp=temp, mu=chem_potential)
            p = self.set_particle(energy=part_energy, mass=part_mass, grid=grid, distribution=stats, side=side,
                                  temp=temp, mu=chem_potential)
            reaction_list.append(p)

        return reaction_list

    def create_matrix_element(self, matrix_element, sterile_nu_mixing):
        const = 8 * self.gf_units**2
        k1 = matrix_element['K1'] * sterile_nu_mixing * const
        k2 = matrix_element['K2'] * sterile_nu_mixing * const
        return M_t(list(matrix_element['order']), k1, k2, 0)

    def set_particle(self, energy, mass, grid, distribution, side=1, temp=15, mu=0, eta=1, in_equil=0):
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
            'energy': energy,
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

    def __init__(self, MOMENTUM_SAMPLES=None, MAX_MOMENTUM=None, MIN_MOMENTUM=None):
        if not MAX_MOMENTUM:
            MAX_MOMENTUM = 500 * Units.MeV
        if not MOMENTUM_SAMPLES:
            MOMENTUM_SAMPLES = 100
        if not MIN_MOMENTUM:
            self.MIN_MOMENTUM = 0. * Units.MeV
        self.MIN_MOMENTUM = MIN_MOMENTUM
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
            MAX_MOMENTUM = 500.* Units.MeV
        if not MOMENTUM_SAMPLES:
            MOMENTUM_SAMPLES = 100

        self.MIN_MOMENTUM = 0. * Units.MeV
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
