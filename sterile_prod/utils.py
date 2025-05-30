

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
