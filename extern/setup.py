import setuptools

setuptools.setup(
    name="boltzmann_integration",
    version="0.1.0",
    description="Code to numerically integrate the Boltzmann integral compiled to integral.so",
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    package_data={
        "boltzmann_integration": ["integral.so"],
    },
)
