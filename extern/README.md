This is the code which inegrates the Boltzmann collision integral
to calculate the sterile neutrino production in the 
supernova core. The calculation is based on the integration 
of the phase space including the relevant matrix element
for the interaction. 

The code is taken from the `pyBBN` project which can be 
found here https://github.com/ckald/pyBBN. The code performs
much more but I only use the four particle interaction found
here in the repository `interactions/four_particle/cpp`. The 
code implements a simplification of the 9d collision integral
and the details can be found in the included user manual or the
associated paper: https://arxiv.org/pdf/2006.07387

To build the code, in the `src` directory run 
```bash
make all 
```
which compiles the code and copies the shared library file
to the `boltzmann_integral` directory. It cna then be installed
by running, in this directory,
```bash
pip install .
```

---
## Compile Requirements

You will need to install the following packages in your conda environment,
assuming you're installing from the conda-forge channel

```bash
conda install clangxx (clang++)
conda install llvm-openmp (OpenMP)
conda install gsl (GSL) 
```

This should give you the required packages to compile the integration code.
You will need to point the `Makefile` to your GSL installation, both the
include and library, for example this is my include, 
```bash
-I/nevis/houston/home/jsen/.conda/envs/sterile_nu_prod/include
```

this should be generalized but is on the ToDo list for now.