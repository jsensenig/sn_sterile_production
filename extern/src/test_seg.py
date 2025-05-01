import sys
sys.path.append("/Users/jonsensenig/work/grams/analysis/sterile_prod_rate/pyBBN/interactions/four_particle/cpp")
print("Set path")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
#from integral import binary_find
import test_pybind

print(test_pybind.distribution_interpolation())
