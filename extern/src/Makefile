
all: test

test: integral.cpp Ds.cpp integral.h
	clang++ -O3 -Wall -fopenmp -shared \
	-o integral.so integral.cpp Ds.cpp -ffast-math -march=native -std=c++11 -undefined dynamic_lookup \
	$(shell python3-config --includes) -I$(PWD)/pybind11/include \
	-L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas -pthread -I/opt/homebrew/opt/gsl/include

	cp integral.so ../boltzmann_integration

clean:
	rm -rf *.o *.so *.c *.so.DSYM
