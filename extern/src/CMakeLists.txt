cmake_minimum_required(VERSION 3.10)
project(integral_module CXX)
set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -ffast-math -march=native")
enable_language(CXX)

# Explicitly set the C compiler to clang
set(CMAKE_CXX_COMPILER clang++)

# Find Python includes
execute_process(COMMAND python3-config --includes
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${PYTHON_INCLUDE_DIRS})

# Include pybind11
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

# Find GSL
find_path(GSL_INCLUDE_DIR NAMES gsl/gsl_version.h HINTS /opt/homebrew/opt/gsl/include)
find_library(GSL_LIBRARIES NAMES gsl gslcblas HINTS /opt/homebrew/opt/gsl/lib)

include_directories(${GSL_INCLUDE_DIR})

# Find OpenMP using provided paths
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling OpenMP for C++.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OpenMP_FOUND TRUE) # Optional: Set a simple flag for later use
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
    set(OpenMP_FOUND FALSE)
endif()
##################
set(OPENMP_INCLUDE /opt/homebrew/Cellar/libomp/20.1.2/include)
find_library(OPENMP_LIBRARIES NAMES libomp HINTS /opt/homebrew/Cellar/libomp/20.1.2/lib)

include_directories(${OPENMP_INCLUDE})

# Create the shared library
add_library(integral SHARED integral.cpp Ds.cpp)

# Link pthread separately if needed
target_link_libraries(integral PRIVATE
        ${GSL_LIBRARIES}
        ${OPENMP_LIBRARIES}
        pthread)

set_target_properties(integral PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
