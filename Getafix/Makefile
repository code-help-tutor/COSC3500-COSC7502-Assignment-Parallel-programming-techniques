# set some defaults.  ?= means don't override the value if it was set already
MPICXX?=mpic++
CXX?=mpic++
CXXFLAGS?=-std=c++11 -O2 -mavx -fopenmp
NVCC?=nvcc
NVFLAGS?=-O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

# all targets
TARGETS = Assignment1_GradeBot

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGETS) : matrixMultiply.h matrixMultiply.o matrixMultiplyGPU.cuh matrixMultiplyGPU.o matrixMultiplyMPI.h matrixMultiplyMPI.o

LIBS_Assignment1_GradeBot = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl -lcudart -lcublas

CXXFLAGS_Assignment1_GradeBot = 

matrixMultiply.o : matrixMultiply.h
matrixMultiplyGPU.o : matrixMultiplyGPU.cuh
matrixMultiplyMPI.o : matrixMultiplyMPI.h

# wildcard rules
#%MPI.o : %MPI.cpp
#	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

#%MPI : %MPI.cpp
#	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%GPU.o : %GPU.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
