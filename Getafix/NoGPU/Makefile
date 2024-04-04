# set some defaults.  ?= means don't override the value if it was set already
MPICXX?=mpic++
CXX?=mpic++
CXXFLAGS?=-std=c++11 -O2 -mavx -fopenmp

# all targets
TARGETS = Assignment1_GradeBot

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGETS) : matrixMultiply.h matrixMultiply.o matrixMultiplyMPI.h matrixMultiplyMPI.o

LIBS_Assignment1_GradeBot = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl

CXXFLAGS_Assignment1_GradeBot = 

matrixMultiply.o : matrixMultiply.h
matrixMultiplyMPI.o : matrixMultiplyMPI.h

# wildcard rules
%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
