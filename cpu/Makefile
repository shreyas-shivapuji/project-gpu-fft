# === Compiler and flags ===
CXX       = mpicxx
CXXFLAGS  = -O3 -std=c++17 -Wall

# === FFTW-MPI paths (adjust if different on your system) ===
# Use `pkg-config --cflags --libs fftw3-mpi` if installed that way
FFTW_INC  = /export/apps/libs/fftw-3.3.10/include
FFTW_LIB = /export/apps/libs/fftw-3.3.10/lib

LDLIBS      = -lfftw3_mpi -lfftw3 -lm

# === Source and target ===
EXE    = poisson_solver_mpi_debug_512
SRC    = cpu_mpi_poisson_solver_debug.cpp

# Build rule
all: $(EXE)

$(EXE): $(SRC)
	$(CXX) $(CXXFLAGS) -I$(FFTW_INC) -L$(FFTW_LIB) $(SRC) -o $(EXE) $(LDLIBS)

# Clean rule
clean:
	rm -f $(EXE) *.o

# Phony targets
.PHONY: all clean
