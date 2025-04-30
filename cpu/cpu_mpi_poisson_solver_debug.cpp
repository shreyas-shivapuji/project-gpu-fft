#include <mpi.h>
#include <fftw3-mpi.h>
#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>  

#define M_PI 3.141592

using dim_t = std::array<ptrdiff_t, 3>;

// fill the local chunk of the charge density array -  gaussian distribution
void fillChargeDensity(fftw_complex* rho, dim_t global_dims,
                       ptrdiff_t local_n0, ptrdiff_t i0_start) {
    const ptrdiff_t nx = global_dims[0],
                     ny = global_dims[1],
                     nz = global_dims[2];
    const double cx = nx / 2.0,
                 cy = ny / 2.0,
                 cz = nz / 2.0;
    const double sigma = std::min({nx, ny, nz}) / 6.0;

    for (ptrdiff_t i = 0; i < local_n0; ++i) {
        ptrdiff_t gi = i + i0_start;            // global i
        for (ptrdiff_t j = 0; j < ny; ++j) {
            for (ptrdiff_t k = 0; k < nz; ++k) {
                double x = gi - cx;
                double y = j  - cy;
                double z = k  - cz;
                double r2 = x*x + y*y + z*z;
                double charge = std::exp(-r2 / (2*sigma*sigma));
                ptrdiff_t idx = (i*ny + j)*nz + k;
                rho[idx][0] = charge;  // real
                rho[idx][1] = 0.0;     // imag
            }
        }
    }
}

// apply poisson operator in reciprocal space on the local chunk
void applyPoissonOperator(fftw_complex* data, dim_t global_dims,
                          ptrdiff_t local_n0, ptrdiff_t i0_start,
                          double dx, double dy, double dz) {
    const ptrdiff_t nx = global_dims[0],
                     ny = global_dims[1],
                     nz = global_dims[2];

    for (ptrdiff_t i = 0; i < local_n0; ++i) {
        ptrdiff_t gi = i + i0_start;
        double kx = (gi <= nx/2 ? gi : gi - nx) * (2*M_PI/(nx*dx));
        for (ptrdiff_t j = 0; j < ny; ++j) {
            double ky = (j <= ny/2 ? j : j - ny) * (2*M_PI/(ny*dy));
            for (ptrdiff_t k = 0; k < nz; ++k) {
                double kz = (k <= nz/2 ? k : k - nz) * (2*M_PI/(nz*dz));
                double k2 = kx*kx + ky*ky + kz*kz;
                ptrdiff_t idx = (i*ny + j)*nz + k;

                if (gi==0 && j==0 && k==0) {
                    data[idx][0] = data[idx][1] = 0.0;
                } else {
                    double scale = 1.0/k2;  // eps = 1 
                    data[idx][0] *= scale;
                    data[idx][1] *= scale;
                }
            }
        }
    }
}

void scaleFFTData(fftw_complex* data, ptrdiff_t local_size, double scale) {
    for (ptrdiff_t idx = 0; idx < local_size; ++idx) {
        data[idx][0] *= scale;
        data[idx][1] *= scale;
    }
}

double calculateSum(fftw_complex* data, ptrdiff_t local_size) {
    double sum = 0.0;
    for (ptrdiff_t idx = 0; idx < local_size; ++idx) {
        sum += data[idx][0];  // Only sum real part
    }
    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (rank == 0) {
        std::cout << "=== Poisson Solver Debug Info ===" << std::endl;
        std::cout << "Running with " << num_procs << " MPI processes" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double t_start = MPI_Wtime();
    double t_checkpoint;

    dim_t global_dims = {1024, 1024, 1024};
    const double dx = 1.0, dy = 1.0, dz = 1.0;

    // determine local slab size for each of the process/rank
    ptrdiff_t local_n0, i0_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(
        global_dims[0], global_dims[1], global_dims[2],
        MPI_COMM_WORLD, &local_n0, &i0_start);
    
    std::cout << "Rank " << rank << ": Local grid size = " << local_n0 
              << " x " << global_dims[1] << " x " << global_dims[2]
              << ", starting at i = " << i0_start << std::endl;
    
    ptrdiff_t local_size = local_n0 * global_dims[1] * global_dims[2];
    std::cout << "Rank " << rank << ": Local elements = " << local_size 
              << " (" << (local_size * sizeof(fftw_complex) / (global_dims[1] * global_dims[2]))
              << " MB)" << std::endl;

    // allocate local arrays
    fftw_complex* rho = fftw_alloc_complex(alloc_local);
    fftw_complex* phi = fftw_alloc_complex(alloc_local);

    if (!rho || !phi) {
        std::cerr << "Rank " << rank << ": Memory allocation failed!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    t_checkpoint = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Initialization time: " << (t_checkpoint - t_start) << " seconds" << std::endl;
    }

    fillChargeDensity(rho, global_dims, local_n0, i0_start);
    
    double local_charge_sum = calculateSum(rho, local_size);
    double global_charge_sum;
    MPI_Reduce(&local_charge_sum, &global_charge_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Total charge: " << global_charge_sum << std::endl;
    }

    double t_init_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Charge density initialization: " << (t_init_end - t_checkpoint) << " seconds" << std::endl;
    }
    t_checkpoint = t_init_end;

    // FFT plan creation on individual ranks/procs
    fftw_plan forward_plan = fftw_mpi_plan_dft_3d(
        global_dims[0], global_dims[1], global_dims[2],
        rho, phi, MPI_COMM_WORLD,
        FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan backward_plan = fftw_mpi_plan_dft_3d(
        global_dims[0], global_dims[1], global_dims[2],
        phi, phi, MPI_COMM_WORLD,
        FFTW_BACKWARD, FFTW_ESTIMATE);

    double t_plan_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "FFT plan creation: " << (t_plan_end - t_checkpoint) << " seconds" << std::endl;
    }
    t_checkpoint = t_plan_end;

    fftw_execute(forward_plan);   // forward FFT

    double t_fft_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Forward FFT: " << (t_fft_end - t_checkpoint) << " seconds" << std::endl;
    }
    t_checkpoint = t_fft_end;

    applyPoissonOperator(phi, global_dims, local_n0, i0_start, dx, dy, dz);   // perform ops in reciprocal space

    double t_poisson_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Poisson operator: " << (t_poisson_end - t_checkpoint) << " seconds" << std::endl;
    }
    t_checkpoint = t_poisson_end;

    fftw_execute(backward_plan);  // backward FFT

    double t_ifft_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Backward FFT: " << (t_ifft_end - t_checkpoint) << " seconds" << std::endl;
    }
    t_checkpoint = t_ifft_end;

    double norm = 1.0 / (global_dims[0]*global_dims[1]*global_dims[2]);   // normalize the result by dividing the result by total N = nx * ny * nz points
    scaleFFTData(phi, alloc_local, norm);

    // min/max potential values
    double local_min = INFINITY, local_max = -INFINITY;
    for (ptrdiff_t idx = 0; idx < local_size; ++idx) {
        local_min = std::min(local_min, phi[idx][0]);
        local_max = std::max(local_max, phi[idx][0]);
    }
    
    double global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Potential range: [" << global_min << ", " << global_max << "]" << std::endl;
    }

    double t_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Total execution time: " << (t_end - t_start) << " seconds" << std::endl;
        std::cout << "================================" << std::endl;
    }

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(rho);
    fftw_free(phi);

    MPI_Finalize();
    return 0;
}
