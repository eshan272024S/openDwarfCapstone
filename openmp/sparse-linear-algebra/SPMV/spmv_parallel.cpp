// spmv_parallel.cpp — parallel SPMV using OpenMP.
// Computes y = A * x for a matrix in CSR format.
// We only time the core SPMV loop (not setup or I/O).

#include <chrono>
#include <omp.h>

double spmv_parallel(const int* row_ptr, const int* col_idx, const double* val,
                     int nrows, const double* x, double* y, int nthreads)
{
  if (nrows <= 0 || !row_ptr || !col_idx || !val || !x || !y) {
    return 0.0;
  }

  // Clear output
  for (int r = 0; r < nrows; ++r) {
    y[r] = 0.0;
  }

  if (nthreads <= 0) {
    nthreads = omp_get_max_threads();
  }

  using clock = std::chrono::high_resolution_clock;
  auto t0 = clock::now();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
  for (int r = 0; r < nrows; ++r) {
    double sum = 0.0;
    int start = row_ptr[r];
    int end   = row_ptr[r + 1];

    for (int k = start; k < end; ++k) {
      int c = col_idx[k];
      sum += val[k] * x[c];
    }
    y[r] = sum;
  }

  auto t1 = clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}