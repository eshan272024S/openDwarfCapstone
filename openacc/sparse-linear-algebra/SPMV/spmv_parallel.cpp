// spmv_parallel.cpp — OpenACC SPMV (GPU)

#include <chrono>
#include <openacc.h>

double spmv_parallel(const int* row_ptr, const int* col_idx, const double* val,
                     int nrows, const double* x, double* y, int nthreads)
{
    (void)nthreads; // ignored for GPU

    if (nrows <= 0 || !row_ptr || !col_idx || !val || !x || !y) {
        return 0.0;
    }

    const int nnz = row_ptr[nrows];   // total nonzeros

    // zero output
    for (int i = 0; i < nrows; ++i) {
        y[i] = 0.0;
    }

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

#pragma acc data copyin(row_ptr[0:nrows+1], col_idx[0:nnz], val[0:nnz], x[0:nrows]) \
                 copy(y[0:nrows])
    {
#pragma acc parallel loop gang vector
        for (int r = 0; r < nrows; ++r) {
            double sum = 0.0;
            int start = row_ptr[r];
            int end   = row_ptr[r + 1];

            for (int k = start; k < end; ++k) {
                sum += val[k] * x[col_idx[k]];
            }
            y[r] = sum;
        }
    }

    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}