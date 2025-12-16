#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstddef>
#include <cstdlib>


void lu_decompose_parallel(std::vector<std::vector<double>>& A)
{
    const int N = (int)A.size();
    if (N == 0) return;

    const double eps = 1e-30;
    const size_t NN = (size_t)N * (size_t)N;

    // Allocate contiguous buffer - tha gpu friendly version
    double* Af = (double*)std::malloc(NN * sizeof(double));
    if (!Af) {
        throw std::runtime_error("malloc failed in lu_decompose_parallel (OpenACC)");
    }

    // Flatten A -> Af
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Af[(size_t)i * (size_t)N + (size_t)j] = A[i][j];
        }
    }

    #pragma acc data copy(Af[0:NN])
    {
        for (int k = 0; k < N; ++k) {

            // Bring pivot back to host for safety check
            const size_t pivot_idx = (size_t)k * (size_t)N + (size_t)k;
            #pragma acc update self(Af[pivot_idx:1])

            const double pivot = Af[pivot_idx];
            if (std::fabs(pivot) < eps) {
                throw std::runtime_error("Zero/tiny pivot in lu_decompose_parallel (OpenACC)");
            }

            // Column scaling
            #pragma acc parallel loop present(Af) firstprivate(k, N, pivot)
            for (int i = k + 1; i < N; ++i) {
                Af[(size_t)i * (size_t)N + (size_t)k] /= pivot;
            }

            // Trailing update
            #pragma acc parallel loop collapse(2) present(Af) firstprivate(k, N)
            for (int i = k + 1; i < N; ++i) {
                for (int j = k + 1; j < N; ++j) {
                    Af[(size_t)i * (size_t)N + (size_t)j] -=
                        Af[(size_t)i * (size_t)N + (size_t)k] *
                        Af[(size_t)k * (size_t)N + (size_t)j];
                }
            }
        }

        // Copy full matrix back once
        #pragma acc update self(Af[0:NN])
    }

    // Unflatten Af -> A
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = Af[(size_t)i * (size_t)N + (size_t)j];
        }
    }

    std::free(Af);
}