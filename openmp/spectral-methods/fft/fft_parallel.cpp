#include <cstddef>
#include <cstdint>
#include <omp.h>

void fft_parallel(float* __restrict real_op, float* __restrict imag_op, int N, const float* __restrict real_weight, const float* __restrict imag_weight, int num_threads = 0)
{
    if (num_threads > 0) {
      omp_set_num_threads(num_threads);
    }
    int iters = 0; {int t = N; while (t > 1) { t >>= 1; ++iters;}}
    int n = 1;
    int a = N / 2;
    for (int j = 0; j < iters; ++j) {
        const int n_local = n;
        const int a_local = a;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            if (!(i & n_local)) {
                std::size_t op_index = ((static_cast<std::size_t>(i) * static_cast<std::size_t>(a_local)) % (static_cast<std::size_t>(n_local) * static_cast<std::size_t>(a_local)));
                const int res_index = i + n_local;
                const float real_i = real_op[i];
                const float imag_i = imag_op[i];
                const float rw = real_weight[op_index];
                const float iw = imag_weight[op_index];
                const float real_r = real_op[res_index];
                const float imag_r = imag_op[res_index];
                const float prod_real = rw * real_r - iw * imag_r;
                const float prod_imag = iw * real_r + rw * imag_r;
                real_op[res_index] = real_i - prod_real;
                imag_op[res_index] = imag_i - prod_imag;
                real_op[i] = real_i + prod_real;
                imag_op[i] = imag_i + prod_imag;
            }
        }
        n <<= 1;
        a >>= 1;
    }
}
