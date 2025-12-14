#include <cstddef>
#include <cstdint>

void fft_parallel(float* __restrict real_op,
                  float* __restrict imag_op,
                  int N,
                  const float* __restrict real_weight,
                  const float* __restrict imag_weight)
{
    int iters = 0; { int t = N; while (t > 1) { t >>= 1; ++iters; } }
    int n = 1;
    int a = N / 2;
    #pragma acc data copy(real_op[0:N], imag_op[0:N]) \
                     copyin(real_weight[0:a], imag_weight[0:a])
    {
        for (int j = 0; j < iters; ++j) {
            #pragma acc parallel loop present(real_op,imag_op,real_weight,imag_weight) independent
            for (int i = 0; i < N; ++i) {
                if (!(i & n)) {
                    std::size_t op_index =
                        ( (static_cast<std::size_t>(i) * static_cast<std::size_t>(a)) %
                          (static_cast<std::size_t>(n) * static_cast<std::size_t>(a)) );
                    int res_index = i + n;
                    float real_temp = real_op[i];
                    float imag_temp = imag_op[i];
                    float rw = real_weight[op_index];
                    float iw = imag_weight[op_index];
                    float real_Temp = rw * real_op[res_index] - iw * imag_op[res_index];
                    float imag_Temp = iw * real_op[res_index] + rw * imag_op[res_index];
                    real_op[res_index] = real_temp - real_Temp;
                    imag_op[res_index] = imag_temp - imag_Temp;
                    real_op[i] = real_temp + real_Temp;
                    imag_op[i] = imag_temp + imag_Temp;
                }
            }
            n <<= 1;
            a >>= 1;
        }
    }
}
void fft_parallel(float* __restrict real_op,
                  float* __restrict imag_op,
                  int N,
                  const float* __restrict real_weight,
                  const float* __restrict imag_weight,
                  int /*nthreads*/)
{
    fft_parallel(real_op, imag_op, N, real_weight, imag_weight);
}
