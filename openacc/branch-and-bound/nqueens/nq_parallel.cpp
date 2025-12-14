#include <cstdint>
static constexpr int MAX_N = 63;
#pragma acc routine seq
static std::uint64_t count_with_first_row_fixed_acc(int N, int r0)
{
    if (N == 1) return 1;
    const std::uint64_t board_mask = (N >= 64) ? ~0ULL : ((1ULL << N) - 1ULL);
    const std::uint64_t first = (1ULL << r0);
    std::uint64_t masks[66] = {};
    std::uint64_t left_masks[66] = {};
    std::uint64_t right_masks[66] = {};
    std::uint64_t ms[66] = {};
    masks[1] = first;
    left_masks[1] = first << 1;
    right_masks[1] = first >> 1;
    ms[1] = masks[1] | left_masks[1] | right_masks[1];
    std::uint64_t solutions = 0;
    int i = 1;
    while (i >= 1) {
        std::uint64_t m = ms[i];
        std::uint64_t ns = (m + 1ULL) & ~m;
        if ((ns & board_mask) != 0ULL) {
            ms[i] |= ns;
            if (i == N - 1) {
                ++solutions;
            } else {
                masks[i + 1] = masks[i] | ns;
                left_masks[i + 1] = (left_masks[i] | ns) << 1;
                right_masks[i + 1] = (right_masks[i] | ns) >> 1;
                ms[i + 1] = masks[i + 1] | left_masks[i + 1] | right_masks[i + 1];
                ++i;
            }
        } else {
            --i;
        }
    }
    return solutions;
}
std::uint64_t count_nqueens_parallel(int n)
{
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (n > MAX_N) return 0;
    std::uint64_t total = 0;
    #pragma acc parallel loop reduction(+:total) copyin(n)
    for (int r0 = 0; r0 < n; ++r0) {
        total += count_with_first_row_fixed_acc(n, r0);
    }
    return total;
}
