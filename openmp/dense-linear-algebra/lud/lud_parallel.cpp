#include <vector>
#include <cmath>
#include <stdexcept>


void lu_decompose_parallel(std::vector<std::vector<double>>& A) {
   const int N = static_cast<int>(A.size());
   if (N == 0) return;


   const double eps = 1e-30;


   for (int k = 0; k < N; ++k) {
       const double pivot = A[k][k];
       if (std::abs(pivot) < eps) {
           throw std::runtime_error("Zero/tiny pivot in lu_decompose_parallel");
       }


       // Column scaling: L(i,k) = A(i,k) / U(k,k)
       #pragma omp parallel for
       for (int i = k + 1; i < N; ++i) {
           A[i][k] /= pivot;
       }


       // Trailing update: A(i,j) -= L(i,k) * U(k,j)
       #pragma omp parallel for
       for (int i = k + 1; i < N; ++i) {
           const double Lik = A[i][k];
           for (int j = k + 1; j < N; ++j) {
               A[i][j] -= Lik * A[k][j];
           }
       }
   }
}
