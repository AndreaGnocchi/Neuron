#include "../include/mat.h"
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <stddef.h>

#define THRESHOLD      64
#define TASK_THRESHOLD (64 * 64 * 64)

static void mat_mul_small(int m, int n, int k,
                        const float* restrict A, int lda,
                        const float* restrict B, int ldb,
                        float* restrict C, int ldc) {
  for (int i = 0; i < m; i++) {
    for (int p = 0; p < k; p++) {
      float a_ip = A[i * lda + p];
      const float * restrict b_row = &B[p * ldb];
      float * restrict c_row = &C[i * ldc];
      for (int j = 0; j < n; j++) {
        c_row[j] += a_ip * b_row[j];
      }
    }
  }
}

static void mat_mul_rec(int m, int n, int k,
                        const float* restrict A, int lda,
                        const float* restrict B, int ldb,
                        float* restrict C, int ldc) {
  if (m <= THRESHOLD && n <= THRESHOLD && k <= THRESHOLD) {
    mat_mul_small(m, n, k, A, lda, B, ldb, C, ldc);
    return;
  }

  // Splitting the matrix
  int m1 = m / 2, m2 = m - m1;
  int n1 = n / 2, n2 = n - n1;
  int k1 = k / 2, k2 = k - k1;

  const float *A_bot = A + m1 * lda;
  const float *A_right = A + k1;
  const float *A_bot_right = A + m1 * lda + k1;
  
  const float *B_right = B + n1;
  const float *B_bot = B + k1 * ldb;
  const float *B_bot_right = B + k1 * ldb + n1;
  
  float *C_right = C + n1;
  float *C_bot = C + m1 * ldc;
  float *C_bot_right = C + m1 * ldc + n1;

  int is_small = ((size_t)m * (size_t)n * (size_t)k) <= TASK_THRESHOLD;

  // k in [0, k1)
  #pragma omp task firstprivate(m1, n1, k1, A, lda, B, ldb, C, ldc) final(is_small)
  mat_mul_rec(m1, n1, k1, A, lda, B, ldb, C, ldc);

  #pragma omp task firstprivate(m1, n2, k1, A, lda, B_right, ldb, C_right, ldc) final(is_small)
  mat_mul_rec(m1, n2, k1, A, lda, B_right, ldb, C_right, ldc);

  #pragma omp task firstprivate(m2, n1, k1, A_bot, lda, B, ldb, C_bot, ldc) final(is_small)
  mat_mul_rec(m2, n1, k1, A_bot, lda, B, ldb, C_bot, ldc);

  #pragma omp task firstprivate(m2, n2, k1, A_bot, lda, B_right, ldb, C_bot_right, ldc) final(is_small)
  mat_mul_rec(m2, n2, k1, A_bot, lda, B_right, ldb, C_bot_right, ldc);

  #pragma omp taskwait

  // k in [k1, k)
  #pragma omp task firstprivate(m1, n1, k2, A_right, lda, B_bot, ldb, C, ldc) final(is_small)
  mat_mul_rec(m1, n1, k2, A_right, lda, B_bot, ldb, C, ldc);

  #pragma omp task firstprivate(m1, n2, k2, A_right, lda, B_bot_right, ldb, C_right, ldc) final(is_small)
  mat_mul_rec(m1, n2, k2, A_right, lda, B_bot_right, ldb, C_right, ldc);

  #pragma omp task firstprivate(m2, n1, k2, A_bot_right, lda, B_bot, ldb, C_bot, ldc) final(is_small)
  mat_mul_rec(m2, n1, k2, A_bot_right, lda, B_bot, ldb, C_bot, ldc);

  #pragma omp task firstprivate(m2, n2, k2, A_bot_right, lda, B_bot_right, ldb, C_bot_right, ldc) final(is_small)
  mat_mul_rec(m2, n2, k2, A_bot_right, lda, B_bot_right, ldb, C_bot_right, ldc);

  #pragma omp taskwait
}

void mat_mul(int m, int n, int k,
             const float* restrict A, int lda,
             const float* restrict B, int ldb,
             float* restrict C, int ldc) {
  assert(m > 0 && n > 0 && k > 0);

  #pragma omp parallel
  #pragma omp single nowait
  mat_mul_rec(m, n, k, A, lda, B, ldb, C, ldc);
}

void mat_mul_set(int m, int n, int k,
                 const float* restrict A, int lda,
                 const float* restrict B, int ldb,
                 float* restrict C, int ldc) {
  assert(m > 0 && n > 0 && k > 0);

  #pragma omp parallel for schedule(static) if ((size_t)m * n > 64 * 64)
  for (int i = 0; i < m; i++)
    memset(&C[i * ldc], 0, (size_t)n * sizeof(float));

  mat_mul(m, n, k, A, lda, B, ldb, C, ldc);
}
