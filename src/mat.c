#include "../include/mat.h"
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <stddef.h>

#define THRESHOLD      128
#define TASK_THRESHOLD (128 * 128 * 128)

static void mat_mul_small(int m, int n, int k,
                          const float* restrict A, int lda,
                          const float* restrict B, int ldb,
                          float* restrict C, int ldc) {
  for (int i = 0; i < m; i++) {
    const float* restrict a_row = &A[i * lda];
    float* restrict c_row = &C[i * ldc];

    int p = 0;
    for (; p + 3 < k; p += 4) {
      float a0 = a_row[p + 0];
      float a1 = a_row[p + 1];
      float a2 = a_row[p + 2];
      float a3 = a_row[p + 3];

      const float* restrict b0_row = &B[(p + 0) * ldb];
      const float* restrict b1_row = &B[(p + 1) * ldb];
      const float* restrict b2_row = &B[(p + 2) * ldb];
      const float* restrict b3_row = &B[(p + 3) * ldb];

      for (int j = 0; j < n; j++) {
        c_row[j] += a0 * b0_row[j] +
                      a1 * b1_row[j] +
                      a2 * b2_row[j] +
                      a3 * b3_row[j];
      }
    }

    for (; p + 1 < k; p += 2) {
      float a0 = a_row[p + 0];
      float a1 = a_row[p + 1];

      const float* restrict b0_row = &B[(p + 0) * ldb];
      const float* restrict b1_row = &B[(p + 1) * ldb];

      for (int j = 0; j < n; j++) {
        c_row[j] += a0 * b0_row[j] + a1 * b1_row[j];
      }
    }

    for (; p < k; p++) {
      float a_ip = a_row[p];
      const float* restrict b_row = &B[p * ldb];
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

  const float* A_bot = A + m1 * lda;
  const float* A_right = A + k1;
  const float* A_bot_right = A + m1 * lda + k1;
  
  const float* B_right = B + n1;
  const float* B_bot = B + k1 * ldb;
  const float* B_bot_right = B + k1 * ldb + n1;
  
  float* C_right = C + n1;
  float* C_bot = C + m1 * ldc;
  float* C_bot_right = C + m1 * ldc + n1;

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

  #pragma omp parallel for schedule(static) if ((size_t)m * n > 128 * 128)
  for (int i = 0; i < m; i++)
    memset(&C[i * ldc], 0, (size_t)n * sizeof(float));

  mat_mul(m, n, k, A, lda, B, ldb, C, ldc);
}
