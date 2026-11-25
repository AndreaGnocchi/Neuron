#include "../include/mat.h"
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <immintrin.h>

#define THRESHOLD      128
#define TASK_THRESHOLD (128 * 128 * 128)

static inline void kernel_6x16(int k,
                               const float* restrict A, int lda,
                               const float* restrict B, int ldb,
                               float* restrict C, int ldc) {
  __m256 c[6][2];

  for (int r = 0; r < 6; r++) {
    c[r][0] = _mm256_loadu_ps(&C[r * ldc]);     // load 8 floats from C
    c[r][1] = _mm256_loadu_ps(&C[r * ldc + 8]); // load next 8 floats from C
  }

  int p = 0;
  for (; p + 3 < k; p += 4) {
    for (int u = 0; u < 4; u++) {
      int pu = p + u;
      __m256 b0 = _mm256_loadu_ps(&B[pu * ldb]);     // load 8 floats from B
      __m256 b1 = _mm256_loadu_ps(&B[pu * ldb + 8]); // load next 8 floats from B

      for (int r = 0; r < 6; r++) {
        __m256 a = _mm256_broadcast_ss(&A[r * lda + pu]); // copy 1 float from A into all 8
        c[r][0] = _mm256_fmadd_ps(a, b0, c[r][0]);        // c = a * b0 + c
        c[r][1] = _mm256_fmadd_ps(a, b1, c[r][1]);        // c = a * b1 + c
      }
    } 
  }

  // Cleanup for left k steps (k % 4 rows B/col A)
  for (; p < k; p++) {
    __m256 b0 = _mm256_loadu_ps(&B[p * ldb]);     // load 8 floats from B
    __m256 b1 = _mm256_loadu_ps(&B[p * ldb + 8]); // load next 8 floats from B
    for (int r = 0; r < 6; r++) {
      __m256 a = _mm256_broadcast_ss(&A[r * lda + p]); // copy 1 float from A into all 8
      c[r][0] = _mm256_fmadd_ps(a, b0, c[r][0]);       // c = a * b0 + c
      c[r][1] = _mm256_fmadd_ps(a, b1, c[r][1]);       // c = a * b1 + c
    }
  }
  
  for (int r = 0; r < 6; r++) {
    _mm256_storeu_ps(&C[r * ldc], c[r][0]);     // store the 8 floats back in C
    _mm256_storeu_ps(&C[r * ldc + 8], c[r][1]); // store the next 8 floats back in C
  }
}

static inline void kernel_6x8(int k,
                              const float* restrict A, int lda,
                              const float* restrict B, int ldb,
                              float* restrict C, int ldc) {
  __m256 c[6];

  for (int r = 0; r < 6; r++) {
    c[r] = _mm256_loadu_ps(&C[r * ldc]);  // load 8 floats from C
  }

  for (int p = 0; p < k; p++) {
    __m256 b = _mm256_loadu_ps(&B[p * ldb]); // load 8 floats from B
    for (int r = 0; r < 6; r++) {
      __m256 a = _mm256_broadcast_ss(&A[r * lda + p]); // copy 1 float from A into all 8
      c[r] = _mm256_fmadd_ps(a, b, c[r]); // c = a * b + c
    }
  }
  
  for (int r = 0; r < 6; r++) {
    _mm256_storeu_ps(&C[r * ldc], c[r]); // store the 8 floats back in C
  }
}

static inline void kernel_1x8(int k,
                              const float* restrict A,
                              const float* restrict B, int ldb,
                              float* restrict C) {
  __m256 c = _mm256_loadu_ps(C); // load 8 floats from C
  
  for (int p = 0; p < k; p++) {
    __m256 a = _mm256_broadcast_ss(&A[p]); // copy 1 float from A into all 8
    __m256 b = _mm256_loadu_ps(&B[p * ldb]); // load 8 floats from B
    c = _mm256_fmadd_ps(a, b, c); // c = a * b + c
  }

  _mm256_storeu_ps(C, c); // store the 8 floats back in C
}

static void mat_mul_small(int m, int n, int k,
                          const float* restrict A, int lda,
                          const float* restrict B, int ldb,
                          float* restrict C, int ldc) {
  int i = 0;

  // Blocks of 6 rows
  for (; i + 5 < m; i += 6) {
    int j = 0;
    for (; j + 15 < n; j += 16) {
      kernel_6x16(k, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
    }

    for (; j + 7 < n; j += 8) {
      kernel_6x8(k, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
    }

    // Scalar tail columns (n % 8)
    for (; j < n; j++) {
      for (int r = 0; r < 6; r++) {
        float acc = C[(i + r) * ldc + j];
        for (int p = 0; p < k; p++)
          acc += A[(i + r) * lda + p] * B[p * ldb + j];
        C[(i + r) * ldc + j] = acc;
      }
    }
  }

  // Rows left (m % 6)
  for (; i < m; i++) {
    int j = 0;
    for (; j + 7 < n; j += 8) {
      kernel_1x8(k, &A[i * lda], &B[j], ldb, &C[i * ldc + j]);
    }

    for (; j < n; j++) {
      float acc = C[i * ldc + j];
      for (int p = 0; p < k; p++)
        acc += A[i * lda + p] * B[p * ldb + j];
      C[i * ldc + j] = acc;
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

void mat_print_gate(const float* X, const float* A2, const float* Y,
                           int batch_size, int input_dim, const char* gate) {
  printf("\nResult:\n");
  for (int i = 0; i < batch_size; i++) {
    printf("  %.0f %s %.0f = %.4f (target: %.0f)\n",
           X[i * input_dim + 0],
           gate,
           X[i * input_dim + 1],
           A2[i],
           Y[i]);
  }
}
