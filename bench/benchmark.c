#include "../include/mat.h"
#include "../src/rng.h"
#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 4096

int main(void) {
  const int m = MAT_SIZE;
  const int n = MAT_SIZE;
  const int k = MAT_SIZE;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  const size_t elems_a = (size_t)m * (size_t)lda;
  const size_t elems_b = (size_t)k * (size_t)ldb;
  const size_t elems_c = (size_t)m * (size_t)ldc;

  float* mat_data = malloc((elems_a + elems_b + elems_c) * sizeof(*mat_data));
  if (!mat_data) {
    perror("malloc");
    return 1;
  }

  float* A = mat_data;
  float* B = mat_data + elems_a;
  float* C = mat_data + elems_a + elems_b;

  fill_random(A, elems_a);
  fill_random(B, elems_b);

  mat_mul(m, n, k, A, lda, B, ldb, C, ldc);

  volatile float sink = C[0];
  (void)sink;

  free(mat_data);
  return 0;
}
