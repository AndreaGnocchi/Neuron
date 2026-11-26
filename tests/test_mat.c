#include "../include/mat.h"
#include "../src/rng.h"
#include "test_main.h"
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float* _malloc(size_t n) {
  float* p = malloc((n ? n : 1) * sizeof(float));
  if (!p)
    exit(1);

  return p;
}

static void* _calloc(size_t n, size_t size) {
  float* p = calloc(n ? n : 1, size ? size : 1);
  if (!p)
    exit(1);

  return p;
}

static void fill_const(float* p, size_t n, float v) {
  for (size_t i = 0; i < n; i++)
    p[i] = v;
}

typedef struct {
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  float* A;
  float* B;
  float* C;
  size_t a_sz;
  size_t b_sz;
  size_t c_sz;
} case_t;

static const float SENTINEL = 31337.0f;

#define T_MC 60
#define T_NC 256
#define T_KC 64
#define T_PAR_DIM 128

static void case_init(case_t* c, int m, int n, int k, int pad_a, int pad_b, int pad_c) {
  c->m = m; c->n = n; c->k = k;
  c->lda = k + pad_a;
  c->ldb = n + pad_b;
  c->ldc = n + pad_c;
  c->a_sz = (size_t)m * c->lda;
  c->b_sz = (size_t)k * c->ldb;
  c->c_sz = (size_t)m * c->ldc;
  c->A = _malloc(c->a_sz);
  c->B = _malloc(c->b_sz);
  c->C = _malloc(c->c_sz);
  fill_const(c->A, c->a_sz, SENTINEL);
  fill_const(c->B, c->b_sz, SENTINEL);
  fill_const(c->C, c->c_sz, SENTINEL);
  for (int i = 0; i < m; i++)
    for (int p = 0; p < k; p++)
      c->A[(size_t)i * c->lda + p] = rng_float();
  
  for (int p = 0; p < k; p++)
    for (int j = 0; j < n; j++)
      c->B[(size_t)p * c->ldb + j] = rng_float();
}
static void case_free(case_t* c) { free(c->A); free(c->B); free(c->C); }

static void case_randomize_C(case_t* c) {
  for (int i = 0; i < c->m; i++)
    for (int j = 0; j < c->n; j++) c->C[(size_t)i * c->ldc + j] = rng_float();
}
static void case_zero_C(case_t* c) {
  for (int i = 0; i < c->m; i++)
    for (int j = 0; j < c->n; j++) c->C[(size_t)i * c->ldc + j] = 0.0f;
}

static int padding_intact(const float* P, int rows, int cols, int ld) {
  for (int i = 0; i < rows; i++)
    for (int j = cols; j < ld; j++)
      if (P[(size_t)i * ld + j] != SENTINEL) return 0;
  return 1;
}

static void ref_mul_acc(int m, int n, int k,
                        const float* A, int lda,
                        const float* B, int ldb,
                        const float* C0, int ldc0,
                        double* out, double* bound) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double s = C0 ? C0[(size_t)i * ldc0 + j] : 0.0;
      double b = C0 ? fabs((double)C0[(size_t)i * ldc0 + j]) : 0.0;
      for (int p = 0; p < k; p++) {
        double t = (double)A[(size_t)i * lda + p] * (double)B[(size_t)p * ldb + j];
        s += t;
        b += fabs(t);
      }
      out[(size_t)i * n + j] = s;
      if (bound) bound[(size_t)i * n + j] = b;
    }
}

static void expect_close(const case_t* c, const double* ref,
                         const double* bound) {
  int bad = 0;
  double worst = 0;
  for (int i = 0; i < c->m && bad < 5; i++)
    for (int j = 0; j < c->n && bad < 5; j++) {
      double got = c->C[(size_t)i * c->ldc + j];
      double want = ref[(size_t)i * c->n + j];
      double tol = (c->k + 2) * 1.2e-7 * bound[(size_t)i * c->n + j] * 2.0 + 1e-6;
      double err = fabs(got - want);
      if (err > worst)
        worst = err;
      if (!(err <= tol)) 
        bad++;
    }
  
  CHECK(bad == 0);
}

static void run_shape(int m, int n, int k, int pad_a, int pad_b, int pad_c) {
  case_t c;
  case_init(&c, m, n, k, pad_a, pad_b, pad_c);
  size_t mn = (size_t)m * n;
  double* ref = _calloc(mn, sizeof(double));
  double* bnd = _calloc(mn, sizeof(double));
  float* A0 = _malloc(c.a_sz), *B0 = _malloc(c.b_sz), *C0 = _malloc(c.c_sz);
  memcpy(A0, c.A, c.a_sz * sizeof(float));
  memcpy(B0, c.B, c.b_sz * sizeof(float));

  // 1) accumulate onto random C
  case_randomize_C(&c);
  memcpy(C0, c.C, c.c_sz * sizeof(float));
  ref_mul_acc(m, n, k, c.A, c.lda, c.B, c.ldb, C0, c.ldc, ref, bnd);
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  expect_close(&c, ref, bnd);
  CHECK(padding_intact(c.C, m, n, c.ldc));
  CHECK(memcmp(A0, c.A, c.a_sz * sizeof(float)) == 0);
  CHECK(memcmp(B0, c.B, c.b_sz * sizeof(float)) == 0);

  // 2) mat_mul onto zeros == plain product
  case_zero_C(&c);
  ref_mul_acc(m, n, k, c.A, c.lda, c.B, c.ldb, NULL, 0, ref, bnd);
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  expect_close(&c, ref, bnd);

  // 3) mat_mul_set must ignore prior garbage
  case_randomize_C(&c);
  for (int i = 0; i < m; i++) c.C[(size_t)i * c.ldc] = 1e30f;
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  expect_close(&c, ref, bnd);
  CHECK(padding_intact(c.C, m, n, c.ldc));

  free(ref); free(bnd); free(A0); free(B0); free(C0);
  case_free(&c);
}

// Tests

static void test_one_by_one(void) {
  float A = 3.0f, B = 4.0f, C = 5.0f;
  mat_mul(1, 1, 1, &A, 1, &B, 1, &C, 1);
  CHECK(C == 17.0f);

  C = 99.0f;
  mat_mul_set(1, 1, 1, &A, 1, &B, 1, &C, 1);
  CHECK(C == 12.0f);
}

static void test_known_2x2(void) {
  float A[] = {1, 2, 3, 4}, B[] = {5, 6, 7, 8}, C[4] = {0};
  mat_mul(2, 2, 2, A, 2, B, 2, C, 2);
  CHECK(C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);
}

static void test_known_rectangular(void) {
  float A[] = {1, 2, 3,
               4, 5, 6};
  float B[] = {1, 0, 2, 1,
               0, 1, 3, 1,
               1, 1, 0, 1};
  float want[] = {4, 5, 8, 6,
                  10, 11, 23, 15};
  float C[8] = {0};
  mat_mul(2, 4, 3, A, 3, B, 4, C, 4);
  for (int i = 0; i < 8; i++)
    CHECK(C[i] == want[i]);
}

static void test_identity(void) {
  const int N = 150;
  float* I = _malloc((size_t)N * N), *X = _malloc((size_t)N * N), *C = _malloc((size_t)N * N);
  fill_const(I, (size_t)N * N, 0.0f);
  for (int i = 0; i < N; i++) I[(size_t)i * N + i] = 1.0f;
  fill_random(X, (size_t)N * N);

  mat_mul_set(N, N, N, I, N, X, N, C, N);
  CHECK(memcmp(C, X, (size_t)N * N * sizeof(float)) == 0);
  mat_mul_set(N, N, N, X, N, I, N, C, N);
  CHECK(memcmp(C, X, (size_t)N * N * sizeof(float)) == 0);
  
  free(I); free(X); free(C);
}

static void test_zero_matrices(void) {
  const int N = 150;
  float* Z = _malloc((size_t)N * N), *X = _malloc((size_t)N * N), *C = _malloc((size_t)N * N);
  fill_const(Z, (size_t)N * N, 0.0f);
  fill_random(X, (size_t)N * N);
  fill_random(C, (size_t)N * N);
  float* C0 = _malloc((size_t)N * N);
  memcpy(C0, C, (size_t)N * N * sizeof(float));

  mat_mul(N, N, N, Z, N, X, N, C, N);
  CHECK(memcmp(C, C0, (size_t)N * N * sizeof(float)) == 0);
  mat_mul(N, N, N, X, N, Z, N, C, N);
  CHECK(memcmp(C, C0, (size_t)N * N * sizeof(float)) == 0);

  mat_mul_set(N, N, N, Z, N, X, N, C, N);
  int nz = 0;
  for (size_t i = 0; i < (size_t)N * N; i++) nz += (C[i] != 0.0f);
  CHECK(nz == 0);
  free(Z); free(X); free(C); free(C0);
}

static void test_accumulate_twice(void) {
  const int m = 70, n = 65, k = 90;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 0);
  float* once = _malloc(c.c_sz);
  case_zero_C(&c);
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  memcpy(once, c.C, c.c_sz * sizeof(float));
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  
  int bad = 0;
  for (size_t i = 0; i < c.c_sz; i++) {
    if (fabsf(c.C[i] - 2.0f * once[i]) > 1e-5f * (1.0f + fabsf(once[i])))
      bad++;
  }
  CHECK(bad == 0);
  free(once);
  case_free(&c);
}

static void test_set_idempotent(void) {
  const int m = 130, n = 270, k = 131;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 0);
  float* first = _malloc(c.c_sz);
  case_randomize_C(&c);
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  memcpy(first, c.C, c.c_sz * sizeof(float));
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  
  CHECK(memcmp(first, c.C, c.c_sz * sizeof(float)) == 0);
  free(first);
  case_free(&c);
}

static void test_small_shapes_exhaustive(void) {
  for (int m = 1; m <= 5; m++)
    for (int n = 1; n <= 5; n++)
      for (int k = 1; k <= 5; k++) {
        run_shape(m, n, k, 0, 0, 0);
        run_shape(m, n, k, 3, 2, 1);
      }
}

static void test_kernel_tile_tails(void) {
  const int ns[] = {1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33};
  const int ks[] = {1, 2, 3, 4, 5, 7, 8, 9};
  for (int m = 1; m <= 13; m++)
    for (size_t b = 0; b < sizeof(ns) / sizeof(ns[0]); b++)
      for (size_t d = 0; d < sizeof(ks) / sizeof(ks[0]); d++)
        run_shape(m, ns[b], ks[d], 0, 0, 0);
}

static void test_block_boundaries(void) {
  const int ms[] = {T_MC - 1, T_MC, T_MC + 1, 2 * T_MC - 1, 2 * T_MC, 2 * T_MC + 1};
  const int ns[] = {T_NC - 1, T_NC, T_NC + 1};
  const int ks[] = {T_KC - 1, T_KC, T_KC + 1, 2 * T_KC, 2 * T_KC + 1};
  for (size_t a = 0; a < sizeof(ms) / sizeof(ms[0]); a++)
    for (size_t b = 0; b < sizeof(ns) / sizeof(ns[0]); b++)
      for (size_t d = 0; d < sizeof(ks) / sizeof(ks[0]); d++)
        run_shape(ms[a], ns[b], ks[d], 0, 0, 0);
}

static void test_parallel_threshold_boundaries(void) {
  const int D = T_PAR_DIM;
  run_shape(D, D, D, 0, 0, 0);
  run_shape(D + 1, D, D, 0, 0, 0);
  run_shape(D, D + 1, D, 0, 0, 0);
  run_shape(D, D, D + 1, 0, 0, 0);
  run_shape(D - 1, D, D, 0, 0, 0);
  run_shape(D, D - 1, D, 0, 0, 0);
  run_shape(D, D, D - 1, 0, 0, 0);
}

static void test_odd_uneven_splits(void) {
  const int shapes[][3] = {
      {1, 1, 200}, {200, 1, 1}, {1, 200, 1}, {1, 200, 200}, {200, 1, 200},
      {200, 200, 1}, {127, 129, 131}, {255, 3, 257}, {3, 255, 129}, {129, 65, 7},
      {7, 129, 65}, {193, 191, 67},
      {121, 513, 129}, {125, 271, 131}, {61, 257, 65}, {179, 519, 193},
  };
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 0, 0, 0);
}

static void test_leading_dimensions(void) {
  const int shapes[][3] = {{10, 12, 14}, {65, 70, 66}, {100, 129, 97}, {130, 130, 130}, {61, 257, 65}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 5, 0, 0);
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 0, 7, 0);
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 0, 0, 9);
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 3, 11, 13);
  }
}

static void test_submatrix_views(void) {
  const int BIG = 200, m = 90, n = 80, k = 70;
  const int ao = 13, bo = 21, co = 34;
  float* A = _malloc((size_t)BIG * BIG), *B = _malloc((size_t)BIG * BIG), *C = _malloc((size_t)BIG * BIG);
  float* Corig = _malloc((size_t)BIG * BIG);
  fill_random(A, (size_t)BIG * BIG);
  fill_random(B, (size_t)BIG * BIG);
  fill_random(C, (size_t)BIG * BIG);
  memcpy(Corig, C, (size_t)BIG * BIG * sizeof(float));

  const float* Ap = A + (size_t)ao * BIG + ao;
  const float* Bp = B + (size_t)bo * BIG + bo;
  float* Cp = C + (size_t)co * BIG + co;

  double* ref = malloc((size_t)m * n * sizeof(double));
  double* bnd = malloc((size_t)m * n * sizeof(double));
  ref_mul_acc(m, n, k, Ap, BIG, Bp, BIG, Cp, BIG, ref, bnd);
  mat_mul(m, n, k, Ap, BIG, Bp, BIG, Cp, BIG);

  int bad_in = 0, bad_out = 0;
  for (int i = 0; i < BIG; i++)
    for (int j = 0; j < BIG; j++) {
      size_t idx = (size_t)i * BIG + j;
      int inside = (i >= co && i < co + m && j >= co && j < co + n);
      if (inside) {
        double w = ref[(size_t)(i - co) * n + (j - co)];
        double tol = (k + 2) * 2.4e-7 * bnd[(size_t)(i - co) * n + (j - co)] + 1e-6;
        if (!(fabs(C[idx] - w) <= tol))
          bad_in++;
      } else if (C[idx] != Corig[idx]) {
        bad_out++;
      }
    }
  
  CHECK(bad_in == 0);
  CHECK(bad_out == 0);
  free(A); free(B); free(C); free(Corig); free(ref); free(bnd);
}

static void test_set_only_touches_block(void) {
  const int m = 100, n = 90, k = 80;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 17);
  case_randomize_C(&c);
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  CHECK(padding_intact(c.C, m, n, c.ldc));
  case_free(&c);
}

static void test_set_large_parallel_memset(void) {
  const int shapes[][3] = {{128, 128, 10}, {129, 128, 10}, {128, 129, 10},
                           {1, 128 * 128 + 1, 3}, {128 * 128 + 1, 1, 3}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
    run_shape(shapes[s][0], shapes[s][1], shapes[s][2], 0, 0, 2);
}

static void test_large_parallel_path(void) {
  run_shape(256, 256, 256, 0, 0, 0);
  run_shape(300, 260, 280, 4, 4, 4);
  run_shape(512, 100, 400, 0, 0, 0);
  run_shape(183, 600, 200, 1, 3, 5);
}

static void test_thread_count_independence(void) {
  const int m = 257, n = 300, k = 263;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 0);
  float* base = _malloc(c.c_sz);

  int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  memcpy(base, c.C, c.c_sz * sizeof(float));

  const int counts[] = {2, 3, 4, 8, 16};
  for (size_t t = 0; t < sizeof(counts) / sizeof(counts[0]); t++) {
    omp_set_num_threads(counts[t]);
    fill_const(c.C, c.c_sz, SENTINEL);
    mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
    CHECK(memcmp(base, c.C, c.c_sz * sizeof(float)) == 0);
  }
  
  omp_set_num_threads(saved);
  free(base);
  case_free(&c);
}

static void test_repeated_runs_stable(void) {
  const int m = 130, n = 130, k = 130;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 0);
  float* base = _malloc(c.c_sz);
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  memcpy(base, c.C, c.c_sz * sizeof(float));
  int diffs = 0;
  
  for (int r = 0; r < 50; r++) {
    mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
    diffs += (memcmp(base, c.C, c.c_sz * sizeof(float)) != 0);
  }
  
  CHECK(diffs == 0);
  free(base);
  case_free(&c);
}

static void test_nested_in_parallel_region(void) {
  const int m = 140, n = 140, k = 140, T = 4;
  case_t cs[4];
  double* refs[4], *bnds[4];
  
  for (int t = 0; t < T; t++) {
    case_init(&cs[t], m, n, k, 0, 0, 0);
    refs[t] = malloc((size_t)m * n * sizeof(double));
    bnds[t] = malloc((size_t)m * n * sizeof(double));
    case_zero_C(&cs[t]);
    ref_mul_acc(m, n, k, cs[t].A, cs[t].lda, cs[t].B, cs[t].ldb, NULL, 0, refs[t], bnds[t]);
  }
   
  #pragma omp parallel for num_threads(4)
  for (int t = 0; t < T; t++)
    mat_mul(m, n, k, cs[t].A, cs[t].lda, cs[t].B, cs[t].ldb, cs[t].C, cs[t].ldc);
  
  for (int t = 0; t < T; t++) {
    expect_close(&cs[t], refs[t], bnds[t]);
    free(refs[t]); free(bnds[t]); case_free(&cs[t]);
  }
}

static void test_special_values(void) {
  const int m = 70, n = 70, k = 70;
  case_t c;
  case_init(&c, m, n, k, 0, 0, 0);

  c.A[(size_t)5 * c.lda + 3] = NAN;
  case_zero_C(&c);
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  int bad = 0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      int isn = isnan(c.C[(size_t)i * c.ldc + j]);
      if ((i == 5) != (isn != 0))
        bad++;
    }
  CHECK(bad == 0);

  c.A[(size_t)5 * c.lda + 3] = 0.5f;
  fill_const(c.C, c.c_sz, NAN);
  mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  int nans = 0;
  for (size_t i = 0; i < c.c_sz; i++)
    nans += isnan(c.C[i]) ? 1 : 0;
  
  CHECK(nans == 0);
  case_free(&c);

  case_init(&c, m, n, k, 0, 0, 0);
  c.B[(size_t)2 * c.ldb + 20] = NAN;
  c.B[(size_t)66 * c.ldb + 66] = NAN;
  case_zero_C(&c);
  mat_mul(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
  bad = 0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      int isn = isnan(c.C[(size_t)i * c.ldc + j]);
      if ((j == 20 || j == 66) != (isn != 0))
        bad++;
    }
  CHECK(bad == 0);
  case_free(&c);
}

static void test_small_integer_exactness(void) {
  const int shapes[][3] = {{67, 71, 73}, {128, 128, 128}, {131, 70, 129}, {61, 257, 65}};
  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    int m = shapes[s][0], n = shapes[s][1], k = shapes[s][2];
    case_t c;
    case_init(&c, m, n, k, 2, 3, 4);
    for (int i = 0; i < m; i++)
      for (int p = 0; p < k; p++)
        c.A[(size_t)i * c.lda + p] = (float)((i * 7 + p * 3) % 5 - 2);
    
    for (int p = 0; p < k; p++)
      for (int j = 0; j < n; j++)
        c.B[(size_t)p * c.ldb + j] = (float)((p * 5 + j * 11) % 7 - 3);
    
    mat_mul_set(m, n, k, c.A, c.lda, c.B, c.ldb, c.C, c.ldc);
    int bad = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        long want = 0;
        for (int p = 0; p < k; p++)
          want += (long)c.A[(size_t)i * c.lda + p] * (long)c.B[(size_t)p * c.ldb + j];
        
        if (c.C[(size_t)i * c.ldc + j] != (float)want)
          bad++;
      }
    
    CHECK(bad == 0);
    case_free(&c);
  }
}

static void test_random_fuzz(void) {
  for (int iter = 0; iter < 60; iter++) {
    int m = 1 + (int)(rng_u32() % 150);
    int n = 1 + (int)(rng_u32() % 300);
    int k = 1 + (int)(rng_u32() % 150);
    run_shape(m, n, k, (int)(rng_u32() % 4), (int)(rng_u32() % 4), (int)(rng_u32() % 4));
  }
}

void run_all_mat_tests(void) {
  printf("[MAT]:");
  test_one_by_one();
  test_known_2x2();
  test_known_rectangular();
  test_identity();
  test_zero_matrices();
  test_accumulate_twice();
  test_set_idempotent();
  test_small_shapes_exhaustive();
  test_kernel_tile_tails();
  test_block_boundaries();
  test_parallel_threshold_boundaries();
  test_odd_uneven_splits();
  test_leading_dimensions();
  test_submatrix_views();
  test_set_only_touches_block();
  test_set_large_parallel_memset();
  test_large_parallel_path();
  test_thread_count_independence();
  test_repeated_runs_stable();
  test_nested_in_parallel_region();
  test_special_values();
  test_small_integer_exactness();
  test_random_fuzz();
}
