#ifndef MAT_H
#define MAT_H

void mat_mul(int m, int n, int k,
             const float* A, int lda,
             const float* B, int ldb,
             float* C, int ldc);

void mat_mul_set(int m, int n, int k,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float* C, int ldc);

void mat_print_gate(const float* X, const float* A2, const float* Y,
                    int batch_size, int input_dim, const char* gate);

#endif // MAT_H
