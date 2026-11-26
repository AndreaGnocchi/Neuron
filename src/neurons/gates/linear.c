#include "train.h"
#include "../../activation.h"
#include "../../rng.h"
#include "../../../include/mat.h"
#include <stdio.h>

#define INPUT_DIM  2
#define OUTPUT_DIM 1
#define BATCH_SIZE 4

void train_linear_gate(const float* Y, const char* name) {
  rng_seed(1);
  
  float X[BATCH_SIZE * INPUT_DIM] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
  };

  float X_T[INPUT_DIM * BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = 0; j < INPUT_DIM; j++) {
      X_T[j * BATCH_SIZE + i] = X[i * INPUT_DIM + j];
    }
  }

  float W[INPUT_DIM * OUTPUT_DIM];
  float b[OUTPUT_DIM];
  fill_random(W, INPUT_DIM * OUTPUT_DIM);
  fill_random(b, OUTPUT_DIM);

  float Z[BATCH_SIZE * OUTPUT_DIM];
  float A[BATCH_SIZE * OUTPUT_DIM];
  float dZ[BATCH_SIZE * OUTPUT_DIM];
  float dW[INPUT_DIM * OUTPUT_DIM];
  float db[OUTPUT_DIM];

  const float lr = 0.5f;
  const int epochs = 40000;
  for (int epoch = 0; epoch <= epochs; epoch++) {
    // Z = X * W + b   ->   A = sigmoid(Z)
    mat_mul_set(BATCH_SIZE, OUTPUT_DIM, INPUT_DIM,
                X, INPUT_DIM, W, OUTPUT_DIM, Z, OUTPUT_DIM);

    float loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < OUTPUT_DIM; j++) {
        int idx = i * OUTPUT_DIM + j;
        Z[idx] += b[j];
        A[idx] = sigmoid(Z[idx]);

        float diff = A[idx] - Y[idx];
        loss += 0.5f * diff * diff;
        dZ[idx] = diff * sigmoid_deriv(A[idx]);
      }
    }

    if (epoch % 8000 == 0) {
      printf("Epoch %5d | Loss: %.6f\n", epoch, loss / BATCH_SIZE);
    }

    // dW = X_T * dZ,  db = sum(dZ, axis=0)
    mat_mul_set(INPUT_DIM, OUTPUT_DIM, BATCH_SIZE,
                X_T, BATCH_SIZE, dZ, OUTPUT_DIM, dW, OUTPUT_DIM);
    
    for (int j = 0; j < OUTPUT_DIM; j++) {
      db[j] = 0.0f;
      for (int i = 0; i < BATCH_SIZE; i++) {
        db[j] += dZ[i * OUTPUT_DIM + j];
      }
    }

    for (int i = 0; i < INPUT_DIM * OUTPUT_DIM; i++) {
      W[i] -= lr * (dW[i] / BATCH_SIZE);
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
      b[i] -= lr * (db[i] / BATCH_SIZE);
    }
  }
  
  mat_print_gate(X, A, Y, BATCH_SIZE, INPUT_DIM, name);
}
