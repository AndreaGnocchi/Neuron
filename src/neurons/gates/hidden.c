#include "train.h"
#include "../../activation.h"
#include "../../rng.h"
#include "../../../include/mat.h"
#include <math.h>
#include <stdio.h>

#define INPUT_DIM  2
#define HIDDEN_DIM 2
#define OUTPUT_DIM 1
#define BATCH_SIZE 4

void train_hidden_gate(const float* Y, const char* name) {
  rng_seed(1);
  
  float X[BATCH_SIZE * INPUT_DIM] = {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
  };
  
  // Network parameters: 2 -> 2 -> 1
  float W1[INPUT_DIM * HIDDEN_DIM];
  float b1[HIDDEN_DIM];
  float W2[HIDDEN_DIM * OUTPUT_DIM];
  float b2[OUTPUT_DIM];

  fill_random(W1, INPUT_DIM * HIDDEN_DIM);
  fill_random(b1, HIDDEN_DIM);
  fill_random(W2, HIDDEN_DIM * OUTPUT_DIM);
  fill_random(b2, OUTPUT_DIM);

  // Forward pass buffers
  float Z1[BATCH_SIZE * HIDDEN_DIM];
  float A1[BATCH_SIZE * HIDDEN_DIM];
  float Z2[BATCH_SIZE * OUTPUT_DIM];
  float A2[BATCH_SIZE * OUTPUT_DIM];

  // Backward pass buffers
  float dZ2[BATCH_SIZE * OUTPUT_DIM];
  float dW2[HIDDEN_DIM * OUTPUT_DIM];
  float db2[OUTPUT_DIM];

  float dA1[BATCH_SIZE * HIDDEN_DIM];
  float dZ1[BATCH_SIZE * HIDDEN_DIM];
  float dW1[INPUT_DIM * HIDDEN_DIM];
  float db1[HIDDEN_DIM];

  // Transposed matrices for backprop
  float A1_T[HIDDEN_DIM * BATCH_SIZE];
  float W2_T[OUTPUT_DIM * HIDDEN_DIM];
  float X_T[INPUT_DIM * BATCH_SIZE];

  float lr = 0.5f;
  const int epochs = 40000;
  for (int epoch = 0; epoch <= epochs; epoch++) {
    // Forward pass:
    //   Z1 = X * W1 + b1   ->   A1 = sigmoid(Z1)
    //   Z2 = A1 * W2 + b2  ->   A2 = tanhf(Z2)
    //   loss += 0.5f * (A2 - Y)^2

    // Z1 = X * W1 + b1
    mat_mul_set(BATCH_SIZE, HIDDEN_DIM, INPUT_DIM,
                X, INPUT_DIM, W1, HIDDEN_DIM, Z1, HIDDEN_DIM);

    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < HIDDEN_DIM; j++) {
        int idx = i * HIDDEN_DIM + j;
        Z1[idx] += b1[j];
        A1[idx] = sigmoid(Z1[idx]);
      }
    }

    // Z2 = A1 * W2 + b2
    mat_mul_set(BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM,
                A1, HIDDEN_DIM, W2, OUTPUT_DIM, Z2, OUTPUT_DIM);

    float loss = 0.0f;
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < OUTPUT_DIM; j++) {
        int idx = i * OUTPUT_DIM + j;
        Z2[idx] += b2[j];
        A2[idx] = tanhf(Z2[idx]);

        float diff = A2[idx] - Y[idx];
        loss += 0.5f * diff * diff;
        dZ2[idx] = diff * tanh_deriv(A2[idx]);
      }
    }

    if (epoch % 8000 == 0) {
      printf("Epoch %5d | Loss: %.6f\n", epoch, loss / BATCH_SIZE);
    }
    
    // Backward pass:
    //   dZ2 = (A2 - Y) * tanh_deriv(A2)
    //   dW2 = A1_T * dZ2
    //   db2 = sum(dZ2, axis=0)
    //   dA1 = dZ2 * W2_T
    //   dZ1 = dA1 * sigmoid_deriv(A1)
    //   dW1 = X_T * dZ1
    //   db1 = sum(dZ1, axis=0)

    // Transponse A1
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < HIDDEN_DIM; j++) {
        A1_T[j * BATCH_SIZE + i] = A1[i * HIDDEN_DIM + j];
      }
    }

    // dW2 = A1_T * dZ2
    mat_mul_set(HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE,
                A1_T, BATCH_SIZE, dZ2, OUTPUT_DIM, dW2, OUTPUT_DIM);
    
    // db2 = sum(dZ2, axis=0)
    for (int j = 0; j < OUTPUT_DIM; j++) {
      db2[j] = 0.0f;
      for (int i = 0; i < BATCH_SIZE; i++) {
        db2[j] += dZ2[i * OUTPUT_DIM + j];
      }
    }

    // Transpose W2
    for (int i = 0; i < HIDDEN_DIM; i++) {
      for (int j = 0; j < OUTPUT_DIM; j++) {
        W2_T[j * HIDDEN_DIM + i] = W2[i * OUTPUT_DIM + j];
      }
    }

    // dA1 = dZ2 * W2_T
    mat_mul_set(BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM,
                dZ2, OUTPUT_DIM, W2_T, HIDDEN_DIM, dA1, HIDDEN_DIM);

    // dZ1 = dA1 * sigmoid_deriv(A1)
    for (int i = 0; i < BATCH_SIZE * HIDDEN_DIM; i++) {
      dZ1[i] = dA1[i] * sigmoid_deriv(A1[i]);
    }

    // Transpose X
    for (int i = 0; i < BATCH_SIZE; i++) {
      for (int j = 0; j < INPUT_DIM; j++) {
        X_T[j * BATCH_SIZE + i] = X[i * INPUT_DIM + j];
      }
    }

    // dW1 = X_T * dZ1
    mat_mul_set(INPUT_DIM, HIDDEN_DIM, BATCH_SIZE,
                X_T, BATCH_SIZE, dZ1, HIDDEN_DIM, dW1, HIDDEN_DIM);

    // db1 = sum(dZ1, axis=0)
    for (int j = 0; j < HIDDEN_DIM; j++) {
      db1[j] = 0.0f;
      for (int i = 0; i < BATCH_SIZE; i++) {
        db1[j] += dZ1[i * HIDDEN_DIM + j];
      }
    }

    // Gradient Descent
    for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) {
      W1[i] -= lr * (dW1[i] / BATCH_SIZE);
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
      b1[i] -= lr * (db1[i] / BATCH_SIZE);
    }
    for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) {
      W2[i] -= lr * (dW2[i] / BATCH_SIZE);
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
      b2[i] -= lr * (db2[i] / BATCH_SIZE);
    }
  }

  mat_print_gate(X, A2, Y, BATCH_SIZE, INPUT_DIM, name);
}
