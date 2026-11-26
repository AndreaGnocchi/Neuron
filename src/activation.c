#include "activation.h"
#include <math.h>

float sigmoid(float x) {
  if (x >= 0.0) {
    float z = exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    float z = exp(x);
    return z / (1.0 + z);
  }
}

float sigmoid_deriv(float x) { return x * (1.0f - x); }

float tanh_deriv(float x) {
  return 1.0f - x * x;
}

void softmax(const float* restrict logits, float* restrict probs, int n) {
    if (n == 0) {
        return;
    }

    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf(logits[i] - max_val);
        probs[i] = e;
        sum += e;
    }

    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] *= inv_sum;
    }
}

void softmax_rows(const float* restrict logits, float* restrict probs, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        softmax(logits + r * cols, probs + r * cols, cols);
    }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float relu_deriv(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}
