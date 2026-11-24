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
