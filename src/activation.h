#ifndef ACTIVATION_H
#define ACTIVATION_H

float sigmoid(float x);
float sigmoid_deriv(float x);
float tanh_deriv(float x);
void softmax(const float* logits, float* probs, int n);
void softmax_rows(const float* logits, float* probs, int rows, int cols);
float relu(float x);
float relu_deriv(float x);

#endif // ACTIVATION_H
