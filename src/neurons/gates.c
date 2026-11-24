#include "train.h"
#include "../../include/neurons.h"

static const float XOR_Y[4]  = {0, 1, 1, 0};
static const float XNOR_Y[4] = {1, 0, 0, 1};

void xor_neuron(void)  { train_hidden_gate(XOR_Y, "XOR"); }
void xnor_neuron(void) { train_hidden_gate(XNOR_Y, "XNOR"); }
