#include "train.h"
#include "../../include/neurons.h"

static const float AND_Y[4]  = {0, 0, 0, 1};
static const float NAND_Y[4] = {1, 1, 1, 0};
static const float OR_Y[4]   = {0, 1, 1, 1};
static const float NOR_Y[4]  = {1, 0, 0, 0};
static const float XOR_Y[4]  = {0, 1, 1, 0};
static const float XNOR_Y[4] = {1, 0, 0, 1};

void and_neuron(void)  { train_linear_gate(AND_Y, "AND"); }
void nand_neuron(void) { train_linear_gate(NAND_Y, "NAND"); }
void or_neuron(void)   { train_linear_gate(OR_Y, "OR"); }
void nor_neuron(void)  { train_linear_gate(NOR_Y, "NOR"); }
void xor_neuron(void)  { train_hidden_gate(XOR_Y, "XOR"); }
void xnor_neuron(void) { train_hidden_gate(XNOR_Y, "XNOR"); }
