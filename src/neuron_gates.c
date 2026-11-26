#include "../include/neuron_gates.h"
#include <stdio.h>

int main(void) {
  printf("[AND neuron]:\n\n");
  and_neuron();

  printf("\n---------------------------------\n");
  printf("\n[NAND neuron]:\n\n");
  nand_neuron();

  printf("\n---------------------------------\n");
  printf("\n[OR neuron]:\n\n");
  or_neuron();

  printf("\n---------------------------------\n");
  printf("\n[NOR neuron]:\n\n");
  nor_neuron();

  printf("\n---------------------------------\n");
  printf("\n[XOR neuron]:\n\n");
  xor_neuron();

  printf("\n---------------------------------\n");
  printf("\n[XNOR neuron]:\n\n");
  xnor_neuron();

  return 0;
} 
