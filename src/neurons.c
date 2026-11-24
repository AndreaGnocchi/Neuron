#include "../include/neurons.h"
#include <iso646.h>
#include <stdio.h>

int main(void) {
  printf("XOR neuron:\n\n");
  xor_neuron();

  printf("\nXNOR neuron:\n\n");
  xnor_neuron();
} 
