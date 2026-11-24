#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <stddef.h>

uint32_t rng_u32(void);
float rng_float(void);
void fill_random(float* p, size_t n);
void rng_seed(uint64_t s);

#endif // RNG_H
