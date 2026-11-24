#include "rng.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

uint32_t rng_u32(void) {
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}

float rng_float(void) {
  return (float)rng_u32() / 2147483648.0f - 1.0f;
}

void fill_random(float* p, size_t n) {
  for (size_t i = 0; i < n; i++)
    p[i] = rng_float();
}
