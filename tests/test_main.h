#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern size_t test_count;
extern size_t test_passed;

#define CHECK(cond) do {                                                       \
  test_count++;                                                                \
  if (cond) {                                                                  \
      test_passed++;                                                           \
  } else {                                                                     \
      fprintf(stderr, "%s:%d: CHECK FAILED: %s\n", __FILE__, __LINE__, #cond); \
  }                                                                            \
} while (0)

// mat.h & mat.c
void run_all_mat_tests(void);

#endif // TEST_MAIN_H
