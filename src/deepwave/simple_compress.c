#include "simple_compress.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* CPU implementation for float */
static void compress_float_cpu(float const *const input, uint8_t *const output,
                               size_t const n_elements) {
  float *const minmax = (float *)output;
  uint8_t *const compressed = output + 2 * sizeof(float);

  /* Find min and max */
  float min_val = input[0];
  float max_val = input[0];
  for (size_t i = 1; i < n_elements; i++) {
    if (input[i] < min_val) min_val = input[i];
    if (input[i] > max_val) max_val = input[i];
  }

  /* Store min and max */
  minmax[0] = min_val;
  minmax[1] = max_val;

  /* Quantize to 8 bits */
  float const range = max_val - min_val;
  float const scale = (range > 0) ? (255.0f / range) : 0.0f;

  for (size_t i = 0; i < n_elements; i++) {
    float const normalized = (input[i] - min_val) * scale;
    compressed[i] = (uint8_t)(normalized + 0.5f);
  }
}

/* CPU implementation for double */
static void compress_double_cpu(double const *const input,
                                uint8_t *const output,
                                size_t const n_elements) {
  double *const minmax = (double *)output;
  uint8_t *const compressed = output + 2 * sizeof(double);

  /* Find min and max */
  double min_val = input[0];
  double max_val = input[0];
  for (size_t i = 1; i < n_elements; i++) {
    if (input[i] < min_val) min_val = input[i];
    if (input[i] > max_val) max_val = input[i];
  }

  /* Store min and max */
  minmax[0] = min_val;
  minmax[1] = max_val;

  /* Quantize to 8 bits */
  double const range = max_val - min_val;
  double const scale = (range > 0) ? (255.0 / range) : 0.0;

  for (size_t i = 0; i < n_elements; i++) {
    double const normalized = (input[i] - min_val) * scale;
    compressed[i] = (uint8_t)(normalized + 0.5);
  }
}

/* CPU implementation for float decompression */
static void decompress_float_cpu(uint8_t const *const input,
                                 float *const output, size_t const n_elements) {
  float const *const minmax = (const float *)input;
  uint8_t const *const compressed = input + 2 * sizeof(float);

  float const min_val = minmax[0];
  float const max_val = minmax[1];
  float const range = max_val - min_val;
  float const scale = range / 255.0f;

  uint8_t const *const in_field = compressed;

  for (size_t i = 0; i < n_elements; i++) {
    output[i] = min_val + in_field[i] * scale;
  }
}

/* CPU implementation for double decompression */
static void decompress_double_cpu(uint8_t const *const input,
                                  double *const output,
                                  size_t const n_elements) {
  double const *const minmax = (const double *)input;
  uint8_t const *const compressed = input + 2 * sizeof(double);

  double const min_val = minmax[0];
  double const max_val = minmax[1];
  double const range = max_val - min_val;
  double const scale = range / 255.0;

  uint8_t const *const in_field = compressed;

  for (size_t i = 0; i < n_elements; i++) {
    output[i] = min_val + in_field[i] * scale;
  }
}

void simple_compress_cpu(void const *input, void *output, size_t n_elements,
                         int is_double) {
  if (is_double) {
    compress_double_cpu((double const *)input, (uint8_t *)output, n_elements);
  } else {
    compress_float_cpu((float const *)input, (uint8_t *)output, n_elements);
  }
}

void simple_decompress_cpu(void const *input, void *output, size_t n_elements,
                           int is_double) {
  if (is_double) {
    decompress_double_cpu((uint8_t const *)input, (double *)output, n_elements);
  } else {
    decompress_float_cpu((uint8_t const *)input, (float *)output, n_elements);
  }
}
