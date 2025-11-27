#include "simple_compress.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* CPU implementation for float */
static void compress_float_cpu(const float *input, uint8_t *output,
                               size_t n_elements) {
  float *minmax = (float *)output;
  uint8_t *compressed = output + 2 * sizeof(float);

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
  float range = max_val - min_val;
  float scale = (range > 0) ? (255.0f / range) : 0.0f;

  for (size_t i = 0; i < n_elements; i++) {
    float normalized = (input[i] - min_val) * scale;
    compressed[i] = (uint8_t)(normalized + 0.5f);
  }
}

/* CPU implementation for double */
static void compress_double_cpu(const double *input, uint8_t *output,
                                size_t n_elements) {
  double *minmax = (double *)output;
  uint8_t *compressed = output + 2 * sizeof(double);

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
  double range = max_val - min_val;
  double scale = (range > 0) ? (255.0 / range) : 0.0;

  for (size_t i = 0; i < n_elements; i++) {
    double normalized = (input[i] - min_val) * scale;
    compressed[i] = (uint8_t)(normalized + 0.5);
  }
}

/* CPU implementation for float decompression */
static void decompress_float_cpu(const uint8_t *input, float *output,
                                 size_t n_elements) {
  const float *minmax = (const float *)input;
  const uint8_t *compressed = input + 2 * sizeof(float);

  float min_val = minmax[0];
  float max_val = minmax[1];
  float range = max_val - min_val;
  float scale = range / 255.0f;

  const uint8_t *in_field = compressed;

  for (size_t i = 0; i < n_elements; i++) {
    output[i] = min_val + in_field[i] * scale;
  }
}

/* CPU implementation for double decompression */
static void decompress_double_cpu(const uint8_t *input, double *output,
                                  size_t n_elements) {
  const double *minmax = (const double *)input;
  const uint8_t *compressed = input + 2 * sizeof(double);

  double min_val = minmax[0];
  double max_val = minmax[1];
  double range = max_val - min_val;
  double scale = range / 255.0;

  const uint8_t *in_field = compressed;

  for (size_t i = 0; i < n_elements; i++) {
    output[i] = min_val + in_field[i] * scale;
  }
}

void simple_compress_cpu(const void *input, void *output, size_t n_elements,
                         int is_double) {
  if (is_double) {
    compress_double_cpu((const double *)input, (uint8_t *)output, n_elements);
  } else {
    compress_float_cpu((const float *)input, (uint8_t *)output, n_elements);
  }
}

void simple_decompress_cpu(const void *input, void *output, size_t n_elements,
                           int is_double) {
  if (is_double) {
    decompress_double_cpu((const uint8_t *)input, (double *)output, n_elements);
  } else {
    decompress_float_cpu((const uint8_t *)input, (float *)output, n_elements);
  }
}
