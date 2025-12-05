#ifndef SIMPLE_COMPRESS_H
#define SIMPLE_COMPRESS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compress a batch of wavefields to 8 bits per sample
 *
 * Parameters:
 *   input: pointer to input data (float or double)
 *   output: pointer to output buffer (uint8_t)
 *   n_batch: number of wavefields in batch
 *   n_elements_per_field: number of elements in each wavefield
 *   is_double: 1 if input is double, 0 if float
 *
 * Output buffer size: n_elements + 2*n_batch*sizeof(float or double) for
 * min/max Returns: actual compressed size in bytes
 */
void simple_compress_cpu(const void *input, void *output,
                         size_t n_elements_per_field, int is_double);
int simple_compress_cuda(const void *input, void *output, size_t n_batch,
                         size_t n_elements_per_field, int is_double);

/* Decompress a batch of wavefields from 8 bits per sample
 *
 * Parameters:
 *   input: pointer to compressed data
 *   output: pointer to output buffer (float or double)
 *   n_batch: number of wavefields in batch
 *   n_elements_per_field: number of elements in each wavefield
 *   is_double: 1 if output is double, 0 if float
 */
void simple_decompress_cpu(const void *input, void *output,
                           size_t n_elements_per_field, int is_double);
int simple_decompress_cuda(const void *input, void *output, size_t n_batch,
                           size_t n_elements_per_field, int is_double);

#ifdef __cplusplus
}
#endif

#endif /* SIMPLE_COMPRESS_H */
