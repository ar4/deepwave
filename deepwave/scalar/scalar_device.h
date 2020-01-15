#ifndef DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#define DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#include <stddef.h>

#include "scalar.h"
void setup(const TYPE *__restrict const fd1,
           const TYPE *__restrict const fd2);
void propagate(TYPE *__restrict const wfn,        /* next wavefield */
               TYPE *__restrict const auxn,       /* next aux_wavefield */
               const TYPE *__restrict const wfc,  /* current wavefield */
               const TYPE *__restrict const wfp,  /* previous wavefield */
               const TYPE *__restrict const auxc, /* current aux_wavefield */
               const TYPE *__restrict const sigma,
               const TYPE *__restrict const model,
               const TYPE *__restrict const fd1, /* 1st difference coeffs */
               const TYPE *__restrict const fd2, /* 2nd difference coeffs */
               const ptrdiff_t *__restrict const shape,
               const ptrdiff_t *__restrict const pml_width,
               const ptrdiff_t num_shots, const TYPE dt);
void add_sources(TYPE *__restrict const next_wavefield,
                 const TYPE *__restrict const model,
                 const TYPE *__restrict const source_amplitudes,
                 const ptrdiff_t *__restrict const source_locations,
                 const ptrdiff_t *__restrict const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot);
void add_scattering(TYPE *__restrict const next_scattered_wavefield,
                    const TYPE *__restrict const next_wavefield,
                    const TYPE *__restrict const current_wavefield,
                    const TYPE *__restrict const previous_wavefield,
                    const TYPE *__restrict const scatter,
                    const ptrdiff_t *__restrict const shape,
                    const ptrdiff_t num_shots);
void record_receivers(TYPE *__restrict const receiver_amplitudes,
                      const TYPE *__restrict const current_wavefield,
                      const ptrdiff_t *__restrict const receiver_locations,
                      const ptrdiff_t *__restrict const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot);
void save_wavefields(TYPE *__restrict const current_saved_wavefield,
                     TYPE *__restrict const current_saved_wavefield_t,
                     TYPE *__restrict const current_saved_wavefield_tt,
                     const TYPE *__restrict const next_wavefield,
                     const TYPE *__restrict const current_wavefield,
                     const TYPE *__restrict const previous_wavefield,
                     const ptrdiff_t *__restrict const shape,
                     const ptrdiff_t num_shots, const TYPE dt,
                     const enum wavefield_save_strategy save_strategy);
void imaging_condition(TYPE *__restrict const model_grad,
                       const TYPE *__restrict const current_wavefield,
                       const TYPE *__restrict const next_saved_wavefield,
                       const TYPE *__restrict const current_saved_wavefield,
                       const TYPE *__restrict const previous_saved_wavefield,
                       const TYPE *__restrict const sigma,
                       const ptrdiff_t *__restrict const shape,
                       const ptrdiff_t *__restrict const pml_width,
                       const ptrdiff_t num_shots);
void model_grad_scaling(TYPE *__restrict const model_grad,
                        const TYPE *__restrict const scaling,
                        const ptrdiff_t *__restrict const shape,
                        const ptrdiff_t *__restrict const pml_width);
#endif /* DEEPWAVE_SCALAR_SCALAR_DEVICE_H */
