#ifndef DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#define DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#include <stddef.h>

#include "scalar.h"
void setup(const TYPE * const fd1,
           const TYPE * const fd2);
void propagate(TYPE * const wfn,        /* next wavefield */
               TYPE * const auxn,       /* next aux_wavefield */
               const TYPE * const wfc,  /* current wavefield */
               const TYPE * const wfp,  /* previous wavefield */
               const TYPE * const auxc, /* current aux_wavefield */
               const TYPE * const sigma,
               const TYPE * const model,
               const TYPE * const fd1, /* 1st difference coeffs */
               const TYPE * const fd2, /* 2nd difference coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const TYPE dt);
void add_sources(TYPE * const next_wavefield,
                 const TYPE * const model,
                 const TYPE * const source_amplitudes,
                 const ptrdiff_t * const source_locations,
                 const ptrdiff_t * const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot);
void add_scattering(TYPE * const next_scattered_wavefield,
                    const TYPE * const next_wavefield,
                    const TYPE * const current_wavefield,
                    const TYPE * const previous_wavefield,
                    const TYPE * const scatter,
                    const ptrdiff_t * const shape,
                    const ptrdiff_t num_shots);
void record_receivers(TYPE * const receiver_amplitudes,
                      const TYPE * const current_wavefield,
                      const ptrdiff_t * const receiver_locations,
                      const ptrdiff_t * const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot);
void save_wavefields(TYPE * const current_saved_wavefield,
                     TYPE * const current_saved_wavefield_t,
                     TYPE * const current_saved_wavefield_tt,
                     const TYPE * const next_wavefield,
                     const TYPE * const current_wavefield,
                     const TYPE * const previous_wavefield,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots, const TYPE dt,
                     const enum wavefield_save_strategy save_strategy);
void imaging_condition(TYPE * const model_grad,
                       const TYPE * const current_wavefield,
                       const TYPE * const next_saved_wavefield,
                       const TYPE * const current_saved_wavefield,
                       const TYPE * const previous_saved_wavefield,
                       const TYPE * const sigma,
                       const ptrdiff_t * const shape,
                       const ptrdiff_t * const pml_width,
                       const ptrdiff_t num_shots);
void model_grad_scaling(TYPE * const model_grad,
                        const TYPE * const scaling,
                        const ptrdiff_t * const shape,
                        const ptrdiff_t * const pml_width);
#endif /* DEEPWAVE_SCALAR_SCALAR_DEVICE_H */
