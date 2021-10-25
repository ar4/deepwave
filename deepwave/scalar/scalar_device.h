#ifndef DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#define DEEPWAVE_SCALAR_SCALAR_DEVICE_H
#include <stddef.h>

#include "scalar.h"
void setup(const DEEPWAVE_TYPE * const fd1,
           const DEEPWAVE_TYPE * const fd2);
void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next aux_wavefield */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current aux_wavefield */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1, /* 1st difference coeffs */
               const DEEPWAVE_TYPE * const fd2, /* 2nd difference coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt);
void add_sources(DEEPWAVE_TYPE * const next_wavefield,
                 const DEEPWAVE_TYPE * const model,
                 const DEEPWAVE_TYPE * const source_amplitudes,
                 const ptrdiff_t * const source_locations,
                 const ptrdiff_t * const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot);
void add_scattering(DEEPWAVE_TYPE * const next_scattered_wavefield,
                    const DEEPWAVE_TYPE * const next_wavefield,
                    const DEEPWAVE_TYPE * const current_wavefield,
                    const DEEPWAVE_TYPE * const previous_wavefield,
                    const DEEPWAVE_TYPE * const scatter,
                    const ptrdiff_t * const shape,
                    const ptrdiff_t num_shots);
void record_receivers(DEEPWAVE_TYPE * const receiver_amplitudes,
                      const DEEPWAVE_TYPE * const current_wavefield,
                      const ptrdiff_t * const receiver_locations,
                      const ptrdiff_t * const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot);
void save_wavefields(DEEPWAVE_TYPE * const current_saved_wavefield,
                     DEEPWAVE_TYPE * const current_saved_wavefield_t,
                     DEEPWAVE_TYPE * const current_saved_wavefield_tt,
                     const DEEPWAVE_TYPE * const next_wavefield,
                     const DEEPWAVE_TYPE * const current_wavefield,
                     const DEEPWAVE_TYPE * const previous_wavefield,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt,
                     const enum wavefield_save_strategy save_strategy);
void imaging_condition(DEEPWAVE_TYPE * const model_grad,
                       const DEEPWAVE_TYPE * const current_wavefield,
                       const DEEPWAVE_TYPE * const next_saved_wavefield,
                       const DEEPWAVE_TYPE * const current_saved_wavefield,
                       const DEEPWAVE_TYPE * const previous_saved_wavefield,
                       const DEEPWAVE_TYPE * const sigma,
                       const ptrdiff_t * const shape,
                       const ptrdiff_t * const pml_width,
                       const ptrdiff_t num_shots);
void model_grad_scaling(DEEPWAVE_TYPE * const model_grad,
                        const DEEPWAVE_TYPE * const scaling,
                        const ptrdiff_t * const shape,
                        const ptrdiff_t * const pml_width);
#endif /* DEEPWAVE_SCALAR_SCALAR_DEVICE_H */
