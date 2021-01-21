#ifndef DEEPWAVE_SCALAR_SCALAR_H
#define DEEPWAVE_SCALAR_SCALAR_H
#include <stddef.h>
enum wavefield_save_strategy {
  /* NONE: Only forward modeling, do not save wavefields.
   * COPY: Copy the wavefields to be saved from the array used during
   *      propagation into a separate storage array.
   */
  STRATEGY_NONE,
  STRATEGY_COPY
};

/* Forward modeling
 * wavefield: At least two time steps of the wavefield for propagation.
 *      3 * num_shots * numel_per_shot if save strategy is NONE or COPY,
 *      where numel_per_shot is the number of elements in the padded model.
 * aux_wavefield: PML auxiliary wavefields.
 *      2 * AUX_SIZE * num_shots * numel_per_shot, where AUX_SIZE = 1 for 1D,
 *      2 for 2D, and 4 for 3D.
 * receiver_amplitudes: Output receiver amplitudes.
 *      num_steps * num_shots * num_receivers_per_shot
 * saved_wavefields: Output saved wavefields for backpropagation.
 *      Unused if save strategy is NONE,
 *      num_steps * num_shots * numel_per_shot for COPY.
 * sigma: PML sigma. Number of elements is the sum of the padded length of
 *      each dimension, e.g. shape[0] + shape[1] + shape[2] in 3D.
 * model: Padded wave speed model. Length in each dimension is the unpadded
 *      length + the finite difference accuracy order + 2 * pml_width. Padded
 *      regions should replicate the edge of the unpadded model.
 * fd1: First derivative finite difference coefficients (one side).
 *      Half finite difference accuracy order * num_dims.
 * fd2: Second derivative finite difference coefficients (positive side).
 *      Half finite difference accuracy order * num_dims + 1. The "+ 1"
 *      contains the coefficient for the central element.
 * source_amplitudes: num_steps * num_shots * num_sources_per_shot
 * source_locations: Locations in units of cells, from origin of model.
 *      num_shots * num_sources_per_shot * num_dims
 * receiver_locations: Locations in units of cells, from origin of model.
 *      num_shots * num_receivers_per_shot * num_dims
 * shape: Padded shape of the model. 3 elements, with minimum value 1, e.g.
 *      100, 1, 1 for 1D.
 * pml_width: Number of PML cells. 6 elements, beginning and end of each
 *      dimension, with minimum value 0, e.g. 10, 10, 0, 0, 0, 0 for 1D.
 * num_steps: Number of time samples in source amplitudes (also the same as
 *      the number of time samples in output receiver amplitudes, and the
 *      number of saved wavefields).
 * step_ratio: The number of inner time steps of the propagator between each
 *      each source amplitude time sample.
 * num_shots
 * num_sources_per_shot
 * num_receivers_per_shot
 * dt: The time interval between propagator time steps (NOT the time interval
 *      between source amplitude samples - that would be dt * step_ratio).
 * save_strategy: Enum specifying how to store wavefield for backpropagation.
 */
void forward(TYPE * const wavefield,
             TYPE * const aux_wavefield,
             TYPE * const receiver_amplitudes,
             TYPE * const saved_wavefields,
             const TYPE * const sigma,
             const TYPE * const model,
             const TYPE * const fd1,
             const TYPE * const fd2,
             const TYPE * const source_amplitudes,
             const ptrdiff_t * const source_locations,
             const ptrdiff_t * const receiver_locations,
             const ptrdiff_t * const shape,
             const ptrdiff_t * const pml_width,
             const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
             const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
             const ptrdiff_t num_receivers_per_shot, const TYPE dt,
             const enum wavefield_save_strategy save_strategy);

/* Born forward modeling
 * Arguments are the same as regular forward modeling, except for the following
 * scattered_wavefield: Like wavefield, but for scattered waves
 * scattered_aux_wavefield: Like aux_wavefield, but for scattered waves
 * scatter: Like model, but contains the scattering amplitude
 */
void forward_born(TYPE * const wavefield,
             TYPE * const aux_wavefield,
    	     TYPE * const scattered_wavefield,
	     TYPE * const scattered_aux_wavefield,
             TYPE * const receiver_amplitudes,
             TYPE * const saved_wavefields,
             const TYPE * const sigma,
             const TYPE * const model,
	     const TYPE * const scatter,
             const TYPE * const fd1,
             const TYPE * const fd2,
             const TYPE * const source_amplitudes,
             const ptrdiff_t * const source_locations,
             const ptrdiff_t * const receiver_locations,
             const ptrdiff_t * const shape,
             const ptrdiff_t * const pml_width,
             const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
             const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
             const ptrdiff_t num_receivers_per_shot, const TYPE dt,
             const enum wavefield_save_strategy save_strategy);

/* Backpropagation
 * wavefield: Two time steps of the wavefield for propagation.
 *      3 * num_shots * numel_per_shot,
 *      where numel_per_shot is the number of elements in the padded model.
 * aux_wavefield: PML auxiliary wavefields.
 *      2 * AUX_SIZE * num_shots * numel_per_shot, where AUX_SIZE = 1 for 1D,
 *      2 for 2D, and 4 for 3D.
 * model_grad: The output image/model gradient.
 *      The same size as the unpadded model. NULL if not doing model inversion.
 * source_grad_amplitudes: The output source amplitudes gradient
 *      The same size as source_amplitudes during forward. NULL if not doing
 *      source inversion.
 * adjoint_wavefields: Saved wavefields from forward modeling.
 *      num_steps * num_shots * numel_per_shot.
 * scaling: The factor that multiplies the zero-lag cross-correlation imaging
 *      condition to turn it into the model gradient. Typically 2 / c^3.
 *      Same shape as image.
 * sigma: PML sigma. Number of elements is the sum of the padded length of
 *      each dimension, e.g. shape[0] + shape[1] + shape[2] in 3D.
 * model: Padded wave speed model. Length in each dimension is the unpadded
 *      length + the finite difference accuracy order + 2 * pml_width. Padded
 *      regions should replicate the edge of the unpadded model.
 * fd1: First derivative finite difference coefficients (one side).
 *      Half finite difference accuracy order * num_dims.
 * fd2: Second derivative finite difference coefficients (positive side).
 *      Half finite difference accuracy order * num_dims + 1. The "+ 1"
 *      contains the coefficient for the central element.
 * receiver_grad_amplitudes: The gradient of the loss wrt the receiver
 *      amplitudes. num_steps * num_shots * num_receivers_per_shot
 * source_locations: Locations in units of cells, from origin of model.
 *      num_shots * num_sources_per_shot * num_dims
 * receiver_locations: Locations in units of cells, from origin of model.
 *      num_shots * num_receivers_per_shot * num_dims
 * shape: Padded shape of the model. 3 elements, with minimum value 1, e.g.
 *      100, 1, 1 for 1D.
 * pml_width: Number of PML cells. 6 elements, beginning and end of each
 *      dimension, with minimum value 0, e.g. 10, 10, 0, 0, 0, 0 for 1D.
 * num_steps: Number of time samples in source amplitudes (also the same as
 *      the number of time samples in output receiver amplitudes, and the
 *      number of saved wavefields).
 * step_ratio: The number of inner time steps of the propagator between each
 *      each source amplitude time sample.
 * num_shots
 * num_sources_per_shot
 * num_receivers_per_shot
 * dt: The time interval between propagator time steps (NOT the time interval
 *      between source amplitude samples - that would be dt * step_ratio).
 */
void backward(TYPE * const wavefield,
              TYPE * const aux_wavefield,
              TYPE * const model_grad,
              TYPE * const source_grad_amplitudes,
              const TYPE * const adjoint_wavefield,
              const TYPE * const scaling,
              const TYPE * const sigma,
              const TYPE * const model,
              const TYPE * const fd1,
              const TYPE * const fd2,
              const TYPE * const receiver_grad_amplitudes,
              const ptrdiff_t * const source_locations,
              const ptrdiff_t * const receiver_locations,
              const ptrdiff_t * const shape,
              const ptrdiff_t * const pml_width,
              const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
              const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
              const ptrdiff_t num_receivers_per_shot, const TYPE dt);

TYPE *set_step_pointer(const TYPE * const origin,
                       const ptrdiff_t step, const ptrdiff_t num_shots,
                       const ptrdiff_t numel_per_shot);

/* Dimension-specific definitions
 *
 * ZPAD/YPAD/XPAD: Number of cells of padding at the beginning and end of
 *      z, y, and x dimensions to make finite difference calculation cleaner
 * AUX_SIZE: Number of auxiliary wavefields
 * */
#if DIM == 1

#define ZPAD 2
#define YPAD 0
#define XPAD 0

#define AUX_SIZE 1

#elif DIM == 2

#define ZPAD 2
#define YPAD 2
#define XPAD 0

#define AUX_SIZE 2

#elif DIM == 3

#define ZPAD 2
#define YPAD 2
#define XPAD 2

#define AUX_SIZE 4

#else
#error "Must specify the dimension, e.g. -D DIM=1"
#endif /* DIM */

#endif /* DEEPWAVE_SCALAR_SCALAR_H */
