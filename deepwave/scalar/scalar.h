enum wavefield_save_strategy
{
        /* NONE: Only forward modeling, do not save wavefields.
         * INPLACE: Save by advancing through an array. E.g., if A, B, C, ...
         *      are pointers into an array, separated by at least the size of
         *      the wavefield, then during the first timestep, the wave
         *      propagator will write the propagated wavefield into C, and use
         *      B and A as the current and previous wavefields (read only).
         *      Then during the second timestep, it will write into D, and use
         *      C and B as the current and previous wavefields. This strategy
         *      avoids copying the wavefield, but is memory inefficient
         *      if step_ratio != 1, that is, if the propagator needs to
         *      do extra "inner" time steps between the wavefields that need
         *      to be stored.
         * COPY: Copy the wavefields to be saved from the array used during
         *      propagation into a separate storage array.
         */
	STRATEGY_NONE,
	STRATEGY_INPLACE, 
	STRATEGY_COPY
};

/* Forward modeling
 * wavefield: At least two time steps of the wavefield for propagation.
 *      2 * num_shots * numel_per_shot if save strategy is NONE or COPY,
 *      (num_steps + 2) * num_shots * numel_per_shot for INPLACE,
 *      where numel_per_shot is the number of elements in the padded model.
 * aux_wavefield: PML auxiliary wavefields.
 *      2 * AUX_SIZE * num_shots * numel_per_shot, where AUX_SIZE = 1 for 1D,
 *      2 for 2D, and 4 for 3D.
 * receiver_amplitudes: Output receiver amplitudes.
 *      num_steps * num_shots * num_receivers_per_shot
 * saved_wavefields: Output saved wavefields for backpropagation.
 *      Unused if save strategy is NONE or INPLACE,
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
void forward(
		float *restrict const wavefield,
		float *restrict const aux_wavefield,
		float *restrict const receiver_amplitudes,
		float *restrict const saved_wavefields,
		const float *restrict const sigma,
		const float *restrict const model, 
		const float *restrict const fd1,
		const float *restrict const fd2,
		const float *restrict const source_amplitudes,
		const ptrdiff_t *restrict const source_locations,
		const ptrdiff_t *restrict const receiver_locations,
		const ptrdiff_t *restrict const shape,
		const ptrdiff_t *restrict const pml_width,
		const ptrdiff_t num_steps,
		const ptrdiff_t step_ratio,
		const ptrdiff_t num_shots,
		const ptrdiff_t num_sources_per_shot,
		const ptrdiff_t num_receivers_per_shot,
		const float dt,
		const enum wavefield_save_strategy save_strategy);

/* Backpropagation
 * wavefield: Two time steps of the wavefield for propagation.
 *      2 * num_shots * numel_per_shot,
 *      where numel_per_shot is the number of elements in the padded model.
 * aux_wavefield: PML auxiliary wavefields.
 *      2 * AUX_SIZE * num_shots * numel_per_shot, where AUX_SIZE = 1 for 1D,
 *      2 for 2D, and 4 for 3D.
 * image: The output image/model gradient.
 *      The same size as the unpadded model.
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
 * source_amplitudes: num_steps * num_shots * num_sources_per_shot
 * source_locations: Locations in units of cells, from origin of model.
 *      num_shots * num_sources_per_shot * num_dims
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
 * dt: The time interval between propagator time steps (NOT the time interval
 *      between source amplitude samples - that would be dt * step_ratio).
 */
void backward(
		float *restrict const wavefield,
		float *restrict const aux_wavefield,
		float *restrict const image,
		const float *restrict const adjoint_wavefield,
		const float *restrict const scaling,
		const float *restrict const sigma,
		const float *restrict const model, 
		const float *restrict const fd1,
		const float *restrict const fd2,
		const float *restrict const source_amplitudes,
		const ptrdiff_t *restrict const source_locations,
		const ptrdiff_t *restrict const shape,
		const ptrdiff_t *restrict const pml_width,
		const ptrdiff_t num_steps,
		const ptrdiff_t step_ratio,
		const ptrdiff_t num_shots,
		const ptrdiff_t num_sources_per_shot,
		const float dt);

