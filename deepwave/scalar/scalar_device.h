#ifndef H_SCALAR_DEVICE
#define H_SCALAR_DEVICE
void setup(
		const TYPE *__restrict__ const fd1,
		const TYPE *__restrict__ const fd2);
void propagate(
                TYPE *__restrict__ const wfn, /* next wavefield */
                TYPE *__restrict__ const auxn, /* next aux_wavefield */
                const TYPE *__restrict__ const wfc, /* current wavefield */
                const TYPE *__restrict__ const wfp, /* previous wavefield */
                const TYPE *__restrict__ const auxc, /* current aux_wavefield */
                const TYPE *__restrict__ const sigma,
                const TYPE *__restrict__ const model,
                const TYPE *__restrict__ const fd1, /* 1st difference coeffs */
                const TYPE *__restrict__ const fd2, /* 2nd difference coeffs */
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_shots,
                const TYPE dt);
void add_sources(
                TYPE *__restrict__ const next_wavefield,
                const TYPE *__restrict__ const model,
                const TYPE *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot);
void record_receivers(
                TYPE *__restrict__ const receiver_amplitudes,
                const TYPE *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot);
void save_wavefields(
                TYPE *__restrict__ const saved_wavefields,
                const TYPE *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t step,
                const enum wavefield_save_strategy save_strategy);
void imaging_condition(
                TYPE *__restrict__ const model_grad,
                const TYPE *__restrict__ const current_wavefield,
                const TYPE *__restrict__ const next_adjoint_wavefield,
                const TYPE *__restrict__ const current_adjoint_wavefield,
                const TYPE *__restrict__ const previous_adjoint_wavefield,
                const TYPE *__restrict__ const sigma,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_shots,
                const TYPE dt);
void model_grad_scaling(
                TYPE *__restrict__ const model_grad,
                const TYPE *__restrict__ const scaling,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width);
#endif /* H_SCALAR_DEVICE */
