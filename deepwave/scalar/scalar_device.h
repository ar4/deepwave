#ifndef H_SCALAR_DEVICE
#define H_SCALAR_DEVICE
void setup(
		const float *__restrict__ const fd1,
		const float *__restrict__ const fd2);
void propagate(
                float *__restrict__ const wfn, /* next wavefield */
                float *__restrict__ const auxn, /* next aux_wavefield */
                const float *__restrict__ const wfc, /* current wavefield */
                const float *__restrict__ const wfp, /* previous wavefield */
                const float *__restrict__ const auxc, /* current aux_wavefield */
                const float *__restrict__ const sigma,
                const float *__restrict__ const model,
                const float *__restrict__ const fd1, /* 1st difference coeffs */
                const float *__restrict__ const fd2, /* 2nd difference coeffs */
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_shots,
                const float dt);
void add_sources(
                float *__restrict__ const next_wavefield,
                const float *__restrict__ const model,
                const float *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot);
void record_receivers(
                float *__restrict__ const receiver_amplitudes,
                const float *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot);
void save_wavefields(
                float *__restrict__ const saved_wavefields,
                const float *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t step,
                const enum wavefield_save_strategy save_strategy);
void imaging_condition(
                float *__restrict__ const model_grad,
                const float *__restrict__ const current_wavefield,
                const float *__restrict__ const next_adjoint_wavefield,
                const float *__restrict__ const current_adjoint_wavefield,
                const float *__restrict__ const previous_adjoint_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_shots,
                const float dt);
void model_grad_scaling(
                float *__restrict__ const model_grad,
                const float *__restrict__ const scaling,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width);
#endif /* H_SCALAR_DEVICE */
