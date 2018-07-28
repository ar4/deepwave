#include <stddef.h>
#include "scalar.h"
#include "scalar_device.h"


/* Static function declarations */
static void advance_step(
                float **__restrict__ next_wavefield,
                float **__restrict__ next_aux_wavefield,
                const float **__restrict__ current_wavefield,
                const float **__restrict__ previous_wavefield,
                const float **__restrict__ current_aux_wavefield,
                const float *__restrict__ const sigma,
                const float *__restrict__ const model,
                const float *__restrict__ const fd1,
                const float *__restrict__ const fd2,
                const float *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const float dt,
                const enum wavefield_save_strategy save_strategy);
static void set_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const float *__restrict__ const wavefield,
                const float *__restrict__ const aux_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy);
static void update_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy);


void forward(
                float *__restrict__ const wavefield,
                float *__restrict__ const aux_wavefield,
                float *__restrict__ const receiver_amplitudes,
                float *__restrict__ const saved_wavefields,
                const float *__restrict__ const sigma,
                const float *__restrict__ const model, 
                const float *__restrict__ const fd1,
                const float *__restrict__ const fd2,
                const float *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_steps,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const ptrdiff_t num_receivers_per_shot,
                const float dt,
                const enum wavefield_save_strategy save_strategy)
{

        float *next_wavefield;
        float *current_wavefield;
        float *previous_wavefield;
        float *next_aux_wavefield;
        float *current_aux_wavefield;

	setup(
			fd1,
			fd2);

        set_pointers(
                        (const float **) &next_wavefield,
                        (const float **) &current_wavefield,
                        (const float **) &previous_wavefield,
                        (const float **) &next_aux_wavefield,
                        (const float **) &current_aux_wavefield,
                        wavefield,
                        aux_wavefield,
                        shape,
                        num_shots,
                        save_strategy);

        for (ptrdiff_t step = 0; step < num_steps; step++)
        {

                float *current_source_amplitudes;
                float *current_receiver_amplitudes;

                current_source_amplitudes = set_step_pointer(
                                source_amplitudes,
                                step,
                                num_shots,
                                num_sources_per_shot);

                current_receiver_amplitudes = set_step_pointer(
                                receiver_amplitudes,
                                step,
                                num_shots,
                                num_receivers_per_shot);

                advance_step(
                                &next_wavefield,
                                &next_aux_wavefield,
                                (const float **) &current_wavefield,
                                (const float **) &previous_wavefield,
                                (const float **) &current_aux_wavefield,
                                sigma,
                                model,
                                fd1,
                                fd2,
                                current_source_amplitudes,
                                source_locations,
                                shape,
                                pml_width,
                                step_ratio,
                                num_shots,
                                num_sources_per_shot,
                                dt,
                                save_strategy);

                record_receivers(
                                current_receiver_amplitudes,
                                current_wavefield,
                                receiver_locations,
                                shape,
                                num_shots,
                                num_receivers_per_shot);

                save_wavefields(
                                saved_wavefields,
                                current_wavefield,
                                shape,
                                num_shots,
                                step,
                                save_strategy);

        }

}


void backward(
                float *__restrict__ const wavefield,
                float *__restrict__ const aux_wavefield,
                float *__restrict__ const model_grad,
                float *__restrict__ const source_grad_amplitudes,
                const float *__restrict__ const adjoint_wavefield,
                const float *__restrict__ const scaling,
                const float *__restrict__ const sigma,
                const float *__restrict__ const model, 
                const float *__restrict__ const fd1,
                const float *__restrict__ const fd2,
                const float *__restrict__ const receiver_grad_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_steps,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const ptrdiff_t num_receivers_per_shot,
                const float dt)
{

        float *next_wavefield;
        float *current_wavefield;
        float *previous_wavefield;
        float *next_aux_wavefield;
        float *current_aux_wavefield;
        const enum wavefield_save_strategy save_strategy = STRATEGY_NONE;

        set_pointers(
                        (const float **) &next_wavefield,
                        (const float **) &current_wavefield,
                        (const float **) &previous_wavefield,
                        (const float **) &next_aux_wavefield,
                        (const float **) &current_aux_wavefield,
                        wavefield,
                        aux_wavefield,
                        shape,
                        num_shots,
                        save_strategy);

        for (ptrdiff_t step = num_steps-1; step >= 0; step--)
        {

                float *current_source_grad_amplitudes;
                float *current_receiver_grad_amplitudes;
                float *next_adjoint_wavefield;
                float *current_adjoint_wavefield;
                float *previous_adjoint_wavefield;

                current_source_grad_amplitudes = set_step_pointer(
                                source_grad_amplitudes,
                                step,
                                num_shots,
                                num_sources_per_shot);

                current_receiver_grad_amplitudes = set_step_pointer(
                                receiver_grad_amplitudes,
                                step,
                                num_shots,
                                num_receivers_per_shot);

                next_adjoint_wavefield = set_step_pointer(
                                adjoint_wavefield,
                                step,
                                num_shots,
                                shape[0] * shape[1] * shape[2]);

                current_adjoint_wavefield = set_step_pointer(
                                adjoint_wavefield,
                                step-1,
                                num_shots,
                                shape[0] * shape[1] * shape[2]);

                previous_adjoint_wavefield = set_step_pointer(
                                adjoint_wavefield,
                                step-2,
                                num_shots,
                                shape[0] * shape[1] * shape[2]);

                advance_step(
                                &next_wavefield,
                                &next_aux_wavefield,
                                (const float **) &current_wavefield,
                                (const float **) &previous_wavefield,
                                (const float **) &current_aux_wavefield,
                                sigma,
                                model,
                                fd1,
                                fd2,
                                current_receiver_grad_amplitudes,
                                receiver_locations,
                                shape,
                                pml_width,
                                step_ratio,
                                num_shots,
                                num_receivers_per_shot,
                                dt,
                                save_strategy);

                record_receivers(
                                current_source_grad_amplitudes,
                                current_wavefield,
                                source_locations,
                                shape,
                                num_shots,
                                num_sources_per_shot);

                if ((step < num_steps) && (step > 1))
                {
                        imaging_condition(
                                        model_grad,
                                        current_wavefield,
                                        next_adjoint_wavefield,
                                        current_adjoint_wavefield,
                                        previous_adjoint_wavefield,
                                        shape,
                                        pml_width,
                                        num_shots,
                                        dt * step_ratio);
                }

        }

        model_grad_scaling(
                        model_grad,
                        scaling,
                        shape,
                        pml_width);

}


static void advance_step(
                float **__restrict__ next_wavefield,
                float **__restrict__ next_aux_wavefield,
                const float **__restrict__ current_wavefield,
                const float **__restrict__ previous_wavefield,
                const float **__restrict__ current_aux_wavefield,
                const float *__restrict__ const sigma,
                const float *__restrict__ const model,
                const float *__restrict__ const fd1,
                const float *__restrict__ const fd2,
                const float *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const float dt,
                const enum wavefield_save_strategy save_strategy)
{

        for (ptrdiff_t inner_step = 0; inner_step < step_ratio; inner_step++)
        {

                propagate(
                                *next_wavefield,
                                *next_aux_wavefield,
                                *current_wavefield,
                                *previous_wavefield,
                                *current_aux_wavefield,
                                sigma,
                                model,
                                fd1,
                                fd2,
                                shape,
                                pml_width,
                                num_shots,
                                dt);

                add_sources(
                                *next_wavefield,
                                model,
                                source_amplitudes,
                                source_locations,
                                shape,
                                num_shots,
                                num_sources_per_shot);

                update_pointers(
                                (const float **) next_wavefield,
                                current_wavefield,
                                previous_wavefield,
                                (const float **) next_aux_wavefield,
                                current_aux_wavefield,
                                shape,
                                num_shots,
                                save_strategy);

        }

}


static void set_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const float *__restrict__ const wavefield,
                const float *__restrict__ const aux_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy)
{

        const ptrdiff_t shot_size = shape[0] * shape[1] * shape[2];

        *previous_wavefield = wavefield;
        *current_wavefield = *previous_wavefield + num_shots * shot_size;

        if (save_strategy == STRATEGY_INPLACE)
        {
                *next_wavefield = *current_wavefield + num_shots * shot_size;
        }
        else
        {
                *next_wavefield = *previous_wavefield;
        }

        *current_aux_wavefield = aux_wavefield;
        *next_aux_wavefield = *current_aux_wavefield +
                AUX_SIZE * num_shots * shot_size;

}


static void update_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy)
{

        if (save_strategy == STRATEGY_INPLACE)
        {
                const ptrdiff_t shot_size = shape[0] * shape[1] * shape[2];
                *next_wavefield += num_shots * shot_size;
                *current_wavefield += num_shots * shot_size;
                *previous_wavefield += num_shots * shot_size;
        }
        else
        {
                /* Before: next_wavefield -> previous_wavefield -> A
                 *         current_wavefield -> B
                 * After: next_wavefield -> previous_wavefield -> B
                 *        current_wavefield -> A */
                *next_wavefield = *current_wavefield;
                *current_wavefield = *previous_wavefield;
                *previous_wavefield = *next_wavefield;
        }

        const float *tmp = *next_aux_wavefield;
        *next_aux_wavefield = *current_aux_wavefield;
        *current_aux_wavefield = tmp;

}


float * set_step_pointer(
                const float *__restrict__ const origin,
                const ptrdiff_t step,
                const ptrdiff_t num_shots,
                const ptrdiff_t numel_per_shot)
{

        if (origin == NULL) return NULL;

        return (float *) origin + step * num_shots * numel_per_shot;

}
