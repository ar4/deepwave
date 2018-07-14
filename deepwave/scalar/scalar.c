#include <stddef.h>
#include <string.h>
#include "scalar.h"

/* First order spatial finite differences */
#define z_deriv(arr) \
        fd1[0] * (arr[si + size_xy] - arr[si - size_xy]) + \
        fd1[1] * (arr[si + 2 * size_xy] - arr[si - 2 * size_xy])

#define y_deriv(arr) \
        fd1[2] * (arr[si + size_x] - arr[si - size_x]) + \
        fd1[3] * (arr[si + 2 * size_x] - arr[si - 2 * size_x])

#define x_deriv(arr) \
        fd1[4] * (arr[si + 1] - arr[si - 1]) + \
        fd1[5] * (arr[si + 2] - arr[si - 2])

/* Second order spatial finite difference (Laplacian) */
#define laplacian1d(arr) \
        fd2[0] * arr[si] + \
        fd2[1] * (arr[si + size_xy] + arr[si - size_xy]) + \
        fd2[2] * (arr[si + 2 * size_xy] + arr[si - 2 * size_xy])
#define laplacian2d(arr) \
        laplacian1d(arr) + \
        fd2[3] * \
        (arr[si + size_x] + arr[si - size_x]) + \
        fd2[4] * \
        (arr[si + 2 * size_x] + arr[si - 2 * size_x])
#define laplacian3d(arr) \
        laplacian2d(arr) + \
        fd2[5] * \
        (arr[si + 1] + arr[si - 1]) + \
        fd2[6] * \
        (arr[si + 2] + arr[si - 2])

/* Dimension-specific definitions
 *
 * LOOP: Loops over shots and spatial dimensions
 * ENDLOOP: Close the loops in LOOP
 * location_index: Convert array of coordinates into index into flat array
 * ZPAD/YPAD/XPAD: Number of cells of padding at the beginning and end of
 *      z, y, and x dimensions to make finite difference calculation cleaner
 * AUX_SIZE: Number of auxiliary wavefields
 * */
#if DIM == 1

#define laplacian(arr) laplacian1d(arr)

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        for (ptrdiff_t shot = 0; shot < num_shots; shot++) \
{ \
        for (ptrdiff_t z = start_z; z < end_z; z++) \
        { \
                ptrdiff_t y = 0; \
                ptrdiff_t x = 0;

#define ENDLOOP \
        } \
}

static inline ptrdiff_t location_index(
                const ptrdiff_t *restrict const arr,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index];

        return z;
}

#define ZPAD 2
#define YPAD 0
#define XPAD 0

#define AUX_SIZE 1

#elif DIM == 2

#define laplacian(arr) laplacian2d(arr)

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        for (ptrdiff_t shot = 0; shot < num_shots; shot++) \
{ \
        for (ptrdiff_t z = start_z; z < end_z; z++) \
        { \
                for (ptrdiff_t y = start_y; y < end_y; y++) \
                { \
                        ptrdiff_t x = 0;


#define ENDLOOP \
                } \
        } \
}

static inline ptrdiff_t location_index(
                const ptrdiff_t *restrict const arr,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index * 2];
        const ptrdiff_t y = arr[index * 2 + 1];

        return z * shape[1] + y;
}

#define ZPAD 2
#define YPAD 2
#define XPAD 0

#define AUX_SIZE 2

#elif DIM == 3

#define laplacian(arr) laplacian3d(arr)

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        for (ptrdiff_t shot = 0; shot < num_shots; shot++) \
{ \
        for (ptrdiff_t z = start_z; z < end_z; z++) \
        { \
                for (ptrdiff_t y = start_y; y < end_y; y++) \
                { \
                        for (ptrdiff_t x = start_x; x < end_x; x++) \
                        { \

#define ENDLOOP \
                        } \
                } \
        } \
}

static inline ptrdiff_t location_index(
                const ptrdiff_t *restrict const arr,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index * 3];
        const ptrdiff_t y = arr[index * 3 + 1];
        const ptrdiff_t x = arr[index * 3 + 2];

        return z * shape[1] * shape[2] + y * shape[2] + x;
}

#define ZPAD 2
#define YPAD 2
#define XPAD 2

#define AUX_SIZE 4

#else
#error "Must specify the dimension, e.g. -D DIM=1"
#endif /* DIM */

/* Static function declarations */
static void advance_step(
                float **restrict next_wavefield,
                float **restrict next_aux_wavefield,
                const float **restrict current_wavefield,
                const float **restrict previous_wavefield,
                const float **restrict current_aux_wavefield,
                const float *restrict const sigma,
                const float *restrict const model,
                const float *restrict const fd1,
                const float *restrict const fd2,
                const float *restrict const source_amplitudes,
                const ptrdiff_t *restrict const source_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const float dt,
                const enum wavefield_save_strategy save_strategy);
static void propagate(
                float *restrict const wfn, /* next wavefield */
                float *restrict const auxn, /* next aux_wavefield */
                const float *restrict const wfc, /* current wavefield */
                const float *restrict const wfp, /* previous wavefield */
                const float *restrict const auxc, /* current aux_wavefield */
                const float *restrict const sigma,
                const float *restrict const model,
                const float *restrict const fd1, /* 1st difference coeffs */
                const float *restrict const fd2, /* 2nd difference coeffs */
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
                const ptrdiff_t num_shots,
                const float dt);
static void add_sources(
                float *restrict const next_wavefield,
                const float *restrict const model,
                const float *restrict const source_amplitudes,
                const ptrdiff_t *restrict const source_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot);
static void set_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const float *restrict const wavefield,
                const float *restrict const aux_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy);
static void update_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const enum wavefield_save_strategy save_strategy);
static float * set_step_pointer(
                const float *restrict const origin,
                const ptrdiff_t step,
                const ptrdiff_t num_shots,
                const ptrdiff_t numel_per_shot);
static void record_receivers(
                float *restrict const receiver_amplitudes,
                const float *restrict const current_wavefield,
                const ptrdiff_t *restrict const receiver_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot);
static void save_wavefields(
                float *restrict const saved_wavefields,
                const float *restrict const current_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t step,
                const enum wavefield_save_strategy save_strategy);
static void imaging_condition(
                float *restrict const image,
                const float *restrict const current_wavefield,
                const float *restrict const next_adjoint_wavefield,
                const float *restrict const current_adjoint_wavefield,
                const float *restrict const previous_adjoint_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
                const ptrdiff_t num_shots,
                const float dt);
static void image_scaling(
                float *restrict const image,
                const float *restrict const scaling,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width);


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
                const enum wavefield_save_strategy save_strategy)
{

        float *next_wavefield;
        float *current_wavefield;
        float *previous_wavefield;
        float *next_aux_wavefield;
        float *current_aux_wavefield;

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

                float *current_source_amplitudes;
                float *next_adjoint_wavefield;
                float *current_adjoint_wavefield;
                float *previous_adjoint_wavefield;

                current_source_amplitudes = set_step_pointer(
                                source_amplitudes,
                                step,
                                num_shots,
                                num_sources_per_shot);

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
                                current_source_amplitudes,
                                source_locations,
                                shape,
                                pml_width,
                                step_ratio,
                                num_shots,
                                num_sources_per_shot,
                                dt,
                                save_strategy);

                if ((step < num_steps) && (step > 1))
                {
                        imaging_condition(
                                        image,
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

        image_scaling(
                        image,
                        scaling,
                        shape,
                        pml_width);

}


static void advance_step(
                float **restrict next_wavefield,
                float **restrict next_aux_wavefield,
                const float **restrict current_wavefield,
                const float **restrict previous_wavefield,
                const float **restrict current_aux_wavefield,
                const float *restrict const sigma,
                const float *restrict const model,
                const float *restrict const fd1,
                const float *restrict const fd2,
                const float *restrict const source_amplitudes,
                const ptrdiff_t *restrict const source_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
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


static void propagate(
                float *restrict const wfn, /* next wavefield */
                float *restrict const auxn, /* next auxilliary */
                const float *restrict const wfc, /* current wavefield */
                const float *restrict const wfp, /* previous wavefield */
                const float *restrict const auxc, /* current auxiliary */
                const float *restrict const sigma,
                const float *restrict const model,
                const float *restrict const fd1, /* 1st difference coeffs */
                const float *restrict const fd2, /* 2nd difference coeffs */
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
                const ptrdiff_t num_shots,
                const float dt)
{

        const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
        const ptrdiff_t size_x = shape[2];
        const ptrdiff_t size_xy = shape[1] * shape[2];
        float *restrict const phizn = auxn;
        const float *restrict const phizc = auxc;
        const float *restrict const sigmaz = sigma;

#if DIM >= 2

        float *restrict const phiyn = auxn + num_shots * numel_shot;
        const float *restrict const phiyc = auxc + num_shots * numel_shot;
        const float *restrict const sigmay = sigma + shape[0];

#endif /* DIM >= 2 */

#if DIM == 3

        float *restrict const phixn = auxn + 2 * num_shots * numel_shot;
        float *restrict const psin = auxn + 3 * num_shots * numel_shot;
        const float *restrict const phixc = auxc + 2 * num_shots * numel_shot;
        const float *restrict const psic = auxc + 3 * num_shots * numel_shot;
        const float *restrict const sigmax = sigma + shape[0] + shape[1];

#endif /* DIM  == 3 */

#pragma omp parallel for default(none) collapse(2)

        LOOP(num_shots,
                        ZPAD, shape[0] - ZPAD,
                        YPAD, shape[1] - YPAD,
                        XPAD, shape[2] - XPAD);

        ptrdiff_t i = z * size_xy + y * size_x + x;
        ptrdiff_t si = shot * numel_shot + i;

        /* Spatial finite differences */
        float lap = laplacian(wfc);
        float wfc_z = z_deriv(wfc);
        float phizc_z = z_deriv(phizc);

#if DIM == 1

        /* Update wavefield */
        wfn[si] = 1 / (1 + dt * sigmaz[z] / 2)
                * (model[i] * (lap + phizc_z)
                                + dt * sigmaz[z] * wfp[si] / 2
                                + (2 * wfc[si] - wfp[si]));

        /* Update phi */
        phizn[si] = phizc[si] - dt * sigmaz[z] * (wfc_z + phizc[si]);

#elif DIM == 2

        float wfc_y = y_deriv(wfc);
        float phiyc_y = y_deriv(phiyc);

        /* Update wavefield */
        wfn[si] = 1 / (1 + dt * (sigmaz[z] + sigmay[y]) / 2) *
                (model[i] * (lap + phizc_z + phiyc_y) +
                 dt * (sigmaz[z] + sigmay[y]) * wfp[si] / 2 +
                 (2 * wfc[si] - wfp[si]) -
                 dt * dt * sigmaz[z] * sigmay[y] * wfc[si]);

        /* Update phi */
        phizn[si] = phizc[si] - dt * (sigmaz[z] * phizc[si] +
                        (sigmaz[z] - sigmay[y]) * wfc_z);
        phiyn[si] = phiyc[si] - dt * (sigmay[y] * phiyc[si] +
                        (sigmay[y] - sigmaz[z]) * wfc_y);

#elif DIM == 3

        float wfc_y = y_deriv(wfc);
        float wfc_x = x_deriv(wfc);
        float phiyc_y = y_deriv(phiyc);
        float phixc_x = x_deriv(phixc);
        float psic_z = z_deriv(psic);
        float psic_y = y_deriv(psic);
        float psic_x = x_deriv(psic);

        /* Update wavefield */
        wfn[si] = 1 / (1 + dt * (sigmaz[z] + sigmay[y] + sigmax[x]) / 2) *
                (model[i] * lap + dt * dt * (phizc_z + phiyc_y + phixc_x -
                                             sigmaz[z] * sigmay[y] * sigmax[x] *
                                             psic[si]) +
                 dt * (sigmaz[z] + sigmay[y] + sigmax[x]) * wfp[si] / 2 +
                 (2 * wfc[si] - wfp[si]) -
                 dt * dt * wfc[si] *
                 (sigmax[x] * sigmay[y] +
                  sigmay[y] * sigmaz[z] +
                  sigmax[x] * sigmaz[z]));

        /* Update phi */
        phizn[si] = phizc[si] - dt * sigmaz[z] * phizc[si] +
                model[i] / dt * (sigmay[y] + sigmax[x]) * wfc_z +
                dt * sigmax[x] * sigmay[y] * psic_z;
        phiyn[si] = phiyc[si] - dt * sigmay[y] * phiyc[si] +
                model[i] / dt * (sigmaz[z] + sigmax[x]) * wfc_y +
                dt * sigmax[x] * sigmaz[z] * psic_y;
        phixn[si] = phixc[si] - dt * sigmax[x] * phixc[si] +
                model[i] / dt * (sigmaz[z] + sigmay[y]) * wfc_x +
                dt * sigmaz[z] * sigmay[y] * psic_x;

        /* Update psi */
        psin[si] = psic[si] + dt * wfc[si];

#endif /* DIM */

        ENDLOOP;

}


static void add_sources(
                float *restrict const next_wavefield,
                const float *restrict const model,
                const float *restrict const source_amplitudes,
                const ptrdiff_t *restrict const source_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot)
{

        for (ptrdiff_t shot = 0; shot < num_shots; shot++)
        {
                for (ptrdiff_t source = 0; source < num_sources_per_shot;
                                source++)
                {

                        ptrdiff_t s = shot * num_sources_per_shot + source;
                        ptrdiff_t i = location_index(source_locations, shape,
                                        s);
                        ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] +
                                i;
                        next_wavefield[si] += source_amplitudes[s] * model[i];

                }
        }

}


static void set_pointers(
                const float **next_wavefield,
                const float **current_wavefield,
                const float **previous_wavefield,
                const float **next_aux_wavefield,
                const float **current_aux_wavefield,
                const float *restrict const wavefield,
                const float *restrict const aux_wavefield,
                const ptrdiff_t *restrict const shape,
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
                const ptrdiff_t *restrict const shape,
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


static float * set_step_pointer(
                const float *restrict const origin,
                const ptrdiff_t step,
                const ptrdiff_t num_shots,
                const ptrdiff_t numel_per_shot)
{

        return (float *) origin + step * num_shots * numel_per_shot;

}


static void record_receivers(
                float *restrict const receiver_amplitudes,
                const float *restrict const current_wavefield,
                const ptrdiff_t *restrict const receiver_locations,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot)
{

        for (ptrdiff_t shot = 0; shot < num_shots; shot++)
        {
                for (ptrdiff_t receiver = 0; receiver < num_receivers_per_shot;
                                receiver++)
                {

                        ptrdiff_t r = shot * num_receivers_per_shot + receiver;
                        ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] +
                                location_index(receiver_locations, shape, r);
                        receiver_amplitudes[r] = current_wavefield[si];

                }
        }

}


static void save_wavefields(
                float *restrict const saved_wavefields,
                const float *restrict const current_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t step,
                const enum wavefield_save_strategy save_strategy)
{

        if (save_strategy == STRATEGY_COPY)
        {
                float *restrict current_saved_wavefield = set_step_pointer(
                                saved_wavefields,
                                step,
                                num_shots,
                                shape[0] * shape[1] * shape[2]);
                memcpy(current_saved_wavefield, current_wavefield,
                                num_shots * shape[0] * shape[1] * shape[2] *
                                sizeof(float));
        }

}


static void imaging_condition(
                float *restrict const image,
                const float *restrict const current_wavefield,
                const float *restrict const next_adjoint_wavefield,
                const float *restrict const current_adjoint_wavefield,
                const float *restrict const previous_adjoint_wavefield,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width,
                const ptrdiff_t num_shots,
                const float dt)
{

        const ptrdiff_t image_size_z = shape[0] - 2 * ZPAD -
                pml_width[0] - pml_width[1];
        const ptrdiff_t image_size_y = shape[1] - 2 * YPAD -
                pml_width[2] - pml_width[3];
        const ptrdiff_t image_size_x = shape[2] - 2 * XPAD -
                pml_width[4] - pml_width[5];
        const ptrdiff_t image_size_xy = image_size_y * image_size_x;

        LOOP(num_shots, 0, image_size_z, 0, image_size_y, 0, image_size_x);

        ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] +
                (z + ZPAD + pml_width[0]) * shape[1] * shape[2] +
                (y + YPAD + pml_width[2]) * shape[2] +
                x + XPAD + pml_width[4];

        float adjoint_wavefield_tt = (next_adjoint_wavefield[si] -
                        2 * current_adjoint_wavefield[si] +
                        previous_adjoint_wavefield[si]) / (dt * dt);

        image[z * image_size_xy + y * image_size_x + x] +=
                current_wavefield[si] *	adjoint_wavefield_tt;

        ENDLOOP;

}


static void image_scaling(
                float *restrict const image,
                const float *restrict const scaling,
                const ptrdiff_t *restrict const shape,
                const ptrdiff_t *restrict const pml_width)
{

        const ptrdiff_t image_size_z = shape[0] - 2 * ZPAD -
                pml_width[0] - pml_width[1];
        const ptrdiff_t image_size_y = shape[1] - 2 * YPAD -
                pml_width[2] - pml_width[3];
        const ptrdiff_t image_size_x = shape[2] - 2 * XPAD -
                pml_width[4] - pml_width[5];
        ptrdiff_t image_size = image_size_z * image_size_y * image_size_x;

        for (ptrdiff_t i = 0; i < image_size; i++)
        {
                image[i] *= scaling[i];
        }

}
