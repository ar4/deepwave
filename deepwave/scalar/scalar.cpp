#include "scalar.h"

#include <stddef.h>

#include "scalar_device.h"

/* Static function declarations */
static void advance_step(
    TYPE * * const next_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const current_aux_wavefield,
    const TYPE * const sigma, const TYPE * const model,
    const TYPE * const fd1, const TYPE * const fd2,
    const TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t step_ratio,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
    const TYPE dt);
static void advance_step_born(
    TYPE * * const next_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const next_scattered_wavefield,
    TYPE * * const next_scattered_aux_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const current_aux_wavefield,
    TYPE * * const current_scattered_wavefield,
    TYPE * * const previous_scattered_wavefield,
    TYPE * * const current_scattered_aux_wavefield,
    const TYPE * const sigma, const TYPE * const model,
    const TYPE * const scatter,
    const TYPE * const fd1, const TYPE * const fd2,
    const TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t step_ratio,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
    const TYPE dt);
static void set_pointers(
    TYPE **const next_wavefield, TYPE **const current_wavefield,
    TYPE **const previous_wavefield, TYPE **const next_aux_wavefield,
    TYPE **const current_aux_wavefield, TYPE * const wavefield,
    TYPE * const aux_wavefield,
    const ptrdiff_t * const shape, const ptrdiff_t num_shots);
static void update_pointers(
    TYPE * * const next_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const current_aux_wavefield);

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
             const enum wavefield_save_strategy save_strategy) {
  TYPE *next_wavefield;
  TYPE *current_wavefield;
  TYPE *previous_wavefield;
  TYPE *next_aux_wavefield;
  TYPE *current_aux_wavefield;

  setup(fd1, fd2);

  set_pointers(&next_wavefield, &current_wavefield, &previous_wavefield,
               &next_aux_wavefield, &current_aux_wavefield, wavefield,
               aux_wavefield, shape, num_shots);

  for (ptrdiff_t step = 0; step < num_steps - 1; step++) {
    const TYPE * const current_source_amplitudes = set_step_pointer(
        source_amplitudes, step * step_ratio, num_shots, num_sources_per_shot);

    /* step + 1 as this step computes the wavefield at step + 1 */
    TYPE * const current_receiver_amplitudes = set_step_pointer(
        receiver_amplitudes, step + 1, num_shots, num_receivers_per_shot);

    TYPE * const current_saved_wavefield = set_step_pointer(
        saved_wavefields, 3 * step, num_shots, shape[0] * shape[1] * shape[2]);

    TYPE * const current_saved_wavefield_t =
        set_step_pointer(saved_wavefields, 3 * step + 1, num_shots,
                         shape[0] * shape[1] * shape[2]);

    TYPE * const current_saved_wavefield_tt =
        set_step_pointer(saved_wavefields, 3 * step + 2, num_shots,
                         shape[0] * shape[1] * shape[2]);

    advance_step(&next_wavefield, &next_aux_wavefield, &current_wavefield,
                 &previous_wavefield, &current_aux_wavefield, sigma, model, fd1,
                 fd2, current_source_amplitudes, source_locations, shape,
                 pml_width, step_ratio, num_shots, num_sources_per_shot, dt);

    record_receivers(current_receiver_amplitudes, current_wavefield,
                     receiver_locations, shape, num_shots,
                     num_receivers_per_shot);

    /* pointers already updated:
     * next_wavefield now in current_wavefield
     * current_wavefield now in previous_wavefield
     * previous_wavefield now in next_wavefield */
    save_wavefields(current_saved_wavefield, current_saved_wavefield_t,
                    current_saved_wavefield_tt, current_wavefield,
                    previous_wavefield, next_wavefield, shape, num_shots, dt,
                    save_strategy);
  }
}

void forward_born(
    TYPE * const wavefield, TYPE * const aux_wavefield,
    TYPE * const scattered_wavefield,
    TYPE * const scattered_aux_wavefield,
    TYPE * const receiver_amplitudes,
    TYPE * const saved_wavefields,
    const TYPE * const sigma, const TYPE * const model,
    const TYPE * const scatter, const TYPE * const fd1,
    const TYPE * const fd2,
    const TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t * const receiver_locations,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t num_steps,
    const ptrdiff_t step_ratio, const ptrdiff_t num_shots,
    const ptrdiff_t num_sources_per_shot,
    const ptrdiff_t num_receivers_per_shot, const TYPE dt,
    const enum wavefield_save_strategy save_strategy) {
  TYPE *next_wavefield;
  TYPE *current_wavefield;
  TYPE *previous_wavefield;
  TYPE *next_aux_wavefield;
  TYPE *current_aux_wavefield;
  TYPE *next_scattered_wavefield;
  TYPE *current_scattered_wavefield;
  TYPE *previous_scattered_wavefield;
  TYPE *next_scattered_aux_wavefield;
  TYPE *current_scattered_aux_wavefield;

  setup(fd1, fd2);

  set_pointers(&next_wavefield, &current_wavefield, &previous_wavefield,
               &next_aux_wavefield, &current_aux_wavefield, wavefield,
               aux_wavefield, shape, num_shots);

  set_pointers(&next_scattered_wavefield, &current_scattered_wavefield,
               &previous_scattered_wavefield, &next_scattered_aux_wavefield,
               &current_scattered_aux_wavefield, scattered_wavefield,
               scattered_aux_wavefield, shape, num_shots);

  for (ptrdiff_t step = 0; step < num_steps - 1; step++) {
    const TYPE * const current_source_amplitudes = set_step_pointer(
        source_amplitudes, step * step_ratio, num_shots, num_sources_per_shot);

    /* step + 1 as this step computes the wavefield at step + 1 */
    TYPE * const current_receiver_amplitudes = set_step_pointer(
        receiver_amplitudes, step + 1, num_shots, num_receivers_per_shot);

    TYPE * const current_saved_wavefield = set_step_pointer(
        saved_wavefields, 3 * step, num_shots, shape[0] * shape[1] * shape[2]);

    TYPE * const current_saved_wavefield_t =
        set_step_pointer(saved_wavefields, 3 * step + 1, num_shots,
                         shape[0] * shape[1] * shape[2]);

    TYPE * const current_saved_wavefield_tt =
        set_step_pointer(saved_wavefields, 3 * step + 2, num_shots,
                         shape[0] * shape[1] * shape[2]);

    advance_step_born(
        &next_wavefield, &next_aux_wavefield, &next_scattered_wavefield,
        &next_scattered_aux_wavefield, &current_wavefield, &previous_wavefield,
        &current_aux_wavefield, &current_scattered_wavefield,
        &previous_scattered_wavefield, &current_scattered_aux_wavefield, sigma,
        model, scatter, fd1, fd2, current_source_amplitudes, source_locations, shape,
        pml_width, step_ratio, num_shots, num_sources_per_shot, dt);

    record_receivers(current_receiver_amplitudes, current_scattered_wavefield,
                     receiver_locations, shape, num_shots,
                     num_receivers_per_shot);

    /* pointers already updated:
     * next_wavefield now in current_wavefield
     * current_wavefield now in previous_wavefield
     * previous_wavefield now in next_wavefield */
    save_wavefields(current_saved_wavefield, current_saved_wavefield_t,
                    current_saved_wavefield_tt, current_wavefield,
                    previous_wavefield, next_wavefield, shape, num_shots, dt,
                    save_strategy);
  }
}

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
              const ptrdiff_t num_receivers_per_shot, const TYPE dt) {
  TYPE *next_wavefield;
  TYPE *current_wavefield;
  TYPE *previous_wavefield;
  TYPE *next_aux_wavefield;
  TYPE *current_aux_wavefield;

  set_pointers(&next_wavefield, &current_wavefield, &previous_wavefield,
               &next_aux_wavefield, &current_aux_wavefield, wavefield,
               aux_wavefield, shape, num_shots);

  for (ptrdiff_t step = num_steps - 1; step > 0; step--) {
    TYPE * const current_source_grad_amplitudes = set_step_pointer(
        source_grad_amplitudes, step - 1, num_shots, num_sources_per_shot);

    const TYPE * const current_receiver_grad_amplitudes =
        set_step_pointer(receiver_grad_amplitudes, step * step_ratio, num_shots,
                         num_receivers_per_shot);

    const TYPE * const current_adjoint_wavefield =
        set_step_pointer(adjoint_wavefield, 3 * (step - 1), num_shots,
                         shape[0] * shape[1] * shape[2]);

    const TYPE * const current_adjoint_wavefield_t =
        set_step_pointer(adjoint_wavefield, 3 * (step - 1) + 1, num_shots,
                         shape[0] * shape[1] * shape[2]);

    const TYPE * const current_adjoint_wavefield_tt =
        set_step_pointer(adjoint_wavefield, 3 * (step - 1) + 2, num_shots,
                         shape[0] * shape[1] * shape[2]);

    advance_step(&next_wavefield, &next_aux_wavefield, &current_wavefield,
                 &previous_wavefield, &current_aux_wavefield, sigma, model, fd1,
                 fd2, current_receiver_grad_amplitudes, receiver_locations,
                 shape, pml_width, step_ratio, num_shots,
                 num_receivers_per_shot, dt);

    record_receivers(current_source_grad_amplitudes, current_wavefield,
                     source_locations, shape, num_shots, num_sources_per_shot);

    if ((step < num_steps) && (step > 1)) {
      imaging_condition(model_grad, current_wavefield,
                        current_adjoint_wavefield, current_adjoint_wavefield_t,
                        current_adjoint_wavefield_tt, sigma, shape, pml_width,
                        num_shots);
    }
  }

  model_grad_scaling(model_grad, scaling, shape, pml_width);
}

static void advance_step(
    TYPE * * const next_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const current_aux_wavefield,
    const TYPE * const sigma, const TYPE * const model,
    const TYPE * const fd1, const TYPE * const fd2,
    const TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t step_ratio,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
    const TYPE dt) {
  for (ptrdiff_t inner_step = 0; inner_step < step_ratio; inner_step++) {
    propagate(*next_wavefield, *next_aux_wavefield, *current_wavefield,
              *previous_wavefield, *current_aux_wavefield, sigma, model, fd1,
              fd2, shape, pml_width, num_shots, dt);

    const TYPE * const current_source_amplitudes = set_step_pointer(
        source_amplitudes, inner_step, num_shots, num_sources_per_shot);

    add_sources(*next_wavefield, model, current_source_amplitudes,
                source_locations, shape, num_shots, num_sources_per_shot);

    update_pointers(next_wavefield, current_wavefield, previous_wavefield,
                    next_aux_wavefield, current_aux_wavefield);
  }
}

static void advance_step_born(
    TYPE * * const next_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const next_scattered_wavefield,
    TYPE * * const next_scattered_aux_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const current_aux_wavefield,
    TYPE * * const current_scattered_wavefield,
    TYPE * * const previous_scattered_wavefield,
    TYPE * * const current_scattered_aux_wavefield,
    const TYPE * const sigma, const TYPE * const model,
    const TYPE * const scatter,
    const TYPE * const fd1, const TYPE * const fd2,
    const TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t step_ratio,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
    const TYPE dt) {
  for (ptrdiff_t inner_step = 0; inner_step < step_ratio; inner_step++) {
    propagate(*next_wavefield, *next_aux_wavefield, *current_wavefield,
              *previous_wavefield, *current_aux_wavefield, sigma, model, fd1,
              fd2, shape, pml_width, num_shots, dt);

    propagate(*next_scattered_wavefield, *next_scattered_aux_wavefield,
              *current_scattered_wavefield, *previous_scattered_wavefield,
              *current_scattered_aux_wavefield, sigma, model, fd1, fd2, shape,
              pml_width, num_shots, dt);

    const TYPE * const current_source_amplitudes = set_step_pointer(
        source_amplitudes, inner_step, num_shots, num_sources_per_shot);

    add_sources(*next_wavefield, model, current_source_amplitudes,
                source_locations, shape, num_shots, num_sources_per_shot);

    add_scattering(*next_scattered_wavefield, *next_wavefield,
                   *current_wavefield, *previous_wavefield, scatter,
		   shape, num_shots);

    update_pointers(next_wavefield, current_wavefield, previous_wavefield,
                    next_aux_wavefield, current_aux_wavefield);

    update_pointers(next_scattered_wavefield, current_scattered_wavefield,
                    previous_scattered_wavefield, next_scattered_aux_wavefield,
                    current_scattered_aux_wavefield);
  }
}

static void set_pointers(
    TYPE **const next_wavefield, TYPE **const current_wavefield,
    TYPE **const previous_wavefield, TYPE **const next_aux_wavefield,
    TYPE **const current_aux_wavefield, TYPE * const wavefield,
    TYPE * const aux_wavefield,
    const ptrdiff_t * const shape, const ptrdiff_t num_shots) {
  const ptrdiff_t shot_size = shape[0] * shape[1] * shape[2];

  *previous_wavefield = wavefield;
  *current_wavefield = *previous_wavefield + num_shots * shot_size;
  *next_wavefield = *current_wavefield + num_shots * shot_size;

  *current_aux_wavefield = aux_wavefield;
  *next_aux_wavefield =
      *current_aux_wavefield + AUX_SIZE * num_shots * shot_size;
}

static void update_pointers(
    TYPE * * const next_wavefield,
    TYPE * * const current_wavefield,
    TYPE * * const previous_wavefield,
    TYPE * * const next_aux_wavefield,
    TYPE * * const current_aux_wavefield) {
  TYPE *tmp;
  /* Before: next_wavefield -> A
   *         current_wavefield -> B
   *	     previous_wavefield -> C
   * After: next_wavefield -> C
   *        current_wavefield -> A
   *        previous_wavefield -> B */
  tmp = *previous_wavefield;
  *previous_wavefield = *current_wavefield;
  *current_wavefield = *next_wavefield;
  *next_wavefield = tmp;

  tmp = *next_aux_wavefield;
  *next_aux_wavefield = *current_aux_wavefield;
  *current_aux_wavefield = tmp;
}

TYPE *set_step_pointer(const TYPE * const origin,
                       const ptrdiff_t step, const ptrdiff_t num_shots,
                       const ptrdiff_t numel_per_shot) {
  if (origin == nullptr) return nullptr;

  return const_cast<TYPE *>(origin) + step * num_shots * numel_per_shot;
}
