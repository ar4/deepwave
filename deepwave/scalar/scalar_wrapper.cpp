#include <torch/extension.h>
#include "scalar.h"

void forward_wrapper(at::Tensor wavefield, at::Tensor aux_wavefield,
                     at::Tensor receiver_amplitudes,
                     at::Tensor saved_wavefields, at::Tensor sigma,
                     at::Tensor model, at::Tensor fd1, at::Tensor fd2,
                     at::Tensor source_amplitudes, at::Tensor source_locations,
                     at::Tensor receiver_locations, at::Tensor shape,
                     at::Tensor pml_width, at::Tensor dt,
                     const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
                     const ptrdiff_t num_shots,
                     const ptrdiff_t num_sources_per_shot,
                     const ptrdiff_t num_receivers_per_shot,
                     const enum wavefield_save_strategy save_strategy) {
  forward(wavefield.data<TYPE>(), aux_wavefield.data<TYPE>(),
          receiver_amplitudes.data<TYPE>(), saved_wavefields.data<TYPE>(),
          sigma.data<TYPE>(), model.data<TYPE>(), fd1.data<TYPE>(),
          fd2.data<TYPE>(), source_amplitudes.data<TYPE>(),
          source_locations.data<ptrdiff_t>(),
          receiver_locations.data<ptrdiff_t>(), shape.data<ptrdiff_t>(),
          pml_width.data<ptrdiff_t>(), num_steps, step_ratio, num_shots,
          num_sources_per_shot, num_receivers_per_shot, dt.data<TYPE>()[0],
          save_strategy);
}

void forward_born_wrapper(
    at::Tensor wavefield, at::Tensor aux_wavefield,
    at::Tensor scattered_wavefield, at::Tensor scattered_aux_wavefield,
    at::Tensor receiver_amplitudes, at::Tensor saved_wavefields,
    at::Tensor sigma, at::Tensor model,
    at::Tensor scatter, at::Tensor fd1, at::Tensor fd2,
    at::Tensor source_amplitudes, at::Tensor source_locations,
    at::Tensor receiver_locations, at::Tensor shape, at::Tensor pml_width,
    at::Tensor dt, const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot,
    const ptrdiff_t num_receivers_per_shot,
    const enum wavefield_save_strategy save_strategy) {
  forward_born(wavefield.data<TYPE>(), aux_wavefield.data<TYPE>(),
	       scattered_wavefield.data<TYPE>(), scattered_aux_wavefield.data<TYPE>(),
               receiver_amplitudes.data<TYPE>(), saved_wavefields.data<TYPE>(),
               sigma.data<TYPE>(),
               model.data<TYPE>(), scatter.data<TYPE>(), fd1.data<TYPE>(),
               fd2.data<TYPE>(), source_amplitudes.data<TYPE>(),
               source_locations.data<ptrdiff_t>(),
               receiver_locations.data<ptrdiff_t>(), shape.data<ptrdiff_t>(),
               pml_width.data<ptrdiff_t>(), num_steps, step_ratio, num_shots,
               num_sources_per_shot, num_receivers_per_shot, dt.data<TYPE>()[0],
               save_strategy);
}

void backward_wrapper(at::Tensor wavefield, at::Tensor aux_wavefield,
                      at::Tensor model_grad, at::Tensor source_grad_amplitudes,
                      at::Tensor adjoint_wavefield, at::Tensor scaling,
                      at::Tensor sigma, at::Tensor model, at::Tensor fd1,
                      at::Tensor fd2, at::Tensor receiver_grad_amplitudes,
                      at::Tensor source_locations,
                      at::Tensor receiver_locations, at::Tensor shape,
                      at::Tensor pml_width, at::Tensor dt,
                      const ptrdiff_t num_steps, const ptrdiff_t step_ratio,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_sources_per_shot,
                      const ptrdiff_t num_receivers_per_shot) {
  backward(wavefield.data<TYPE>(), aux_wavefield.data<TYPE>(),
           model_grad.data<TYPE>(), source_grad_amplitudes.data<TYPE>(),
           adjoint_wavefield.data<TYPE>(), scaling.data<TYPE>(),
           sigma.data<TYPE>(), model.data<TYPE>(), fd1.data<TYPE>(),
           fd2.data<TYPE>(), receiver_grad_amplitudes.data<TYPE>(),
           source_locations.data<ptrdiff_t>(),
           receiver_locations.data<ptrdiff_t>(), shape.data<ptrdiff_t>(),
           pml_width.data<ptrdiff_t>(), num_steps, step_ratio, num_shots,
           num_sources_per_shot, num_receivers_per_shot, dt.data<TYPE>()[0]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "forward");
  m.def("forward_born", &forward_born_wrapper, "forward_born");
  m.def("backward", &backward_wrapper, "backward");
  py::enum_<wavefield_save_strategy>(m, "wavefield_save_strategy",
                                     py::module_local())
      .value("STRATEGY_NONE", STRATEGY_NONE)
      .value("STRATEGY_COPY", STRATEGY_COPY)
      .export_values();
}
