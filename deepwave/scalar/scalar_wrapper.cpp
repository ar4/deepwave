#include <torch/torch.h>
#include "scalar.h"

void forward_wrapper(
		at::Tensor wavefield,
		at::Tensor aux_wavefield,
		at::Tensor receiver_amplitudes,
		at::Tensor saved_wavefields,
		at::Tensor sigma,
		at::Tensor model, 
		at::Tensor fd1,
		at::Tensor fd2,
		at::Tensor source_amplitudes,
		at::Tensor source_locations,
		at::Tensor receiver_locations,
		at::Tensor shape,
		at::Tensor pml_width,
		const ptrdiff_t num_steps,
		const ptrdiff_t step_ratio,
		const ptrdiff_t num_shots,
		const ptrdiff_t num_sources_per_shot,
		const ptrdiff_t num_receivers_per_shot,
		const float dt,
		const enum wavefield_save_strategy save_strategy)
{
        forward(
                        wavefield.data<float>(),
                        aux_wavefield.data<float>(),
                        receiver_amplitudes.data<float>(),
                        saved_wavefields.data<float>(),
                        sigma.data<float>(),
                        model.data<float>(), 
                        fd1.data<float>(),
                        fd2.data<float>(),
                        source_amplitudes.data<float>(),
                        source_locations.data<ptrdiff_t>(),
                        receiver_locations.data<ptrdiff_t>(),
                        shape.data<ptrdiff_t>(),
                        pml_width.data<ptrdiff_t>(),
                        num_steps,
                        step_ratio,
                        num_shots,
                        num_sources_per_shot,
                        num_receivers_per_shot,
                        dt,
                        save_strategy);
}


void backward_wrapper(
                at::Tensor wavefield,
                at::Tensor aux_wavefield,
                at::Tensor model_grad,
                at::Tensor source_grad_amplitudes,
                at::Tensor adjoint_wavefield,
                at::Tensor scaling,
                at::Tensor sigma,
                at::Tensor model, 
                at::Tensor fd1,
                at::Tensor fd2,
                at::Tensor receiver_grad_amplitudes,
                at::Tensor source_locations,
                at::Tensor receiver_locations,
                at::Tensor shape,
                at::Tensor pml_width,
                const ptrdiff_t num_steps,
                const ptrdiff_t step_ratio,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot,
                const ptrdiff_t num_receivers_per_shot,
                const float dt)
{
        backward(
                        wavefield.data<float>(),
                        aux_wavefield.data<float>(),
                        model_grad.data<float>(),
                        source_grad_amplitudes.data<float>(),
                        adjoint_wavefield.data<float>(),
                        scaling.data<float>(),
                        sigma.data<float>(),
                        model.data<float>(), 
                        fd1.data<float>(),
                        fd2.data<float>(),
                        receiver_grad_amplitudes.data<float>(),
                        source_locations.data<ptrdiff_t>(),
                        receiver_locations.data<ptrdiff_t>(),
                        shape.data<ptrdiff_t>(),
                        pml_width.data<ptrdiff_t>(),
                        num_steps,
                        step_ratio,
                        num_shots,
                        num_sources_per_shot,
                        num_receivers_per_shot,
                        dt);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

        m.def("forward", &forward_wrapper, "forward");
        m.def("backward", &backward_wrapper, "backward");
        py::enum_<wavefield_save_strategy>(m, "wavefield_save_strategy",
			py::module_local())
                .value("STRATEGY_NONE", STRATEGY_NONE)
                .value("STRATEGY_INPLACE", STRATEGY_INPLACE)
                .value("STRATEGY_COPY", STRATEGY_COPY)
                .export_values();
}
