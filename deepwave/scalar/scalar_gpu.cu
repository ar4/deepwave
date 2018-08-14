#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "scalar.h"
#include "scalar_device.h"

__constant__ TYPE fd1[2 * DIM ];
__constant__ TYPE fd2[2 * DIM + 1] ;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr, "GPUassert: %s %s %d\n",
                                cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

/* Dimension-specific definitions
 *
 * LOOP: Loops over shots and spatial dimensions
 * ENDLOOP: Close the loops in LOOP
 * location_index: Convert array of coordinates into index into flat array
 * */
#if DIM == 1

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y; \
        ptrdiff_t z = blockIdx.x * blockDim.x + threadIdx.x + start_z; \
        ptrdiff_t y = 0; \
        ptrdiff_t x = 0; \
        if ((shot < num_shots) && (z < end_z)) \
{

#define ENDLOOP \
}

inline __device__ ptrdiff_t location_index(
                const ptrdiff_t *__restrict__ const arr,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index];

        return z;
}

#elif DIM == 2

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        ptrdiff_t shot = blockIdx.z * blockDim.z + threadIdx.z; \
        ptrdiff_t z = blockIdx.y * blockDim.y + threadIdx.y + start_z; \
        ptrdiff_t y = blockIdx.x * blockDim.x + threadIdx.x + start_y; \
        ptrdiff_t x = 0; \
        if ((shot < num_shots) && (z < end_z) && (y < end_y)) \
{


#define ENDLOOP \
}

inline __device__ ptrdiff_t location_index(
                const ptrdiff_t *__restrict__ const arr,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index * 2];
        const ptrdiff_t y = arr[index * 2 + 1];

        return z * shape_y + y;
}

#elif DIM == 3

#define LOOP(num_shots, start_z, end_z, start_y, end_y, start_x, end_x) \
        ptrdiff_t threadz = blockIdx.z * blockDim.z + threadIdx.z; \
        ptrdiff_t shot = threadz / (end_z - start_z + 1); \
        ptrdiff_t z = threadz % (end_z - start_z + 1) + start_z; \
        ptrdiff_t y = blockIdx.y * blockDim.y + threadIdx.y + start_y; \
        ptrdiff_t x = blockIdx.x * blockDim.x + threadIdx.x + start_x; \
        if ((shot < num_shots) && (z < end_z) && (y < end_y) && (x < end_x)) \
{

#define ENDLOOP \
}

inline __device__ ptrdiff_t location_index(
                const ptrdiff_t *__restrict__ const arr,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t index)
{
        const ptrdiff_t z = arr[index * 3];
        const ptrdiff_t y = arr[index * 3 + 1];
        const ptrdiff_t x = arr[index * 3 + 2];

        return z * shape_y * shape_x + y * shape_x + x;
}

#else
#error "Must specify the dimension, e.g. -D DIM=1"
#endif /* DIM */


void setup(
                const TYPE *__restrict__ const fd1_d,
                const TYPE *__restrict__ const fd2_d)
{
        gpuErrchk(cudaMemcpyToSymbol(fd1, fd1_d, 2 * DIM * sizeof(TYPE)));
        gpuErrchk(cudaMemcpyToSymbol(fd2, fd2_d, (2 * DIM + 1) * sizeof(TYPE)));
}


#if DIM == 1

__global__ void propagate_kernel(
                TYPE *__restrict__ const wfn,
                TYPE *__restrict__ const phizn,
                const TYPE *__restrict__ const wfc,
                const TYPE *__restrict__ const wfp,
                const TYPE *__restrict__ const phizc,
                const TYPE *__restrict__ const sigmaz,
                const TYPE *__restrict__ const model,
                const ptrdiff_t shape_z,
                const ptrdiff_t numel_shot,
                const ptrdiff_t size_xy,
                const ptrdiff_t num_shots,
                const ptrdiff_t pmlz0,
                const ptrdiff_t pmlz1,
                const TYPE dt)
{

        LOOP(
                        num_shots,
                        ZPAD, shape_z - ZPAD,
                        YPAD, 1 - YPAD,
                        XPAD, 1 - XPAD);

        ptrdiff_t i = z;
        ptrdiff_t si = shot * numel_shot + i;

        TYPE lap = laplacian(wfc);

        if (
                        (z >= pmlz0 + 2 * ZPAD) &&
                        (z < shape_z - pmlz1 - 2 * ZPAD)
           )
        {

                /* Update wavefield */
                wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
        }
        else
        {
                /* Inside PML region */

                TYPE wfc_z = z_deriv(wfc);
                TYPE phizc_z = z_deriv(phizc);

                /* Update wavefield */
                wfn[si] = 1 / (1 + dt * sigmaz[z] / 2) *
                        (model[i] * (lap + phizc_z)
                         + dt * sigmaz[z] * wfp[si] / 2
                         + (2 * wfc[si] - wfp[si]));

                /* Update phi */
                phizn[si] = phizc[si] - dt * sigmaz[z] * (wfc_z + phizc[si]);
        }

        ENDLOOP;
}

#elif DIM == 2

__global__ void propagate_kernel(
                TYPE *__restrict__ const wfn,
                TYPE *__restrict__ const phizn,
                TYPE *__restrict__ const phiyn,
                const TYPE *__restrict__ const wfc,
                const TYPE *__restrict__ const wfp,
                const TYPE *__restrict__ const phizc,
                const TYPE *__restrict__ const sigmaz,
                const TYPE *__restrict__ const phiyc,
                const TYPE *__restrict__ const sigmay,
                const TYPE *__restrict__ const model,
                const ptrdiff_t shape_z,
                const ptrdiff_t shape_y,
                const ptrdiff_t numel_shot,
                const ptrdiff_t size_x,
                const ptrdiff_t size_xy,
                const ptrdiff_t num_shots,
                const ptrdiff_t pmlz0,
                const ptrdiff_t pmlz1,
                const ptrdiff_t pmly0,
                const ptrdiff_t pmly1,
                const TYPE dt)
{

        LOOP(
                        num_shots,
                        ZPAD, shape_z - ZPAD,
                        YPAD, shape_y - YPAD,
                        XPAD, 1 - XPAD);

        ptrdiff_t i = z * size_xy + y;
        ptrdiff_t si = shot * numel_shot + i;

        TYPE lap = laplacian(wfc);

        if (
                        (z >= pmlz0 + 2 * ZPAD) &&
                        (z < shape_z - pmlz1 - 2 * ZPAD) &&
                        (y >= pmly0 + 2 * YPAD) &&
                        (y < shape_y - pmly1 - 2 * YPAD)
           )
        {

                /* Update wavefield */
                wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
        }
        else
        {
                /* Inside PML region */

                TYPE wfc_z = z_deriv(wfc);
                TYPE phizc_z = z_deriv(phizc);
                TYPE wfc_y = y_deriv(wfc);
                TYPE phiyc_y = y_deriv(phiyc);

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
        }

        ENDLOOP;
}

#elif DIM == 3

__global__ void propagate_kernel(
                TYPE *__restrict__ const wfn,
                TYPE *__restrict__ const phizn,
                TYPE *__restrict__ const phiyn,
                TYPE *__restrict__ const phixn,
                TYPE *__restrict__ const psin,
                const TYPE *__restrict__ const wfc,
                const TYPE *__restrict__ const wfp,
                const TYPE *__restrict__ const phizc,
                const TYPE *__restrict__ const sigmaz,
                const TYPE *__restrict__ const phiyc,
                const TYPE *__restrict__ const sigmay,
                const TYPE *__restrict__ const phixc,
                const TYPE *__restrict__ const sigmax,
                const TYPE *__restrict__ const psic,
                const TYPE *__restrict__ const model,
                const ptrdiff_t shape_z,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t numel_shot,
                const ptrdiff_t size_x,
                const ptrdiff_t size_xy,
                const ptrdiff_t num_shots,
                const ptrdiff_t pmlz0,
                const ptrdiff_t pmlz1,
                const ptrdiff_t pmly0,
                const ptrdiff_t pmly1,
                const ptrdiff_t pmlx0,
                const ptrdiff_t pmlx1,
                const TYPE dt)
{
        LOOP(
                        num_shots,
                        ZPAD, shape_z - ZPAD,
                        YPAD, shape_y - YPAD,
                        XPAD, shape_x - XPAD);

        ptrdiff_t i = z * size_xy + y * size_x + x;
        ptrdiff_t si = shot * numel_shot + i;

        TYPE lap = laplacian(wfc);

        if (
                        (z >= pmlz0 + 2 * ZPAD) &&
                        (z < shape_z - pmlz1 - 2 * ZPAD) &&
                        (y >= pmly0 + 2 * YPAD) &&
                        (y < shape_y - pmly1 - 2 * YPAD) &&
                        (x >= pmlx0 + 2 * XPAD) &&
                        (x < shape_x - pmlx1 - 2 * XPAD)
           )
        {

                /* Update wavefield */
                wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
        }
        else
        {
                /* Inside PML region */

                TYPE wfc_z = z_deriv(wfc);
                TYPE phizc_z = z_deriv(phizc);
                TYPE wfc_y = y_deriv(wfc);
                TYPE wfc_x = x_deriv(wfc);
                TYPE phiyc_y = y_deriv(phiyc);
                TYPE phixc_x = x_deriv(phixc);
                TYPE psic_z = z_deriv(psic);
                TYPE psic_y = y_deriv(psic);
                TYPE psic_x = x_deriv(psic);

                /* Update wavefield */
                wfn[si] =
                        1 / (1 + dt * (sigmaz[z] + sigmay[y] + sigmax[x]) / 2) *
                        (model[i] * lap + dt * dt *
                         (phizc_z + phiyc_y + phixc_x -
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

                ENDLOOP;
        }
}

#endif /* DIM */


void propagate(
                TYPE *__restrict__ const wfn, /* next wavefield */
                TYPE *__restrict__ const auxn, /* next auxiliary */
                const TYPE *__restrict__ const wfc, /* current wavefield */
                const TYPE *__restrict__ const wfp, /* previous wavefield */
                const TYPE *__restrict__ const auxc, /* current auxiliary */
                const TYPE *__restrict__ const sigma,
                const TYPE *__restrict__ const model,
                const TYPE *__restrict__ const fd1_d, /* 1st diff coeffs */
                const TYPE *__restrict__ const fd2_d, /* 2nd diff coeffs */
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width,
                const ptrdiff_t num_shots,
                const TYPE dt)
{

        const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
        const ptrdiff_t size_x = shape[2];
        const ptrdiff_t size_xy = shape[1] * shape[2];
        TYPE *__restrict__ const phizn = auxn;
        const TYPE *__restrict__ const phizc = auxc;
        const TYPE *__restrict__ const sigmaz = sigma;

#if DIM >= 2

        TYPE *__restrict__ const phiyn = auxn + num_shots * numel_shot;
        const TYPE *__restrict__ const phiyc = auxc + num_shots * numel_shot;
        const TYPE *__restrict__ const sigmay = sigma + shape[0];

#endif /* DIM >= 2 */

#if DIM == 3

        TYPE *__restrict__ const phixn = auxn + 2 * num_shots * numel_shot;
        TYPE *__restrict__ const psin = auxn + 3 * num_shots * numel_shot;
        const TYPE *__restrict__ const phixc =
                auxc + 2 * num_shots * numel_shot;
        const TYPE *__restrict__ const psic =
                auxc + 3 * num_shots * numel_shot;
        const TYPE *__restrict__ const sigmax = sigma + shape[0] + shape[1];

#endif /* DIM  == 3 */

        dim3 dimBlock(32, 32, 1);
#if DIM == 1
        int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
        int gridz = 1;
        dim3 dimGrid(gridx, gridy, gridz);
        propagate_kernel<<<dimGrid, dimBlock>>>(
                        wfn,
                        phizn,
                        wfc,
                        wfp,
                        phizc,
                        sigmaz,
                        model,
                        shape[0],
                        numel_shot,
                        size_xy,
                        num_shots,
                        pml_width[0],
                        pml_width[1],
                        dt);
#elif DIM == 2
        int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
        int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;
        dim3 dimGrid(gridx, gridy, gridz);
        propagate_kernel<<<dimGrid, dimBlock>>>(
                        wfn,
                        phizn,
                        phiyn,
                        wfc,
                        wfp,
                        phizc,
                        sigmaz,
                        phiyc,
                        sigmay,
                        model,
                        shape[0],
                        shape[1],
                        numel_shot,
                        size_x,
                        size_xy,
                        num_shots,
                        pml_width[0],
                        pml_width[1],
                        pml_width[2],
                        pml_width[3],
                        dt);
#elif DIM == 3
        int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
        int gridz = (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) /
                dimBlock.z;
        dim3 dimGrid(gridx, gridy, gridz);
        propagate_kernel<<<dimGrid, dimBlock>>>(
                        wfn,
                        phizn,
                        phiyn,
                        phixn,
                        psin,
                        wfc,
                        wfp,
                        phizc,
                        sigmaz,
                        phiyc,
                        sigmay,
                        phixc,
                        sigmax,
                        psic,
                        model,
                        shape[0],
                        shape[1],
                        shape[2],
                        numel_shot,
                        size_x,
                        size_xy,
                        num_shots,
                        pml_width[0],
                        pml_width[1],
                        pml_width[2],
                        pml_width[3],
                        pml_width[4],
                        pml_width[5],
                        dt);
#endif /* DIM */

        gpuErrchk( cudaPeekAtLastError() );

}


void __global__ add_sources_kernel(
                TYPE *__restrict__ const next_wavefield,
                const TYPE *__restrict__ const model,
                const TYPE *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t shape_z,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot)
{

        ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
        ptrdiff_t source = blockIdx.x * blockDim.x + threadIdx.x;

        if ((shot < num_shots) && (source < num_sources_per_shot))
        {

                ptrdiff_t s = shot * num_sources_per_shot + source;
                ptrdiff_t i = location_index(source_locations, shape_y,
                                shape_x, s);
                ptrdiff_t si = shot * shape_z * shape_y * shape_x + i;
                atomicAdd(next_wavefield + si, source_amplitudes[s] * model[i]);

        }

}


void add_sources(
                TYPE *__restrict__ const next_wavefield,
                const TYPE *__restrict__ const model,
                const TYPE *__restrict__ const source_amplitudes,
                const ptrdiff_t *__restrict__ const source_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_sources_per_shot)
{

        dim3 dimBlock(32, 1, 1);
        int gridx = (num_sources_per_shot + dimBlock.x - 1) / dimBlock.x;
        int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
        int gridz = 1;
        dim3 dimGrid(gridx, gridy, gridz);

        add_sources_kernel<<<dimGrid, dimBlock>>>(
                        next_wavefield,
                        model,
                        source_amplitudes,
                        source_locations,
                        shape[0],
                        shape[1],
                        shape[2],
                        num_shots,
                        num_sources_per_shot);

        gpuErrchk( cudaPeekAtLastError() );

}


void __global__ record_receivers_kernel(
                TYPE *__restrict__ const receiver_amplitudes,
                const TYPE *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t shape_z,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot)
{

        ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
        ptrdiff_t receiver = blockIdx.x * blockDim.x + threadIdx.x;

        if ((shot < num_shots) && (receiver < num_receivers_per_shot))
        {

                ptrdiff_t r = shot * num_receivers_per_shot + receiver;
                ptrdiff_t si = shot * shape_z * shape_y * shape_x +
                        location_index(receiver_locations, shape_y,
                                        shape_x, r);
                receiver_amplitudes[r] = current_wavefield[si];

        }

}


void record_receivers(
                TYPE *__restrict__ const receiver_amplitudes,
                const TYPE *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const receiver_locations,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t num_receivers_per_shot)
{

        if (receiver_amplitudes == NULL) return; /* no source inversion */

        dim3 dimBlock(32, 1, 1);
        int gridx = (num_receivers_per_shot + dimBlock.x - 1) / dimBlock.x;
        int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
        int gridz = 1;
        dim3 dimGrid(gridx, gridy, gridz);

        record_receivers_kernel<<<dimGrid, dimBlock>>>(
                        receiver_amplitudes,
                        current_wavefield,
                        receiver_locations,
                        shape[0],
                        shape[1],
                        shape[2],
                        num_shots,
                        num_receivers_per_shot);

        gpuErrchk( cudaPeekAtLastError() );

}


void save_wavefields(
                TYPE *__restrict__ const saved_wavefields,
                const TYPE *__restrict__ const current_wavefield,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t num_shots,
                const ptrdiff_t step,
                const enum wavefield_save_strategy save_strategy)
{

        if (save_strategy == STRATEGY_COPY)
        {
                TYPE *__restrict__ current_saved_wavefield = set_step_pointer(
                                saved_wavefields,
                                step,
                                num_shots,
                                shape[0] * shape[1] * shape[2]);
                gpuErrchk(cudaMemcpy(
                                        current_saved_wavefield,
                                        current_wavefield,
                                        num_shots *
                                        shape[0] * shape[1] * shape[2] *
                                        sizeof(TYPE),
                                        cudaMemcpyDeviceToDevice));
        }

}


void __global__ imaging_condition_kernel(
                TYPE *__restrict__ const model_grad,
                const TYPE *__restrict__ const current_wavefield,
                const TYPE *__restrict__ const next_adjoint_wavefield,
                const TYPE *__restrict__ const current_adjoint_wavefield,
                const TYPE *__restrict__ const previous_adjoint_wavefield,
                const TYPE *__restrict__ const sigmaz,
                const TYPE *__restrict__ const sigmay,
                const TYPE *__restrict__ const sigmax,
                const ptrdiff_t shape_z,
                const ptrdiff_t shape_y,
                const ptrdiff_t shape_x,
		const ptrdiff_t numel_shot,
                const ptrdiff_t size_x,
                const ptrdiff_t size_xy,
                const ptrdiff_t num_shots,
                const TYPE dt)
{
        LOOP(
                        num_shots,
                        ZPAD, shape_z - ZPAD,
                        YPAD, shape_y - YPAD,
                        XPAD, shape_x - XPAD);

        ptrdiff_t i = z * size_xy + y * size_x + x;
        ptrdiff_t si = shot * numel_shot + i;

        TYPE adjoint_wavefield_tt = (next_adjoint_wavefield[si] -
                        2 * current_adjoint_wavefield[si] +
                        previous_adjoint_wavefield[si]) / (dt * dt);

        TYPE adjoint_wavefield_t = (next_adjoint_wavefield[si] -
                        previous_adjoint_wavefield[si]) / (2 * dt);

#if DIM == 1

        atomicAdd(model_grad + i,
                        current_wavefield[si] *
                        (adjoint_wavefield_tt +
                         sigmaz[z] * adjoint_wavefield_t));

#elif DIM == 2

        atomicAdd(model_grad + i,
                        current_wavefield[si] *
                        (adjoint_wavefield_tt +
                         (sigmaz[z] + sigmay[y]) * adjoint_wavefield_t +
                        sigmaz[z] * sigmay[y] * current_adjoint_wavefield[si]));

#elif DIM == 3

	/* NOTE: There should be an additional term here (involving spatial
	 * derivative of phi, sigma, and psi), but it is neglected due to
	 * the additional computational cost it would cause. */
        atomicAdd(model_grad + i,
                        current_wavefield[si] *
                        (adjoint_wavefield_tt +
                         (sigmaz[z] + sigmay[y] + sigmax[x]) *
                         adjoint_wavefield_t +
                        (sigmax[x] * sigmay[y] +
                          sigmay[y] * sigmaz[z] +
                          sigmax[x] * sigmaz[z]) *
                        current_adjoint_wavefield[si]));

#endif

        ENDLOOP;

}


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
                const TYPE dt)
{

        if (model_grad == NULL) return; /* Not doing model inversion */

        dim3 dimBlock(32, 32, 1);
#if DIM == 1
        int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
        int gridz = 1;
#elif DIM == 2
        int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
        int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;
#elif DIM == 3
        int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
        int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
        int gridz = (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) /
                dimBlock.z;
#endif /* DIM */

        dim3 dimGrid(gridx, gridy, gridz);
        const TYPE *__restrict__ const sigmaz = sigma;
        const TYPE *__restrict__ const sigmay = sigma + shape[0];
        const TYPE *__restrict__ const sigmax = sigma + shape[0] + shape[1];

        imaging_condition_kernel<<<dimGrid, dimBlock>>>(
                        model_grad,
                        current_wavefield,
                        next_adjoint_wavefield,
                        current_adjoint_wavefield,
                        previous_adjoint_wavefield,
                        sigmaz,
                        sigmay,
                        sigmax,
                        shape[0],
                        shape[1],
                        shape[2],
			shape[0] * shape[1] * shape[2],
                        shape[2],
                        shape[1] * shape[2],
                        num_shots,
                        dt);

        gpuErrchk( cudaPeekAtLastError() );

}


void __global__ model_grad_scaling_kernel(
                TYPE *__restrict__ const model_grad,
                const TYPE *__restrict__ const scaling,
                const ptrdiff_t numel_shot)
{

        ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < numel_shot)
        {
                model_grad[i] *= scaling[i];
        }

}


void model_grad_scaling(
                TYPE *__restrict__ const model_grad,
                const TYPE *__restrict__ const scaling,
                const ptrdiff_t *__restrict__ const shape,
                const ptrdiff_t *__restrict__ const pml_width)
{

        if (model_grad == NULL) return; /* Not doing model inversion */

        const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];

        dim3 dimBlock(32, 1, 1);
        int gridx = (numel_shot + dimBlock.x - 1) / dimBlock.x;
        int gridy = 1;
        int gridz = 1;
        dim3 dimGrid(gridx, gridy, gridz);

        model_grad_scaling_kernel<<<dimGrid, dimBlock>>>(
                        model_grad,
                        scaling,
                        numel_shot);

        gpuErrchk( cudaPeekAtLastError() );

}
