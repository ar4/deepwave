"""Minimal repro: torch.compile silently evaluates `1 / <dynamic int>` in
float32, corrupting a float64 computation by ~8 significant digits.

Environment: torch 2.13.0+cu130, inductor, reproduces on CUDA and CPU.

When a Python *int* argument becomes a dynamic ``SymInt`` (either via
``dynamic=True`` or automatically, after a second call with a different value),
a float expression derived from it -- here ``1 / n`` -- is emitted by inductor
as a division of the *integer* kernel parameter (``ks0: i64``).  Triton's
integer true-division yields fp32, and the fp32 value is then widened to fp64:

    tmp2 = (tl.broadcast_to(1 / ks0, [XBLOCK])).to(tl.float64)

so the whole float64 computation silently inherits float32 precision.  The
resulting relative error is exactly 2**-26 = 1.4901161e-08 for n = 5
(float32(0.2) = 0.20000000298023224).

Expected: relative error ~1e-16 (everything is float64).
Actual:   relative error ~1.5e-08 whenever the int is dynamic.

Generated code (TORCH_LOGS=output_code), note ``ks0`` is declared ``i64`` and
the fp32 result of the integer division is only *then* widened to fp64::

    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64',
                               'ks0': 'i64', ...}}
    def triton_poi_fused_mul_0(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK):
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = (1 / ks0).to(tl.float64)   # <-- Triton int/int truediv is fp32
        tmp2 = (tmp1).to(tl.float64)
        tmp3 = tmp0 * tmp2
        tl.store(out_ptr0 + (x0), tmp3, xmask)

A dynamic *float* argument is unaffected (it is passed to the kernel in double
precision), so ``f(x, a) = x * (1 / a)`` with ``a = 5.0`` is exact.  The CPU
inductor backend is also unaffected -- its C++ codegen divides in double.

Workarounds: pass the scalar as a Python float rather than an int; pass the
reciprocal as a float64 tensor; or keep the int static
(``torch._dynamo.config.automatic_dynamic_shapes = False`` /
``torch.compile(..., dynamic=False)``).
"""

import torch


def f(x: torch.Tensor, n: int) -> torch.Tensor:
    return x * (1 / n)


def check(device: str, dynamic) -> None:
    x = torch.randn(1024, dtype=torch.float64, device=device)
    torch._dynamo.reset()
    cf = torch.compile(f, fullgraph=True, dynamic=dynamic)
    if dynamic is None:
        # Automatic dynamic: first call specialises, second promotes to SymInt.
        cf(x, 4)
    got = cf(x, 5)
    ref = x * (1 / 5)
    rel = ((got - ref).abs() / ref.abs()).max().item()
    print(f"  device={device:5s} dynamic={dynamic!s:5s}  max rel err = {rel:.6e}")


def main() -> None:
    torch.manual_seed(0)
    print("f(x, n) = x * (1 / n), x float64, n int")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for dynamic in (False, True, None):
            check(device, dynamic)
    print()
    print("2**-26 =", 2.0**-26, " (= float32 rounding error of 1/5)")


main()
