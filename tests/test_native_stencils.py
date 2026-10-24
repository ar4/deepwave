"""Exercise the actual C stencil macros under the native fast-math flags."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", params=["gcc", "clang"])
def stencil_compiler(request):
    """Use each available compiler independently of the installed wheel build."""
    compiler = shutil.which(request.param)
    if compiler is None:
        pytest.skip(f"{request.param} is unavailable")
    return compiler


@pytest.fixture(scope="module", params=[2, 4, 6, 8])
def native_stencil(request, stencil_compiler, tmp_path_factory):
    """Compile float and double probes without loading a fast-math constructor."""
    directory = tmp_path_factory.mktemp("native-stencil")
    source = directory / "probe.c"
    source.write_text(
        '#include "regular_grid.h"\n'
        "void probe(const DW_DTYPE *field, DW_DTYPE *out) {\n"
        "  const DW_DTYPE rdz2 = 0.04, rdy2 = 0.04, rdx2 = 0.04;\n"
        "#define CELL(z,y,x) field[364 + (z)*81 + (y)*9 + (x)]\n"
        "  out[0] = DIFFZ2(CELL);\n"
        "  out[1] = DIFFY2(CELL);\n"
        "  out[2] = DIFFX2(CELL);\n"
        "}\n"
    )
    include = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    optimisation = (
        ["-O3", "-ffast-math", "-fstrict-aliasing"]
        if "clang" in Path(stencil_compiler).name
        else ["-Ofast"]
    )
    functions = {}
    for dtype, spelling, ctype in (
        (torch.float32, "float", ctypes.c_float),
        (torch.float64, "double", ctypes.c_double),
    ):
        obj = directory / f"{spelling}.o"
        library = directory / f"{spelling}.so"
        subprocess.run(  # noqa: S603 - compiler and locally generated source
            [
                stencil_compiler,
                *optimisation,
                "-fno-associative-math",
                "-Werror",
                "-fPIC",
                f"-DDW_ACCURACY={request.param}",
                f"-DDW_DTYPE={spelling}",
                "-I",
                str(include),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
        )
        subprocess.run(  # noqa: S603 - link only the local probe object
            [stencil_compiler, "-shared", str(obj), "-o", str(library)],
            check=True,
        )
        function = ctypes.CDLL(str(library)).probe
        function.argtypes = [ctypes.POINTER(ctype), ctypes.POINTER(ctype)]
        function.restype = None
        functions[dtype] = (function, ctype)
    return functions, request.param


def _evaluate(native_stencil, field):
    function, ctype = native_stencil[0][field.dtype]
    values = (ctype * 3)()
    function(ctypes.cast(field.data_ptr(), ctypes.POINTER(ctype)), values)
    return torch.tensor(list(values), dtype=field.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_second_derivative_preserves_constants(native_stencil, dtype):
    """Zero curvature survives both signs and the representable finite limits."""
    for magnitude in (0.0, 0.1, 1.0, torch.finfo(dtype).tiny, torch.finfo(dtype).max):
        for sign in (-1, 1):
            field = torch.full((9, 9, 9), sign * magnitude, dtype=dtype)
            assert torch.equal(_evaluate(native_stencil, field), torch.zeros(3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_second_derivative_of_quadratic(native_stencil, dtype):
    """Check the stencil scale and axis indexing against exact polynomial calculus."""
    coords = torch.arange(-4, 5, dtype=dtype)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    field = z.square() + 2 * y.square() + 3 * x.square()
    torch.testing.assert_close(
        _evaluate(native_stencil, field),
        torch.tensor([0.08, 0.16, 0.24], dtype=dtype),
        rtol=8 * torch.finfo(dtype).eps,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_native_stencil_avoids_neighbour_sum_overflow(native_stencil, dtype, sign):
    """Finite neighbour differences survive an overflowing neighbour sum."""
    from decimal import Decimal, localcontext

    coefficients = {
        2: [1],
        4: [4 / 3, -1 / 12],
        6: [3 / 2, -3 / 20, 1 / 90],
        8: [8 / 5, -1 / 5, 8 / 315, -1 / 560],
    }[native_stencil[1]]
    field = torch.full((9, 9, 9), sign * 0.7 * torch.finfo(dtype).max, dtype=dtype)
    field[4, 4, 4] = sign * 0.4 * torch.finfo(dtype).max
    with localcontext() as context:
        context.prec = 400
        delta = Decimal(field[0, 0, 0].item()) - Decimal(field[4, 4, 4].item())
        weights = [Decimal(torch.tensor(c, dtype=dtype).item()) for c in coefficients]
        expected = float(2 * delta * sum(weights) * Decimal.from_float(0.04))
    torch.testing.assert_close(
        _evaluate(native_stencil, field),
        torch.full((3,), expected, dtype=dtype),
        rtol=16 * torch.finfo(dtype).eps,
        atol=0,
    )
