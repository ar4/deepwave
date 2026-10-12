"""Measured lines of code, build time and binary size for the spike."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dwgen import build as buildmod
from dwgen.specs import elastic as elastic_spec
from dwgen.specs import scalar as scalar_spec

ROOT = pathlib.Path(__file__).resolve().parent

HAND_WRITTEN = {
    "generator (dwgen/)": sorted(ROOT.glob("dwgen/**/*.py")),
    "hand-written runtime (runtime/)": sorted(ROOT.glob("runtime/*")),
    "torch binding (spike/)": sorted(ROOT.glob("spike/*.py")),
    "tests + harness": [
        ROOT / "tests" / "test_parity.py",
        ROOT / "run_benchmark.py",
        ROOT / "build_all.py",
        ROOT / "measure.py",
        ROOT / "check_determinism.py",
        ROOT / "demo_injected_bug.py",
    ],
}


def count(paths):
    n = 0
    for p in paths:
        if p.is_file():
            n += len(p.read_text().splitlines())
    return n


def main():
    print("=== hand-maintained lines of code ===")
    tot = 0
    for name, paths in HAND_WRITTEN.items():
        c = count(paths)
        tot += c
        print(f"  {name:34s} {c:6d}")
    specs = count(sorted(ROOT.glob("dwgen/specs/*.py")))
    print(f"  (of which physics specs)            {specs:6d}")
    print(f"  {'TOTAL hand-maintained':34s} {tot:6d}")

    print("\n=== generated + built (cold: build dir cleared first) ===")
    if buildmod.OUT.exists():
        shutil.rmtree(buildmod.OUT)
    vs = [
        buildmod.Variant(s.make(2), a, d)
        for s in (scalar_spec, elastic_spec)
        for a in (2, 4)
        for d in ("f32", "f64")
    ]
    r = buildmod.build(vs, verify=True, jobs=4, cuda=True)
    gen_c = sum(len(p.read_text().splitlines()) for p in buildmod.OUT.glob("*.c"))
    gen_cu = sum(len(p.read_text().splitlines()) for p in buildmod.OUT.glob("*.cu"))
    gen_h = sum(len(p.read_text().splitlines()) for p in buildmod.OUT.glob("*_p.h"))
    print(f"  variants                           {len(vs):6d}")
    print(f"  generated C (CPU + CUDA host)      {gen_c:6d} lines")
    print(f"  generated CUDA device code         {gen_cu:6d} lines")
    print(f"  generated headers                  {gen_h:6d} lines")
    print(f"  generation wall time               {r.gen_seconds:6.2f} s")
    print(f"  doubly-derived adjoint check       {r.check_seconds:6.2f} s")
    print(f"  gcc -j4                            {r.compile_seconds:6.2f} s")
    print(f"  nvcc -j2 (sm_86 + sm_89 + PTX)     {r.cuda_compile_seconds:6.2f} s")
    print(f"  libdwspike.so                      {r.lib_bytes:9d} bytes")
    print(
        f"  fatbins ({len(r.fatbins)} files)                 {r.fatbin_bytes:9d} bytes"
    )
    print(
        f"  total shipped binary               {r.lib_bytes + r.fatbin_bytes:9d} bytes"
    )
    out = subprocess.run(
        ["readelf", "-d", str(r.lib)], capture_output=True, text=True
    ).stdout
    needed = [
        ln.split("[")[1].split("]")[0] for ln in out.splitlines() if "NEEDED" in ln
    ]
    print(f"  libdwspike.so DT_NEEDED             {needed}")


if __name__ == "__main__":
    main()
