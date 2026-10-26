"""Generate, verify and compile every in-scope variant; print measured sizes."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dwgen import build as buildmod
from dwgen.specs import scalar as scalar_spec


def variants(families=("scalar",)):
    out = []
    if "scalar" in families:
        out += [
            buildmod.Variant(scalar_spec.make(2), a, d)
            for a in (2, 4)
            for d in ("f32", "f64")
        ]
    if "elastic" in families:
        from dwgen.specs import elastic as elastic_spec

        out += [
            buildmod.Variant(elastic_spec.make(2), a, d)
            for a in (2, 4)
            for d in ("f32", "f64")
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--families", default="scalar")
    args = ap.parse_args()
    vs = variants(tuple(args.families.split(",")))
    r = buildmod.build(vs, verify=not args.no_verify, jobs=4, cuda=not args.no_cuda)
    print(f"variants          : {len(vs)}")
    print(f"generated lines   : {r.generated_lines}")
    print(f"generate seconds  : {r.gen_seconds:.2f}")
    print(f"symcheck seconds  : {r.check_seconds:.2f}")
    print(f"gcc seconds (-j4) : {r.compile_seconds:.2f}")
    print(f"nvcc seconds      : {r.cuda_compile_seconds:.2f}")
    print(f"libdwspike.so     : {r.lib_bytes} bytes")
    print(f"fatbins           : {r.fatbin_bytes} bytes ({len(r.fatbins)} files)")


if __name__ == "__main__":
    main()
