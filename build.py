import os
from cffi import FFI

scalar1d = FFI()
scalar2d = FFI()
scalar3d = FFI()

scalar_dir = os.path.join('deepwave', 'scalar')

with open(os.path.join(scalar_dir, 'scalar.h'), 'r') as headerfile:
    header = headerfile.read()
    scalar1d.cdef(header)
    scalar2d.cdef(header)
    scalar3d.cdef(header)

with open(os.path.join(scalar_dir, 'scalar.c'), 'r') as sourcefile:
    source = sourcefile.read()
    scalar1d.set_source('deepwave.scalar1d', source,
                        define_macros=[('DIM', '1')],
                        include_dirs=[scalar_dir],
                        extra_compile_args=['-Ofast', '-march=native',
                                            '-fopenmp', '-std=c99'],
                        extra_link_args=['-fopenmp'])
    scalar2d.set_source('deepwave.scalar2d', source,
                        define_macros=[('DIM', '2')],
                        include_dirs=[scalar_dir],
                        extra_compile_args=['-Ofast', '-march=native',
                                            '-fopenmp', '-std=c99'],
                        extra_link_args=['-fopenmp'])
    scalar3d.set_source('deepwave.scalar3d', source,
                        define_macros=[('DIM', '3')],
                        include_dirs=[scalar_dir],
                        extra_compile_args=['-Ofast', '-march=native',
                                            '-fopenmp', '-std=c99'],
                        extra_link_args=['-fopenmp'])
