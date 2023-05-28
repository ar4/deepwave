#!/bin/sh

set -e

CFLAGS="-Wall -O2 -fp:fast -arch:AVX2 -openmp"
CUDAFLAGS="--restrict --use_fast_math -O3 -gencode=arch=compute_52,code=sm_52, -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80"
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar.c -Foscalar_cpu_iso_2_float.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar.c -Foscalar_cpu_iso_4_float.obj
cl $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar.c -Foscalar_cpu_iso_6_float.obj
cl $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar.c -Foscalar_cpu_iso_8_float.obj
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar.c -Foscalar_cpu_iso_2_double.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar.c -Foscalar_cpu_iso_4_double.obj
cl $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar.c -Foscalar_cpu_iso_6_double.obj
cl $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar.c -Foscalar_cpu_iso_8_double.obj
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar_born.c -Foscalar_born_cpu_iso_2_float.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar_born.c -Foscalar_born_cpu_iso_4_float.obj
cl $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar_born.c -Foscalar_born_cpu_iso_6_float.obj
cl $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar_born.c -Foscalar_born_cpu_iso_8_float.obj
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar_born.c -Foscalar_born_cpu_iso_2_double.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar_born.c -Foscalar_born_cpu_iso_4_double.obj
cl $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar_born.c -Foscalar_born_cpu_iso_6_double.obj
cl $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar_born.c -Foscalar_born_cpu_iso_8_double.obj
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c elastic.c -Foelastic_cpu_iso_2_float.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c elastic.c -Foelastic_cpu_iso_4_float.obj
cl $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c elastic.c -Foelastic_cpu_iso_2_double.obj
cl $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c elastic.c -Foelastic_cpu_iso_4_double.obj
cl $CFLAGS -LD scalar_born_cpu_iso_2_float.obj scalar_born_cpu_iso_4_float.obj scalar_born_cpu_iso_6_float.obj scalar_born_cpu_iso_8_float.obj scalar_born_cpu_iso_2_double.obj scalar_born_cpu_iso_4_double.obj scalar_born_cpu_iso_6_double.obj scalar_born_cpu_iso_8_double.obj scalar_cpu_iso_2_float.obj scalar_cpu_iso_4_float.obj scalar_cpu_iso_6_float.obj scalar_cpu_iso_8_float.obj scalar_cpu_iso_2_double.obj scalar_cpu_iso_4_double.obj scalar_cpu_iso_6_double.obj scalar_cpu_iso_8_double.obj elastic_cpu_iso_2_float.obj elastic_cpu_iso_4_float.obj elastic_cpu_iso_2_double.obj elastic_cpu_iso_4_double.obj libiomp5md.lib -Felibdeepwave_cpu_windows_x86_64.dll -link -nodefaultlib:vcomp
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar.cu -o scalar_cuda_iso_2_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar.cu -o scalar_cuda_iso_4_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar.cu -o scalar_cuda_iso_6_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar.cu -o scalar_cuda_iso_8_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar.cu -o scalar_cuda_iso_2_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar.cu -o scalar_cuda_iso_4_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar.cu -o scalar_cuda_iso_6_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar.cu -o scalar_cuda_iso_8_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar_born.cu -o scalar_born_cuda_iso_2_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar_born.cu -o scalar_born_cuda_iso_4_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar_born.cu -o scalar_born_cuda_iso_6_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar_born.cu -o scalar_born_cuda_iso_8_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar_born.cu -o scalar_born_cuda_iso_2_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar_born.cu -o scalar_born_cuda_iso_4_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar_born.cu -o scalar_born_cuda_iso_6_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar_born.cu -o scalar_born_cuda_iso_8_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c elastic.cu -o elastic_cuda_iso_2_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c elastic.cu -o elastic_cuda_iso_4_float.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c elastic.cu -o elastic_cuda_iso_2_double.obj
nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c elastic.cu -o elastic_cuda_iso_4_double.obj
nvcc $CUDAFLAGS -shared scalar_born_cuda_iso_2_float.obj scalar_born_cuda_iso_4_float.obj scalar_born_cuda_iso_6_float.obj scalar_born_cuda_iso_8_float.obj scalar_born_cuda_iso_2_double.obj scalar_born_cuda_iso_4_double.obj scalar_born_cuda_iso_6_double.obj scalar_born_cuda_iso_8_double.obj scalar_cuda_iso_2_float.obj scalar_cuda_iso_4_float.obj scalar_cuda_iso_6_float.obj scalar_cuda_iso_8_float.obj scalar_cuda_iso_2_double.obj scalar_cuda_iso_4_double.obj scalar_cuda_iso_6_double.obj scalar_cuda_iso_8_double.obj elastic_cuda_iso_2_float.obj elastic_cuda_iso_4_float.obj elastic_cuda_iso_2_double.obj elastic_cuda_iso_4_double.obj -o libdeepwave_cuda_windows_x86_64.dll
rm *.obj
