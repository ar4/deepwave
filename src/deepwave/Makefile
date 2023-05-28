CFLAGS = -Wall -Wextra -pedantic -std=c89 -fPIC -Ofast -mavx2 -fopenmp
CUDAFLAGS = --restrict --use_fast_math -O3 \
  -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_75,code=compute_75 \
  --compiler-options -fPIC
#CFLAGS = -fPIC -fopenmp -Ofast -mavx2

all: libdeepwave_cpu.so libdeepwave_cuda.so

libdeepwave_cpu.so: scalar_cpu_iso_2_float.o scalar_cpu_iso_4_float.o scalar_cpu_iso_6_float.o scalar_cpu_iso_8_float.o scalar_cpu_iso_2_double.o scalar_cpu_iso_4_double.o scalar_cpu_iso_6_double.o scalar_cpu_iso_8_double.o
	gcc $(CFLAGS) -shared $^ -o $@

scalar_cpu_iso_2_float.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=2 -DDW_DTYPE=float $^ -o $@

scalar_cpu_iso_4_float.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=4 -DDW_DTYPE=float $^ -o $@

scalar_cpu_iso_6_float.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=6 -DDW_DTYPE=float $^ -o $@

scalar_cpu_iso_8_float.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=8 -DDW_DTYPE=float $^ -o $@

scalar_cpu_iso_2_double.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=2 -DDW_DTYPE=double $^ -o $@

scalar_cpu_iso_4_double.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=4 -DDW_DTYPE=double $^ -o $@

scalar_cpu_iso_6_double.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=6 -DDW_DTYPE=double $^ -o $@

scalar_cpu_iso_8_double.o: scalar.c
	gcc $(CFLAGS) -c -DDW_ACCURACY=8 -DDW_DTYPE=double $^ -o $@

libdeepwave_cuda.so: scalar_cuda_iso_2_float.o scalar_cuda_iso_4_float.o scalar_cuda_iso_6_float.o scalar_cuda_iso_8_float.o scalar_cuda_iso_2_double.o scalar_cuda_iso_4_double.o scalar_cuda_iso_6_double.o scalar_cuda_iso_8_double.o
	nvcc $(CUDAFLAGS) -shared $^ -o $@

scalar_cuda_iso_2_float.o: scalar.c
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=2 -DDW_DTYPE=float $^ -o $@

scalar_cuda_iso_4_float.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=4 -DDW_DTYPE=float $^ -o $@

scalar_cuda_iso_6_float.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=6 -DDW_DTYPE=float $^ -o $@

scalar_cuda_iso_8_float.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=8 -DDW_DTYPE=float $^ -o $@

scalar_cuda_iso_2_double.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=2 -DDW_DTYPE=double $^ -o $@

scalar_cuda_iso_4_double.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=4 -DDW_DTYPE=double $^ -o $@

scalar_cuda_iso_6_double.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=6 -DDW_DTYPE=double $^ -o $@

scalar_cuda_iso_8_double.o: scalar.cu
	nvcc $(CUDAFLAGS) -c -DDW_ACCURACY=8 -DDW_DTYPE=double $^ -o $@

clean:
	rm *.o *.so

.PHONY: clean
