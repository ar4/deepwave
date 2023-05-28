#!/bin/sh

set -e

CFLAGS="-Wall -Wextra -pedantic -fPIC -Ofast"
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_6_float.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_8_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_4_double.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_6_double.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_8_double.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_6_float.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_8_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_4_double.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_6_double.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_8_double.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c elastic.c -o elastic_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c elastic.c -o elastic_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c elastic.c -o elastic_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c elastic.c -o elastic_cpu_iso_4_double.o
clang $CFLAGS -dynamiclib scalar_born_cpu_iso_2_float.o scalar_born_cpu_iso_4_float.o scalar_born_cpu_iso_6_float.o scalar_born_cpu_iso_8_float.o scalar_born_cpu_iso_2_double.o scalar_born_cpu_iso_4_double.o scalar_born_cpu_iso_6_double.o scalar_born_cpu_iso_8_double.o scalar_cpu_iso_2_float.o scalar_cpu_iso_4_float.o scalar_cpu_iso_6_float.o scalar_cpu_iso_8_float.o scalar_cpu_iso_2_double.o scalar_cpu_iso_4_double.o scalar_cpu_iso_6_double.o scalar_cpu_iso_8_double.o elastic_cpu_iso_2_float.o elastic_cpu_iso_4_float.o elastic_cpu_iso_2_double.o elastic_cpu_iso_4_double.o -o libdeepwave_cpu_macos_x86_64.dylib
rm *.o
CFLAGS="-Wall -Wextra -pedantic -fPIC -Ofast -arch arm64"
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_6_float.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar.c -o scalar_cpu_iso_8_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_4_double.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_6_double.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar.c -o scalar_cpu_iso_8_double.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_6_float.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -c scalar_born.c -o scalar_born_cpu_iso_8_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_4_double.o
clang $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_6_double.o
clang $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -c scalar_born.c -o scalar_born_cpu_iso_8_double.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -c elastic.c -o elastic_cpu_iso_2_float.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -c elastic.c -o elastic_cpu_iso_4_float.o
clang $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -c elastic.c -o elastic_cpu_iso_2_double.o
clang $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -c elastic.c -o elastic_cpu_iso_4_double.o
clang $CFLAGS -shared scalar_born_cpu_iso_2_float.o scalar_born_cpu_iso_4_float.o scalar_born_cpu_iso_6_float.o scalar_born_cpu_iso_8_float.o scalar_born_cpu_iso_2_double.o scalar_born_cpu_iso_4_double.o scalar_born_cpu_iso_6_double.o scalar_born_cpu_iso_8_double.o scalar_cpu_iso_2_float.o scalar_cpu_iso_4_float.o scalar_cpu_iso_6_float.o scalar_cpu_iso_8_float.o scalar_cpu_iso_2_double.o scalar_cpu_iso_4_double.o scalar_cpu_iso_6_double.o scalar_cpu_iso_8_double.o elastic_cpu_iso_2_float.o elastic_cpu_iso_4_float.o elastic_cpu_iso_2_double.o elastic_cpu_iso_4_double.o -o libdeepwave_cpu_macos_arm64.dylib
rm *.o
