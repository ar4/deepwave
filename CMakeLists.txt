cmake_minimum_required(VERSION 3.18)
# If scikit-build-core sets SKBUILD_PROJECT_NAME / VERSION, prefer those.
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
else()
    project(deepwave LANGUAGES C)
endif()

# Detect and enable CUDA if present
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
else()
    message(WARNING "CUDA not found. Building without CUDA support.")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find OpenMP quietly (optional)
find_package(OpenMP QUIET)

# AVX2 test: use CMAKE_REQUIRED_FLAGS (safer than clobbering CMAKE_C_FLAGS)
include(CheckCSourceCompiles)
set(AVX2_TEST_CODE "
    #include <immintrin.h>
    int main() {
        __m256 vec = _mm256_set1_ps(42.0f);
        return 0;
    }")

# Prepare flags for the test based on compiler
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif()

check_c_source_compiles("${AVX2_TEST_CODE}" HAVE_AVX2)

# Clear any test-only flags
unset(CMAKE_REQUIRED_FLAGS)

if(HAVE_AVX2)
    message(STATUS "AVX2 is supported.")
else()
    message(STATUS "AVX2 is not supported.")
endif()

# Compiler optimization flags — prefer target/compile-options but keep simple here
if(CMAKE_BUILD_TYPE MATCHES Release)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        add_compile_options(-Ofast)
        if(HAVE_AVX2)
            add_compile_options(-mavx2)
        endif()
        # Ensure -Ofast in link step
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast")
    elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/O2 /fp:fast)
        if(HAVE_AVX2)
            add_compile_options(/arch:AVX2)
        endif()
    endif()
endif()

# Prepare lists that will collect the object files
set(DEEPWAVE_OBJECTS)
set(SCALAR_ACCURACIES 2 4 6 8)
set(ELASTIC_ACCURACIES 2 4)
set(DTYPES float double)

# CPU object libraries (scalar and scalar_born)
foreach(ACCURACY ${SCALAR_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(scalar_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.c)
        target_compile_definitions(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_${ACCURACY}_${DTYPE}_obj>)

        add_library(scalar_born_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.c)
        target_compile_definitions(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_${ACCURACY}_${DTYPE}_obj>)

        if(OpenMP_C_FOUND)
            target_link_libraries(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE OpenMP::OpenMP_C)
            target_link_libraries(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE OpenMP::OpenMP_C)
        endif()
    endforeach()
endforeach()

# CPU object libraries (elastic)
foreach(ACCURACY ${ELASTIC_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(elastic_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.c)
        target_compile_definitions(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_${ACCURACY}_${DTYPE}_obj>)

        if(OpenMP_C_FOUND)
            target_link_libraries(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE OpenMP::OpenMP_C)
        endif()
    endforeach()
endforeach()

# CUDA object libraries (if CUDA available)
if(CMAKE_CUDA_COMPILER)
    foreach(ACCURACY ${SCALAR_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(scalar_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.cu)
            target_compile_definitions(scalar_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_cu_${ACCURACY}_${DTYPE}_obj>)

            add_library(scalar_born_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.cu)
            target_compile_definitions(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_cu_${ACCURACY}_${DTYPE}_obj>)

            if(CMAKE_BUILD_TYPE MATCHES Release)
                target_compile_options(scalar_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3)
                target_compile_options(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3)
                set_target_properties(scalar_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
                set_target_properties(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
            endif()
        endforeach()
    endforeach()

    foreach(ACCURACY ${ELASTIC_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(elastic_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.cu)
            target_compile_definitions(elastic_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_cu_${ACCURACY}_${DTYPE}_obj>)

            if(CMAKE_BUILD_TYPE MATCHES Release)
                target_compile_options(elastic_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3)
                set_target_properties(elastic_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
            endif()
        endforeach()
    endforeach()
endif()

# Build shared lib from object files (this is what Python/ctypes will load)
add_library(deepwave_C SHARED ${DEEPWAVE_OBJECTS})
set_target_properties(deepwave_C PROPERTIES
    C_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Link libs
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)  # prefer CUDA::cudart via the toolkit package
    target_link_libraries(deepwave_C PRIVATE CUDA::cudart)
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(deepwave_C PRIVATE OpenMP::OpenMP_C)
endif()

if(HAVE_AVX2)
    target_compile_definitions(deepwave_C PRIVATE HAVE_AVX2)
endif()

# Install
# If scikit-build-core drives the build, install into the package name — scikit-build-core
# will map that to platlib/site-packages for wheels/editable redirects.
if(DEFINED SKBUILD)
    message(STATUS "scikit-build-core detected; installing into package '${SKBUILD_PROJECT_NAME}'.")
    install(TARGETS deepwave_C
        LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
        ARCHIVE DESTINATION ${SKBUILD_PROJECT_NAME}
        RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
    )
else()
    message(WARNING "scikit-build-core not detected; using fallback install directories under CMAKE_INSTALL_PREFIX.")
    install(TARGETS deepwave_C
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
endif()
