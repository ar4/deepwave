cmake_minimum_required(VERSION 3.21)
# If scikit-build-core sets SKBUILD_PROJECT_NAME / VERSION, prefer those.
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
else()
    project(deepwave LANGUAGES C)
endif()

# Detect and enable CUDA if present
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
else()
    message(WARNING "CUDA not found. Building without CUDA support.")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# --- OpenMP Configuration ---
add_library(Deepwave_OpenMP_Interface INTERFACE)
set(OPENMP_CONFIGURED FALSE)

if(WIN32)
    # On Windows, prioritize the Intel OpenMP library to avoid conflicts with PyTorch.
    set(INTEL_OMP_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/deepwave/libiomp5md.lib")
    if(EXISTS "${INTEL_OMP_LIB_PATH}")
        message(STATUS "Found Intel OpenMP library for Windows build.")
        # Use /openmp:experimental, link the Intel lib, and exclude the default MSVC lib.
        target_compile_options(Deepwave_OpenMP_Interface INTERFACE /openmp:experimental)
        target_link_libraries(Deepwave_OpenMP_Interface INTERFACE "${INTEL_OMP_LIB_PATH}")
        target_link_options(Deepwave_OpenMP_Interface INTERFACE "/nodefaultlib:vcomp")
        set(OPENMP_CONFIGURED TRUE)
    else()
        # Fallback for users building from source without the Intel lib.
        # Use standard MSVC OpenMP, but with the user-required experimental flag.
        message(STATUS "Intel OpenMP library not found. Falling back to standard MSVC OpenMP.")
        find_package(OpenMP QUIET)
        if(OpenMP_C_FOUND)
            target_link_libraries(Deepwave_OpenMP_Interface INTERFACE OpenMP::OpenMP_C)
            target_compile_options(Deepwave_OpenMP_Interface INTERFACE /openmp:experimental)
            set(OPENMP_CONFIGURED TRUE)
        endif()
    endif()
else()
    # For non-Windows, use the standard find_package
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(Deepwave_OpenMP_Interface INTERFACE OpenMP::OpenMP_C)
        set(OPENMP_CONFIGURED TRUE)
    endif()
endif()

if(OPENMP_CONFIGURED)
    message(STATUS "OpenMP enabled.")
else()
    message(STATUS "OpenMP not found or not configured.")
endif()
# --- End OpenMP Configuration ---


# AVX2 test
include(CheckCSourceCompiles)
set(AVX2_TEST_CODE "
    #include <immintrin.h>
    int main() {
        __m256 vec = _mm256_set1_ps(42.0f);
        return 0;
    }")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
else()
    if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    endif()
endif()

check_c_source_compiles("${AVX2_TEST_CODE}" HAVE_AVX2)
unset(CMAKE_REQUIRED_FLAGS)

if(HAVE_AVX2)
    message(STATUS "AVX2 is supported.")
else()
    message(STATUS "AVX2 is not supported.")
endif()

# Add warning flags for GCC/Clang
add_compile_options(
    $<$<C_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<C_COMPILER_ID:GNU,Clang>:-Wextra>
    $<$<C_COMPILER_ID:GNU,Clang>:-pedantic>
)

if(CMAKE_BUILD_TYPE MATCHES Release)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast")
    endif()
endif()

# Prepare lists that will collect the object files
set(DEEPWAVE_OBJECTS)
set(SCALAR_ACCURACIES 2 4 6 8)
set(ELASTIC_ACCURACIES 2 4)
set(DTYPES float double)

# --- CPU object libraries ---
# All CPU object libraries will link to our unified OpenMP target if it was configured.
set(CPU_TARGETS)

# scalar and scalar_born
foreach(ACCURACY ${SCALAR_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(scalar_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.c)
        target_compile_definitions(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_${ACCURACY}_${DTYPE}_obj>)
        list(APPEND CPU_TARGETS scalar_${ACCURACY}_${DTYPE}_obj)

        add_library(scalar_born_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.c)
        target_compile_definitions(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_${ACCURACY}_${DTYPE}_obj>)
        list(APPEND CPU_TARGETS scalar_born_${ACCURACY}_${DTYPE}_obj)

        if(CMAKE_BUILD_TYPE MATCHES Release)
            if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "Intel")
		target_compile_options(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE -Ofast -fPIC)
		target_compile_options(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE -Ofast -fPIC)
            elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
                target_compile_options(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE /O2 /fp:fast)
                target_compile_options(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE /O2 /fp:fast)
            endif()
        endif()

        if(HAVE_AVX2)
            if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "Intel")
                target_compile_options(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE -mavx2)
                target_compile_options(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE -mavx2)
            elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
                target_compile_options(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE /arch:AVX2)
                target_compile_options(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE /arch:AVX2)
            endif()
        endif()
    endforeach()
endforeach()

# elastic
foreach(ACCURACY ${ELASTIC_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(elastic_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.c)
        target_compile_definitions(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_${ACCURACY}_${DTYPE}_obj>)
        list(APPEND CPU_TARGETS elastic_${ACCURACY}_${DTYPE}_obj)

        if(CMAKE_BUILD_TYPE MATCHES Release)
            if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "Intel")
		target_compile_options(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE -Ofast -fPIC)
            elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
                target_compile_options(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE /O2 /fp:fast)
            endif()
        endif()

        if(HAVE_AVX2)
            if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "Intel")
                target_compile_options(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE -mavx2)
            elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
                target_compile_options(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE /arch:AVX2)
            endif()
        endif()
    endforeach()
endforeach()

if(OPENMP_CONFIGURED)
    # Link all CPU targets to our unified OpenMP interface library in one go.
    foreach(CPU_TARGET ${CPU_TARGETS})
        target_link_libraries(${CPU_TARGET} PRIVATE Deepwave_OpenMP_Interface)
    endforeach()
endif()
# --- End CPU object libraries ---


# --- CUDA object libraries ---
if(CUDAToolkit_FOUND)
    # scalar and scalar_born
    foreach(ACCURACY ${SCALAR_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(scalar_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.cu)
            target_compile_definitions(scalar_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_cu_${ACCURACY}_${DTYPE}_obj>)

            add_library(scalar_born_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.cu)
            target_compile_definitions(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_cu_${ACCURACY}_${DTYPE}_obj>)

            if(CMAKE_BUILD_TYPE MATCHES Release)
                target_compile_options(scalar_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3 --restrict --compiler-options -fPIC)
                target_compile_options(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3 --restrict --compiler-options -fPIC)
                set_target_properties(scalar_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
                set_target_properties(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
            endif()
        endforeach()
    endforeach()

    # elastic
    foreach(ACCURACY ${ELASTIC_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(elastic_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.cu)
            target_compile_definitions(elastic_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_cu_${ACCURACY}_${DTYPE}_obj>)

            if(CMAKE_BUILD_TYPE MATCHES Release)
                target_compile_options(elastic_cu_${ACCURACY}_${DTYPE}_obj PRIVATE --use_fast_math -O3 --restrict --compiler-options -fPIC)
                set_target_properties(elastic_cu_${ACCURACY}_${DTYPE}_obj PROPERTIES CUDA_ARCHITECTURES all-major)
            endif()
        endforeach()
    endforeach()
endif()
# --- End CUDA object libraries ---


# --- Final Library Build ---
add_library(deepwave_C SHARED ${DEEPWAVE_OBJECTS})
set_target_properties(deepwave_C PROPERTIES
    C_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(OPENMP_CONFIGURED)
    target_link_libraries(deepwave_C PRIVATE Deepwave_OpenMP_Interface)
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(deepwave_C PRIVATE CUDA::cudart)
endif()

if(HAVE_AVX2)
    target_compile_definitions(deepwave_C PRIVATE HAVE_AVX2)
endif()

# Install
install(TARGETS deepwave_C
    LIBRARY DESTINATION deepwave
    ARCHIVE DESTINATION deepwave
    RUNTIME DESTINATION deepwave
)
