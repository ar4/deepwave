cmake_minimum_required(VERSION 3.23)
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


# --- Compiler Feature Detection and Flags ---
# AVX2
include(CheckCSourceCompiles)
set(AVX2_TEST_CODE "
    #include <immintrin.h>
    int main() {
        __m256 vec = _mm256_set1_ps(42.0f);
        return 0;
    }")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    set(C_AVX2_FLAG "-mavx2")
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(C_AVX2_FLAG "/arch:AVX2")
endif()

if(C_AVX2_FLAG)
    set(CMAKE_REQUIRED_FLAGS "${C_AVX2_FLAG}")
    check_c_source_compiles("${AVX2_TEST_CODE}" HAVE_AVX2)
    unset(CMAKE_REQUIRED_FLAGS)
else()
    set(HAVE_AVX2 FALSE)
endif()

if(HAVE_AVX2)
    message(STATUS "AVX2 is supported.")
else()
    message(STATUS "AVX2 is not supported.")
endif()

# Release flags
if(CMAKE_BUILD_TYPE MATCHES Release)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        set(C_RELEASE_FLAGS "-Ofast")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast")
    elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
        set(C_RELEASE_FLAGS "/O2" "/fp:fast")
    endif()
endif()
# --- End Compiler Feature Detection and Flags ---

# --- Helper Macros for Object Libraries ---
macro(add_deepwave_cpu_object_library BASENAME NDIM ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_${NDIM}_${ACCURACY}_${DTYPE}_obj")
    add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
    list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    # Set PIC for shared library objects
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if(C_RELEASE_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_RELEASE_FLAGS})
    endif()

    if(HAVE_AVX2 AND C_AVX2_FLAG)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()

if(CUDAToolkit_FOUND)
    macro(add_deepwave_cuda_object_library BASENAME NDIM ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_cu_${NDIM}_${ACCURACY}_${DTYPE}_obj")
        add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES all)

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_RELEASE_OPTIONS})
        endif()
    endmacro()
endif()
# --- End Helper Macros ---


# Prepare lists that will collect the object files
set(DEEPWAVE_OBJECTS)
set(CPU_TARGETS)
set(NDIMS 1 2 3)
set(ACCURACIES 2 4 6 8)
set(DTYPES float double)

# --- CPU object libraries ---
# Simple compress is generic (not dependent on NDIM/ACCURACY/DTYPE macros)

foreach(NDIM ${NDIMS})
    foreach(DTYPE ${DTYPES})
        # storage_utils and simple_compress depend on NDIM and DTYPE
	add_deepwave_cpu_object_library(storage_utils ${NDIM} 2 ${DTYPE})
        add_deepwave_cpu_object_library(simple_compress ${NDIM} 2 ${DTYPE})

    	foreach(ACCURACY ${ACCURACIES})
		add_deepwave_cpu_object_library(scalar ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(scalar_born ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(acoustic ${NDIM} ${ACCURACY} ${DTYPE})
        endforeach()
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
        if(CMAKE_BUILD_TYPE MATCHES Release)
            set(CUDA_RELEASE_OPTIONS --use_fast_math -O3 --restrict)
        endif()
        if(NOT WIN32)
            # -fPIC is not a valid option for the MSVC compiler on Windows.
            list(APPEND CMAKE_CUDA_FLAGS -Xcompiler=-fPIC)
        endif()

        # Simple compress CUDA is generic

        foreach(NDIM ${NDIMS})
            foreach(DTYPE ${DTYPES})
                # storage_utils and simple_compress depend on NDIM and DTYPE
		add_deepwave_cuda_object_library(storage_utils ${NDIM} 2 ${DTYPE})
                add_deepwave_cuda_object_library(simple_compress ${NDIM} 2 ${DTYPE})

            	foreach(ACCURACY ${ACCURACIES})
			add_deepwave_cuda_object_library(scalar ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(scalar_born ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(acoustic ${NDIM} ${ACCURACY} ${DTYPE})
                endforeach()
            endforeach()
        endforeach()
endif()
# --- End CUDA object libraries ---

# --- Final Library Build ---
add_library(deepwave_C SHARED ${DEEPWAVE_OBJECTS})
if(WIN32)
    set_target_properties(deepwave_C PROPERTIES OUTPUT_NAME "libdeepwave_C")
endif()
set_target_properties(deepwave_C PROPERTIES
    C_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(OPENMP_CONFIGURED)
    target_link_libraries(deepwave_C PRIVATE Deepwave_OpenMP_Interface)
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
