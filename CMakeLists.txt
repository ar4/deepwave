cmake_minimum_required(VERSION 3.23)
# If scikit-build-core sets SKBUILD_PROJECT_NAME / VERSION, prefer those.
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
else()
    project(deepwave LANGUAGES C)
endif()

# DW_SANITIZE=ON builds the CPU object libraries with AddressSanitizer +
# UndefinedBehaviorSanitizer instead of the normal release flags (see the
# "Release flags" section below for the actual flag swap). ASan/UBSan
# instrumentation of CPU code and CUDA device code don't mix cleanly (the
# sanitizer runtime has no visibility into device-side memory, and nvcc's host
# pass would need matching instrumentation), so a sanitizer build always forces
# a CPU-only build.
option(DW_SANITIZE "Build the CPU object libraries with AddressSanitizer + UndefinedBehaviorSanitizer instead of normal release optimisation (GCC/Clang only; forces DW_SKIP_CUDA=ON)" OFF)

# Detect and enable CUDA if present. DW_SKIP_CUDA=ON forces a CPU-only build
# (used for fast kernel iteration; the CUDA translation units, which cannot be
# unit-tested on a CPU-only host, are skipped -- do a full build before release).
option(DW_SKIP_CUDA "Skip building the CUDA object libraries (CPU-only build)" OFF)

# DW_FAST_MATH controls whether the CPU/CUDA kernels build with
# -Ofast/--use_fast_math (IEEE-noncompliant NaN/Inf handling, in exchange
# for extra speed) or precise -O3. Reassociation is excluded from the
# fast-math set either way; see the release-flag block below. Default ON:
# a same-commit A/B (branch HEAD built both ways, nothing else varying)
# measured fast-math's speed advantage directly rather than inferring it.
# On a many-shots/small-model CPU sweep -- the regime where per-step
# spatial-domain OpenMP parallelism is most exposed to per-thread numerical
# shortcuts -- precise -O3 cost a median 25.6% more wall-clock time than
# fast-math (worst case +36.2%); the meso-tier CPU and GPU benchmarks
# (all compiled propagators) showed smaller but still positive medians in
# the same direction. The accuracy-sensitive test suite (convergence +
# gradcheck) passes identically under both flag sets, so there is no
# correctness finding forcing precise math by default; set DW_FAST_MATH=OFF
# to build IEEE-precise -O3 kernels instead (see docs/accuracy.rst).
option(DW_FAST_MATH "Build the CPU/CUDA kernels with -Ofast/--use_fast_math instead of precise -O3 (GNU/Clang/Intel only)" ON)
if(DW_SANITIZE AND NOT DW_SKIP_CUDA)
    message(WARNING "DW_SANITIZE=ON forces a CPU-only build (ASan/UBSan CPU instrumentation and CUDA device code do not mix); enabling DW_SKIP_CUDA.")
    set(DW_SKIP_CUDA ON)
endif()
if(DW_SKIP_CUDA)
    message(STATUS "DW_SKIP_CUDA=ON: building without CUDA support.")
else()
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CXX)
        enable_language(CUDA)
    else()
        message(WARNING "CUDA not found. Building without CUDA support.")
    endif()
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Export compile_commands.json for editor/clang-tooling integration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- ccache ---
# If ccache is available on the system, use it to cache compiler invocations
# and speed up rebuilds. This is purely an optimisation: when ccache is not
# found, the build proceeds exactly as before with no hard failure.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
else()
    message(STATUS "ccache not found: compiler invocations will not be cached.")
endif()
# --- End ccache ---

# --- OpenMP Configuration ---
add_library(Deepwave_OpenMP_Interface INTERFACE)
set(OPENMP_CONFIGURED FALSE)

if(WIN32)
    # On Windows, prioritise the Intel OpenMP library to avoid conflicts with PyTorch.
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
    message(STATUS "Compiler accepts AVX2 code.")
else()
    message(STATUS "Compiler does not accept AVX2 code.")
endif()

# --- x86 runtime ISA dispatch (GCC/Clang function multi-versioning) ---
# Baking -mavx2 into every kernel translation unit makes the wheel abort with a
# bare "Illegal instruction (core dumped)" on pre-AVX2 x86 CPUs. Runtime
# dispatch is available on x86 with a supporting compiler: compile hot kernel
# TUs at the baseline ISA and annotate their
# exported forward/backward functions with
#   __attribute__((target_clones("default,avx2,avx512f")))
# (macro DW_TARGET_CLONES, enabled by DW_X86_MULTIVERSION -- see the kernel .c
# files). GCC/Clang then emit one clone per ISA and an ifunc resolver that picks
# the best the running CPU supports (via CPUID). This keeps the wheel a single
# .so linking only libc + libgomp, and does NOT change any exported symbol name
# or signature, so the ctypes ABI (tests/test_abi_*) is untouched. The stencil
# loops are inlined into those exported functions, so cloning the function
# clones the vectorised hot path. Non-kernel helper TUs (storage_utils,
# simple_compress) and the cpu_features probe stay at the baseline ISA (SSE2 on
# x86-64), which is universally executable, so the entire .so is SIGILL-free on
# any x86-64 CPU. target_clones is supported for C since GCC 6 and by Clang; the
# manylinux_2_28 images used by cibuildwheel ship gcc-toolset-13, so the
# compiler support this needs is present in release builds too. Support is
# still always PROBED (below), but NOT enabled by default -- see the
# DW_X86_MULTIVERSION_OPT_IN option below for the measured size-vs-benefit
# evidence behind that default.
set(DW_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64|i[3-6]86")
    set(DW_X86 TRUE)
endif()

if(DW_X86 AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # The kernels annotate *external* (exported) functions, so probe with an
    # external function too -- some compilers restrict multiversioning on static
    # functions, and we want the probe to match real usage exactly.
    #
    # Do NOT pre-set HAVE_TARGET_CLONES before this call: check_c_source_compiles
    # (via cmake_check_source_compiles) only runs its try_compile probe when the
    # result variable is not yet DEFINED in this scope. A `set(HAVE_TARGET_CLONES
    # FALSE)` here would make the variable DEFINED (any set(), even to a false
    # value, counts), so the probe would be silently skipped on every configure,
    # on every platform -- which is exactly what happened before this fix: the
    # variable simply falls through to the plain `if(HAVE_TARGET_CLONES)` below
    # as undefined (falsy) when this outer `if()` isn't entered, with no probe
    # needed in that case anyway.
    set(TARGET_CLONES_TEST_CODE "
        __attribute__((target_clones(\"default,avx2,avx512f\")))
        int dw_probe(int x) { return x + 1; }
        int main(void) { return dw_probe(0) - 1; }")
    check_c_source_compiles("${TARGET_CLONES_TEST_CODE}" HAVE_TARGET_CLONES)
endif()


# Compiler support for AVX2 does not imply that the wheel's destination CPU
# supports it. The portable x86-64 baseline is the default. Builders targeting
# a known AVX2 fleet may opt into a smaller fixed-ISA build, while runtime
# dispatch remains available for builds that can afford its larger code size.
option(DW_X86_MULTIVERSION_OPT_IN "Actually use target_clones runtime ISA dispatch on the compiled kernels when the compiler supports it, instead of only detecting support. OFF by default: the kernels are memory-bandwidth-bound, so the measured speed-up did not pay for the code-size increase" OFF)
option(DW_FIXED_AVX2_OPT_IN "Build CPU kernels requiring AVX2; only for known AVX2 target CPUs" OFF)
if(DW_X86_MULTIVERSION_OPT_IN AND DW_FIXED_AVX2_OPT_IN)
    message(FATAL_ERROR "Choose only one of DW_X86_MULTIVERSION_OPT_IN and DW_FIXED_AVX2_OPT_IN")
endif()
if(DW_FIXED_AVX2_OPT_IN AND NOT HAVE_AVX2)
    message(FATAL_ERROR "DW_FIXED_AVX2_OPT_IN requires compiler AVX2 support")
endif()
set(DW_ENABLE_X86_MULTIVERSION FALSE)
if(DW_X86_MULTIVERSION_OPT_IN AND NOT HAVE_TARGET_CLONES)
    message(FATAL_ERROR "DW_X86_MULTIVERSION_OPT_IN requires target_clones support")
elseif(DW_X86_MULTIVERSION_OPT_IN)
    set(DW_ENABLE_X86_MULTIVERSION TRUE)
endif()

if(HAVE_TARGET_CLONES)
    if(DW_ENABLE_X86_MULTIVERSION)
        message(STATUS "x86 runtime ISA dispatch enabled (target_clones on kernels).")
    else()
        message(STATUS "x86 runtime ISA dispatch available but not enabled.")
    endif()
else()
    message(STATUS "x86 runtime ISA dispatch not used (non-x86 or unsupported compiler).")
endif()

# The absolute ISA baseline the compiled kernels require. Empty when kernels use
# runtime dispatch (baseline clone runs anywhere) or when no ISA is forced;
# "avx2" only for a fixed-ISA build that bakes -mavx2 into the kernels. The
# cpu_features probe reports this so the Python layer can raise a clear error
# instead of letting the process SIGILL.
if(DW_ENABLE_X86_MULTIVERSION)
    set(DW_KERNEL_BASELINE_ISA "")
elseif(DW_FIXED_AVX2_OPT_IN AND DW_X86)
    set(DW_KERNEL_BASELINE_ISA "avx2")
else()
    set(DW_KERNEL_BASELINE_ISA "")
endif()

# Release flags
if(DW_SANITIZE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # Use -O1 by default; Clang's large 3D sanitizer objects use -O0
        # below to bound compiler memory. -fno-omit-frame-pointer, rather
        # than the optimisation level, keeps sanitizer stack traces walkable.
        # -g adds debug info so those traces resolve to source lines.
        set(C_RELEASE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address,undefined")
    else()
        message(WARNING "DW_SANITIZE=ON requires GCC or Clang as the C compiler; ignoring on ${CMAKE_C_COMPILER_ID}.")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    # Reassociation is excluded from the fast-math set because it makes
    # results depend on the number of threads. The kernel loops are
    # `omp for simd` with no `schedule` clause, so the runtime splits the
    # iteration space into one chunk per thread; reassociation then lets the
    # compiler compute a chunk's vectorised body and its scalar remainder in
    # different orders, and moving the chunk boundaries moves which cells
    # take which path. The loops are elementwise rather than reductions, so
    # forbidding reassociation costs little. The flag removes the main cause
    # but does not on its own guarantee that a vectorised cell performs the
    # scalar cell's operations in the scalar cell's order -- a minimal
    # `for simd collapse(2)` stencil built with it still diverges across
    # thread counts. The invariance is therefore a measured property of these
    # kernel bodies, not one the flag enforces: tests/test_determinism.py
    # checks it across thread counts.
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        if(DW_FAST_MATH)
            if(CMAKE_C_COMPILER_ID MATCHES "Clang")
                # Clang deprecates -Ofast. Spell out its optimisation policy
                # rather than suppressing the diagnostic; keep reassociation
                # disabled for thread-count invariance.
                set(C_RELEASE_FLAGS "-O3" "-ffast-math" "-fstrict-aliasing" "-fno-associative-math")
            else()
                set(C_RELEASE_FLAGS "-Ofast" "-fno-associative-math")
            endif()
        else()
            set(C_RELEASE_FLAGS "-O3" "-fno-math-errno")
        endif()
    elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
        # MSVC's only reassociation control is /fp:, whose `fast` level
        # reassociates, so the fast-math build keeps the default /fp:precise.
        set(C_RELEASE_FLAGS "/O2")
    endif()
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND DW_FIXED_AVX2_OPT_IN)
        # Keep the main AVX2 loop and scalar remainder, without also emitting
        # a smaller-width SIMD remainder for each specialised stencil region.
        # These extra bodies substantially inflate the packaged CPU kernels.
        include(CheckCCompilerFlag)
        check_c_compiler_flag("--param=vect-epilogues-nomask=0" DW_HAVE_NO_VECTOR_EPILOGUES)
        if(DW_HAVE_NO_VECTOR_EPILOGUES)
            list(APPEND C_RELEASE_FLAGS "--param=vect-epilogues-nomask=0")
        endif()
    endif()
endif()
# --- End Compiler Feature Detection and Flags ---

# --- Warning flags for the kernel object libraries ---
# These warning options apply to GCC/Clang; the Windows release workflow
# uses MSVC and retains its own compiler defaults.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # -pedantic is CPU-only: this project ships Windows wheels built with
    # MSVC (see .github/workflows/build.yml's windows-2022 matrix leg), which
    # has no equivalent of GNU/Clang language extensions, so catching
    # non-standard-C usage here (while we have gcc/clang available) protects
    # that build even though it cannot be run on this box. In practice
    # -pedantic adds zero findings beyond plain -Wall -Wextra on this
    # codebase (verified on both gcc and clang).
    #
    # -Wno-unused-parameter: every NDIM/ACCURACY/DTYPE/M-PML/custom-FD/GSLS/
    # PM variant of a kernel driver function shares one uniform parameter
    # list across all object-library instantiations (see the
    # add_deepwave_cpu_*_object_library macros below), so a given variant
    # routinely doesn't touch a subset of its own parameters (e.g. an
    # NDIM==1 build's unused y/x-axis arguments, or a gradient-disabled
    # build's unused store buffers) by design, not by omission. This is the
    # single largest warning class this codebase produces under -Wall
    # -Wextra (this class alone was ~19900 instances across the full CPU
    # object-library matrix); forcing every such parameter to be referenced
    # would mean scattering per-site (void)x; casts through the kernel
    # sources for a non-bug, purely structural pattern, which is exactly
    # what this suppression avoids.
    #
    # -Wno-unused-function: the same shared kernel headers define a family of
    # static per-axis/per-variant inline helpers (e.g. the boundary-fill
    # helpers behind elastic.c's/elastic_born.c's *_ADJOINT macros); a given
    # NDIM/ACCURACY/DTYPE/variant translation unit only calls the subset its
    # own preprocessor branch needs, leaving the rest unused in that TU by
    # the same cross-variant design as the unused-parameter case above
    # (observed on the Clang build; GCC did not report this class on this
    # codebase).
    #
    # -Wno-pass-failed (Clang-only diagnostic, silently accepted by GCC): the
    # kernel loops are annotated with `#pragma omp simd collapse(DW_NDIM)`
    # throughout (a vectorisation HINT to the compiler, not a correctness
    # requirement -- OpenMP still executes the loop correctly either way).
    # Clang treats a failure to satisfy an explicitly-requested transform as
    # a build warning ("loop not vectorized"), unlike GCC, which does not
    # warn about this at all; it is a pure optimiser-feedback diagnostic
    # about the multi-dimensional (NDIM>=2) collapsed loops specifically,
    # not a functional problem, and fires on essentially every kernel hot
    # loop given how pervasive the pragma is.
    set(C_WARNING_FLAGS
        -Wall
        -Wextra
        -pedantic
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-pass-failed
    )
endif()
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    # Check nvcc's frontend independently of its host compiler. Unused local
    # variables remain visible: declarations should match the active variant.
    # Missing launch bounds are not enabled: they are an optional tuning hint,
    # not a correctness requirement, and need GPU measurements before use.
    set(CUDA_WARNING_FLAGS
        --Wreorder
        --Wdefault-stream-launch
        --Wext-lambda-captures-this
        --Werror=all-warnings
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Shared cell-function signatures and header helper families include
        # parameters/functions unused by individual variants, as on the CPU.
        # These GNU-style options must not be forwarded to MSVC on Windows.
        list(APPEND CUDA_WARNING_FLAGS
            -Xcompiler=-Wall
            -Xcompiler=-Wextra
            -Xcompiler=-Werror
            -Xcompiler=-Wno-unused-parameter
            -Xcompiler=-Wno-unused-function
        )
    endif()
endif()
# --- End warning flags ---

# The 3D staggered kernels account for most of the CPU code-size reduction.
# Matched GCC/AVX2 forward/adjoint timings support this policy there. The
# viscoacoustic adjoint also needs this policy in 1D/2D: GCC 13's smaller-width
# vector remainder changes its arithmetic with the OpenMP chunk boundaries.
# Other regular grids, elastic Born and smaller dimensional kernels retain
# normal vector tails to avoid the measured slowdowns on those workloads.
function(configure_deepwave_cpu_release_flags TARGET_NAME BASENAME NDIM)
    set(_CPU_RELEASE_FLAGS ${C_RELEASE_FLAGS})
    if(DW_SANITIZE AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND NDIM EQUAL 3)
        # Optimising these instrumented stencils can exceed 10 GiB for a
        # single object. Keep both sanitizers and debug information, while
        # avoiding that optimiser expansion in diagnostic builds.
        list(REMOVE_ITEM _CPU_RELEASE_FLAGS "-O1")
        list(APPEND _CPU_RELEASE_FLAGS "-O0")
    endif()
    if(NOT BASENAME STREQUAL "viscoacoustic" AND
       (NOT NDIM EQUAL 3 OR BASENAME MATCHES "^(scalar|scalar_born|elastic_born)$"))
        list(REMOVE_ITEM _CPU_RELEASE_FLAGS "--param=vect-epilogues-nomask=0")
    endif()
    # Compression must preserve subnormals even when long double has only
    # double's range (Apple Silicon and MSVC). Fast-math can turn its bounded
    # tiny-value division into an overflowing reciprocal. Keep this small
    # helper precise without changing the propagator kernel optimisation.
    if(BASENAME STREQUAL "simple_compress" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        list(APPEND _CPU_RELEASE_FLAGS "-fno-fast-math")
    endif()
    if(_CPU_RELEASE_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${_CPU_RELEASE_FLAGS})
    endif()
endfunction()

# --- Helper Macros for Object Libraries ---
macro(add_deepwave_cpu_object_library BASENAME NDIM ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_${NDIM}_${ACCURACY}_${DTYPE}_obj")
    add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
    if(DW_FAST_MATH)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_FAST_MATH_BUILD)
    endif()
    list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    # Set PIC for shared library objects
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    configure_deepwave_cpu_release_flags(${TARGET_NAME} ${BASENAME} ${NDIM})
    if(C_WARNING_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_WARNING_FLAGS})
    endif()

    # Is this a kernel TU whose exported forward/backward carry the
    # target_clones annotation? (storage_utils / simple_compress are generic
    # helpers with no such annotation.)
    set(_IS_KERNEL_TU FALSE)
    # NB: quote-expand -- inside a macro() the parameter is a string
    # substitution, not a real variable, so bare BASENAME would test the literal
    # text "BASENAME".
    if("${BASENAME}" MATCHES "^(scalar|scalar_born|acoustic|acoustic_born|acoustic_vti|viscoacoustic|elastic|elastic_born|elastic_vti)$")
        set(_IS_KERNEL_TU TRUE)
    endif()

    if(DW_ENABLE_X86_MULTIVERSION AND _IS_KERNEL_TU)
        # Runtime ISA dispatch: compile at the baseline ISA (NO -mavx2 -- that
        # would make even the "default" clone AVX2 and defeat the purpose) and
        # let DW_TARGET_CLONES emit the per-ISA clones.
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_X86_MULTIVERSION)
    elseif(DW_FIXED_AVX2_OPT_IN AND C_AVX2_FLAG)
        # Explicit fixed-ISA build for a known AVX2 target CPU.
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()

# Multiaxial-PML (M-PML) build variant of a CPU kernel TU: identical to
# add_deepwave_cpu_object_library but compiles the same source with -DDW_MPML,
# yielding the additional *_mpml exported symbols. Used only for elastic
# (ndim >= 2). The default (non-mpml) TUs above are completely untouched.
macro(add_deepwave_cpu_mpml_object_library BASENAME NDIM ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_mpml_${NDIM}_${ACCURACY}_${DTYPE}_obj")
    add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu DW_MPML)
    if(DW_FAST_MATH)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_FAST_MATH_BUILD)
    endif()
    list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    configure_deepwave_cpu_release_flags(${TARGET_NAME} ${BASENAME} ${NDIM})
    if(C_WARNING_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_WARNING_FLAGS})
    endif()

    # The elastic kernel TU carries the target_clones annotation on its exported
    # forward/backward functions, so it gets the same ISA-dispatch treatment as
    # the default variant.
    if(DW_ENABLE_X86_MULTIVERSION)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_X86_MULTIVERSION)
    elseif(DW_FIXED_AVX2_OPT_IN AND C_AVX2_FLAG)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()


# Multi-mechanism GSLS (DW_GSLS) build variant of a CPU kernel TU: identical
# to add_deepwave_cpu_object_library but compiles the same source with
# -DDW_GSLS, yielding the additional *_gsls exported symbols (one trailing
# int64 n_mechanisms parameter; see viscoacoustic.c). Used only for
# viscoacoustic. Mirrors add_deepwave_cpu_mpml_object_library.
macro(add_deepwave_cpu_gsls_object_library BASENAME NDIM ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_gsls_${NDIM}_${ACCURACY}_${DTYPE}_obj")
    add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu DW_GSLS)
    if(DW_FAST_MATH)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_FAST_MATH_BUILD)
    endif()
    list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    configure_deepwave_cpu_release_flags(${TARGET_NAME} ${BASENAME} ${NDIM})
    if(C_WARNING_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_WARNING_FLAGS})
    endif()

    # The viscoacoustic kernel TU carries the target_clones annotation on its
    # exported forward/backward functions, so it gets the same ISA-dispatch
    # treatment as the default variant.
    if(DW_ENABLE_X86_MULTIVERSION)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_X86_MULTIVERSION)
    elseif(DW_FIXED_AVX2_OPT_IN AND C_AVX2_FLAG)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()

# Parameter-modified free-surface (DW_PM) build variant of a CPU kernel TU:
# yields the additional *_pm exported symbols (trailing per-component surface-
# stiffness arrays and their gradients; see elastic.c). Used only for elastic,
# 2-D only. Mirrors add_deepwave_cpu_gsls_object_library.
macro(add_deepwave_cpu_pm_object_library BASENAME NDIM ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_pm_${NDIM}_${ACCURACY}_${DTYPE}_obj")
    add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu DW_PM)
    if(DW_FAST_MATH)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_FAST_MATH_BUILD)
    endif()
    list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    configure_deepwave_cpu_release_flags(${TARGET_NAME} ${BASENAME} ${NDIM})
    if(C_WARNING_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_WARNING_FLAGS})
    endif()

    # The elastic kernel TU carries the target_clones annotation on its
    # exported forward/backward functions, so it gets the same ISA-dispatch
    # treatment as the default variant.
    if(DW_ENABLE_X86_MULTIVERSION)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_X86_MULTIVERSION)
    elseif(DW_FIXED_AVX2_OPT_IN AND C_AVX2_FLAG)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()

if(CUDAToolkit_FOUND)
    # CUDA arch fatbin diet. "all" (CMake's built-in "every
    # architecture this CMake/CUDA release knows about" preset) bakes SASS
    # for architectures no supported PyTorch build can ever dispatch a
    # kernel to in the first place: PyTorch's own official wheels only ship
    # kernels for a curated per-CUDA-version subset, and a user must already
    # have a matching PyTorch build to import Deepwave's compiled extension
    # at all. Match that precedent instead of exhaustive coverage.
    #
    # The specific architectures a compiler even accepts change across CUDA
    # releases -- this project's release CI (.github/workflows/build.yml)
    # is pinned to CUDA 13.3, but a dev box may have a newer toolkit (CUDA
    # 13.x dropped sm_50/60/70 entirely: `nvcc --list-gpu-arch` on a CUDA
    # 13.3 install starts at compute_75) -- so the curated list is chosen
    # per detected CUDA major version rather than hardcoded once. The
    # oldest architecture each bucket's toolchain actually supports is also
    # kept as "-virtual" (PTX only, no SASS), so a GPU this curated list
    # doesn't have exact SASS for -- whether that's the toolchain's own
    # floor or anything newer than the curated real entries -- can still
    # JIT rather than hard-failing outright.
    #
    # A caller may pass -DDW_CUDA_ARCHITECTURES=... on the cmake command
    # line to override this per-version default outright.
    if(NOT DEFINED DW_CUDA_ARCHITECTURES)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
            # Not independently size/perf-measured on this toolchain (no
            # CUDA 12.4 environment available); derived by applying the same
            # Ampere+Ada-SASS-plus-oldest-virtual-JIT-fallback policy the
            # CUDA>=13 branch below was measured against. Ampere (sm_86) and
            # Ada (sm_89) SASS cover the great majority of currently-deployed
            # CUDA GPUs; Maxwell (sm_50) -- the oldest architecture this
            # toolchain branch supports (matching PyTorch's own cu121/cu124
            # wheel precedent of Maxwell through Hopper) -- is kept as
            # PTX-only ("-virtual") so anything this list doesn't have exact
            # SASS for, from Maxwell itself up through Hopper, still works
            # via forward-compatible PTX JIT on first use.
            set(DW_CUDA_ARCHITECTURES
                "86-real;89-real;50-virtual")
        else()
            # Ampere (sm_86) and Ada (sm_89) SASS cover the great majority of
            # currently-deployed CUDA GPUs -- consumer RTX 30/40 series,
            # A-series/L-series datacentre cards -- kept as real (no JIT
            # latency). Turing (sm_75), the oldest architecture CUDA 13's
            # toolchain accepts at all, is kept as PTX-only ("-virtual") so
            # any GPU this list doesn't have exact SASS for -- older Turing
            # cards, or anything newer than Ada (Hopper, Blackwell) -- still
            # can use forward-compatible PTX JIT on first use when the
            # installed driver and PyTorch support that device and PTX.
            # Restricting SASS to these two architectures reduces the CUDA
            # payload; the complete repaired wheel must still pass its size
            # budget because compiler choices also affect CPU code size.
            set(DW_CUDA_ARCHITECTURES
                "86-real;89-real;75-virtual")
        endif()
    endif()

    macro(add_deepwave_cuda_mpml_object_library BASENAME NDIM ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_mpml_cu_${NDIM}_${ACCURACY}_${DTYPE}_obj")
        add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda DW_MPML)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${DW_CUDA_ARCHITECTURES}")

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_RELEASE_OPTIONS})
        endif()
        if(CUDA_WARNING_FLAGS)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_WARNING_FLAGS})
        endif()
    endmacro()

    macro(add_deepwave_cuda_object_library BASENAME NDIM ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_cu_${NDIM}_${ACCURACY}_${DTYPE}_obj")
        add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${DW_CUDA_ARCHITECTURES}")

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_RELEASE_OPTIONS})
        endif()
        if(CUDA_WARNING_FLAGS)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_WARNING_FLAGS})
        endif()
    endmacro()

    # Multi-mechanism GSLS (DW_GSLS) CUDA build variant. Mirrors
    # add_deepwave_cpu_gsls_object_library / the CUDA -DDW_MPML macro
    # above; used only for viscoacoustic.
    macro(add_deepwave_cuda_gsls_object_library BASENAME NDIM ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_gsls_cu_${NDIM}_${ACCURACY}_${DTYPE}_obj")
        add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda DW_GSLS)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${DW_CUDA_ARCHITECTURES}")

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_RELEASE_OPTIONS})
        endif()
        if(CUDA_WARNING_FLAGS)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_WARNING_FLAGS})
        endif()
    endmacro()

    # Parameter-modified free-surface (DW_PM) build variant of a CUDA kernel
    # TU: yields the additional *_pm exported symbols (trailing per-component
    # surface-stiffness arrays and their gradients; see elastic.cu). Used only
    # for elastic, 2-D only. Mirrors add_deepwave_cuda_gsls_object_library.
    macro(add_deepwave_cuda_pm_object_library BASENAME NDIM ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_pm_cu_${NDIM}_${ACCURACY}_${DTYPE}_obj")
        add_library(${TARGET_NAME} OBJECT src/deepwave/${BASENAME}.cu)
        target_compile_definitions(${TARGET_NAME} PRIVATE DW_NDIM=${NDIM} DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda DW_PM)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${DW_CUDA_ARCHITECTURES}")

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_RELEASE_OPTIONS})
        endif()
        if(CUDA_WARNING_FLAGS)
            target_compile_options(${TARGET_NAME} PRIVATE ${CUDA_WARNING_FLAGS})
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
				# Parameter-modified free-surface variant (Cao and Chen 2018):
				# 2-D only (matches the eager scope), so gated with an exact
				# NDIM == 2 guard rather than the NDIM >= 2 M-PML/VTI guard.
				if(NDIM EQUAL 2)
					add_deepwave_cpu_pm_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
				endif()
            	add_deepwave_cpu_object_library(acoustic ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(acoustic_born ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(elastic_born ${NDIM} ${ACCURACY} ${DTYPE})
            	add_deepwave_cpu_object_library(viscoacoustic ${NDIM} ${ACCURACY} ${DTYPE})
					# Multi-mechanism GSLS (DW_GSLS) variant: viscoacoustic only,
					# every ndim viscoacoustic itself supports (1-3), built
					# alongside the default variant (never combined with it on
					# one TU -- see the #error guard in viscoacoustic.c/.cu).
					add_deepwave_cpu_gsls_object_library(viscoacoustic ${NDIM} ${ACCURACY} ${DTYPE})
                # Elastic multiaxial-PML (M-PML) variant: only ndim >= 2 (the
                # cross-axis damping is meaningless in 1D). Elastic VTI is also
                # 2-D/3-D only (no 1-D anisotropy), with the same M-PML variant.
                # Acoustic VTI (pseudo-acoustic V_S=0 reduction of elastic_vti)
                # mirrors elastic_vti's 2-D/3-D-only, M-PML-variant build.
                if(NDIM GREATER_EQUAL 2)
                    add_deepwave_cpu_mpml_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
                    add_deepwave_cpu_object_library(elastic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                    add_deepwave_cpu_mpml_object_library(elastic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                    add_deepwave_cpu_object_library(acoustic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                    add_deepwave_cpu_mpml_object_library(acoustic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                endif()
        endforeach()
    endforeach()
endforeach()

if(OPENMP_CONFIGURED)
    # Link all CPU targets to our unified OpenMP interface library in one go.
    foreach(CPU_TARGET ${CPU_TARGETS})
        target_link_libraries(${CPU_TARGET} PRIVATE Deepwave_OpenMP_Interface)
    endforeach()
endif()

# --- CPU feature probe object library ---
# Compiled for the baseline ISA ONLY: it gets the release optimisation flags but
# deliberately never receives -mavx2 / DW_X86_MULTIVERSION, so calling it is
# guaranteed SIGILL-free even on the oldest supported x86 CPU. It is generic (no
# NDIM/ACCURACY/DTYPE specialisation) so it is added once, outside the macro.
add_library(cpu_features_obj OBJECT src/deepwave/cpu_features.c)
set_target_properties(cpu_features_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(C_RELEASE_FLAGS)
    target_compile_options(cpu_features_obj PRIVATE ${C_RELEASE_FLAGS})
endif()
if(C_WARNING_FLAGS)
    target_compile_options(cpu_features_obj PRIVATE ${C_WARNING_FLAGS})
endif()
if(DW_KERNEL_BASELINE_ISA)
    target_compile_definitions(cpu_features_obj PRIVATE
        DW_BASELINE_ISA="${DW_KERNEL_BASELINE_ISA}")
endif()
list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:cpu_features_obj>)
# --- End CPU feature probe object library ---

# --- End CPU object libraries ---


# --- CUDA object libraries ---
if(CUDAToolkit_FOUND)
        if(CMAKE_BUILD_TYPE MATCHES Release)
            if(DW_FAST_MATH)
                set(CUDA_RELEASE_OPTIONS --use_fast_math -O3 --restrict)
            else()
                set(CUDA_RELEASE_OPTIONS -O3 --restrict)
            endif()
            # Keep all SASS/PTX targets while reducing the wheel payload.
            # CUDA 13 supports size compression on every compatible driver;
            # leave older toolkits on their default to retain driver support.
            if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
               CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
                list(APPEND CUDA_RELEASE_OPTIONS --compress-mode=size)
            endif()
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
						# Parameter-modified free-surface variant (Cao and
						# Chen 2018): 2-D only, exact NDIM == 2 guard.
						if(NDIM EQUAL 2)
							add_deepwave_cuda_pm_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
						endif()
                    	add_deepwave_cuda_object_library(acoustic ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(acoustic_born ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(elastic_born ${NDIM} ${ACCURACY} ${DTYPE})
                    	add_deepwave_cuda_object_library(viscoacoustic ${NDIM} ${ACCURACY} ${DTYPE})
							# Multi-mechanism GSLS (DW_GSLS) variant; see the CPU
							# loop's comment above.
							add_deepwave_cuda_gsls_object_library(viscoacoustic ${NDIM} ${ACCURACY} ${DTYPE})
                        # Elastic M-PML and elastic VTI variants: only ndim >= 2.
                        # Acoustic VTI mirrors elastic_vti's build.
                        if(NDIM GREATER_EQUAL 2)
                            add_deepwave_cuda_mpml_object_library(elastic ${NDIM} ${ACCURACY} ${DTYPE})
                            add_deepwave_cuda_object_library(elastic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                            add_deepwave_cuda_mpml_object_library(elastic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                            add_deepwave_cuda_object_library(acoustic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                            add_deepwave_cuda_mpml_object_library(acoustic_vti ${NDIM} ${ACCURACY} ${DTYPE})
                        endif()
                endforeach()
            endforeach()
        endforeach()
endif()
# --- End CUDA object libraries ---

# --- Final Library Build ---
if(CUDAToolkit_FOUND)
    list(APPEND DEEPWAVE_OBJECTS src/deepwave/cuda_config_lock.cpp)
endif()
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

# Install
install(TARGETS deepwave_C
    LIBRARY DESTINATION deepwave
    ARCHIVE DESTINATION deepwave
    RUNTIME DESTINATION deepwave
)

# The Windows workflow supplies the DLL and its license from the same NuGet
# package as declared platform inputs, after extracting the verified sdist.
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/deepwave/libiomp5md.dll")
    set(DW_INTEL_OPENMP_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/runtime-notices/INTEL_OPENMP_LICENSE.txt")
    if(NOT EXISTS "${DW_INTEL_OPENMP_LICENSE}")
        message(FATAL_ERROR "The bundled Intel OpenMP runtime requires its package license in runtime-notices/INTEL_OPENMP_LICENSE.txt")
    endif()
endif()

# Wheel repair can bundle platform runtimes after CMake installs the library.
# Builders supply notices from those actual runtime packages alongside the
# verified source distribution, without adding platform binaries to the sdist.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime-notices")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtime-notices/"
        DESTINATION deepwave/licenses
        FILES_MATCHING PATTERN "*.txt"
    )
endif()

# CUDA's static runtime is linked into CUDA-enabled wheels. Preserve the
# notices from the actual toolkit used to build that artifact rather than
# copying a license from a different toolkit version into the source tree.
if(CUDAToolkit_FOUND)
    find_file(DW_CUDA_LICENSE
        NAMES EULA.txt EULA.pdf
        PATHS "${CUDAToolkit_BIN_DIR}/.." "${CUDAToolkit_BIN_DIR}/../doc"
        NO_DEFAULT_PATH
    )
    if(DW_CUDA_LICENSE)
        get_filename_component(DW_CUDA_LICENSE_EXT "${DW_CUDA_LICENSE}" LAST_EXT)
        install(FILES "${DW_CUDA_LICENSE}"
            DESTINATION deepwave/licenses
            RENAME "NVIDIA_CUDA_EULA${DW_CUDA_LICENSE_EXT}"
        )
    else()
        message(WARNING "CUDA toolkit redistribution notices were not found; wheel qualification must supply and verify the notices for this toolkit.")
    endif()
endif()
