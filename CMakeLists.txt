cmake_minimum_required(VERSION 3.18)
project(deepwave LANGUAGES C CXX)

find_package(OpenMP)

set(DEEPWAVE_OBJECTS)

set(CMAKE_CUDA_STANDARD 11)

set(SCALAR_ACCURACIES 2 4 6 8)
set(ELASTIC_ACCURACIES 2 4)
set(DTYPES float double)

foreach(ACCURACY ${SCALAR_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(scalar_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.c)
        target_compile_definitions(scalar_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_${ACCURACY}_${DTYPE}_obj>)

        add_library(scalar_born_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.c)
        target_compile_definitions(scalar_born_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_${ACCURACY}_${DTYPE}_obj>)
    endforeach()
endforeach()

foreach(ACCURACY ${ELASTIC_ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_library(elastic_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.c)
        target_compile_definitions(elastic_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cpu)
        list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_${ACCURACY}_${DTYPE}_obj>)
    endforeach()
endforeach()

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    foreach(ACCURACY ${SCALAR_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(scalar_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar.cu)
            target_compile_definitions(scalar_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_cu_${ACCURACY}_${DTYPE}_obj>)

            add_library(scalar_born_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/scalar_born.cu)
            target_compile_definitions(scalar_born_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:scalar_born_cu_${ACCURACY}_${DTYPE}_obj>)
        endforeach()
    endforeach()

    foreach(ACCURACY ${ELASTIC_ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_library(elastic_cu_${ACCURACY}_${DTYPE}_obj OBJECT src/deepwave/elastic.cu)
            target_compile_definitions(elastic_cu_${ACCURACY}_${DTYPE}_obj PRIVATE DW_ACCURACY=${ACCURACY} DW_DTYPE=${DTYPE} DW_DEVICE=cuda)
            list(APPEND DEEPWAVE_OBJECTS $<TARGET_OBJECTS:elastic_cu_${ACCURACY}_${DTYPE}_obj>)
        endforeach()
    endforeach()
endif()

add_library(deepwave_C SHARED ${DEEPWAVE_OBJECTS})
set_target_properties(deepwave_C PROPERTIES C_VISIBILITY_PRESET default CXX_VISIBILITY_PRESET default)

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(deepwave_C PRIVATE CUDA::cudart)
endif()

if (OpenMP_FOUND)
    target_link_libraries(deepwave_C PRIVATE OpenMP::OpenMP_C)
endif()

set_target_properties(deepwave_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS deepwave_C
    DESTINATION ${SKBUILD_PLATLIB_DIR}/deepwave
)
