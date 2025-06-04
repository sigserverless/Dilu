cmake_minimum_required(VERSION 3.5)
project(hijack C CXX)

add_definitions(-D_GNU_SOURCE)
add_compile_options(-Wall -Wshadow -Werror -Wno-format)

include_directories(${CMAKE_SOURCE_DIR})

if (${ENABLE_DEBUG})
    add_compile_options(-g -O0)
else ()
    add_compile_options(-g -O2)
endif ()

set(STATIC_C_LIBRARIES -static-libgcc -static-libstdc++)
set(CMAKE_C_STANDARD 99)

add_library(cuda-control-container-client SHARED
        src/container_client_hijack_call.c
        include/hijack.h
        include/cuda-subset.h
        include/nvml-subset.h
        include/cuda-helper.h
        include/nvml-helper.h
        src/cuda_originals.c
        src/nvml_entry.c
        src/loader.c)

target_link_libraries(cuda-control-container-client ${STATIC_C_LIBRARIES})

target_compile_options(cuda-control-container-client PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)