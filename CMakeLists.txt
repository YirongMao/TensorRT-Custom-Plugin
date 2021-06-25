cmake_minimum_required(VERSION 3.14)
include_directories(./)

file(GLOB SRC_FILES "./flattenConcatCustom.cpp")

add_compile_options(-std=c++11)
add_library(flatten_concat SHARED ${SRC_FILES})

target_include_directories(flatten_concat PUBLIC /usr/include/x86_64-linux-gnu)
target_include_directories(flatten_concat PUBLIC /usr/local/cuda/include)

target_link_libraries(flatten_concat -ldl -lpthread -lrt)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/libnvparsers_static.a)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/lib/libnvinfer_static.a)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/lib/libnvinfer_plugin_static.a)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/libcudnn.a ${CUDA_LIB}/libcublas.a)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/libculibos.a)
target_link_libraries(flatten_concat /usr/lib/x86_64-linux-gnu/libcudart.a)

