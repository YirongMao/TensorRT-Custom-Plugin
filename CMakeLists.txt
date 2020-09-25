cmake_minimum_required(VERSION 3.14)
include_directories(./)

file(GLOB SRC_FILES "./flattenConcatCustom.cpp")
set(TRT_HOME "path-to-tensorrt/TensorRT-6.0.1.5")
set(CUDA_LIB "path-to-cuda-9.0-lib")

add_compile_options(-std=c++11)
add_library(flatten_concat SHARED ${SRC_FILES})

target_include_directories(flatten_concat PUBLIC ${TRT_HOME}/include)
target_include_directories(flatten_concat PUBLIC ${TRT_HOME})
target_include_directories(flatten_concat PUBLIC ${TRT_HOME}/cuda/include)

target_link_libraries(flatten_concat -ldl -lpthread -lrt)
target_link_libraries(flatten_concat ${TRT_HOME}/lib/libnvparsers_static.a)
target_link_libraries(flatten_concat ${TRT_HOME}/lib/libnvinfer_static.a)
target_link_libraries(flatten_concat ${TRT_HOME}/lib/libnvinfer_plugin_static.a)
target_link_libraries(flatten_concat path-to-conda/anaconda3/lib/libstdc++.so)
target_link_libraries(flatten_concat ${CUDA_LIB}/libcudnn_v9.a ${CUDA_LIB}/libcublas_v9.a)
target_link_libraries(flatten_concat ${CUDA_LIB}/libculibos_v9.a)
target_link_libraries(flatten_concat ${CUDA_LIB}/libcudart_v9.a)

