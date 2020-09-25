

rm -rf load_trt_engine
g++ load_trt_engine.cpp flattenConcatCustom.cpp -o load_trt_engine\
    -I path-to-tensorrt/TensorRT-6.0.1.5/include \
    -I path-to-tensorrt/TensorRT-6.0.1.5 \
    -I path-to-cuda-9.0/cuda/include \
    -std=c++11 -ldl -lpthread -lrt\
    -Lpath-to-tensorrt/TensorRT-6.0.1.5/lib \
    -lnvparsers_static -lnvinfer_static -lnvinfer_plugin_static \
    -Lpath-to-conda/anaconda3/lib \
    -lstdc++ \
    -Lpath-to-cuda-0.9/cuda/lib -lcudnn_v9 -lcublas_v9 -lculibos_v9 -lcudart_v9 
