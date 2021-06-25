rm -rf load_trt_engine
g++ load_trt_engine.cpp flattenConcatCustom.cpp -o load_trt_engine\
    -I /usr/include/x86_64-linux-gnu \
    -I /usr/local/cuda/include \
    -std=c++11 -ldl -lpthread -lrt\
    -L/usr/lib/x86_64-linux-gnu \
    -lnvparsers_static -lnvinfer_static -lnvinfer_plugin_static \
    -lstdc++ \
    -L -lcudnn -lcublas -lculibos -lcudart
