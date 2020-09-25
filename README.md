# TensorRT-Custom-Plugin
This repository describes:   

    (1) how to add a custom TensorRT plugin in c++,    
    
    (2) how to build and serialize network with the custom plugin in python   
    
    (3) how to load and forward the network in c++.
    
## Add custom TensorRT plugin in c++
We follow [flattenconcat plugin](https://github.com/NVIDIA/TensorRT/tree/release/6.0/plugin/flattenConcat) to create flattenConcat plugin. 

Since the flattenConcat plugin is already in TensorRT, we renamed the class name.
The corresponding source codes are in flattenConcatCustom.cpp flattenConcatCustom.h.
We use file [CMakeLists.txt](https://github.com/YirongMao/TensorRT-Custom-Plugin/blob/master/CMakeLists.txt) to build shared lib: libflatten_concat.so

## Build network and serialize engine in python
Please follow [builder.py](https://github.com/YirongMao/TensorRT-Custom-Plugin/blob/master/builder.py).

You should configure the path to libnvinfer_plugin.so

## Load network in c++
Please follow load_trt_engine.cpp 
To load the engine with custom plugin, its header *.h file should be included.

## env. requirements
TensorRT-6.0.1.5, cuda-9.0

## Contacts
If you encounter any problem, be free to create an issue.
  
  
  
