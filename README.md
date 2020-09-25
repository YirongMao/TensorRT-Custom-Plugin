# TensorRT-Custom-Plugin
This repository describes:   

    (1) how to add a custom TensorRT plugin in c++,    
    
    (2) how to build and serialize network with the custom plugin in python   
    
    (3) how to load and forward the network in c++.
    
## Add custom TensorRT plugin in c++
We follow https://github.com/NVIDIA/TensorRT/tree/release/6.0/plugin/flattenConcat to create flattenConcat plugin. Since the flattenConcat plugin is already in TensorRT, we rename the class name.
Source codes are in flattenConcatCustom.cpp flattenConcatCustom.h
we use CMakeLists.txt to build libflatten_concat.so

## Build network and serialize engine in python
Please follow builder.py

## Load network in c++
Please follow load_trt_engine.cpp 


## requirements
TensorRT-6.0.1.5, cuda-9.0
  
  
  
