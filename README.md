# TensorRT-Custom-Plugin
This repository describes:   

    (1) how to add a custom TensorRT plugin in c++,    
    
    (2) how to build and serialize network with the custom plugin in python   
    
    (3) how to load and forward the network in c++.
    
## Add custom TensorRT plugin in c++
We follow [flattenconcat plugin](https://github.com/NVIDIA/TensorRT/tree/release/6.0/plugin/flattenConcat) to create flattenConcat plugin. 

Since the flattenConcat plugin is already in TensorRT, we renamed the class name.
The corresponding source codes are in flattenConcatCustom.cpp flattenConcatCustom.h
We use file [CMakeLists.txt](https://github.com/YirongMao/TensorRT-Custom-Plugin/blob/master/CMakeLists.txt) to build shared lib: libflatten_concat.so


## Build network and serialize engine in python
Please follow [builder.py](https://github.com/YirongMao/TensorRT-Custom-Plugin/blob/master/builder.py).

You should configure the path to libnvinfer_plugin.so
```python
nvinfer = ctypes.CDLL("/path-to-tensorrt/TensorRT-6.0.1.5/lib/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
print('load nvinfer')
pg = ctypes.CDLL("./libflatten_concat.so", mode=ctypes.RTLD_GLOBAL)
print('load customed plugin')

#TensorRT Initialization
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
plg_creator = plg_registry.get_plugin_creator("FlattenConcatCustom", "1", "")
print(plg_creator)

axis_pf = trt.PluginField("axis", np.array([1], np.int32), trt.PluginFieldType.INT32)
batch_pf = trt.PluginField("ignoreBatch", np.array([0], np.int32), trt.PluginFieldType.INT32)

pfc = trt.PluginFieldCollection([axis_pf, batch_pf])
fn = plg_creator.create_plugin("FlattenConcatCustom1", pfc)
print(fn)
```

## Load network in c++
Please follow load_trt_engine.cpp. To load the engine with custom plugin, its header *.h file should be included.

## env. requirements
TensorRT-6.0.1.5, cuda-9.0

## Contacts
If you encounter any problem, be free to create an issue.
  
  
  
