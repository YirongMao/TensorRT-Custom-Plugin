import tensorrt as trt
import ctypes
import numpy as np
import torch

nvinfer = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7", mode=ctypes.RTLD_GLOBAL)
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

builder = trt.Builder(TRT_LOGGER)
builder.max_batch_size = 10
builder.max_workspace_size = 5000 * (1024 * 1024)
builder.strict_type_constraints = False
network = builder.create_network()

input_1 = network.add_input(name="input_1", dtype=trt.float32, shape=(4, 2, 2))
input_2 = network.add_input(name="input_2", dtype=trt.float32, shape=(2, 2, 2))

inputs = [input_1, input_2]
emb_layer = network.add_plugin_v2(inputs, fn)
print(emb_layer)
embeddings = emb_layer.get_output(0)
network.mark_output(embeddings)
embeddings_shape = embeddings.shape
engine = builder.build_cuda_engine(network)
serialized_engine = engine.serialize()
# TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving the engine....")
with open('flattenconcat.engine', 'wb') as fout:
    fout.write(serialized_engine)
print('engine serialized')
