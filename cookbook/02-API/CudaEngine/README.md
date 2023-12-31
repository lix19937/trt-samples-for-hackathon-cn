# 引擎 

## 运行命令   

```shell
python3 main.py
```

## 说明  
```python
shape = [1, 4, 8, 8]
data = (np.arange(1, 1 + np.prod(shape), dtype=np.float32) / np.prod(shape) * 128).astype(np.float32).reshape(shape)
np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

# 构建器
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

# 搭建网络 
inputT0 = network.add_input("inputT0", trt.float32, [-1] + shape[1:])
profile.set_shape(inputT0.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
config.add_optimization_profile(profile)
layer = network.add_identity(inputT0)
layer.name = "MyIdentityLayer"
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)  # use a uncommon data format
layer.get_output(0).dynamic_range = [-128, 128]
network.mark_output(layer.get_output(0))

# 从网络和config构建序列化引擎  
engineString = builder.build_serialized_network(network, config)

# 反序列化引擎
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

# 引擎信息  
print("engine.__len__() = %d" % len(engine))
print("engine.__sizeof__() = %d" % engine.__sizeof__())
print("engine.__str__() = %s" % engine.__str__())

print("\nEngine related ========================================================")
# All member functions with "binding" in name are deprecated since TensorRT 8.5
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]

# count of input / output tensor
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT) 
nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

print("engine.name = %s" % engine.name)
print("engine.device_memory_size = %d" % engine.device_memory_size)
print("engine.engine_capability = %d" % engine.engine_capability)  # refer to 02-API/BuilderConfig
print("engine.hardware_compatibility_level = %d" % engine.hardware_compatibility_level)
print("engine.num_aux_streams = %d" % engine.num_aux_streams)
print("engine.has_implicit_batch_dimension = %s" % engine.has_implicit_batch_dimension)

# io tensor 数目 
print("engine.num_io_tensors = %d" % engine.num_io_tensors)
#print("engine.num_bindings = %d" % engine.num_bindings)  # deprecated since TensorRT 8.5
print("engine.num_layers = %d" % engine.num_layers)
print("engine.num_optimization_profiles = %d" % engine.num_optimization_profiles)
print("engine.refittable = %s" % engine.refittable)  # refer to 02-API/Refit
print("engine.tactic_sources = %d" % engine.tactic_sources)  # refer to 02-API/TacticSource

print("\nLayer related =========================================================")
print("engine.get_tensor_location(%s): %s" % (layer.get_output(0).name, engine.get_tensor_location(layer.get_output(0).name)))

print("\nInput / Output tensor related =========================================")
print("No. Input  output:                   %s 0,%s 1" % (" " * 56, " " * 56))
print("engine.get_tensor_name():            %58s,%58s" % (engine.get_tensor_name(0), engine.get_tensor_name(1)))
#print("get_binding_name():                  %58s,%58s" % (engine.get_binding_name(0), engine.get_binding_name(1)))
print("get_tensor_shape():                  %58s,%58s" % (engine.get_tensor_shape(lTensorName[0]), engine.get_tensor_shape(lTensorName[1])))
#print("get_binding_shape():                 %58s,%58s" % (engine.get_binding_shape(0), engine.get_binding_shape(1)))
print("get_tensor_dtype():                  %58s,%58s" % (engine.get_tensor_dtype(lTensorName[0]), engine.get_tensor_dtype(lTensorName[1])))
#print("get_binding_dtype():                 %58s,%58s" % (engine.get_binding_dtype(0), engine.get_binding_dtype(1)))
print("get_tensor_format():                 %58s,%58s" % (engine.get_tensor_format(lTensorName[0]), engine.get_tensor_format(lTensorName[1])))
#print("get_binding_format():                %58s,%58s" % (engine.get_binding_format(0), engine.get_binding_format(1)))
print("get_tensor_format_desc():            %58s,%58s" % (engine.get_tensor_format_desc(lTensorName[0]), engine.get_tensor_format_desc(lTensorName[1])))
#print("get_binding_format_desc():           %58s,%58s" % (engine.get_binding_format_desc(0), engine.get_binding_format_desc(1)))
print("get_tensor_bytes_per_component():    %58d,%58d" % (engine.get_tensor_bytes_per_component(lTensorName[0]), engine.get_tensor_bytes_per_component(lTensorName[1])))
#print("get_binding_bytes_per_component():   %58d,%58d" % (engine.get_binding_bytes_per_component(0), engine.get_binding_bytes_per_component(1)))
print("get_tensor_components_per_element(): %58d,%58d" % (engine.get_tensor_components_per_element(lTensorName[0]), engine.get_tensor_components_per_element(lTensorName[1])))
#print("get_binding_components_per_element():%58d,%58d" % (engine.get_binding_components_per_element(0), engine.get_binding_components_per_element(1)))
print("get_tensor_vectorized_dim():         %58d,%58d" % (engine.get_tensor_vectorized_dim(lTensorName[0]), engine.get_tensor_vectorized_dim(lTensorName[1])))
#print("get_binding_vectorized_dim():        %58d,%58d" % (engine.get_binding_vectorized_dim(0), engine.get_binding_vectorized_dim(1)))
print("")
print("get_tensor_mode():                   %58s,%58s" % (engine.get_tensor_mode(lTensorName[0]), engine.get_tensor_mode(lTensorName[1])))
#print("binding_is_input():                  %58s,%58s" % (engine.binding_is_input(0), engine.binding_is_input(1)))
print("get_tensor_location():               %58s,%58s" % (engine.get_tensor_location(lTensorName[0]), engine.get_tensor_location(lTensorName[0])))
print("Comment: Execution input / output tensor is on Device, while Shape input / output tensor is on CPU")
#print("get_location(int):                   %58s,%58s" % (engine.get_location(0), engine.get_location(1)))
#print("get_location(str):                   %58s,%58s" % (engine.get_location(lTensorName[0]), engine.get_location(lTensorName[1])))
print("is_shape_inference_io():             %58s,%58s" % (engine.is_shape_inference_io(lTensorName[0]), engine.is_shape_inference_io(lTensorName[0])))
#print("is_execution_binding():              %58s,%58s" % (engine.is_execution_binding(0), engine.is_execution_binding(1)))
#print("is_shape_binding():                  %58s,%58s" % (engine.is_shape_binding(0), engine.is_shape_binding(1)))
print("get_tensor_profile_shape():          %58s,%58s" % (engine.get_tensor_profile_shape(lTensorName[0], 0), "Optimization Profile is only for input tensor"))
#print("get_profile_shape():                 %58s,%58s" % (engine.get_profile_shape(0, 0), "Optimization Profile is only for input tensor"))
#print("get_profile_shape_input():           %58s,%58s" % ("No input shape tensor in this network", ""))
print("__getitem__(int):                    %58s,%58s" % (engine[0], engine[1]))
print("__getitem__(str):                    %58d,%58d" % (engine[lTensorName[0]], engine[lTensorName[1]]))

```