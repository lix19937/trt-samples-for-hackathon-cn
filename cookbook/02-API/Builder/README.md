#

## 运行命令  

```shell
python3 main.py
```

## 说明  
```python
# 日志级别  然后生成构建器
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)

# 构建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 获得config 对象
config = builder.create_builder_config()

# 搭建网络  
inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

# 打印构建器信息 
print("builder.__sizeof__() = %d" % builder.__sizeof__())
print("builder.__str__() = %s" % builder.__str__())

print("\nDevice type part ======================================================")
print("builder.platform_has_tf32 = %s" % builder.platform_has_tf32)
print("builder.platform_has_fast_fp16 = %s" % builder.platform_has_fast_fp16)
print("builder.platform_has_fast_int8 = %s" % builder.platform_has_fast_int8)
print("builder.num_DLA_cores = %d" % builder.num_DLA_cores)
print("builder.max_DLA_batch_size = %d" % builder.max_DLA_batch_size)

print("\nEngine build part =====================================================")
print("builder.logger = %s" % builder.logger)
print("builder.is_network_supported() = %s" % builder.is_network_supported(network, config))
print("builder.get_plugin_registry().plugin_creator_list =", builder.get_plugin_registry().plugin_creator_list)
builder.max_threads = 16  # The maximum thread that can be used by the Builder 

# 序列化网络  
engineString = builder.build_serialized_network(network, config)

```
