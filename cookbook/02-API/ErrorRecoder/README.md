# 错误记录器    

## 运行命令    

```shell
python3 main.py
```

## 说明   

### 构建期   
```python

trtFile = "./model.plan"

class MyErrorRecorder(trt.IErrorRecorder):
    def get_error_code(self, index):
        print("[MyErrorRecorder::get_error_code]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return trt.ErrorCodeTRT.SUCCESS
        return self.errorList[index][0]

    def get_error_desc(self, index):
        print("[MyErrorRecorder::get_error_desc]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return ""
        # Error number in self.errorList[index][0]:
        # trt.ErrorCodeTRT.SUCCESS                # 0
        # trt.ErrorCodeTRT.UNSPECIFIED_ERROR      # 1
        # trt.ErrorCodeTRT.INTERNAL_ERROR         # 2
        # trt.ErrorCodeTRT.INVALID_ARGUMENT       # 3
        # trt.ErrorCodeTRT.INVALID_CONFIG         # 4
        # trt.ErrorCodeTRT.FAILED_ALLOCATION      # 5
        # trt.ErrorCodeTRT.FAILED_INITIALIZATION  # 6
        # trt.ErrorCodeTRT.FAILED_EXECUTION       # 7
        # trt.ErrorCodeTRT.FAILED_COMPUTATION     # 8
        # trt.ErrorCodeTRT.INVALID_STATE          # 9
        # trt.ErrorCodeTRT.UNSUPPORTED_STATE      # 10
        return self.errorList[index][1]

    def report_error(self, errorCode, errorDescription):
        print("[MyErrorRecorder::report_error]\n\tNumber=%d,Code=%d,Information=%s" % (self.nError, int(errorCode), errorDescription))
        self.nError += 1
        self.errorList.append([errorCode, errorDescription])
        if self.has_overflowed():
            print("Error Overflow!")
        return

myErrorRecorder = MyErrorRecorder()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
 # 自定义的错误记录器给构建器  ！！！   类似注册函数  
builder.error_recorder = myErrorRecorder 
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

# 构建器和网络使用同一个错误记录器  once assigned, Builder and Network share the same Error Recorder
print("Builder.error_recorder:", builder.error_recorder.helloWorld())  
print("Network.error_recorder:", network.error_recorder.helloWorld())

# 搭建网络  
inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
#network.mark_output(identityLayer.get_output(0))  # TensorRT raises a error without this line

print("Report error during building serialized network -------------------------")
# 构建序列化引擎 
engineString = builder.build_serialized_network(network, config)

if engineString == None:
    print("Failed building serialized engine!")
    print("Report error after all other work -----------------------------------")
    print("There is %d error" % myErrorRecorder.num_errors())
    for i in range(myErrorRecorder.num_errors()):
        print("\tNumber=%d,Code=%d,Information=%s" % (i, int(myErrorRecorder.get_error_code(i)), myErrorRecorder.get_error_desc(i)))
    # clear all error information    
    myErrorRecorder.clear()  
else:
    print("Succeeded building serialized engine!")
```

### 运行时  
```python 
class MyErrorRecorder(trt.IErrorRecorder):
    def get_error_code(self, index):
        print("[MyErrorRecorder::get_error_code]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return trt.ErrorCodeTRT.SUCCESS
        return self.errorList[index][0]

    def get_error_desc(self, index):
        print("[MyErrorRecorder::get_error_desc]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return ""
        # Error number in self.errorList[index][0]:
        # trt.ErrorCodeTRT.SUCCESS            # 0
        # trt.ErrorCodeTRT.UNSPECIFIED_ERROR  # 1
        # trt.ErrorCodeTRT.INTERNAL_ERROR     # 2
        # trt.ErrorCodeTRT.INVALID_ARGUMENT   # 3
        # trt.ErrorCodeTRT.INVALID_CONFIG     # 4
        # trt.ErrorCodeTRT.FAILED_ALLOCATION  # 5
        # trt.ErrorCodeTRT.FAILED_INITIALIZATION  # 6
        # trt.ErrorCodeTRT.FAILED_EXECUTION       # 7
        # trt.ErrorCodeTRT.FAILED_COMPUTATION     # 8
        # trt.ErrorCodeTRT.INVALID_STATE          # 9
        # trt.ErrorCodeTRT.UNSUPPORTED_STATE      # 10
        return self.errorList[index][1]

    def report_error(self, errorCode, errorDescription):
        print("[MyErrorRecorder::report_error]\n\tNumber=%d,Code=%d,Information=%s" % (self.nError, int(errorCode), errorDescription))
        self.nError += 1
        self.errorList.append([errorCode, errorDescription])
        if self.has_overflowed():
            print("Error Overflow!")
        return

    def helloWorld(self):  # not required API, just for fun
        return "Hello World!"

myErrorRecorder = MyErrorRecorder()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config() 

# 搭建网络  
inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

# 序列化
engineString = builder.build_serialized_network(network, config)
runtime = trt.Runtime(logger)

# 错误记录器注册 ErrorRecorder for runtime, it can be assigned to Runtime or Engine or ExecutionContext
runtime.error_recorder = myErrorRecorder  

# 反序列化 
engine = runtime.deserialize_cuda_engine(engineString)
#engine.error_recorder = myErrorRecorder 

# 创建执行器上下文 
context = engine.create_execution_context()
#context.error_recorder = myErrorRecorder

print("Runtime.error_recorder:", runtime.error_recorder, runtime.error_recorder.helloWorld())
print("Engine.error_recorder:", engine.error_recorder, engine.error_recorder.helloWorld())
print("Context.error_recorder:", context.error_recorder, context.error_recorder.helloWorld())

# use null pointer to do inference, TensorRT raises a error
context.execute_v2([int(0), int(0)])  

print("Failed doing inference!")
print("Report error after all other work ---------------------------------------")
print("There is %d error" % myErrorRecorder.num_errors())

for i in range(myErrorRecorder.num_errors()):
    print("\tNumber=%d,Code=%d,Information=%s" % (i, int(myErrorRecorder.get_error_code(i)), myErrorRecorder.get_error_desc(i)))

# clear all error information    
myErrorRecorder.clear()  

```