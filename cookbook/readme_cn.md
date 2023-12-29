`以下内容来自 lix19937 梳理精简，版权归属 NVIDIA `     

# TensorRT 菜谱   

## 有用的链接 

+ [TensorRT Download](https://developer.nvidia.com/nvidia-tensorrt-download)
+ [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
+ [Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) and [Document Archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
+ [Supporting Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) and [Supporting Matrix (Old version)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/support-matrix/index.html)
+ [C++ API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api) and [Python API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/) and [API Change](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)
+ [Operator algorithm](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)
+ [Onnx Operator Supporting Matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
+ [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)
+ [NSight- Systems](https://developer.nvidia.com/nsight-systems)

+ 其他:
  + [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt)
  + [TF-TRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
  + [Torch-TensorRT](https://pytorch.org/TensorRT/)
  + [ONNX model zoo](https://github.com/onnx/models)
  + [tensorrtx (build engine by API)](https://github.com/wang-xinyu/tensorrtx)

---

## 目录  

### include

+ 例程使用的通用文件   

### 01-SimpleDemo -- 使用TensorRT的完整示例    

+ 网络构建，引擎构建，序列化/反序列化，推理计算等         

### 02-API -- TessnorRT核心对象和API的示例   

+ API使用   

### 03-BuildEngineByTensorRTAPI -- 使用layer API重建模型的示例   

+ 不依赖onnx，在TensorRT中使用layer API从各种机器学习框架中重建训练模型的过程，包括从原始模型中提取权重、模型重建和权重加载等    

### 04-BuildEngineByONNXParser -- 使用ONNX Parser重建模型的示例    

+ 将模型导出为ONNX格式，然后导出为TensorRT的过程    

### 05-Plugin -- 编写自定义插件的示例     

### 06-框架中的TRT -- 在机器学习框架中使用内置TensorRT API的示例    

### 07-工具 -- 开发辅助工具    

### 08-高阶 -- TensorRT中高级功能的示例     

### 09-最佳实践 -- 有趣的TensorRT优化示例     

### 10-问题解决中   

+ 将模型部署到TensorRT时的一些错误消息和相应的解决方案    

### 50-资源 -- 文档资源  

+ TensorRT中文使用说明，以及其他一些有用的参考资料     

### 51-未分类  

+ 未分类的内容     

### 51-废弃接口    

### 99-未完成      

## 仓库目录树  

```txt
├── 00-MNISTData
│   ├── test
│   └── train
├── 01-SimpleDemo  // 完整的使用示例   
│   └── TensorRT8.5
├── 02-API // 接口 
│   ├── AlgorithmSelector    // 算法策略选择器   
│   ├── AuxStream            // 辅助流   
│   ├── Builder              // 构建器
│   ├── BuilderConfig        // 构建器配置  
│   ├── CudaEngine           // cuda 引擎  
│   ├── EngineInspector      // 引擎检查工具  
│   ├── ErrorRecoder         // 错误记录
│   ├── ExecutionContext     // 执行器上下文
│   ├── GPUAllocator         // gpu内存分配器
│   ├── HostMemory           // 主机内存
│   ├── INT8-PTQ             // 后量化 
│   │   └── C++
│   ├── Layer // 网络层
│   │   ├── ActivationLayer
│   │   ├── AssertionLayer
│   │   ├── CastLayer
│   │   ├── ConcatenationLayer
│   │   ├── ConstantLayer
│   │   ├── ConvolutionNdLayer
│   │   ├── DeconvolutionNdLayer
│   │   ├── EinsumLayer
│   │   ├── ElementwiseLayer
│   │   ├── FillLayer
│   │   ├── FullyConnectedLayer
│   │   ├── GatherLayer
│   │   ├── GridSampleLayer
│   │   ├── IdentityLayer
│   │   ├── IfConditionStructure
│   │   ├── LoopStructure
│   │   ├── LRNLayer
│   │   ├── MatrixMultiplyLayer
│   │   ├── NMSLayer
│   │   ├── NonZeroLayer
│   │   ├── NormalizationLayer
│   │   ├── OneHotLayer
│   │   ├── PaddingNdLayer
│   │   ├── ParametricReLULayer
│   │   ├── PluginV2Layer
│   │   ├── PoolingNdLayer
│   │   ├── QuantizeDequantizeLayer
│   │   ├── RaggedSoftMaxLayer
│   │   ├── ReduceLayer
│   │   ├── ResizeLayer
│   │   ├── ReverseSequenceLayer
│   │   ├── RNNv2Layer
│   │   ├── ScaleLayer
│   │   ├── ScatterLayer
│   │   ├── SelectLayer
│   │   ├── ShapeLayer
│   │   ├── ShuffleLayer
│   │   ├── SliceLayer
│   │   ├── SoftmaxLayer
│   │   ├── TopKLayer
│   │   └── UnaryLayer
│   ├── Logger // 日志器
│   ├── Network // 网络
│   ├── ONNXParser // 解析器
│   ├── OptimizationProfile  // 优化分析
│   ├── OutputAllocator // 输出分配器
│   ├── Profiler // 分析器
│   ├── ProfilingVerbosity  // 分析显示   
│   ├── Refit // 重装配 
│   ├── Runtime // 运行时 
│   ├── TacticSource   // 策略来源   
│   ├── Tensor  // 张量  
│   └── TimingCache  // 时序缓存  
├── 03-BuildEngineByTensorRTAPI  // 通过API 构建引擎 
│   ├── MNISTExample-pyTorch
│   │   └── C++
│   ├── TypicalAPI-pyTorch
├── 04-BuildEngineByONNXParser  // 通过onnx 解析器构建引擎  
│   ├── pyTorch-ONNX-TensorRT
│   │   └── C++
│   ├── pyTorch-ONNX-TensorRT-QAT
├── 05-Plugin  // 插件
│   ├── API
│   ├── C++-UsePluginInside
│   ├── C++-UsePluginOutside
│   ├── LoadDataFromNpz
│   ├── MultipleVersion
│   ├── PluginProcess
│   ├── PluginReposity
│   │   ├── AddScalarPlugin-TRT8
│   │   ├── BatchedNMS_TRTPlugin-TRT8
│   │   ├── CCLPlugin-TRT6-StaticShape
│   │   ├── CCLPlugin-TRT7-DynamicShape
│   │   ├── CumSumPlugin-V2.1-TRT8
│   │   ├── GruPlugin
│   │   ├── LayerNormPlugin-TRT8
│   │   ├── Mask2DPlugin
│   │   ├── MaskPugin
│   │   ├── MaxPlugin
│   │   ├── MMTPlugin
│   │   ├── MultinomialDistributionPlugin-cuRAND-TRT8
│   │   ├── MultinomialDistributionPlugin-thrust-TRT8
│   │   ├── OneHotPlugin-TRT8
│   │   ├── ReducePlugin
│   │   ├── Resize2DPlugin-TRT8
│   │   ├── ReversePlugin
│   │   ├── SignPlugin
│   │   ├── SortPlugin-V0.0-useCubAlone
│   │   ├── SortPlugin-V1.0-float
│   │   ├── SortPlugin-V2.0-float4
│   │   ├── TopKAveragePlugin
│   │   └── WherePlugin
│   ├── PluginSerialize-TODO
│   ├── PythonPlugin
│   │   └── circ_plugin_cpp
│   ├── UseCuBLAS
│   ├── UseFP16
│   ├── UseINT8-PTQ
│   ├── UseINT8-QDQ
│   ├── UseONNXParserAndPlugin-pyTorch
│   ├── UsePluginV2DynamicExt
│   ├── UsePluginV2Ext
│   └── UsePluginV2IOExt
├── 06-UseFrameworkTRT  // 框架中使用TensorRT     
│   └── Torch-TensorRT
├── 07-Tool  // 工具
│   ├── FP16FineTuning  // fp16微调 
│   ├── Netron    // netron 可视化模型文件  
│   ├── NetworkInspector // 网络检查器  
│   │   └── C++
│   ├── NetworkPrinter  // 网络打印  
│   ├── NsightSystems  nsys  
│   ├── nvtx
│   ├── OnnxGraphSurgeon  
│   │   └── API
│   ├── Onnxruntime
│   ├── Polygraphy-API
│   ├── Polygraphy-CLI
│   │   ├── convertExample
│   │   ├── dataExample
│   │   ├── debugExample
│   │   ├── HelpInformation
│   │   ├── inspectExample
│   │   ├── runExample
│   │   ├── surgeonExample
│   │   └── templateExample
│   ├── trex  // 引擎浏览器  
│   │   ├── model
│   │   ├── trex
│   │   └── trex.egg-info
│   └── trtexec
├── 08-Advance  // 高阶 
│   ├── BuilderOptimizationLevel       // 构建器优化级别 
│   ├── CreateExecutionContextWithoutDeviceMemory  // 不使用设备内存构建执行器上下文  
│   ├── C++StaticCompilation // cpp 静态编译 
│   ├── CudaGraph  // cuda graph
│   ├── DataFormat // 数据格式
│   ├── DynamicShapeOutput // 输出shape为动态 
│   ├── EmptyTensor   // 空tensor 
│   ├── Event   // 事件 
│   ├── ExternalSource  // 外部源   
│   ├── HardwareCompatibility // 硬件兼容性 
│   ├── LabeledDimension  // 维度被标记
│   ├── MultiContext  // 多上下文
│   ├── MultiOptimizationProfile  // 多优化分析配置
│   ├── MultiStream // 多流
│   ├── Safety-TODO // 安全性
│   ├── Sparsity // 稀疏
│   │   └── pyTorch-ONNX-TensorRT-ASP
│   ├── StreamAndAsync   // 流与异步
│   ├── StrictType  // 严格类型
│   ├── TensorRTGraphSurgeon // TensorRT图检查
│   ├── TorchOperation  // torch算子
│   └── VersionCompatibility  // 版本兼容性
├── 09-BestPractice
│   ├── AdjustReduceLayer
│   ├── AlignSize
│   ├── ComputationInAdvance
│   │   └── Convert3DMMTo2DMM
│   ├── ConvertTranposeMultiplicationToConvolution
│   ├── EliminateSqueezeUnsqueezeTranspose
│   ├── IncreaseBatchSize
│   ├── UsingMultiOptimizationProfile
│   ├── UsingMultiStream
│   └── WorkFlowOnONNXModel
├── 10-ProblemSolving
│   ├── ParameterCheckFailed
│   ├── SliceNodeWithBoolIO
│   ├── WeightsAreNotPermittedSinceTheyAreOfTypeInt32
│   └── WeightsHasCountXButYWasExpected
├── 51-Uncategorized
├── 52-Deprecated
│   ├── BindingEliminate-TRT8
│   ├── ConcatenationLayerBUG-TRT8.4
│   ├── ErrorWhenParsePadNode-TRT-8.4
│   ├── FullyConnectedLayer-TRT8.4
│   ├── FullyConnectedLayerWhenUsingParserTRT-8.4
│   ├── MatrixMultiplyDeprecatedLayer-TRT8
│   ├── max_workspace_size-TRT8.4
│   ├── MultiContext-TRT8
│   ├── ResizeLayer-TRT8
│   ├── RNNLayer-TRT8
│   └── ShapeLayer-TRT8
├── 99-NotFinish
│   └── TensorRTElementwiseBug
└── include
```
