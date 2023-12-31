`以下内容来自 lix19937 梳理精简，版权归属 NVIDIA `     

# TensorRT 食谱精讲          

## 目录  

### include

+ 例程使用的通用文件

### 00-MNISTData -- mnist 数据集获取     

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
├── 02-API         // 接口 
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
│   ├── Layer     // 网络层
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
│   ├── Logger               // 日志器
│   ├── Network              // 网络
│   ├── ONNXParser           // 解析器
│   ├── OptimizationProfile  // 优化分析
│   ├── OutputAllocator      // 输出分配器
│   ├── Profiler             // 分析器
│   ├── ProfilingVerbosity   // 分析显示   
│   ├── Refit                // 重装配  常用于GNN 网络    
│   ├── Runtime              // 运行时 
│   ├── TacticSource         // 策略来源   
│   ├── Tensor               // 张量  
│   └── TimingCache          // timing缓存  trtexec --timingCacheFile=  
├── 03-BuildEngineByTensorRTAPI  // 通过API 构建引擎 
│   ├── MNISTExample-pyTorch
│   │   └── C++
│   ├── TypicalAPI-pyTorch
├── 04-BuildEngineByONNXParser  // 通过onnx 解析器构建引擎  
│   ├── pyTorch-ONNX-TensorRT
│   │   └── C++
│   ├── pyTorch-ONNX-TensorRT-QAT
├── 05-Plugin  // 自定义插件
│   ├── C++-UsePluginInside
│   ├── C++-UsePluginOutside
│   ├── LoadDataFromNpz
│   ├── MultipleVersion
│   ├── PluginProcess
│   ├── PluginReposity
│   │   ├── AddScalarPlugin-TRT8
│   │   ├── BatchedNMS_TRTPlugin-TRT8
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
├── 06-UseFrameworkTRT   // 框架中使用TensorRT     
│   └── Torch-TensorRT
├── 07-Tool              // 工具
│   ├── FP16FineTuning   // fp16微调 
│   ├── Netron           // netron 可视化模型文件  
│   ├── NetworkInspector // 网络检查器  
│   │   └── C++
│   ├── NetworkPrinter   // 网络打印  
│   ├── NsightSystems    // nsys  
│   ├── nvtx             // 每一段时间对应的是哪一段程序 libnvToolsExt.so  https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/NVTX     
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
│   ├── trex      // 引擎浏览器    如看 layerinfo 的svg   
│   │   ├── model
│   │   ├── trex
│   │   └── trex.egg-info
│   └── trtexec  // 转换与资源分析工具，可以轻量化    
├── 08-Advance   // 高阶 
│   ├── BuilderOptimizationLevel                   // 构建器优化级别 
│   ├── CreateExecutionContextWithoutDeviceMemory  // 不使用设备内存构建执行器上下文  
│   ├── C++StaticCompilation      // cpp 静态编译 
│   ├── CudaGraph                 // cuda graph
│   ├── DataFormat                // 数据格式 ，如 HC/4HW4
│   ├── DynamicShapeOutput        // 输出shape为动态 
│   ├── EmptyTensor               // 空tensor 
│   ├── Event                     // cuda 事件 
│   ├── ExternalSource            // 外部源   
│   ├── HardwareCompatibility     // 硬件兼容性 
│   ├── LabeledDimension          // 维度被标记
│   ├── MultiContext              // 多上下文
│   ├── MultiOptimizationProfile  // 多优化分析配置
│   ├── MultiStream               // 多流
│   ├── Safety-TODO               // 安全性 qnx 系统  
│   ├── Sparsity                  // 稀疏，如稀疏训练   
│   │   └── pyTorch-ONNX-TensorRT-ASP 
│   ├── StreamAndAsync            // 流与异步
│   ├── StrictType                // 严格类型， 如设置部分网络层精度   
│   ├── TensorRTGraphSurgeon      // TensorRT图检查
│   ├── TorchOperation            // torch算子
│   └── VersionCompatibility      // 版本兼容性
├── 09-BestPractice               // 最佳实践，包括图优化
│   ├── AdjustReduceLayer         // 调整reduce 层
│   ├── AlignSize                 //  对齐尺寸
│   ├── ComputationInAdvance      // 高阶计算
│   │   └── Convert3DMMTo2DMM     // 3D矩阵乘法到2D矩阵乘法 
│   ├── ConvertTranposeMultiplicationToConvolution     //  转置乘法到卷积
│   ├── EliminateSqueezeUnsqueezeTranspose             // 消除转置中的squeeze/unsqueeze
│   ├── IncreaseBatchSize                              // 增大batch size
│   ├── UsingMultiOptimizationProfile                  // 使用多个优化分析文件  
│   ├── UsingMultiStream                               // 使用多流
│   └── WorkFlowOnONNXModel                            // onnx 工作流
├── 10-ProblemSolving                                  // 问题解决中  
│   ├── ParameterCheckFailed                           // 参数检查失败  
│   ├── SliceNodeWithBoolIO                            // slice 层带有bool 量  
│   ├── WeightsAreNotPermittedSinceTheyAreOfTypeInt32  // 权重类型是INT32
│   └── WeightsHasCountXButYWasExpected                // 权重数目有误  
├── 51-Uncategorized
├── 52-Deprecated  // 废弃的接口，针对8.4 版本     
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

## 有用的链接 

+ [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
+ [Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) and [Document Archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
+ [Supporting Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)      
+ [C++ API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api) and [Python API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/) and [API Change](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)
+ [Operator algorithm](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)
+ [Onnx Operator Supporting Matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
+ [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)
+ [NSight- Systems](https://developer.nvidia.com/nsight-systems)
+ [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt)
+ [ONNX model zoo](https://github.com/onnx/models)
---
