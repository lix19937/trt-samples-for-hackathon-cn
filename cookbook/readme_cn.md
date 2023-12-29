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
├── 01-SimpleDemo
│   └── TensorRT8.5
├── 02-API
│   ├── AlgorithmSelector
│   ├── AuxStream
│   ├── Builder
│   ├── BuilderConfig
│   ├── CudaEngine
│   ├── EngineInspector
│   ├── ErrorRecoder
│   ├── ExecutionContext
│   ├── GPUAllocator
│   ├── HostMemory
│   ├── INT8-PTQ
│   │   └── C++
│   ├── Layer
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
│   ├── Logger
│   ├── Network
│   ├── ONNXParser
│   ├── OptimizationProfile
│   ├── OutputAllocator
│   ├── Profiler
│   ├── ProfilingVerbosity
│   ├── Refit
│   ├── Runtime
│   ├── TacticSource
│   ├── Tensor
│   └── TimingCache
├── 03-BuildEngineByTensorRTAPI
│   ├── MNISTExample-pyTorch
│   │   └── C++
│   ├── TypicalAPI-pyTorch
├── 04-BuildEngineByONNXParser
│   ├── pyTorch-ONNX-TensorRT
│   │   └── C++
│   ├── pyTorch-ONNX-TensorRT-QAT
├── 05-Plugin
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
├── 06-UseFrameworkTRT
│   └── Torch-TensorRT
├── 07-Tool
│   ├── FP16FineTuning
│   ├── Netron
│   ├── NetworkInspector
│   │   └── C++
│   ├── NetworkPrinter
│   ├── NsightSystems
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
│   ├── trex
│   │   ├── model
│   │   ├── trex
│   │   └── trex.egg-info
│   └── trtexec
├── 08-Advance
│   ├── BuilderOptimizationLevel
│   ├── CreateExecutionContextWithoutDeviceMemory
│   ├── C++StaticCompilation
│   ├── CudaGraph
│   ├── DataFormat
│   ├── DynamicShapeOutput
│   ├── EmptyTensor
│   ├── Event
│   ├── ExternalSource
│   ├── HardwareCompatibility
│   ├── LabeledDimension
│   ├── MultiContext
│   ├── MultiOptimizationProfile
│   ├── MultiStream
│   ├── Safety-TODO
│   ├── Sparsity
│   │   └── pyTorch-ONNX-TensorRT-ASP
│   ├── StreamAndAsync
│   ├── StrictType
│   ├── TensorRTGraphSurgeon
│   ├── TorchOperation
│   └── VersionCompatibility
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
