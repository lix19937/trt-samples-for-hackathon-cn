

使用TRT API的基本示例     

## 编译命令    
```shell
make test
```    

## 说明    
```cpp  
// 设置 TRT log 级别  
static Logger     gLogger(ILogger::Severity::kERROR);

// 如果引擎文件（本地，已经系列化了）已经存在  
    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize); // 得到序列化引擎（位于内存中）

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize); // 反序列化，实际就是结构化 
    }
 // 本地无引擎文件   从API 方式构建网络      
    else
    {
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}}); // 增加网络的输入 NCHW CHW 为动态维度    
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        // 设置优化分析配置  
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor); // 直连层  
        network->markOutput(*identityLayer->getOutput(0)); // 标记直连层的输出为网络输出  
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config); // 构建序列化引擎 （位于内存中）

        IRuntime *runtime {createInferRuntime(gLogger)}; // log 作为运行时的 logger
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size()); // 反序列化  

        std::ofstream engineFile(trtFile, std::ios::binary); //  创建trtFile， 将引擎内容写到 trtFile 

        engineFile.write(static_cast<char *>(engineString->data()), engineString->size()); // 序列化内容写到plan 
    }

    // 输入输出 tensor 数目  
    long unsigned int        nIO     = engine->getNbIOTensors(), nInput  = 0, nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    //  执行器上下文, 注意 和cuda ctx不是一回事  
    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(vTensorName[0].c_str(), Dims32 {3, {3, 4, 5}});

    for (int i = 0; i < nIO; ++i) {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" "); // tensor 数据类型  
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" "); // tensor 维度 
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");// ctx 维度 
        std::cout << vTensorName[i] << std::endl;
    }

    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i) {
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j) {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    std::vector<void *> vBufferH {nIO, nullptr}, vBufferD {nIO, nullptr};
    for (int i = 0; i < nIO; ++i) {
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    float *pData = (float *)vBufferH[0];
    for (int i = 0; i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str())); ++i) {
        pData[i] = float(i);
    }
    // 主机到设备  vBufferD为input buff 
    for (int i = 0; i < nInput; ++i){
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    // 设置地址  
    for (int i = 0; i < nIO; ++i) {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }

    //  enqueueV3 版本  
    context->enqueueV3(0);

    // 设备到主机   
    for (int i = nInput; i < nIO; ++i) {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
    }

}
```
