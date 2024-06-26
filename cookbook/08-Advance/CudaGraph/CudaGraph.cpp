/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string trtFile {"./model.plan"};
static Logger     gLogger(ILogger::Severity::kERROR);

void run(){
    ICudaEngine *engine = nullptr;
    
    // 如果有plan 
    if (access(trtFile.c_str(), F_OK) == 0) {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        // 从磁盘读文件  
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0){
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        // 构建rt 对象  
        IRuntime *runtime {createInferRuntime(gLogger)};  
        // 反序列化内存数据(序列化的)
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else {// 无plan，从trt api 搭建网络 
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);
        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));

        // 构建序列化网络 
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0) {
            std::cout << "Failed building serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // 反序列化引擎  
        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        // 保存plan文件 
        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile){
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail()){
            std::cout << "Failed saving .plan file!" << std::endl;
            return;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }
    
    // 创建执行器上下文 （TRT管理引擎所需的gpu内存）  
    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {3, {3, 4, 5}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i) {
        nInput += int(engine->bindingIsInput(i));
    }
    int nOutput = nBinding - nInput;
    for (int i = 0; i < nBinding; ++i) {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i) {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j) {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // 主机指针  
    std::vector<void *> vBufferH {nBinding, nullptr};
    // 设备指针 
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i) {
        vBufferH[i] = (void *)new char[vBindingSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    // 赋值  
    float *pData = (float *)vBufferH[0];
    for (int i = 0; i < vBindingSize[0] / dataTypeToSize(engine->getBindingDataType(0)); ++i) {
        pData[i] = float(i);
    }

    // 给主机buff分配空间  
    int  inputSize = 3 * 4 * 5, outputSize = 1;
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; ++i) {
        outputSize *= outputShape.d[i];
    }
    std::vector<float>  inputH0(inputSize, 1.0f);
    std::vector<float>  outputH0(outputSize, 0.0f);
    
    // bindings 即输入输出指针 
    std::vector<void *> binding = {nullptr, nullptr};
    CHECK(cudaMalloc(&binding[0], sizeof(float) * inputSize));
    CHECK(cudaMalloc(&binding[1], sizeof(float) * outputSize));
    for (int i = 0; i < inputSize; ++i) {
        inputH0[i] = (float)i;
    }

    // 运行推理和使用 CUDA Graph 要用的流 ； 类的构造或init  
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 捕获 CUDA Graph 之前要运行一次推理
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a2f4429652736e8ef6e19f433400108c7
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }
    
    // TRT8510 支持V3
    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i) {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream); // 同步操作不用在 graph 内同步

    for (int i = 0; i < nBinding; ++i){
        printArrayInformation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true, true);
    }

    //----------------------------------------------------------------------   
    // 首次捕获 CUDA Graph 并运行推理
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = nullptr;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i) {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    //-----------------------捕获结束-----------------------------------------------   

    //-----------------------graph 执行 -----------------------------------------------
    cudaGraphLaunch(graphExec, stream);
    
    // 等待执行完   
    cudaStreamSynchronize(stream);

    // 输入尺寸改变后，也需要首先运行一次推理，然后重新捕获 CUDA Graph，最后再运行推理
    context->setBindingDimensions(0, Dims32 {3, {2, 3, 4}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;

    for (int i = 0; i < nBinding; ++i) {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j){
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    
    // 这里偷懒，因为本次推理绑定的输入输出数据形状不大于上一次推理，所以这里不再重新准备所有 buffer
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i) {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream); // 注意，不用在 graph 内同步

    for (int i = 0; i < nBinding; ++i) {
        printArrayInformation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true, true);
    }

    //----------------------------------------------------------------------   
    // 再次捕获 CUDA Graph 并运行推理
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i) {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    //-----------------------捕获结束-----------------------------------------------   

    //-----------------------graph 执行 -----------------------------------------------
    cudaGraphLaunch(graphExec, stream);

    // 等待   
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    for (int i = 0; i < nBinding; ++i) {
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    return;
}

int main(){
    CHECK(cudaSetDevice(0));
    // 热身一次   
    run();
    
    run();
    return 0;
}
