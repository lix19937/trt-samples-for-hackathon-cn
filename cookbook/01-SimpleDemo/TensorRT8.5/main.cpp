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

    // 如果有 plan 文件  
    if (access(trtFile.c_str(), F_OK) == 0) {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0){
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr){
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else {  // 没有plan,没有onnx  用户自己使用trt api搭建网络，  从这里我们可以看到完整过程   
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        // 搭建网络 
        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);
        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));

        // 序列化网络  
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0){
            std::cout << "Failed building serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // 反序列化网络  
        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr){
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        // 存储plan 到本地 
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

    long unsigned int        nIO     = engine->getNbIOTensors();
    long unsigned int        nInput  = 0, nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    // 统计 in out tensor数目 
    for (int i = 0; i < nIO; ++i){
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    // 创建上下文  
    IExecutionContext *context = engine->createExecutionContext();  

    // 设置 输入shape 
    context->setInputShape(vTensorName[0].c_str(), Dims32 {3, {3, 4, 5}});

    // 打印tensor属性  
    for (int i = 0; i < nIO; ++i) {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    std::vector<int> vTensorSize(nIO, 0);
    // 计算每一个tensor的size 以字节数为单位   
    for (int i = 0; i < nIO; ++i) {
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j){
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    // 给每一个tensor分配空间  
    std::vector<void *> vBufferH {nIO, nullptr}, vBufferD {nIO, nullptr};
    for (int i = 0; i < nIO; ++i) {
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    // 给输入tensor赋值  实际情况可能就是输入图像   
    auto pData = (float *)vBufferH[0];
    for (int i = 0; i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str())); ++i) {
        pData[i] = float(i);
    }
    // 主机到设备  
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    // 绑定tensor地址   
    for (int i = 0; i < nIO; ++i) {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }

    // 推理执行  
    context->enqueueV3(0);

    // 设备到主机  
    for (int i = nInput; i < nIO; ++i) {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
    }

    // 打印结果  
    for (int i = 0; i < nIO; ++i) {
        printArrayInformation((float *)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, true);
    }

    // 资源释放  
    for (int i = 0; i < nIO; ++i) {
        delete[] (char *)vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
}

int main(){
    CHECK(cudaSetDevice(0));
    // 1次热身    
    run();
    
    run();
    return 0;
}
