# 后量化  

+ 使用TensorRT 进行INT8 PTQ  

+ 更多内容见 03-BuildEngineByTensorRTAPI/MNISTExample-pyTorch

+ 输出: result.log

## 运行命令    

```shell
python3 main.py
```

## 说明  
```python 

IInt8Calibrator
IInt8LegacyCalibrator     基本废弃 
IInt8EntropyCalibrator
IInt8EntropyCalibrator2   最常见  
IInt8MinMaxCalibrator

见py  https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html  

c++   https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html

```