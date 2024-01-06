# 属于计算图优化范畴   

## 说明  

+ test the code in two scenarios.

  1. Conv -> Conv -> Unsqueeze -> Add -> Squeeze -> ReLU -> ... -> Conv ->Transpose -> Add -> Transpose -> ReLU -> ... -> Conv.
      conv 合并， Unsqueeze消除， Squeeze 消除，    
  3. All Squeeze / Unsqueeze / Transpose layers in scenario 1 are removed.

## 结果  

+ 所有的 Conv+Add+ReLU 合并为 1个kernel by TensorRT


+ Nearly doubled performance after optimization
