#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorrt as trt

# 这里可以和 ../WeightsAreNotPermittedSinceTheyAreOfTypeInt32/main.py 对比   
kernel = np.ones([32, 3, 5, 5], dtype=np.float32)  # count of weight is incorrect
bias = np.ones(7, dtype=np.float32) # fp32  

# 构建器 网络 config   
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# 搭建网络  
inputT0 = network.add_input("inputT0", trt.float32, (1, 3, 600, 800))  
convolutionLayer = network.add_convolution_nd(inputT0, 32, [5, 5], kernel, bias)

network.mark_output(convolutionLayer.get_output(0))

# 序列化引擎  
engineString = builder.build_serialized_network(network, config)
