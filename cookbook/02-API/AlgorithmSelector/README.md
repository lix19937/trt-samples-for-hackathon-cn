

## 运行命令  

```shell
python3 main.py
```

## 说明   

```python
   def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        # we print the alternative algorithms of each layer here
        nInput = layerAlgorithmContext.num_inputs
        nOutput = layerAlgorithmContext.num_outputs
        print("Layer %s,in=%d,out=%d" % (layerAlgorithmContext.name, nInput, nOutput))

        for i in range(nInput + nOutput):
            print("    %s    %2d: shape=%s" % ("Input " if i < nInput else "Output", i if i < nInput else i - nInput, layerAlgorithmContext.get_shape(i)))

        for i, algorithm in enumerate(layerAlgorithmList):
            print("    algorithm%3d:implementation[%10d], tactic[%20d], timing[%7.3fus], workspace[%s]" % ( \
                i,
                algorithm.algorithm_variant.implementation,
                algorithm.algorithm_variant.tactic,
                algorithm.timing_msec * 1000,
                getSizeString(algorithm.workspace_size)))

        # 全局最短时间  TRT默认方式  
        if self.iStrategy == 0:  # choose the algorithm spending shortest time, the same as TensorRT
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmin(timeList)]

        # 最长时间  
        elif self.iStrategy == 1:  # choose the algorithm spending longest time to get a TensorRT engine with worst performance, just for fun :)
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmax(timeList)]

        # 使用最小workspace 
        elif self.iStrategy == 2:  # choose the algorithm using smallest workspace
            workspaceSizeList = [algorithm.workspace_size for algorithm in layerAlgorithmList]
            result = [np.argmin(workspaceSizeList)]

        # 使用用户指定的某种算法   
        elif self.iStrategy == 3:  # choose one certain algorithm we have known
            # This strategy can be a workaround for building the exactly same engine for many times, but Timing Cache is more recommended to do so.
            # The reason is that function select_algorithms is called after the performance test of all algorithms of a layer is finished (you can find algorithm.timing_msec > 0),
            # so it will not save the time of the test.
            # On the contrary, performance test of the algorithms will be skiped using Timing Cache (though performance test of Reformating can not be skiped),
            # so it surely saves a lot of time comparing with Algorithm Selector.
            if layerAlgorithmContext.name == "(Unnamed Layer* 0) [Convolution] + (Unnamed Layer* 1) [Activation]":
                # the number 2147483648 is from VERBOSE log, marking the certain algorithm
                result = [index for index, algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 2147483648]
            else:  # keep all algorithms for other layers
                result = list(range(len(layerAlgorithmList)))

        else:  # default behavior: keep all algorithms
            result = list(range(len(layerAlgorithmList)))

        return result
```
