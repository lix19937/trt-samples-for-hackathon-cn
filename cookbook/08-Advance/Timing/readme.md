
# Timing

让候选算子在目标GPU上直接跑跑，统计出性能，最后通过比对选出最优解。   
TensorRT把这个过程叫做Timing，TensorRT甚至可以将优化的中间过程存储下来供分析，
叫做timing caching（通过trtexec --timingCacheFile=<file>）      
