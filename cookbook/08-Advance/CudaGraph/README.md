#

## 运行命令  

```shell
make test
```

## 输出参考: ./result-*.log

## 注意  

可提取为类的方式，graph捕获的执行一次放在init中，launch 在运行时     
可更新图 cudaGraphExecUpdate     

**cudaStreamCaptureModeGlobal** vs **cudaStreamCaptureModeThreadLocal**         
```cpp 
void beginCapture(TrtCudaStream& stream){
    cudaCheck(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
}
```
```cpp  
void beginCapture(cudaStream_t& stream){
    // cudaStreamCaptureModeGlobal is the only allowed mode in SAFE CUDA
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}
```


the DLA suggraph as a single big kernel. CUDA graph don't support it.      


更多内容 https://github.com/lix19937/history/blob/main/cuda/cudagraph.md     
