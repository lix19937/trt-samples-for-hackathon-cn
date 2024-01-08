## cudaHostAlloc

```cpp
auto nbytes = data_size * sizeof(int);
if ((cudaHostAlloc((void **)&h_out_data, nbytes, cudaHostAllocMapped)) != 0){
    printf("ERROR, cudaHostAlloc in %s at line %d", __FILE__, __LINE__);
    return false;
}else{
    printf("cudaHostAlloc output buffer [%d] success!", nbytes);
}

cudaHostGetDevicePointer((void **)&d_out_data, (void *)h_out_data, 0);     
```          

```c
cudaError_t cudaHostAlloc	(void ** 	pHost,
                             size_t 	size,
                             unsigned int 	flags)	
```

**cudaHostAlloc** vs  **cudaMallocHost**        

![v](./cha_cmh.png)     


## cudaHostRegister   
```cpp
__host__​cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags )   
Registers an existing host memory range for use by CUDA.
```

是否修改了ptr 空间的内容？  
ptr值没有改变   

更多内容 https://github.com/lix19937/history/blob/main/cuda/%E9%94%81%E9%A1%B5%E5%86%85%E5%AD%98.md    

------------------------------------

ref   

https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd5c991beb38e2b8419f50285707ae87e       
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory       
https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g15a3871f15f8c38f5b7190946845758c.html    
