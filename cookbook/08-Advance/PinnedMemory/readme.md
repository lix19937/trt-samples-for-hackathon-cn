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
cudaError_t cudaHostAlloc	(	void ** 	pHost,
                                size_t 	size,
                                unsigned int 	flags	)	
```

**cudaHostAlloc** vs  **cudaMallocHost**        

![](./cha_cmh.png)     

------------------------------------

ref   

https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd5c991beb38e2b8419f50285707ae87e    
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory    
https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g15a3871f15f8c38f5b7190946845758c.html    
