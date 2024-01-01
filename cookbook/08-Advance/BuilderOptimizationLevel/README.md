# 构建器优化级别    

+ 示例代码创建了一个硬件兼容性TensorRT引擎，该引擎可以在Ampere或更高版本的GPU上运行      

## 运行命令  

```shell
python main.py
```

## 结果    

```txt
Test <Level=0>
Time of building: 1570.186331ms
Time of inference: 60.682590ms
Test <Level=0> finish!

Test <Level=1>
Time of building: 62134.697218ms
Time of inference: 22.540259ms
Test <Level=1> finish!

Test <Level=2>
Time of building: 60654.759621ms
Time of inference: 22.544082ms
Test <Level=2> finish!

Test <Level=3>
Time of building: 127966.520685ms
Time of inference: 23.019233ms
Test <Level=3> finish!

Test <Level=4>
Time of building: 135040.661363ms
Time of inference: 16.593110ms
Test <Level=4> finish!

Test <Level=5>
Time of building: 254460.420299ms
Time of inference: 16.055930ms
Test <Level=5> finish!
```
