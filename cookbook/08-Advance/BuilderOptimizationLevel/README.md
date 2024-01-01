# 构建器优化级别    

+ 示例代码创建了一个硬件兼容性TensorRT引擎，该引擎可以在Ampere或更高版本的GPU上运行      

设置生成器优化级别。设置更高的优化级别允许优化器花费更多时间搜索优化机会。与以较低优化水平构建的引擎相比，所得到的发动机可以具有更好的性能。     
**默认优化级别为3**。有效值包括从0到最大优化级别（当前为5）的整数。将其设置为大于最大级别会导致与最大级别相同的行为。

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
