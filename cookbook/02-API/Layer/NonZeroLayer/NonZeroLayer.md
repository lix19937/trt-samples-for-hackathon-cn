# Non Zero Layer

+ Simple example

---

## Simple example

+ Refer to SimpleExample.py

```txt
[ 0]Input -> DataType.FLOAT (3, 4, 5) (3, 4, 5) inputT0
[ 1]Output-> DataType.INT32 (3, -1) (3, -1) (Unnamed Layer* 0) [NonZero]_output
inputT0
[[[ 0.  1.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.]
  [ 0.  0.  0.  2.  0.]
  [ 0.  0.  0.  0.  3.]]

 [[ 0.  0.  0.  0.  0.]
  [ 4.  5.  6.  7.  8.]
  [ 0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.]]

 [[ 0.  9.  0.  0.  0.]
  [ 0. 10.  0.  0.  0.]
  [ 0. 11.  0.  0.  0.]
  [ 0. 12.  0.  0.  0.]]]
(Unnamed Layer* 0) [NonZero]_output
[[0 0 0 1 1 1 1 1 2 2 2 2]
 [0 2 3 1 1 1 1 1 0 1 2 3]
 [1 3 4 0 1 2 3 4 1 1 1 1]]
```

+ Notice that before the first inference, **context.get_tensor_shape(lTensorName[1])** returns (3,-1) though the shape of input tensor is set.