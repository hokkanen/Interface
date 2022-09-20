Compile for host execution with 
```
nvcc -x cu --extended-lambda main.cpp
```
and GPU execution with
```
nvcc -x cu --extended-lambda -DHAVE_CUDA=1 main.cpp
```