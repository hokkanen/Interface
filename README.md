## Compile 

Sequential host execution: 
```
nvcc -x cu --extended-lambda main.cpp
```
Parallel GPU execution using CUDA:
```
nvcc -x cu --extended-lambda -DHAVE_CUDA=1 main.cpp
```
Parallel GPU execution using CUDA with memory pool allocation:
```
nvcc -x cu --extended-lambda -DHAVE_CUDA=1 -DCUDA_MEMPOOL=1 main.cpp
```