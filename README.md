## Compile 

Sequential host execution: 
```
nvcc -x cu --extended-lambda main.cpp
```
Parallel GPU execution using CUDA:
```
nvcc -x cu --extended-lambda -DHAVE_CUDA=1 main.cpp
```
Parallel GPU execution using CUDA and UMPIRE:
```
nvcc -x cu --extended-lambda -DHAVE_CUDA=1 -DHAVE_UMPIRE=1 main.cpp -I/umpire/include -L/umpire/lib -lcamp -lumpire
```