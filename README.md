# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)


AITER is AMD’s centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.
 

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.



## Installation
`git clone --recursive https://github.com/ROCm/aiter.git`

or

`git submodule sync ; git submodule update --init --recursive`

Then
```
cd aiter
python3 setup.py develop
```

## Run operators supported by aiter

There are number of op test, you can run them with: `python3 op_tests/test_layernorm2d.py`
|  **Ops**   | **Description**                                                                             |
|------------|---------------------------------------------------------------------------------------------|
|GEMM        | D=AxB+C                                                                                     |
|FusedMoE    | bf16 balabala                                                                               |
|WIP         | coming soon...                                                                              |
