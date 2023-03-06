# Matmul Examples: Python, C++, CUDA and BLAS

## Requirements

Prefer to use your package manager on Linux instead of manually installing each program.
- [Python](https://www.python.org/downloads/)
- [CUDA](https://developer.nvidia.com/cuda-downloads) and [cudNN](https://developer.nvidia.com/rdp/cudnn-download) (make sure you have matching cudNN and CUDA versions)
- [openblas](https://www.openblas.net/)
- `cblas` package

## Installation

1. Clone this repository
```
git clone https://...
```
2. Create a virtual environment and source it
```
python -m venv env
source env/bin/activate
```
3. Install the pip requirements
```
pip install -r requirements.txt
```

## Building
The script `matmul-build.py` will try to build all of the requested scripts. If it was not able to compile one of the scripts, the relevent error will be shown. To build everything, use:
```
python matmul-build.py -cpp -cuda -blas
```

## Running
The script `matmul-script.py` will try to run all of the requested programs. To run everything, use:
```
python matmul-run.py -all <matrix size>
```
Beware of `<matrix size>`, if it is too large some scripts will hang for a very long time. Prefer use `-blas` instead of `-all` for sizes larger than 1000.

## Results

This is a copy of the results presented in the intervention.

Matrix Size | Python Basic | C++ Basic | CUDA Basic | NumPy | OpenBLAS | Tensorflow | CuBLAS
---|---|---|---|---|---|---|---
32 | 17.5 ms | | | | | |
64 | 130 ms | 4.5 ms | | | | |
128 | 1.01 s | 35.5 ms | 1.4 ms | | | |
256 | 8.05 s | 272 ms | 10.7 ms | 1.5 ms | 1.6 ms | 230 ms | 150 µs
512 | 65.8 s | 2.5 ms | 79.4 ms | 3.0 ms | 228 ms | 228 ms | 670 µs
1024 | | 22.5 ms | 588 ms | 10.3 ms | 12.7 ms | 225 ms | 3.36 ms
2048 | | | 4.64 s | 69.5 ms | 75.0 ms | 238 ms | 20.1 ms
4096 | | | | 530 ms | 556 ms | 350 ms | 143 ms

Specs:
- *CPU: Intel i5-6300HQ*
- *GPU: NVidia GeForce GTX 960M*