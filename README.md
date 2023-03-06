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