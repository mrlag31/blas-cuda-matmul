# python    : python matmul_python.py
# numpy     : python matmul_numpy.py
# tensorflow: python matmul_tensorflow.py
# cpp       : g++ matmul_cpp.cpp -o main
# openblas  : g++ matmul_openblas.cpp -o main -l cblas
# cuda      : nvcc matmul_cuda.cu -arch=? -o main
# cublas    : nvcc matmul_cublas.cu -arch=? -l cublas -o main

import os
import sys
import subprocess as sp
import argparse as ap

GPP = "g++"
NVCC = "nvcc"
BUILDCPP = "build-cpp"
BUILDCU = "build-cu"

CUDA_PATH = ""
ARCH = "all"

MATMUL_PYTHON = "src" + os.sep + "matmul_python.py"
MATMUL_NUMPY = "src" + os.sep + "matmul_numpy.py"
MATMUL_TESORFLOW = "src" + os.sep + "matmul_tensorflow.py"
MATMUL_CPP = BUILDCPP + os.sep + "matmul_cpp"
MATMUL_OPENBLAS = BUILDCPP + os.sep + "matmul_openblas"
MATMUL_CUDA = BUILDCU + os.sep + "matmul_cuda"
MATMUL_CUBLAS = BUILDCU + os.sep + "matmul_cublas"

PYTHON = sys.orig_argv[0]

def local_run(*args):
    print(f"Running '{' '.join(args)}'")
    c = sp.run(args, capture_output = True, encoding = "utf8")
    if c.returncode != 0:
        raise Exception(f"Command '{' '.join(args)}' failed with the following error:\n{c.stderr}")
    return c

def gpp_check():
    local_run(GPP, "--version")
    if not os.path.exists(BUILDCPP):
        os.mkdir(BUILDCPP)

def cuda_check():
    global CUDA_PATH
    global ARCH 

    CUDA_PATH = os.environ["CUDA_PATH"]
    print(f"CUDA found at {CUDA_PATH}")
    
    try:
        local_run(NVCC, "--version")
    except FileNotFoundError:
        print(f"'{NVCC}' not found in PATH, manually adding it.")
        os.environ["PATH"] += os.pathsep + CUDA_PATH + os.sep + "bin"
        local_run(NVCC, "--version")
    
    try:
        c = local_run(CUDA_PATH + os.sep + "extras" + os.sep + "demo_suite" + os.sep + "deviceQuery")
        ARCH = "compute_" + c.stdout.splitlines(False)[8].split()[-1].replace('.', '')
    except FileNotFoundError:
        print("Unable to find 'queryDevice' to query your GPU's compute capability. Cannot properly set 'arch'")
    
    print(f"Cuda arch set to {ARCH}.")
    
    if not os.path.exists(BUILDCU):
        os.mkdir(BUILDCU)

def build_cpp():
    local_run(GPP, "src/matmul_cpp.cpp", "-o", MATMUL_CPP)

def build_openblas():
    local_run(GPP, "src/matmul_openblas.cpp", "-l", "cblas", "-o", MATMUL_OPENBLAS)

def build_cuda():    
    local_run(NVCC, "src/matmul_cuda.cu", f"-arch={ARCH}", "-o", MATMUL_CUDA)

def build_cublas():
    local_run(NVCC, "src/matmul_cublas.cu", f"-arch={ARCH}", "-l", "cublas", "-o", MATMUL_CUBLAS)    

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog = f"{PYTHON} matmul-build.py",
        description = "Builds the matmul example programs written in Python, C++ and CUDA, with or without BLAS"
    )
    
    parser.add_argument("-cpp", action = "store_true",
                        help = "Build the 'cpp' matmul.")
    parser.add_argument("-cuda", action = "store_true",
                        help = "Build the 'cuda' matmul.")
    parser.add_argument("-blas", action = "store_true",
                        help = "Adds the BLAS variants ('openblas' if 'cpp' is set, 'cublas' if 'cuda' is set).")
    args = parser.parse_args()
    
    action = False

    if args.cpp:
        gpp_check()
        build_cpp()
        if args.blas:
            build_openblas()
        action = True
    
    if args.cuda:
        cuda_check()
        build_cuda()
        if args.blas:
            build_cublas()
        action = True
    
    if not action:
        print("Nothing to do.")
        parser.print_help()
else:
    print(f"Do not import {__file__}, use it directly!")
    exit()