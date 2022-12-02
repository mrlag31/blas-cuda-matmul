import os
import sys
import subprocess as sp
import argparse as ap

BUILDCPP = "build-cpp"
BUILDCU = "build-cu"

MATMUL_PYTHON = "src" + os.sep + "matmul_python.py"
MATMUL_NUMPY = "src" + os.sep + "matmul_numpy.py"
MATMUL_TESORFLOW = "src" + os.sep + "matmul_tensorflow.py"
MATMUL_CPP = BUILDCPP + os.sep + "matmul_cpp"
MATMUL_OPENBLAS = BUILDCPP + os.sep + "matmul_openblas"
MATMUL_CUDA = BUILDCU + os.sep + "matmul_cuda"
MATMUL_CUBLAS = BUILDCU + os.sep + "matmul_cublas"

MATSIZE_ENV = "YOB_DEMO_MS"

PYTHON = sys.orig_argv[0]

def local_run(*args):
    try:
        c = sp.run(args, capture_output = True, encoding = "utf8")
    except FileNotFoundError as e:
        print(f"{e.filename} not found, skipped.")
    else:
        if c.returncode != 0:
            raise Exception(f"Command '{' '.join(args)}' failed with the following error:\n{c.stderr}")
        print(c.stdout, end='')

if __name__ == "__main__":   
    parser = ap.ArgumentParser(
        prog = f"{PYTHON} matmul-run.py",
        description = "Executes the matmul example programs written in Python, C++ and CUDA, with or without BLAS"
    )
    
    parser.add_argument("-py", action = "store_true",
                        help = "Run matmul_python.")
    parser.add_argument("-np", action = "store_true",
                        help = "Run matmul_numpy.")
    parser.add_argument("-tf", action = "store_true",
                        help = "Run matmul_tensorflow.")
    parser.add_argument("-cp", action = "store_true",
                        help = "Run matmul_cpp.")
    parser.add_argument("-ob", action = "store_true",
                        help = "Run matmul_openblas.")
    parser.add_argument("-cu", action = "store_true",
                        help = "Run matmul_cuda.")
    parser.add_argument("-cb", action = "store_true",
                        help = "Run matmul_cublas.")
    parser.add_argument("-all", action = "store_true",
                        help = "Run everything available")
    parser.add_argument("-blas", action = "store_true",
                        help = "Run everything using blas (not python, not cpp, not cuda)")
    parser.add_argument("MS", type = int, nargs = "?", default = 0,
                        help = f"Matrix size to use (deaults to 128 or to the environment variable {MATSIZE_ENV}).")
    args = parser.parse_args()
    
    if args.MS <= 0:
        os.environ[MATSIZE_ENV] = os.environ.get(MATSIZE_ENV, "128")
    else:
        os.environ[MATSIZE_ENV] = str(args.MS)
    MATSIZE = os.environ[MATSIZE_ENV]
    print(f"Matrix size: {MATSIZE}")
    
    action = False
    if args.py or args.all:
        local_run(PYTHON, MATMUL_PYTHON)
        action = True
    if args.np or args.all or args.blas:
        local_run(PYTHON, MATMUL_NUMPY)
        action = True
    if args.tf or args.all or args.blas:
        local_run(PYTHON, MATMUL_TESORFLOW)
        action = True
    if args.cp or args.all:
        local_run(MATMUL_CPP)
        action = True
    if args.ob or args.all or args.blas:
        local_run(MATMUL_OPENBLAS)
        action = True
    if args.cu or args.all:
        local_run(MATMUL_CUDA)
        action = True
    if args.cb or args.all or args.blas:
        local_run(MATMUL_CUBLAS)
        action = True
    
    if not action:
        print("Nothing to do.")
        parser.print_help()
else:
    print(f"Do not import {__file__}, use it directly!")
    exit()