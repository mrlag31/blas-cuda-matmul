from os import environ, path
from typing import List
from random import random

from utils.chrono import Chrono

FILE = path.basename(__file__)

class Matrix:
    data: List[float]
    size: int

def random_matrix(n: int) -> Matrix:
    a = Matrix()
    a.data = list(random() for _ in range(n * n))
    a.size = n
    return a
   
def matmul(a: Matrix, b: Matrix) -> Matrix:
    n = a.size
    
    c = Matrix()
    c.data = [0.0] * (n * n)
    c.size = n
    
    pos = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c.data[pos(i, j)] += a.data[pos(i, k)] * b.data[pos(k, j)]
    
    return c

def main():
    MATRIX_SIZE = int(environ.get("YOB_DEMO_MS"))
    
    a = random_matrix(MATRIX_SIZE)
    b = random_matrix(MATRIX_SIZE)
    
    chr = Chrono()
    with chr:
        c = matmul(a, b)
    chr.print(FILE)

if __name__ == "__main__":
    main()