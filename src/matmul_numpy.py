from os import environ, path
from numpy.typing import NDArray
import numpy as np

from utils.chrono import Chrono

FILE = path.basename(__file__)

def random_matrix(n: int) -> NDArray[np.float32]:
    return np.random.default_rng().random((n, n), dtype=np.float32)
   
def matmul(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.matmul(a, b)

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