from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from utils.chrono import Chrono
from time import time

FILE = path.basename(__file__)

def random_matrix(n: int) -> tf.Tensor:
    return tf.random.uniform((n, n))
   
def matmul(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    c = tf.matmul(a, b)
    float(c[0, 0]) # This is done so we are SURE that the matmul has completed
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