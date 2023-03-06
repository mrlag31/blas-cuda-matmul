#include <filesystem>
#include <vector>
#include <random>
#include <cstdlib>

#include "./utils/chrono.cpp"
#include "cuda.h"
#include "cublas_v2.h"

/*
    This is the matrix multiplication using CuBLAS.
    It is the fastest method.
*/

static const std::string file = std::filesystem::path(__FILE__).filename();

struct Matrix {
    float* data;
    std::size_t size;
};

Matrix random_matrix(std::size_t n)
{
    auto mat = Matrix{
        .data = new float[n * n] {0.0},
        .size = n
    };

    // This is for generating a random matrix
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> urd(0.0, 1.0);
    for (std::size_t i = 0; i < n * n; i++)
        mat.data[i] = urd(gen);
    
    // This is allocating it on the GPU and copying it
    float* dMat; std::size_t size = sizeof(float) * n * n;
    cudaMalloc(&dMat, size);
    cudaMemcpy(dMat, mat.data, size, cudaMemcpyHostToDevice);
    delete mat.data; mat.data = dMat;

    return mat;
}

Matrix matmul(const Matrix& a, const Matrix& b, cublasHandle_t handle)
{
    auto n = a.size;

    auto c = Matrix{
        .data = new float[n * n] {0.0},
        .size = n
    };

    // Allocate C on the GPU
    float* dC; std::size_t size = sizeof(float) * n * n;
    cudaMalloc(&dC, size);
    cudaMemcpy(dC, c.data, size, cudaMemcpyHostToDevice);
    delete c.data; c.data = dC;

    // BLAS call. alpha and beta are pointers instead of values
    float alpha = 1.0, beta = 0.0;
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        a.data, n,
        b.data, n,
        &beta,
        c.data, n);
    cudaDeviceSynchronize();
    
    return c;
}

int main()
{
    const std::size_t MATRIX_SIZE = std::strtoul(std::getenv("YOB_DEMO_MS"), nullptr, 10);
    cublasHandle_t handle; cublasCreate(&handle);

    auto a = random_matrix(MATRIX_SIZE);
    auto b = random_matrix(MATRIX_SIZE);
    Matrix c {};

    auto chr = Chrono();
    {
        auto _ = chr.start();
        c = matmul(a, b, handle);
    }
    chr.print(file);

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(c.data);
    return 0;
}