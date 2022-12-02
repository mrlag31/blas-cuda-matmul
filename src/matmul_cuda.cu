#include <filesystem>
#include <vector>
#include <random>
#include <cstdlib>

#include "./utils/chrono.cpp"
#include "cuda.h"

static const std::string file = std::filesystem::path(__FILE__).filename();

struct Matrix {
    float* data;
    std::size_t size;
};

__global__ void matmulKer(float* a, float* b, float* c, std::size_t n)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n)
        return;

    for (std::size_t k = 0; k < n; k++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

Matrix random_matrix(std::size_t n)
{
    auto mat = Matrix{
        .data = new float[n * n] {0.0},
        .size = n
    };

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> urd(0.0, 1.0);
    for (std::size_t i = 0; i < n * n; i++)
        mat.data[i] = urd(gen);
    
    float* dMat; std::size_t size = sizeof(float) * n * n;
    cudaMalloc(&dMat, size);
    cudaMemcpy(dMat, mat.data, size, cudaMemcpyHostToDevice);
    delete mat.data; mat.data = dMat;

    return mat;
}

Matrix matmul(const Matrix& a, const Matrix& b)
{
    auto n = a.size;

    auto c = Matrix{
        .data = new float[n * n] {0.0},
        .size = n
    };

    float* dC; std::size_t size = sizeof(float) * n * n;
    cudaMalloc(&dC, size);
    cudaMemcpy(dC, c.data, size, cudaMemcpyHostToDevice);
    delete c.data; c.data = dC;

    unsigned int dimBlock = 32;
    unsigned int dimGrid = n / dimBlock;
    if (dimGrid * dimBlock < n) dimGrid += 1;

    dim3 numBlocks { dimGrid, dimGrid };
    dim3 numThreads { dimBlock, dimBlock };
    matmulKer<<<numBlocks, numThreads>>>(a.data, b.data, c.data, n);
    cudaDeviceSynchronize();
    
    return c;
}

int main()
{
    const std::size_t MATRIX_SIZE = std::strtoul(std::getenv("YOB_DEMO_MS"), nullptr, 10);

    auto a = random_matrix(MATRIX_SIZE);
    auto b = random_matrix(MATRIX_SIZE);
    Matrix c {};

    auto chr = Chrono();
    {
        auto _ = chr.start();
        c = matmul(a, b);
    }
    chr.print(file);

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(c.data);
    return 0;
}