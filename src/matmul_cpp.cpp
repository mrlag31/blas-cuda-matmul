#include <filesystem>
#include <vector>
#include <random>
#include <cstdlib>

#include "./utils/chrono.cpp"

static const std::string file = std::filesystem::path(__FILE__).filename();

struct Matrix {
    std::vector<float> data;
    std::size_t size;
};

Matrix random_matrix(std::size_t n)
{
    auto mat = Matrix{
        .data = std::vector<float>(n * n, 0.0),
        .size = n
    };

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> urd(0.0, 1.0);
    for (std::size_t i = 0; i < n * n; i++)
        mat.data[i] = urd(gen);
    
    return mat;
}

Matrix matmul(const Matrix& a, const Matrix& b)
{
    auto n = a.size;

    auto c = Matrix{
        .data = std::vector<float>(n * n, 0.0),
        .size = n
    };

    auto pos = [=](std::size_t i, std::size_t j){ return i * n + j; };
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++)
            for (std::size_t k = 0; k < n; k++)
                c.data[pos(i, j)] += a.data[pos(i, k)] * b.data[pos(k, j)];
    
    return c;
}

int main()
{
    const std::size_t MATRIX_SIZE = std::strtoul(std::getenv("YOB_DEMO_MS"), nullptr, 10);

    auto a = random_matrix(MATRIX_SIZE);
    auto b = random_matrix(MATRIX_SIZE);

    auto chr = Chrono();
    {
        auto _ = chr.start();
        auto c = matmul(a, b);
    }
    chr.print(file);

    return 0;
}