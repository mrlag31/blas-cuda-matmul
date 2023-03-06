#include <chrono>
#include <iostream>
#include <iomanip>

/*
    This is a helper class to compute timings
    via RAII, kind of like Python's with statement.
*/
class Chrono {
public:
    struct Ref {
        Chrono& ref_;

        ~Ref() { ref_.stop(); }
    };

    Ref start()
    {
        start_ = std::chrono::steady_clock::now();
        return Ref { *this };
    }

    void stop()
    {
        end_ = std::chrono::steady_clock::now();
    }

    void print(const std::string& file)
    {
        double t = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_
        ).count();

        std::stringstream ts; ts << std::fixed << t;
        std::string tss; ts >> std::setw(7) >> tss;

        std::cout << '[' << std::left << std::setw(20) << file << ']'
            << ": " << tss << "ms"
            << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::time_point<std::chrono::steady_clock> end_;
};