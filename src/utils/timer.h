#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() { reset(); }
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double elapsed_seconds() const {
        using namespace std::chrono;
        return duration_cast<duration<double>>(steady_clock::now() - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

#endif // TIMER_H
