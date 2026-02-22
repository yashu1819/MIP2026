#ifndef SOLUTION_H
#define SOLUTION_H

#include <vector>
#include <limits>

struct Solution {
    bool feasible = false;
    std::vector<double> x;
    double obj_value = std::numeric_limits<double>::infinity();

    void clear() {
        feasible = false;
        x.clear();
        obj_value = std::numeric_limits<double>::infinity();
    }
};

#endif

