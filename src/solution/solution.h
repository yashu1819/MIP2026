#ifndef SOLUTION_H
#define SOLUTION_H

#include <vector>
#include <iostream>
#include <limits>
#include <iomanip>

namespace milp {

struct Solution {
    std::vector<double> x;
    double obj_value = std::numeric_limits<double>::infinity();
    bool feasible = false;

    void print(std::ostream& os = std::cout) const {
        os << "Solution feasible=" << (feasible ? "yes" : "no") << " obj=" << obj_value << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            os << " x" << i << " = " << std::setprecision(6) << x[i] << "\n";
        }
    }
};

} // namespace milp

#endif // SOLUTION_H
