#ifndef MILP_PROBLEM_H
#define MILP_PROBLEM_H

#include <vector>
#include <string>
#include <iostream>
#include <limits>

namespace milp {

struct Variable {
    double lb = 0.0;
    double ub = std::numeric_limits<double>::infinity();
    bool is_integer = false;
    std::string name;
};

struct Constraint {
    // sum(coeff[i] * x[var_idx[i]]) between lb and ub
    std::vector<int> var_idx;
    std::vector<double> coeff;
    double lb = -std::numeric_limits<double>::infinity();
    double ub = std::numeric_limits<double>::infinity();
    std::string name;
};

struct MILPProblem {
    std::vector<Variable> vars;
    std::vector<Constraint> cons;
    std::vector<double> obj; // objective coefficients (minimization by default)
    bool minimize = true;
    std::string name;

    int num_vars() const { return static_cast<int>(vars.size()); }
    int num_cons() const { return static_cast<int>(cons.size()); }

    int add_variable(double lb=0.0, double ub=std::numeric_limits<double>::infinity(),
                     bool is_integer=false, const std::string& name="") {
        Variable v; v.lb = lb; v.ub = ub; v.is_integer = is_integer; v.name = name;
        vars.push_back(v);
        obj.push_back(0.0);
        return static_cast<int>(vars.size()) - 1;
    }

    int add_constraint(const std::vector<int>& idx,
                       const std::vector<double>& coeff,
                       double lb=-std::numeric_limits<double>::infinity(),
                       double ub=std::numeric_limits<double>::infinity(),
                       const std::string& name = "") {
        Constraint c;
        c.var_idx = idx;
        c.coeff = coeff;
        c.lb = lb; c.ub = ub; c.name = name;
        cons.push_back(c);
        return static_cast<int>(cons.size()) - 1;
    }

    void set_obj_coef(int var, double coef) {
        if (var >= 0 && var < num_vars()) obj[var] = coef;
    }

    void print_summary(std::ostream& os = std::cout) const {
        os << "MILPProblem: " << name << "\n";
        os << "Variables: " << num_vars() << (minimize ? " (minimize)\n":" (maximize)\n");
        for (int i = 0; i < num_vars(); ++i) {
            os << " v" << i << " (" << vars[i].name << ") lb=" << vars[i].lb << " ub=" << vars[i].ub
               << " int=" << vars[i].is_integer << " obj=" << obj[i] << "\n";
        }
        os << "Constraints: " << num_cons() << "\n";
        for (int i = 0; i < num_cons(); ++i) {
            os << " c" << i << " (" << cons[i].name << "): ";
            for (size_t j = 0; j < cons[i].var_idx.size(); ++j) {
                os << cons[i].coeff[j] << "*v" << cons[i].var_idx[j] << (j+1<cons[i].var_idx.size() ? " + " : "");
            }
            os << " in [" << cons[i].lb << ", " << cons[i].ub << "]\n";
        }
    }
};

} // namespace milp

#endif // MILP_PROBLEM_H
