#ifndef MPS_READER_H
#define MPS_READER_H

#include "milp_problem.h"
#include <string>

namespace milp {

struct MpsReader {
    // Very small helper: for now we support a built-in "example" instance.
    // Real MPS parsing is not implemented here.
    // Returns true if loaded, false otherwise.
    bool loadFromFile(const std::string& filename, MILPProblem& out_problem);
};

} // namespace milp

#endif // MPS_READER_H
