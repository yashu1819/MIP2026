#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iomanip>

int main() {
    for (int idx = 1; idx <= 50; idx++) {
        std::ostringstream fname;
        fname << "test_set/instances/instance_"
              << std::setw(2) << std::setfill('0') << idx
              << ".mps";

        std::ifstream fin(fname.str());
        if (!fin.is_open()) {
            std::cerr << "Could not open " << fname.str() << "\n";
            continue;
        }

        std::string line, section;
        std::string obj_row;

        std::unordered_set<std::string> rows;   // constraints
        std::unordered_set<std::string> cols;   // variables
        long long nnz = 0;

        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '*') continue;

            std::stringstream ss(line);
            std::string a, b, c, d, e;

            ss >> a;
            if (a == "ROWS" || a == "COLUMNS" || a == "RHS" ||
                a == "BOUNDS" || a == "ENDATA") {
                section = a;
                continue;
            }

            if (section == "ROWS") {
                // format: type row_name
                char type = a[0];
                ss >> b;
                if (type == 'N') {
                    obj_row = b;   // objective row
                } else {
                    rows.insert(b);
                }
            }
            else if (section == "COLUMNS") {
                // format:
                // col row val [row val]
                std::string col = a;
                cols.insert(col);

                ss >> b >> c;
                if (b != obj_row) nnz++;

                if (ss >> d >> e) {
                    if (d != obj_row) nnz++;
                }
            }
        }

        fin.close();

        long long m = rows.size();
        long long n = cols.size();

        double density = (m > 0 && n > 0)
                         ? static_cast<double>(nnz) / (m * n)
                         : 0.0;

        std::cout << fname.str()
                  << " | m=" << m
                  << " n=" << n
                  << " nnz=" << nnz
                  << " density=" << density
                  << "\n";
    }

    return 0;
}

// g++ -std=c++17 mps_density.cpp -O2 -o mps_density
// ./mps_density
