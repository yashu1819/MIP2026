#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

enum VarType { CONTINUOUS, BINARY, INTEGER };

std::string categorize_mps(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string line;
    bool in_columns = false, in_bounds = false, int_marker = false;
    bool has_cont = false, has_bin = false, has_int = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '*') continue;
        
        // Section Switching
        if (line.find("COLUMNS") == 0) { in_columns = true; in_bounds = false; continue; }
        if (line.find("BOUNDS") == 0)  { in_columns = false; in_bounds = true; continue; }
        if (line.find("RHS") == 0 || line.find("ENDATA") == 0) break;

        // Column Logic (Markers)
        if (in_columns) {
            if (line.find("'MARKER'") != std::string::npos) {
                int_marker = (line.find("'INTORG'") != std::string::npos);
            } else {
                if (int_marker) has_int = true;
                else has_cont = true;
            }
        }
        // Bounds Logic
        if (in_bounds) {
            if (line.find("BV") != std::string::npos) has_bin = true;
            else if (line.find("LI") != std::string::npos || line.find("UI") != std::string::npos) has_int = true;
            else if (line.find("FR") != std::string::npos || line.find("LO") != std::string::npos) has_cont = true;
        }
    }

    if (!has_cont) return (has_bin && !has_int) ? "IBP" : "IP";
    return (has_bin && !has_int) ? "MBP" : "MIP";
}

int main() {
    std::string base = "test_set/";
    std::vector<std::string> dirs = {"IBP", "IP", "MBP", "MIP"};
    
    for (const auto& d : dirs) fs::create_directories(base + d);

    for (int i = 1; i <= 50; ++i) {
        char buf[20];
        snprintf(buf, sizeof(buf), "instance_%02d.mps", i);
        std::string filename = buf;
        fs::path src = fs::path(base) / "instances" / filename;

        if (fs::exists(src)) {
            std::string cat = categorize_mps(src.string());
            fs::copy(src, fs::path(base) / cat / filename, fs::copy_options::overwrite_existing);
            std::cout << "Moved " << filename << " to " << cat << std::endl;
        }
    }
    return 0;
}
