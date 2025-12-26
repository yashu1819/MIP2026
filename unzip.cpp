#include <iostream>
#include <filesystem>
#include <fstream>
#include <zlib.h>
#include <string>

namespace fs = std::filesystem;

// Changed parameter name to 'inputPath' to avoid conflict with 'gzFile' type
int decompressGzip(const std::string& inputPath) {
    // Better way to handle extension: use filesystem path
    fs::path p(inputPath);
    
    // Logic to change the name: 
    // p.stem() removes the .gz, then we can add a prefix or keep as is
    std::string newFileName = p.stem().string(); 
    fs::path outFile = p.parent_path() / newFileName;

    // gzFile is a type from zlib.h; infile is the variable name
    gzFile infile = gzopen(inputPath.c_str(), "rb");
    if (!infile) {
        std::cerr << "Failed to open: " << inputPath << std::endl;
        return -1;
    }

    std::ofstream outfile(outFile, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to create: " << outFile << std::endl;
        gzclose(infile);
        return -1;
    }

    char buffer[4096];
    int bytes;
    while ((bytes = gzread(infile, buffer, sizeof(buffer))) > 0) {
        outfile.write(buffer, bytes);
    }

    gzclose(infile);
    outfile.close();
    return 0;
}

int main() {
    std::string dir = "test_set/instances";

    // Check if directory exists to avoid crashes
    if (!fs::exists(dir)) {
        std::cerr << "Directory not found: " << dir << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".gz") {
            std::cout << "Decompressing: " << entry.path().filename() << std::endl;
            decompressGzip(entry.path().string());
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
