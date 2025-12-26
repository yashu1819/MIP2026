#include <iostream>
#include <filesystem>
#include <fstream>
#include <zlib.h>

namespace fs = std::filesystem;

int decompressGzip(const std::string& gzFile) {
    std::string outFile = gzFile.substr(0, gzFile.size() - 3); // remove .gz
    
    gzFile infile = gzopen(gzFile.c_str(), "rb");
    if (!infile) {
        std::cerr << "Failed to open: " << gzFile << std::endl;
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

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".gz") {
            std::cout << "Decompressing: " << entry.path() << std::endl;
            decompressGzip(entry.path().string());
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
