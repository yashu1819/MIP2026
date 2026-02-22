#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

int main() {

    std::map<std::string, std::string> instanceType;  
    // key: "01" -> value: "Pure BP"

    std::map<std::string, int> totalCount;
    std::map<std::string, int> successCount;

    // -------------------------
    // Read partitions.csv
    // -------------------------
    std::ifstream partFile("partitions.csv");
    if (!partFile.is_open()) {
        std::cerr << "Could not open partitions.csv\n";
        return 1;
    }

    std::string line;
    while (std::getline(partFile, line)) {
        std::stringstream ss(line);
        std::string id, type;

        std::getline(ss, id, ',');
        std::getline(ss, type);

        instanceType[id] = type;
        totalCount[type]++;   // count total per type
    }
    partFile.close();

    // -------------------------
    // Read FJresults.csv
    // -------------------------
    std::ifstream fjFile("FJresults.csv");
    if (!fjFile.is_open()) {
        std::cerr << "Could not open FJresults.csv\n";
        return 1;
    }

    while (std::getline(fjFile, line)) {
        std::stringstream ss(line);
        std::string instanceName, flag;

        std::getline(ss, instanceName, ',');
        std::getline(ss, flag, ',');

        // Extract "01" from "instance_01"
        std::string id = instanceName.substr(instanceName.find('_') + 1);

        if (instanceType.count(id) && flag == "1") {
            std::string type = instanceType[id];
            successCount[type]++;
        }
    }
    fjFile.close();

    // -------------------------
    // Write summary.txt
    // -------------------------
    std::ofstream out("summary.txt");
    if (!out.is_open()) {
        std::cerr << "Could not create summary.txt\n";
        return 1;
    }

    for (const auto& p : totalCount) {
        const std::string& type = p.first;
        int total = p.second;
        int success = successCount[type];

        out << type << ": "
            << success << " out of "
            << total << " had 1\n";
    }

    out.close();

    std::cout << "Summary written to summary.txt\n";
    return 0;
}
