#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <set>

using namespace std;

string classifyInstance(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open " << filename << endl;
        return "UNKNOWN";
    }

    string line;
    bool inColumns = false;
    bool inBounds = false;
    bool inIntegerBlock = false;

    set<string> allVars;
    set<string> binaryVars;
    set<string> integerVars;

    while (getline(file, line)) {

        if (line.find("COLUMNS") != string::npos) {
            inColumns = true;
            continue;
        }

        if (line.find("BOUNDS") != string::npos) {
            inColumns = false;
            inBounds = true;
            continue;
        }

        if (line.find("ENDATA") != string::npos)
            break;

        // ---- Parse COLUMNS ----
        if (inColumns) {

            if (line.find("INTORG") != string::npos) {
                inIntegerBlock = true;
                continue;
            }

            if (line.find("INTEND") != string::npos) {
                inIntegerBlock = false;
                continue;
            }

            stringstream ss(line);
            string var;
            ss >> var;

            if (var.empty()) continue;

            allVars.insert(var);

            if (inIntegerBlock)
                integerVars.insert(var);
        }

        // ---- Parse BOUNDS ----
        if (inBounds) {
            stringstream ss(line);
            string boundType, boundName, var;
            ss >> boundType >> boundName >> var;

            if (boundType == "BV") {
                binaryVars.insert(var);
            }
        }
    }

    file.close();

    // Remove binary vars from general integer set
    for (const auto& v : binaryVars)
        integerVars.erase(v);

    bool hasBinary = !binaryVars.empty();
    bool hasInteger = !integerVars.empty();

    bool hasContinuous = false;
    for (const auto& v : allVars) {
        if (!binaryVars.count(v) && !integerVars.count(v)) {
            hasContinuous = true;
            break;
        }
    }

    // ---- Classification ----

    if (hasInteger) {
        if (hasContinuous)
            return "MILP";
        else
            return "Pure IP";
    }

    if (hasBinary && hasContinuous)
        return "MBP";

    if (hasBinary && !hasContinuous)
        return "Pure BP";

    return "LP";
}

int main() {
    ofstream out("partitions.csv");
    if (!out.is_open()) {
        cerr << "Could not create partitions.csv\n";
        return 1;
    }

    for (int i = 1; i <= 50; i++) {
        stringstream filename;
        filename << "instances/instance_"
                 << setw(2) << setfill('0') << i
                 << ".mps";

        string type = classifyInstance(filename.str());

        out << setw(2) << setfill('0') << i
            << "," << type;

        if (i != 50)
            out << "\n";
    }

    out.close();
    return 0;
}
