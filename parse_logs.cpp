#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

int main() {
    vector<string> files;

    for (const auto& entry : fs::directory_iterator("logs_pdlp_1e-06")) {
        if (entry.path().string().find("relaxed_") != string::npos && 
            entry.path().extension() == ".log") {
            files.push_back(entry.path().string());
        }
    }

    sort(files.begin(), files.end());

    ofstream csv("summary.csv");
    csv << "filename,optimal_objective,time\n";

    regex objRegex(R"(Objective = ([\-\d\.]+))");
    regex timeRegex(R"(Time: ([\d\.]+)s)");

    for (const auto& file : files) {
        ifstream f(file);
        string content((istreambuf_iterator<char>(f)), {});
        smatch objMatch, timeMatch;

        string objective = "";
        string time = "";

        if (regex_search(content, objMatch, objRegex)) {
            objective = objMatch[1];
        }
        if (regex_search(content, timeMatch, timeRegex)) {
            time = timeMatch[1];
        }

        csv << file << "," << objective << "," << time << "\n";
    }

    cout << "CSV created: summary.csv" << endl;
    return 0;
}
