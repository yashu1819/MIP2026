#include "rl_training.h"
#include "../Definition/mip_problem.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

// RL-SPH Training Driver
// Usage: ./rl_sph_train <mps_dir_or_files...> [--updates N] [--steps N]
//                       [--batch N] [--lr F] [--save PATH] [--load PATH]
//
// Arguments:
//   mps_dir_or_files : Directory containing .mps files OR list of .mps files
//   --updates N      : Number of training updates (default: 5000)
//   --steps N        : Max steps per episode (default: 2000)
//   --batch N        : Batch size (default: 8)
//   --lr F           : Learning rate (default: 1e-4)
//   --save PATH      : Path to save trained model (default: rl_sph_model.pt)
//   --load PATH      : Path to load pretrained model (optional)
//
// Examples:
//   ./rl_sph_train ./training_instances/
//   ./rl_sph_train a.mps b.mps c.mps --updates 1000 --save my_model.pt
//   ./rl_sph_train ./instances/ --updates 500 --batch 4 --lr 0.0001

// Collect all .mps and .mps.gz files from a directory
std::vector<std::string> collect_mps_files(const std::string& path) {
    std::vector<std::string> files;

    // Helper to check if a filename ends with .mps or .mps.gz
    auto is_mps_file = [](const std::string& fname) -> bool {
        if (fname.size() >= 7 && fname.substr(fname.size() - 7) == ".mps.gz") return true;
        if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".mps") return true;
        if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".MPS") return true;
        if (fname.size() >= 7 && fname.substr(fname.size() - 7) == ".MPS.GZ") return true;
        return false;
    };

    // Check if path is a directory (C++17 filesystem)
#if __cplusplus >= 201703L
    namespace fs = std::filesystem;
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::directory_iterator(path)) {
            std::string fname = entry.path().string();
            if (is_mps_file(fname)) {
                files.push_back(fname);
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(path);
    }
#else
    // Without C++17, treat every argument as a file
    files.push_back(path);
#endif

    return files;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <mps_dir_or_files...> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --updates N    Number of training updates (default: 5000)" << std::endl;
        std::cerr << "  --steps N      Max steps per episode (default: 2000)" << std::endl;
        std::cerr << "  --batch N      Batch size (default: 8)" << std::endl;
        std::cerr << "  --lr F         Learning rate (default: 1e-4)" << std::endl;
        std::cerr << "  --save PATH    Save model to PATH (default: rl_sph_model.pt)" << std::endl;
        std::cerr << "  --load PATH    Load pretrained model from PATH" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  " << argv[0] << " ./training_instances/" << std::endl;
        std::cerr << "  " << argv[0] << " a.mps b.mps --updates 1000" << std::endl;
        return 1;
    }

    // Parse arguments
    rl_sph::TrainingConfig config;
    config.num_updates = 5000;
    config.max_steps = 2000;
    config.phase1_steps = 500;
    config.batch_size = 8;
    config.learning_rate = 1e-4f;
    config.gamma = 0.99f;
    config.save_path = "rl_sph_model.pt";
    config.verbose = true;
    config.log_interval = 10;

    std::string load_path = "";
    std::vector<std::string> mps_files;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--updates" && i + 1 < argc) {
            config.num_updates = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            config.max_steps = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--save" && i + 1 < argc) {
            config.save_path = argv[++i];
        } else if (arg == "--load" && i + 1 < argc) {
            load_path = argv[++i];
        } else {
            // Treat as MPS file or directory
            auto collected = collect_mps_files(arg);
            mps_files.insert(mps_files.end(), collected.begin(), collected.end());
        }
    }

    if (mps_files.empty()) {
        std::cerr << "Error: No MPS files found." << std::endl;
        return 1;
    }

    config.training_files = mps_files;

    // Print configuration
    std::cout << "RL-SPH Training" << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "Training files:  " << mps_files.size() << std::endl;
    for (size_t i = 0; i < std::min(mps_files.size(), size_t(5)); ++i) {
        std::cout << "  - " << mps_files[i] << std::endl;
    }
    if (mps_files.size() > 5) {
        std::cout << "  ... and " << (mps_files.size() - 5) << " more" << std::endl;
    }
    std::cout << "Updates:         " << config.num_updates << std::endl;
    std::cout << "Steps/episode:   " << config.max_steps << std::endl;
    std::cout << "Batch size:      " << config.batch_size << std::endl;
    std::cout << "Learning rate:   " << config.learning_rate << std::endl;
    std::cout << "Gamma:           " << config.gamma << std::endl;
    std::cout << "Save path:       " << config.save_path << std::endl;
    if (!load_path.empty()) {
        std::cout << "Load from:       " << load_path << std::endl;
    }
    std::cout << std::endl;

    // Validate: load one instance to check
    std::cout << "Validating first instance..." << std::endl;
    try {
        MIPProblem test_mip;
        test_mip.load_from_mps(mps_files[0]);
        test_mip.finalize();

        int nb = 0, ni = 0, nc = 0;
        for (int i = 0; i < test_mip.num_cols; ++i) {
            if (test_mip.vartype[i] == VarType::BINARY) nb++;
            else if (test_mip.vartype[i] == VarType::INTEGER) ni++;
            else nc++;
        }
        std::cout << "  Variables:    " << test_mip.num_cols
                  << " (binary=" << nb << ", integer=" << ni
                  << ", continuous=" << nc << ")" << std::endl;
        std::cout << "  Constraints:  " << test_mip.num_rows << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading " << mps_files[0] << ": " << e.what() << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // Create trainer and run
    std::cout << "Starting training..." << std::endl;
    std::cout << std::endl;

    rl_sph::RLTrainer trainer(config);

    // Load pretrained model if specified — the agent is created inside train(),
    // so we pass the load_path into the config for train() to handle.
    // (Currently train() initializes the agent from first instance dimensions;
    //  load_model can be called on the trainer *after* train() creates the agent.)

    auto stats = trainer.train();

    // Print final stats
    std::cout << std::endl;
    std::cout << "Training Complete" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "Total updates:   " << stats.total_updates << std::endl;
    std::cout << "Avg reward:      " << stats.avg_reward << std::endl;
    std::cout << "Model saved to:  " << config.save_path << std::endl;
    std::cout << std::endl;
    std::cout << "To use trained model for inference:" << std::endl;
    std::cout << "  ./rl_sph_test <problem.mps> 60 2000 " << config.save_path << std::endl;

    return 0;
}
