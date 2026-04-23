# CMakeLists.txt for RL-SPH Heuristic
cmake_minimum_required(VERSION 3.14)
project(RL_SPH LANGUAGES CXX)

# Default to C++11; upgraded to C++17 when LibTorch is enabled
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch support (optional)
option(USE_LIBTORCH "Build with LibTorch for full neural network" OFF)

if(USE_LIBTORCH)
    # LibTorch requires C++17
    set(CMAKE_CXX_STANDARD 17)
    find_package(Torch REQUIRED)
endif()

# RL-SPH source files
set(RL_SPH_SOURCES
    rl_graph.cpp
    rl_heuristic.cpp
    rl_agent.cpp
    rl_training.cpp
)

set(RL_SPH_HEADERS
    rl_graph.h
    rl_state.h
    rl_reward.h
    rl_variable_selection.h
    rl_heuristic.h
    rl_agent.h
    rl_training.h
    rl_features.h
)

# MIP Definition library
set(MIP_DEF_SOURCES
    ../Definition/mip_problem.cpp
    ../Utiils/mps_reader.cpp
)

# Build MIP definition as static library
add_library(mip_definition STATIC ${MIP_DEF_SOURCES})
target_include_directories(mip_definition PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Find COIN-OR libraries (required by MIP definition)
find_path(OSI_INCLUDE_DIR NAMES coin/OsiClpSolverInterface.hpp)
find_library(OSICLP_LIBRARY NAMES OsiClp)
find_library(OSI_LIBRARY NAMES Osi)
find_library(CLP_LIBRARY NAMES Clp)
find_library(COIN_UTILS_LIBRARY NAMES CoinUtils)

if(OSI_INCLUDE_DIR AND OSICLP_LIBRARY AND OSI_LIBRARY AND CLP_LIBRARY AND COIN_UTILS_LIBRARY)
    target_include_directories(mip_definition PUBLIC ${OSI_INCLUDE_DIR})
    target_link_libraries(mip_definition
        ${OSICLP_LIBRARY}
        ${OSI_LIBRARY}
        ${CLP_LIBRARY}
        ${COIN_UTILS_LIBRARY}
    )
endif()

# Main executable
add_executable(rl_sph_test
    main_rl.cpp
    ${RL_SPH_SOURCES}
)

target_include_directories(rl_sph_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# target_link_libraries(rl_sph_test PRIVATE mip_definition)
target_link_libraries(rl_sph_test PRIVATE
    mip_definition
    ${OSICLP_LIBRARY}
    ${OSI_LIBRARY}
    ${CLP_LIBRARY}
    ${COIN_UTILS_LIBRARY}
)

# Compiler warnings
target_compile_options(rl_sph_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# Link LibTorch if enabled
if(USE_LIBTORCH)
    target_link_libraries(rl_sph_test PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(rl_sph_test PRIVATE -DUSE_LIBTORCH)
    target_compile_definitions(mip_definition PRIVATE -DUSE_LIBTORCH)
endif()
