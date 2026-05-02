# CMakeLists.txt for RL-SPH Heuristic
cmake_minimum_required(VERSION 3.14)
project(RL_SPH LANGUAGES CXX)

# Default to C++17 (needed for std::filesystem in training driver)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch support (optional)
option(USE_LIBTORCH "Build with LibTorch for full neural network" OFF)

if(USE_LIBTORCH)
    # Find LibTorch - expects TORCH_INSTALL_PREFIX to be set or libtorch in CMAKE_PREFIX_PATH
    # On Ubuntu: cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
    find_package(Torch REQUIRED)

    # Set additional compiler flags for LibTorch
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    # Define USE_LIBTORCH globally
    add_definitions(-DUSE_LIBTORCH)
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
    rl_lp_subproblem.h
    rl_logger.h
)

# MIP Definition library
set(MIP_DEF_SOURCES
    ../mip_problem.cpp
)

# Build MIP definition as static library
add_library(mip_definition STATIC ${MIP_DEF_SOURCES})
target_include_directories(mip_definition PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Find COIN-OR libraries (required)
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
else()
    message(WARNING "COIN-OR libraries not found. Build may fail.")
endif()

# Common link libraries
set(COMMON_LIBS
    mip_definition
    ${OSICLP_LIBRARY}
    ${OSI_LIBRARY}
    ${CLP_LIBRARY}
    ${COIN_UTILS_LIBRARY}
)

# ==================== Inference executable ====================
add_executable(rl_sph_test
    main_rl.cpp
    ${RL_SPH_SOURCES}
)

target_include_directories(rl_sph_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(rl_sph_test PRIVATE ${COMMON_LIBS})

target_compile_options(rl_sph_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# ==================== Training executable ====================
add_executable(rl_sph_train
    main_train.cpp
    ${RL_SPH_SOURCES}
)

target_include_directories(rl_sph_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(rl_sph_train PRIVATE ${COMMON_LIBS})

target_compile_options(rl_sph_train PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# ==================== LibTorch linking ====================
if(USE_LIBTORCH)
    # Link LibTorch to both executables
    target_link_libraries(rl_sph_test PRIVATE "${TORCH_LIBRARIES}")
    target_link_libraries(rl_sph_train PRIVATE "${TORCH_LIBRARIES}")

    # Include LibTorch headers
    target_include_directories(rl_sph_test PRIVATE ${TORCH_INCLUDE_DIRS})
    target_include_directories(rl_sph_train PRIVATE ${TORCH_INCLUDE_DIRS})

    # Link C++ ABI library for LibTorch compatibility on Linux
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(rl_sph_test PRIVATE torch torch_cpu c10)
        target_link_libraries(rl_sph_train PRIVATE torch torch_cpu c10)
    endif()
endif()

# std::filesystem linkage (GCC < 9 needs explicit -lstdc++fs)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(rl_sph_train PRIVATE stdc++fs)
endif()
