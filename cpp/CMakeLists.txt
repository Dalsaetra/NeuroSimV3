# CMakeLists.txt (at project root)

cmake_minimum_required(VERSION 3.10)
project(neuralsim VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(neuralsim
    src/IzhikevichNeuron.cpp
    # … add other .cpp sources here as you implement them …
)
target_include_directories(neuralsim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Simple test executable
add_executable(test_izhikevich
    tests/test_izhikevich.cpp
)
target_link_libraries(test_izhikevich PRIVATE neuralsim)
