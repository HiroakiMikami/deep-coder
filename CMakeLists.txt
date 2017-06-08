cmake_minimum_required(VERSION 3.7)
project(deep_coder)

# common settings
enable_testing()

# C++ settings
set(CMAKE_CXX_STANDARD 14)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(test)
