cmake_minimum_required(VERSION 3.0)
project(deep_coder)

# common settings
enable_testing()

# C++ settings
set(CMAKE_CXX_STANDARD 14)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(src)