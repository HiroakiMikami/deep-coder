# Enable google test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB files *.cc **/*.cc)

# Compile the test files
message(${PROJECT_BINARY_DIR})
add_executable(test-deep-coder ${files})
target_link_libraries(test-deep-coder ${GTEST_BOTH_LIBRARIES})
target_link_libraries(test-deep-coder pthread)
target_link_libraries(test-deep-coder deepcoder)
add_test(
        NAME google-test-deep-coder
        COMMAND test-deep-coder)
