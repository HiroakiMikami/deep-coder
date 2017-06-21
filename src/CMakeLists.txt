add_executable(gen_dataset generate_dataset.cc)
target_link_libraries(gen_dataset deepcoder)

add_executable(gen_program generate_program.cc)
target_link_libraries(gen_program deepcoder)
