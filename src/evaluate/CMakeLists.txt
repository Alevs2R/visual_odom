
add_library(evaluate_odometry SHARED "evaluate_odometry.cpp")
add_library(matrix SHARED "matrix.cpp")
target_link_libraries( evaluate_odometry matrix)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_executable( evaluate evaluate_odometry.cpp )
target_link_libraries( evaluate matrix)


