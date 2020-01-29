cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 17)
project(VisualOdometry)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

llvm_map_components_to_libnames(llvm_libs support core irreader )

include_directories( ${LLVM_INCLUDE_DIRS})

set(CMAKE_C_COMPILER   /usr/local/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)

# find_package (Eigen3 REQUIRED NO_MODULE)
# include_directories (${EIGEN3_INCLUDE_DIR})
# message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
