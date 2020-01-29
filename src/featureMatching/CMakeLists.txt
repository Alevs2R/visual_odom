find_package( OpenCV REQUIRED )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_library(featureMatching circularMatching.cpp circularMatching.h)
target_link_libraries( featureMatching PRIVATE ${OpenCV_LIBS} ${llvm_libs} /usr/local/opt/llvm/lib/libomp.dylib)
