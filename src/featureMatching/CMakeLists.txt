find_package( OpenCV REQUIRED )
add_library(featureMatching circularMatching.cpp circularMatching.h)
target_link_libraries( featureMatching ${OpenCV_LIBS} )