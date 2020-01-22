find_package( OpenCV REQUIRED )
add_library(featureProcessing filters.cpp)
target_link_libraries( featureProcessing ${OpenCV_LIBS})