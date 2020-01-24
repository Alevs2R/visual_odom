find_package( OpenCV REQUIRED )
add_library(featureProcessing filters.cpp nms.cpp filters.h nms.h)
target_link_libraries( featureProcessing ${OpenCV_LIBS})