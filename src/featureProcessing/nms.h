#ifndef NMS_INCLUDE

#define NMS_INCLUDE

#include <vector>
#include <stdlib.h>     /* srand, rand */
#include "../config.h"

#define BLOB_MIN_CLASS 1
#define BLOB_MAX_CLASS 2
#define CORNER_MIN_CLASS 3
#define CORNER_MAX_CLASS 4

std::vector<KeyPoint> nonMaximaSuppression(cv::Mat& blobF, cv::Mat& cornerF);
bool maximumValidityFailed (cv::Mat& I, double detectedMax, cv::Point2i& p);
bool minimumValidityFailed (cv::Mat& I, double detectedMin, cv::Point2i& p);

#endif