#include <vector>
#include "opencv2/core/core.hpp"

#define BLOB_MIN_CLASS 1
#define BLOB_MAX_CLASS 2
#define CORNER_MIN_CLASS 3
#define CORNER_MAX_CLASS 4

struct KeyPoint {
    cv::Point2f point;
    int value;
    int pointClass;
};

std::vector<KeyPoint> nonMaximaSuppression(cv::Mat blobF, cv::Mat cornerF);
bool checkMinimumValidity (cv::Mat I, int fmin, int fmin_i, int fmin_j);
bool checkMaximumValidity (cv::Mat I, int fmax, int fmax_i, int fmax_j);