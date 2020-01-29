#include "../config.h"
#include <vector>

std::vector<Match> removeOutliers(cv::Mat i1_l, cv::Mat i1_r, cv::Mat i2_l, cv::Mat i2_r, std::vector<Match> matches);