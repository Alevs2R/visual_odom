#ifndef CIRCULAR_MATCHING_INCLUDE

#define CIRCULAR_MATCHING_INCLUDE

#include "../config.h"
#include <vector>

std::vector<Match> performCircularMatching(cv::Mat& img_r, int width, int height, std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r, cv::Mat& projMatrl, cv::Mat& projMatrr, cv::Mat transform);

cv::Point2d predictKeypointPosition(KeyPoint& keypoint_left,
                                    KeyPoint& keypoint_right,
                                    cv::Mat& projMatrl, 
                                    cv::Mat& projMatrr,
                                    cv::Mat& transform,
                                    cv::Mat& cameraRvec,
                                    cv::Mat& cameraTvec,
                                    cv::Mat& distCoeffs);

#endif