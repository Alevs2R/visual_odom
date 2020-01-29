#include "nms.h"
#include <iostream>

int nms_n = 8;
int nms_tau = 90;
int margin = 21;

std::vector<KeyPoint> nonMaximaSuppression(cv::Mat& blobF, cv::Mat& cornerF){
    int width = blobF.cols;
    int height = blobF.rows;

    std::vector<KeyPoint> keypoints = std::vector<KeyPoint>();

    int i,j;

    for (i = nms_n + margin; i <= height - nms_n - margin; i+=nms_n+1){
        for (j = nms_n + margin; j <= width - nms_n - margin; j+=nms_n+1){
            int16_t f1min_i = i, f1min_j = j, f1max_i = i, f1max_j = j;
            int16_t f2min_i = i, f2min_j = j, f2max_i = i, f2max_j = j;

            int16_t f1min_val = blobF.at<int16_t>(i, j);
            int16_t f1max_val = f1min_val;
            int16_t f2min_val = cornerF.at<int16_t>(i, j);
            int16_t f2max_val = f2min_val;

            // Step a)
            // Partitions input image into blocks of sizes (n + 1) x (n + 1).
            // Search for maxima/minima within each block.
            for (int i2 = i; i2 <=i + nms_n; i2++){
                for (int j2 = j; j2 <= j + nms_n; j2++) {
                    // for blob detector
                    int16_t currval = blobF.at<int16_t>(i2, j2);
                    if (currval < f1min_val) {        
                        f1min_i   = i2;
                        f1min_j   = j2;
                        f1min_val = currval;
                    }
                    else if (currval > f1max_val) {
                        f1max_i   = i2;
                        f1max_j   = j2;
                        f1max_val = currval;
                    }
                    // for corner detector
                    currval = cornerF.at<int16_t>(i2, j2);
                    if (currval < f2min_val) {        
                        f2min_i   = i2;
                        f2min_j   = j2;
                        f2min_val = currval;
                    }
                    else if (currval > f2max_val) {
                        f2max_i   = i2;
                        f2max_j   = j2;
                        f2max_val = currval;
                    }
                }
            }


            bool failedBlobMin = checkMinimumValidity(blobF, f1min_val, f1min_i, f1min_j);
            if (!failedBlobMin) {
                if (f1min_val <= -nms_tau) {
                    KeyPoint newPoint {
                        cv::Point2i(f1min_j, f1min_i),
                        f1min_val,
                        BLOB_MIN_CLASS
                    };
                    keypoints.push_back(newPoint);
                }
            }
            bool failedBlobMax = checkMaximumValidity(blobF, f1max_val, f1max_i, f1max_j);
            if (!failedBlobMax) {
                if (f1max_val >= nms_tau) {
                    KeyPoint newPoint {
                        cv::Point2i(f1max_j, f1min_i),
                        f1max_val,
                        BLOB_MAX_CLASS
                    };
                    keypoints.push_back(newPoint);                
                }
            }
            bool failedCornerMin = checkMinimumValidity(cornerF, f2min_val, f2min_i, f2min_j);
            if (!failedCornerMin) {
                if (f2min_val <= -nms_tau) {
                    KeyPoint newPoint{ 
                        cv::Point2i(f2min_j, f2min_i),
                        f2min_val,
                        CORNER_MIN_CLASS
                    };
                    keypoints.push_back(newPoint);
                }
            }
            bool failedCornerMax = checkMaximumValidity(cornerF, f2max_val, f2max_i, f2max_j);
            if (!failedCornerMax) {
                if (f2max_val >= nms_tau) {
                    KeyPoint newPoint {
                        cv::Point2i(f2max_j, f2max_i),
                        f2max_val,
                        CORNER_MAX_CLASS
                    };
                    keypoints.push_back(newPoint);
                }
            }
        }
    }
    return keypoints;
}

bool checkMaximumValidity (cv::Mat& I, int fmax, int fmax_i, int fmax_j) {
    int width = I.cols;
    int height = I.rows;
    int i_last = std::min(fmax_i + nms_n, height - 1 - margin);
    int j_last = std::min(fmax_j + nms_n, width - 1 - margin);
    int16_t maxval = 0;
    int max_i, max_j;

    for (int i = fmax_i - nms_n; i <= i_last; i++){
        for (int j = fmax_j - nms_n; j <= j_last; j++){
            int16_t currval = I.at<int16_t>(i, j);
            if (currval > maxval) {
                maxval = currval;
                max_i = i;
                max_j = j;
            }
        }        
    }
    // TODO check condition and that NMS works correctly
    if (maxval > fmax && (max_i != fmax_i || max_j != fmax_j)) {
        return true; // failed
    }
    return false; // not failed
}

bool checkMinimumValidity (cv::Mat& I, int fmin, int fmin_i, int fmin_j) {
    int width = I.cols;
    int height = I.rows;
    int i_last = std::min(fmin_i + nms_n, height - 1 - margin);
    int j_last = std::min(fmin_j + nms_n, width - 1 - margin);
    int16_t minval = INT16_MAX; // MAXIMUM FOR int!
    int min_i, min_j;

    for (int i = fmin_i - nms_n; i <= i_last; i++){
        for (int j = fmin_j - nms_n; j <= j_last; j++){
            int16_t currval = I.at<int16_t>(i, j);
            if (currval < minval) {
                minval = currval;
                min_i = i;
                min_j = j;
            }
        }        
    }
    // TODO check condition and that NMS works correctly
    if (minval < fmin && (min_i != fmin_i || min_j != fmin_j)) {
        return true; // failed
    }
    return false; // not failed
}

