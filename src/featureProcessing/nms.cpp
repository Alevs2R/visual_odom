#include "nms.h"
#include <iostream>

std::vector<KeyPoint> nonMaximaSuppression(cv::Mat& blobF, cv::Mat& cornerF){
    int width = blobF.cols;
    int height = blobF.rows;

    std::vector<KeyPoint> keypoints = std::vector<KeyPoint>();

    int i,j;

    for (i = NMS_N + NMS_MARGIN; i <= height - NMS_N - NMS_MARGIN; i+=NMS_N+1){
        for (j = NMS_N + NMS_MARGIN; j <= width - NMS_N - NMS_MARGIN; j+=NMS_N+1){
            // (2n + 1) × (2n + 1)-Block Algorithm
            // Step a)
            // Partitions input image into blocks of sizes (n + 1) x (n + 1).
            // Search for maxima/minima within each block.

            auto rect = cv::Rect(j, i, NMS_N, NMS_N);
            cv::Mat currentBlobF = blobF(rect);
            cv::Mat currentCornerF = cornerF(rect);

            double minv1, maxv1, minv2, maxv2;
            cv::Point minl1, maxl1, minl2, maxl2;

            cv::minMaxLoc(currentBlobF, &minv1, &maxv1, &minl1, &maxl1);
            cv::minMaxLoc(currentCornerF, &minv2, &maxv2, &minl2, &maxl2);

            minl1.x += j;
            minl2.x += j;
            maxl1.x += j;
            maxl2.x += j;
            minl1.y += i;
            minl2.y += i;
            maxl1.y += i;
            maxl2.y += i;

            // std::cout << "current blob " << currentBlob << "\n";

            // Steb b)
            // Full neighborhood of each candidate is tested

            if(!minimumValidityFailed(blobF, minv1, minl1)) {
                keypoints.push_back(KeyPoint(
                    minl1,
                    (uint16_t)minv1,
                    BLOB_MIN_CLASS
                ));
            }
            if(!maximumValidityFailed(blobF, maxv1, maxl1)) {
                keypoints.push_back(KeyPoint(
                    maxl1,
                    (uint16_t)maxv1,
                    BLOB_MAX_CLASS
                ));
            }
            if(!minimumValidityFailed(cornerF, minv2, minl2)) {
                keypoints.push_back(KeyPoint(
                    minl2,
                    (uint16_t)minv2,
                    CORNER_MIN_CLASS
                ));
            }
            if(!maximumValidityFailed(cornerF, maxv2, maxl2)) {
                keypoints.push_back(KeyPoint(
                    maxl2,
                    (uint16_t)maxv2,
                    CORNER_MAX_CLASS
                ));
            }          
        }
    }
    return keypoints;
}

bool maximumValidityFailed (cv::Mat& I, double detectedMax, cv::Point2i& p) {
    if (detectedMax < NMS_TAU) return true;

    int width = I.cols;
    int height = I.rows;
    int y_last = std::min(p.y + NMS_N, height - 1 - NMS_MARGIN);
    int x_last = std::min(p.x + NMS_N, width - 1 - NMS_MARGIN);
    int x = p.x - NMS_N;
    int y = p.y - NMS_N;
    auto currentPatch = cv::Rect(x, y, x_last-x+1, y_last-y+1);
    double minv, maxv;

    cv::minMaxLoc(I(currentPatch), &minv, &maxv);
    
    if (maxv > detectedMax) {
        return true; // failed
    }
    return false; // not failed
}


bool minimumValidityFailed (cv::Mat& I, double detectedMin, cv::Point2i& p) {
    if (detectedMin > -NMS_TAU) return true;

    int width = I.cols;
    int height = I.rows;
    int y_last = std::min(p.y + NMS_N, height - 1 - NMS_MARGIN);
    int x_last = std::min(p.x + NMS_N, width - 1 - NMS_MARGIN);
    int x = p.x - NMS_N;
    int y = p.y - NMS_N;

    auto currentPatch = cv::Rect(x, y, x_last-x+1, y_last-y+1);
    double minv, maxv;

    cv::minMaxLoc(I(currentPatch), &minv, &maxv);
    
    if (minv < detectedMin) {
        return true; // failed
    }
    return false; // not failed
}

