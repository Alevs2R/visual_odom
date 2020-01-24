#include "computeDescriptors.h"
#include "nms.h"

void fillDescriptor(int startindex, int* ar, int x, int y, cv::Mat& I) {
        ar[startindex] = I.at<int>(x-1, y-5);
        ar[startindex+1] = I.at<int>(x+1, y-5);
        ar[startindex+2] = I.at<int>(x-5, y-3);
        ar[startindex+3] = I.at<int>(x+5, y-3);
        ar[startindex+4] = I.at<int>(x-3, y-1);
        ar[startindex+5] = I.at<int>(x-1, y-1);
        ar[startindex+6] = I.at<int>(x+1, y-1);
        ar[startindex+7] = I.at<int>(x+3, y-1);
        ar[startindex+8] = I.at<int>(x-3, y+1);
        ar[startindex+9] = I.at<int>(x-1, y+1);
        ar[startindex+10] = I.at<int>(x+1, y+1);
        ar[startindex+11] = I.at<int>(x+3, y+1);
        ar[startindex+12] = I.at<int>(x-5, y+3);
        ar[startindex+13] = I.at<int>(x+5, y+3);
        ar[startindex+14] = I.at<int>(x-1, y+5);
        ar[startindex+15] = I.at<int>(x+1, y+5);
}

void computeDescriptors(cv::Mat& I_dx, cv::Mat& I_dy, std::vector<KeyPoint>& keypts_in) {
    for (auto& kp: keypts_in) {
        int x = kp.point.x;
        int y = kp.point.y;
        fillDescriptor(0, kp.descriptor, x, y, I_dx);
        fillDescriptor(16, kp.descriptor, x, y, I_dy);
    }
}