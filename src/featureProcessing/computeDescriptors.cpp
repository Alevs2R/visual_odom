#include "computeDescriptors.h"
#include "nms.h"
#include <iostream>

void fillDescriptor(int startindex, int16_t* ar, int x, int y, cv::Mat& I) {
        ar[startindex] = I.at<int16_t>(y-5, x-1);
        ar[startindex+1] = I.at<int16_t>(y-5, x+1);
        ar[startindex+2] = I.at<int16_t>(y-3, x-5);
        ar[startindex+3] = I.at<int16_t>(y-3, x+5);
        ar[startindex+4] = I.at<int16_t>(y-1, x-3);
        ar[startindex+5] = I.at<int16_t>(y-1, x-1);
        ar[startindex+6] = I.at<int16_t>(y-1, x+1);
        ar[startindex+7] = I.at<int16_t>(y-1, x+3);
        ar[startindex+8] = I.at<int16_t>(y+1, x-3);
        ar[startindex+9] = I.at<int16_t>(y+1, x-1);
        ar[startindex+10] = I.at<int16_t>(y+1, x+1);
        ar[startindex+11] = I.at<int16_t>(y+1, x+3);
        ar[startindex+12] = I.at<int16_t>(y+3, x-5);
        ar[startindex+13] = I.at<int16_t>(y+3, x+5);
        ar[startindex+14] = I.at<int16_t>(y+5, x-1);
        ar[startindex+15] = I.at<int16_t>(y+5, x+1);
}

void computeDescriptors(cv::Mat& I_dx, cv::Mat& I_dy, std::vector<KeyPoint>& keypts_in) {
    int counter = 0;
    for (auto& kp: keypts_in) {
        int x = kp.point.x;
        int y = kp.point.y;
        fillDescriptor(0, kp.descriptor, x, y, I_dx);
        fillDescriptor(16, kp.descriptor, x, y, I_dy);
        // for (int i = 0; i < 32; i++){
        //     std::cout << kp.descriptor[i] << "\n";
        // }
        // std::cout << "finish\n";
    }

}