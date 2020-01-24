#include "filters.h"
#include "opencv2/imgproc/imgproc.hpp"

float blobKernel[] = {
    -1, -1, -1, -1,-1,
    -1, 1, 1, 1, -1,
    -1, 1, 8, 1, -1,
    -1, 1, 1, 1, -1,
    -1, -1, -1, -1,-1
};

float cornerKernel[] = {
    -1, -1, 0, 1, 1,
    -1, -1, 0, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 0, -1, -1,
    1, 1, 0, -1, -1
};

float sobelKernelX[] = {1, 4, 6, 4, 1};

float sobelKernelY[] = {1, 2, 0, -2, 1};

int divisor = 48;

cv::Mat blob5x5(cv::Mat& image)  
{
    cv::Mat dst;
    cv::Mat blobFilter = cv::Mat(5, 5, CV_32F, blobKernel);
    cv::filter2D(image, dst, -1, blobFilter, cv::Point(-1, -1), 0,
             cv::BORDER_DEFAULT);
    return dst;         
}

cv::Mat corner5x5(cv::Mat& image)  
{
    cv::Mat dst;
    cv::Mat cornerFilter = cv::Mat(5, 5, CV_32F, cornerKernel);
    cv::filter2D(image, dst, -1, cornerFilter, cv::Point(-1, -1), 0,
             cv::BORDER_DEFAULT);
    return dst;         
}

cv::Mat gradientX(cv::Mat& image)
{
    cv::Mat grad, abs_grad;
    cv::Scharr( image, grad, -1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(grad, abs_grad);  
    return abs_grad; 
}

cv::Mat gradientY(cv::Mat& image)
{
    cv::Mat grad, abs_grad;
    cv::Scharr( image, grad, -1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(grad, abs_grad);  
    return abs_grad; 
}


