#include "stereo_rectifier.h"
#include <opencv2/imgproc.hpp>

stereo_rectifier::stereo_rectifier(cv::FileStorage cfg) {
    const cv::Size img_size((int)cfg["Camera.cols"], (int)cfg["Camera.rows"]);

    cv::Mat K_l, K_r, R_l, R_r, D_l, D_r;

    cfg["StereoRectifier.K_left"] >> K_l;
    cfg["StereoRectifier.K_right"] >> K_r;

    cfg["StereoRectifier.R_left"] >> R_l;
    cfg["StereoRectifier.R_right"] >> R_r;

    cfg["StereoRectifier.D_left"] >> D_l;
    cfg["StereoRectifier.D_right"] >> D_r;
        
    double fx = cfg["Camera.fx"];
    double fy = cfg["Camera.fy"];
    double cx = cfg["Camera.cx"];
    double cy = cfg["Camera.cy"];
    double bf = cfg["Camera.bf"];

    cv::Mat K_rect = (cv::Mat_<double>(3, 3) << fx, 0., cx, 0., fy, cy, 0.,  0., 1.);

    cv::initUndistortRectifyMap(K_l, D_l, R_l, K_rect, img_size, CV_32F, undist_map_x_l_, undist_map_y_l_);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, K_rect, img_size, CV_32F, undist_map_x_r_, undist_map_y_r_);
}

stereo_rectifier::~stereo_rectifier() {
    
}

void stereo_rectifier::rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r,
                               cv::Mat& out_img_l, cv::Mat& out_img_r) const {
    cv::remap(in_img_l, out_img_l, undist_map_x_l_, undist_map_y_l_, cv::INTER_LINEAR);
    cv::remap(in_img_r, out_img_r, undist_map_x_r_, undist_map_y_r_, cv::INTER_LINEAR);
}