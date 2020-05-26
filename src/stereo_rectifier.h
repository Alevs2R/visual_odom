#ifndef UTIL_STEREO_RECTIFIER_H
#define UTIL_STEREO_RECTIFIER_H

#include <memory>
#include <opencv2/core/core.hpp>

class stereo_rectifier {
public:

    //! Constructor
    stereo_rectifier(cv::FileStorage cfg);

    //! Destructor
    virtual ~stereo_rectifier();

    //! Apply stereo-rectification
    void rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r,
                 cv::Mat& out_img_l, cv::Mat& out_img_r) const;

private:

    //! undistortion map for x-axis in left image
    cv::Mat undist_map_x_l_;
    //! undistortion map for y-axis in left image
    cv::Mat undist_map_y_l_;
    //! undistortion map for x-axis in right image
    cv::Mat undist_map_x_r_;
    //! undistortion map for y-axis in right image
    cv::Mat undist_map_y_r_;
};

#endif // UTIL_STEREO_RECTIFIER_H