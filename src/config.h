#ifndef VO_CONFIG

#define VO_CONFIG

#include "opencv2/core/core.hpp"

#define NMS_N 8
#define NMS_TAU 90
#define NMS_MARGIN 21

#define MATCH_BINSIZE 50            // matching bin width/height (affects efficiency only)
#define MATCH_RADIUS 100          // matching radius (du/dv in pixels)
#define MATCH_DISP_TOLERANCE 1   // dx tolerance for stereo matches (in pixels)
// TO-DO: add subpixel-refinement using parabolic fitting
#define MATCH_REFINEMENT 2       // refinement (0=none,1=pixel,2=subpixel)

#define MATCH_NCC_WINDOW 31     // window size of the patch for normalized cross-correlation
#define MATCH_NCC_TOLERANCE 0.33 // threshold for normalized cross-correlation

#define OPENCV_TRAITS_ENABLE_DEPRECATED

struct KeyPoint {
    cv::Point2i point;
    int16_t value;
    int pointClass;
    int16_t descriptor[32];
    int age = 0;
    int id = rand();
    int parentKeyPointInd = -1;
    KeyPoint(cv::Point2i point, int16_t value, int pointClass) : point(point), value(value), pointClass(pointClass) {}
    KeyPoint(){}
};

struct Match {
    KeyPoint* pt1_l;
    KeyPoint* pt1_r;
    KeyPoint* pt2_l;
    KeyPoint* pt2_r;
};

#endif
