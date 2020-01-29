#include <iostream>

#include "removeOutliers.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

int NCC_WINDOW_RADIUS = MATCH_NCC_WINDOW / 2;

bool isValidMatchNCC(cv::Mat& i1, cv::Mat& i2, cv::Point2i& p1, cv::Point2i& p2) {
    auto patch1 = cv::Rect(p1.x-NCC_WINDOW_RADIUS, p1.y-NCC_WINDOW_RADIUS, MATCH_NCC_WINDOW, MATCH_NCC_WINDOW);
    auto patch2 = cv::Rect(p2.x-NCC_WINDOW_RADIUS, p2.y-NCC_WINDOW_RADIUS, MATCH_NCC_WINDOW, MATCH_NCC_WINDOW);
    cv::Mat subimg1 = i1(patch1);
    cv::Mat subimg2 = i2(patch2);
    cv::Mat result;
    cv::Scalar mean1, std1, mean2, std2;
    cv::meanStdDev(subimg1, mean1, std1);
    cv::meanStdDev(subimg2, mean2, std2);

    if (std1[0] < 1e-3 || std2[0] < 1e-3) {
        return false;
    }
    // cv::subtract(subimg1, mean1, subimg1);
    // cv::subtract(subimg2, mean2, subimg2);
    cv::matchTemplate( subimg1, subimg2, result, CV_TM_CCOEFF_NORMED );
    double minv, maxv;
    cv::minMaxLoc(result, &minv, &maxv);
    // std::cout << maxv << " ";
    // if (maxv > 0.7) {
    //     cv::Mat vis;
    //     cv::cvtColor(subimg1, vis, CV_GRAY2BGR, 3);
    //     cv::imshow("img1", vis);

    //     cv::Mat vis2;
    //     cv::cvtColor(subimg2, vis2, CV_GRAY2BGR, 3);
    //     cv::imshow("img2", vis2);

    //     cv::waitKey(0);
    // }
    return (maxv > MATCH_NCC_TOLERANCE);
}
std::vector<Match> removeOutliers(cv::Mat i1_l, cv::Mat i1_r, cv::Mat i2_l, cv::Mat i2_r, std::vector<Match> matches) {
    std::vector<Match> filteredMatches;
    for (Match match: matches) {
        if (!isValidMatchNCC(i1_l, i1_r, match.pt1_l->point, match.pt1_r->point)) {
            continue;
        }
        if (!isValidMatchNCC(i1_r, i2_r, match.pt1_r->point, match.pt2_r->point)) {
            continue;
        }
        if (!isValidMatchNCC(i2_r, i2_l, match.pt2_r->point, match.pt2_l->point)) {
            continue;
        }
        if (!isValidMatchNCC(i2_l, i1_l, match.pt2_l->point, match.pt1_l->point)) {
            continue;
        }           
        filteredMatches.push_back(match);
    }
    return filteredMatches;
}