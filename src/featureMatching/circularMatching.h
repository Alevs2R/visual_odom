#ifndef CIRCULAR_MATCHING_INCLUDE

#define CIRCULAR_MATCHING_INCLUDE

#include "../featureProcessing/nms.h"
#include <vector>

struct Match {
    KeyPoint* pt1_l;
    KeyPoint* pt1_r;
    KeyPoint* pt2_l;
    KeyPoint* pt2_r;
};

std::vector<Match> performCircularMatching(int width, int height, std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r);

#endif