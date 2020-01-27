#include "../featureProcessing/nms.h"
#include <vector>

struct Match {
    KeyPoint& pt1_l;
    KeyPoint& pt1_r;
    KeyPoint& pt2_l;
    KeyPoint& pt2_r;
};

std::vector<Match> performCircularMatching(std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r);