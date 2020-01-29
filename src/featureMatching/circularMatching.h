#ifndef CIRCULAR_MATCHING_INCLUDE

#define CIRCULAR_MATCHING_INCLUDE

#include "../config.h"
#include <vector>

std::vector<Match> performCircularMatching(int width, int height, std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r);

#endif