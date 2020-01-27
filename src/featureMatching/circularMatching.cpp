#define MATCH_BINSIZE 50            // matching bin width/height (affects efficiency only)
#define MATCH_RADIUS 200            // matching radius (du/dv in pixels)
#define MATCH_DISP_TOLERANCE 1   // dx tolerance for stereo matches (in pixels)
#define MATCH_NCC_WINDOW 21      // window size of the patch for normalized cross-correlation
#define MATCH_NCC_TOLERANCE 0.3  // threshold for normalized cross-correlation
// TO-DO: add subpixel-refinement using parabolic fitting
#define MATCH_REFINEMENT 2       // refinement (0=none,1=pixel,2=subpixel)

#include "circularMatching.h"

int findMatchSAD(KeyPoint& keypt1, std::vector<KeyPoint>& keypts2, bool stereo) {
    int matched_ind = -1;
    int matched_sad = 0;

    int x1 = keypt1.point.x;
    int y1 = keypt1.point.y;
    int x_min = x1 - MATCH_RADIUS;
    int x_max = x1 + MATCH_RADIUS;
    int y_min = y1 - MATCH_RADIUS;
    int y_max = y1 + MATCH_RADIUS;

    int i, j;

    // TODO add division into bins
    for (j = 0; j < keypts2.size(); j++){
        KeyPoint& kp2 = keypts2[j];

        if (kp2.point.x < x_min || kp2.point.x > x_max || kp2.point.y < y_min || kp2.point.y > y_max) {
            continue;
        }

        if (kp2.pointClass != keypt1.pointClass) {
            continue;
        }

        int sad = 0;
        for (i = 0; i < 32; i++) {
            sad += abs(kp2.descriptor[i] - keypt1.descriptor[i]);
        }
        if (sad < matched_sad || matched_ind == -1) {
            matched_ind = j;
            matched_sad = sad;
        }
    }
    return matched_ind;
}

std::vector<Match> performCircularMatching(std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r) {
    // TODO execute in parallel
    std::vector<Match> matches;
    for (int i = 0; i < pts1_l.size(); i++) {
        int pt1_r_match = findMatchSAD(pts1_l[i], pts1_r, true);
        if (pt1_r_match == -1) continue;

        int pt2_r_match = findMatchSAD(pts1_r[pt1_r_match], pts2_r, false);  
        if (pt2_r_match == -1) continue;

        int pt2_l_match = findMatchSAD(pts2_r[pt2_r_match], pts2_l, true);  
        if (pt2_l_match == -1) continue;

        int pt1_l_match = findMatchSAD(pts2_l[pt2_l_match], pts1_l, false);  
        if (pt1_l_match == -1) continue;

        if (pt1_l_match == i) {
            // calculate disparity
            int disp1 = pts1_l[pt1_l_match].point.x - pts1_r[pt1_r_match].point.x;
            int disp2 = pts2_l[pt2_l_match].point.x - pts2_r[pt2_r_match].point.x;
            if (disp1 > 0 && disp2 > 0) {
                pts2_l[pt2_l_match].age = pts1_l[pt1_l_match].age + 1; // increase age 
                memcpy(pts2_l[pt2_l_match].descriptor, pts1_l[pt1_l_match].descriptor, 32*sizeof(uint16_t)); // keep original descriptor
                matches.push_back({ pts1_l[pt1_l_match], pts1_r[pt1_r_match], pts2_l[pt2_l_match], pts2_r[pt2_r_match] });
            }
        }
    }
    return matches;
}