#include "circularMatching.h"
#include <iostream>
#include <cmath>
#include "omp.h"

typedef std::vector<std::vector<std::vector<KeyPoint*>>> BIN_VECTOR;

void findMatchSAD(KeyPoint& keypt1, BIN_VECTOR& bins2, bool stereo, KeyPoint* &mathedKp, bool& success) {
    int matched_sad = 0;
    success = false;

    int x1 = keypt1.point.x;
    int y1 = keypt1.point.y;

    int y_min;
    int y_max;

    int x_min = x1 - MATCH_RADIUS;
    int x_max = x1 + MATCH_RADIUS;

    if (stereo) {
        y_min = y1 - MATCH_DISP_TOLERANCE;
        y_max = y1 + MATCH_DISP_TOLERANCE;
    } else {
        y_min = y1 - MATCH_RADIUS;
        y_max = y1 + MATCH_RADIUS;
    }
    //std::cout << "y_max y_min " << y_max << " " << y_min << "\n";
    int x_bin_min = std::max(0, x_min/MATCH_BINSIZE);
    int x_bin_max = std::min((int)bins2[0].size()-1, x_max/MATCH_BINSIZE);
    int y_bin_min = std::max(0, y_min/MATCH_BINSIZE);
    int y_bin_max = std::min((int)bins2.size()-1, y_max/MATCH_BINSIZE);

    int i, m, n;

    // TODO add division into bins

    for (m = y_bin_min; m <= y_bin_max; m++){
        for (n = x_bin_min; n <= x_bin_max; n++) {
            //std::cout << "checking bin, size " << bins2[m][n].size() << "\n";
            for (auto kp2: bins2[m][n]) {
            if (kp2->point.x < x_min || kp2->point.x > x_max || kp2->point.y < y_min || kp2->point.y > y_max) {
                continue;
            }

            if (kp2->pointClass != keypt1.pointClass) {
                continue;
            }
            int sad = 0;
            for (i = 0; i < 32; i++) {
                sad += abs(kp2->descriptor[i] - keypt1.descriptor[i]);
            }
            if (sad < matched_sad || success == false) {
                matched_sad = sad;
                mathedKp = kp2;
                success = true;
            }
            }
        }
    }
}

void createIndexVector(std::vector<KeyPoint>& points, BIN_VECTOR& bins) {

    for (KeyPoint& kp: points) {
        int bin_x = kp.point.x/MATCH_BINSIZE;
        int bin_y = kp.point.y/MATCH_BINSIZE;
        bins[bin_y][bin_x].push_back(&kp);
    }
}

std::vector<Match> performCircularMatching(int width, int height, std::vector<KeyPoint>& pts1_l, std::vector<KeyPoint>& pts2_l,
                             std::vector<KeyPoint>& pts1_r, std::vector<KeyPoint>& pts2_r) {
    // TODO execute in parallel
    #pragma omp declare reduction (merge : std::vector<Match> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    std::vector<Match> matches;

    int x_bin_num = width / MATCH_BINSIZE + 1;
    int y_bin_num = height / MATCH_BINSIZE + 1;

    BIN_VECTOR bins1_l(y_bin_num,std::vector<std::vector<KeyPoint*> >(x_bin_num,std::vector <KeyPoint*>()));
    BIN_VECTOR bins1_r(y_bin_num,std::vector<std::vector<KeyPoint*> >(x_bin_num,std::vector <KeyPoint*>()));
    BIN_VECTOR bins2_l(y_bin_num,std::vector<std::vector<KeyPoint*> >(x_bin_num,std::vector <KeyPoint*>()));
    BIN_VECTOR bins2_r(y_bin_num,std::vector<std::vector<KeyPoint*> >(x_bin_num,std::vector <KeyPoint*>()));

    createIndexVector(pts1_l, bins1_l);
    createIndexVector(pts1_r, bins1_r);
    createIndexVector(pts2_l, bins2_l);
    createIndexVector(pts2_r, bins2_r);

    #pragma omp parallel reduction(merge: matches) num_threads(3)
    {
    #pragma omp for
    for (int i = 0; i < pts1_l.size(); i++) {
        // std::cout << "Greetings from thread "<<omp_get_thread_num()<<std::endl;

        KeyPoint* pt1_r, *pt1_l, *pt2_r, *pt2_l;
        bool pt1_r_matched, pt2_r_matched, pt2_l_matched, pt1_l_mathed;
        findMatchSAD(pts1_l[i], bins1_r, 1, pt1_r, pt1_r_matched);

        if (!pt1_r_matched) continue;


        findMatchSAD(*pt1_r, bins2_r, 0, pt2_r, pt2_r_matched);  
 
        if (!pt2_r_matched) continue;


        findMatchSAD(*pt2_r, bins2_l, 1, pt2_l, pt2_l_matched);  
        if (!pt2_l_matched) continue;

        findMatchSAD(*pt2_l, bins1_l, 0, pt1_l, pt1_l_mathed);  
        if (!pt1_l_mathed) continue;

//        float frame_time = 1000*(double)(clock()-t_a)/CLOCKS_PER_SEC;
        // std::cout << "[Info] findMatchSAD times (ms): " << frame_time << std::endl;


        if (pt1_l->id == pts1_l[i].id) {
            // calculate disparity
            int disp1 = pt1_l->point.x - pt1_r->point.x;
            int disp2 = pt2_l->point.x - pt2_r->point.x;
            if (disp1 > 0 && disp2 > 0) {
                pt2_l->age = pt1_l->age + 1; // increase age 
                memcpy(pt2_l->descriptor, pt1_l->descriptor, 32*sizeof(uint16_t)); // keep original descriptor
                matches.push_back({ pt1_l, pt1_r, pt2_l, pt2_r });
            }
        }        
    }
    }
    return matches;
}
