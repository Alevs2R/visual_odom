
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <filesystem>
#include <time.h>       /* time */
#include <stdlib.h>     /* srand, rand */


#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "Frame.h"
#include "featureProcessing/filters.h"
#include "omp.h"

using namespace std;
namespace fs = std::__fs::filesystem;

int main(int argc, char **argv)
{
    // float testAr[] = {
    //     1,1,1,0,1,
    //     1,1,1,0,1,
    //     0,0,0,0,0,
    //     1,1,1,0,1,
    //     1,1,1,0,1,
    // };
    // cv::Mat testImg = cv::Mat(5, 5, CV_32F, testAr);
    // cv::Mat grad_x, grad_y;
    // cv::Mat abs_grad_x, abs_grad_y;   
    // cv::Scharr( testImg, grad_x, -1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    // cv::Scharr( testImg, grad_y, -1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    // convertScaleAbs( grad_x, abs_grad_x );
    // convertScaleAbs( grad_y, abs_grad_y );

    // cout << "sourc: " << testImg << endl;
    // cout << "grad y: " << abs_grad_y << endl;
    // cout << "grad x: " << abs_grad_x << endl;

    // return 0; 
    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    bool display_ground_truth = false;
    srand (time(NULL)); // rand for keypoint ids
    std::vector<Matrix> pose_matrix_gt;
    if(argc == 4)
    {   display_ground_truth = true;
        cerr << "Display ground truth trajectory" << endl;
        // load ground truth pose
        string filename_pose = string(argv[3]);
        pose_matrix_gt = loadPoses(filename_pose);

    }
    if(argc < 3)
    {
        cerr << "Usage: ./run path_to_sequence path_to_calibration [optional]path_to_ground_truth_pose" << endl;
        return 1;
    }

    // Sequence
    string filepath = string(argv[1]);
    cout << "Filepath: " << filepath << endl;

    // Camera calibration
    string strSettingPath = string(argv[2]);
    cout << "Calibration Filepath: " << strSettingPath << endl;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat cameraMatrl = projMatrl(cv::Rect( 0, 0, 3, 3 ));
    cv::Mat cameraMatrr = projMatrr(cv::Rect( 0, 0, 3, 3 ));

    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;
    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(1200, 1400, CV_8UC3);
    cv::Mat points4D, points3D;
    int init_frame_id = 0;

    vector<string> imagenames;
    std::string path = filepath + "image_0/";
    for (const auto & entry : fs::directory_iterator(path))
        imagenames.push_back(entry.path().filename());
    std::sort(imagenames.begin(), imagenames.end());

    FILE* result_poses_file = fopen ("./result.txt","w");

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageLeft_t0_color,  imageLeft_t0;
    loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath, imagenames);
    
    cv::Mat imageRight_t0_color, imageRight_t0;  
    loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath, imagenames);
    clock_t t_a, t_b;

    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;

    std::vector<KeyPoint> pts1_l, pts1_r, pts2_l, pts2_r;

    pts1_l = featureDetectionGeiger(imageLeft_t0);
    pts1_r = featureDetectionGeiger(imageRight_t0);

    double rmatr_data[9] = { 1,0,0,0,1,0,0,0,1};
    cv::Mat rmatrix = cv::Mat(3, 3, CV_64F, rmatr_data); // rvec to project on left camera

    cv::Mat tvec_l = projMatrl.col(3);

    cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64F);

    for (int frame_id = init_frame_id+1; frame_id < imagenames.size(); frame_id++)
    {

        std::cout << std::endl << "frame_id " << frame_id << std::endl;

        double very_start = omp_get_wtime();
        // ------------
        // Load images
        // ------------
        cv::Mat imageLeft_t1_color,  imageLeft_t1;
        loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath, imagenames);        
        cv::Mat imageRight_t1_color, imageRight_t1;  
        loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath, imagenames);


        cv::Mat transformation;
        cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
        cv::hconcat(rotation, -translation, transformation);
        cv::vconcat(transformation, addup, transformation);

        // transformation = transformation;

        std::vector<Match> matches = matchingFeatures( imageLeft_t0, imageRight_t0,
                          imageLeft_t1, imageRight_t1, 
                          pts1_l, pts1_r, pts2_l, pts2_r, projMatrr, projMatrl, transformation);

        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  

        for (auto& match: matches) {
            pointsLeft_t0.push_back(cv::Point2f(match.pt1_l->point));
            pointsRight_t0.push_back(cv::Point2f(match.pt1_r->point));
            pointsLeft_t1.push_back(cv::Point2f(match.pt2_l->point));
            pointsRight_t1.push_back(cv::Point2f(match.pt2_r->point));
        }

        // continue;

        // ---------------------
        // Triangulate 3D Points
        // ---------------------
        cv::Mat points3D_t0, points4D_t0;

        double start = omp_get_wtime();
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
        printf("triangulating %f \n",omp_get_wtime() - start);  
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

        displayDepthMap(points3D_t0, imageLeft_t0, rmatrix, tvec_l, cameraMatrl, distCoeffs, 30.0f);

        cv::Mat points3D_t1, points4D_t1;
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t1,  pointsRight_t1,  points4D_t1);
        cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);

        // // ---------------------
        // // Tracking transfomation
        // // ---------------------
        start = omp_get_wtime();
        trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation, false);
        printf("calculate transformation %f \n",omp_get_wtime() - start);  
        // displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);


        points4D = points4D_t0;
        frame_pose.convertTo(frame_pose32, CV_32F);
        points4D = frame_pose32 * points4D;
        cv::convertPointsFromHomogeneous(points4D.t(), points3D);

        // // ------------------------------------------------
        // // Intergrating and display
        // // ------------------------------------------------

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);


        cv::Mat rigid_body_transformation;

        std::cout << "rotation euler " << rotation_euler << "\n";

        if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);

        } else {

            std::cout << "Too large rotation"  << std::endl;
        }

        printf("overall time %f \n",omp_get_wtime() - very_start);  

        // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;
        // std::cout << "rotation: " << rotation_euler << std::endl;
        // std::cout << "translation: " << translation.t() << std::endl;
        // std::cout << "frame_pose" << frame_pose << std::endl;


        cv::Mat xyz = frame_pose.col(3).clone();
        display(frame_id, trajectory, xyz, pose_matrix_gt, 10, display_ground_truth);
        // logToFile(result_poses_file, frame_pose);
        // cv::waitKey(1);

        pts1_l = std::move(pts2_l);
        pts1_r = std::move(pts2_r);

    }
    fclose(result_poses_file);
    cv::waitKey(0);
    return 0;
}

