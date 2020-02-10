#include "visualOdometry.h"
#include "featureProcessing/nms.h"
#include "featureMatching/circularMatching.h"
#include "featureMatching/removeOutliers.h"
#include "omp.h"



cv::Mat euler2rot(cv::Mat& rotationMatrix, const cv::Mat & euler)
{

  double x = euler.at<double>(0);
  double y = euler.at<double>(1);
  double z = euler.at<double>(2);

  // Assuming the angles are in radians.
  double ch = cos(z);
  double sh = sin(z);
  double ca = cos(y);
  double sa = sin(y);
  double cb = cos(x);
  double sb = sin(x);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = ch * ca;
  m01 = sh*sb - ch*sa*cb;
  m02 = ch*sa*sb + sh*cb;
  m10 = sa;
  m11 = ca*cb;
  m12 = -ca*sb;
  m20 = -sh*ca;
  m21 = sh*sa*cb + ch*sb;
  m22 = -sh*sa*sb + ch*cb;

  rotationMatrix.at<double>(0,0) = m00;
  rotationMatrix.at<double>(0,1) = m01;
  rotationMatrix.at<double>(0,2) = m02;
  rotationMatrix.at<double>(1,0) = m10;
  rotationMatrix.at<double>(1,1) = m11;
  rotationMatrix.at<double>(1,2) = m12;
  rotationMatrix.at<double>(2,0) = m20;
  rotationMatrix.at<double>(2,1) = m21;
  rotationMatrix.at<double>(2,2) = m22;

  return rotationMatrix;
}

std::vector<Match> matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      std::vector<KeyPoint>&  pts1_l, 
                      std::vector<KeyPoint>&  pts1_r, 
                      std::vector<KeyPoint>&  pts2_l, 
                      std::vector<KeyPoint>&  pts2_r,
                      cv::Mat& projMatrl,
                      cv::Mat& projMatrr,
                      cv::Mat& transform
                      )
{
    double start = omp_get_wtime();

    pts2_l = featureDetectionGeiger(imageLeft_t1);
    pts2_r = featureDetectionGeiger(imageRight_t1);
    std::cout << "left features detected size " << pts2_l.size() << std::endl;
    printf("feature detecting %f sec\n",omp_get_wtime() - start);  
    start = omp_get_wtime();

    std::vector<Match> matches = performCircularMatching(imageRight_t0, imageLeft_t0.cols, imageLeft_t0.rows, pts1_l, pts2_l, pts1_r, pts2_r, projMatrl, projMatrr, transform);
    std::cout << "Match set size: " << matches.size() << std::endl;
    printf("circular matching %f sec\n",omp_get_wtime() - start);  
    start = omp_get_wtime();
    auto filteredMatches = removeOutliers(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1, matches);
    printf("removing outliers %f sec\n",omp_get_wtime() - start);  
    std::cout << "After filtering there is a match size: " << filteredMatches.size() << std::endl;
      // -----------------------------------------
    // displayKeypoints(imageLeft_t1, pts2_l);
    displayTracking(imageLeft_t1, filteredMatches);
    return filteredMatches;

    // cv::waitKey();

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    // int bucket_size = 50;
    // int features_per_bucket = 5;
    // bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket);

    // pointsLeft_t0 = currentVOFeatures.points;
    
    // circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
    //                  pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);

    // std::vector<bool> status;
    // checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);

    // removeInvalidPoints(pointsLeft_t0, status);
    // removeInvalidPoints(pointsLeft_t1, status);
    // removeInvalidPoints(pointsRight_t0, status);
    // removeInvalidPoints(pointsRight_t1, status);

    // currentVOFeatures.points = pointsLeft_t1;

}


void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t0,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,
                         bool mono_rotation)
{

      // Calculate frame to frame transformation

      // -----------------------------------------------------------
      // Rotation(R) estimation using Nister's Five Points Algorithm
      // -----------------------------------------------------------
      double focal = projMatrl.at<float>(0, 0);
      cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));

      //recovering the pose and the essential cv::matrix
      cv::Mat E, mask;
      cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
      E = cv::findEssentialMat(pointsLeft_t0, pointsLeft_t1, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
      cv::recoverPose(E, pointsLeft_t0, pointsLeft_t1, rotation, translation_mono, focal, principle_point, mask);
      // std::cout << "recoverPose rotation: " << rotation << std::endl;

      // ------------------------------------------------
      // Translation (t) estimation by use solvePnPRansac
      // ------------------------------------------------
      cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  
      cv::Mat inliers;  
      cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
      cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                   projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                   projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

      int iterationsCount = 500;        // number of Ransac iterations.
      float reprojectionError = .5;    // maximum allowed distance to consider it an inlier.
      float confidence = 0.999;          // RANSAC successful confidence.
      bool useExtrinsicGuess = true;
      int flags =cv::SOLVEPNP_ITERATIVE;

      cv::solvePnPRansac( points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation,
                          useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                          inliers, flags );

      if (!mono_rotation)
      {
        cv::Rodrigues(rvec, rotation);
      }

      std::cout << "[trackingFrame2Frame] inliers size: " << inliers.size() << std::endl;

}

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<Match>&  matches)
{
      // -----------------------------------------
      // Display feature racking
      // -----------------------------------------
      int radius = 2;
      cv::Mat vis;

      cv::cvtColor(imageLeft_t1, vis, CV_GRAY2BGR, 3);
      for (int i = 0; i < matches.size(); i++)
      {
        cv::circle(vis, cvPoint(matches[i].pt1_l->point.x, matches[i].pt1_l->point.y), radius, CV_RGB(0,255,0));
        cv::circle(vis, cvPoint(matches[i].pt2_l->point.x, matches[i].pt2_l->point.y), radius, CV_RGB(255,0,0));
        cv::line(vis, matches[i].pt1_l->point, matches[i].pt2_l->point, CV_RGB(0,255,0));
      }

      cv::imshow("vis ", vis );  
      cv::waitKey(1);
}
