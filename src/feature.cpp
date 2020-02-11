#include "feature.h"
#include "bucket.h"
#include "featureProcessing/filters.h"
#include "featureProcessing/computeDescriptors.h"
#include "omp.h"


void deleteUnmatchFeatures(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<uchar>& status)
{
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points1.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))   
        {
              if((pt.x<0)||(pt.y<0))    
              {
                status.at(i) = 0;
              }
              points0.erase (points0.begin() + (i - indexCorrection));
              points1.erase (points1.begin() + (i - indexCorrection));
              indexCorrection++;
        }
     }
}

std::vector<KeyPoint> featureDetectionGeiger(cv::Mat& image)  
{  
    cv::Mat gradX, gradY, cornerF, blobF;
    gradX = gradientX(image);
    gradY = gradientY(image);
    cornerF = corner5x5(image);
    blobF = blob5x5(image);
    // double start = omp_get_wtime();
    std::vector<KeyPoint> keypts = nonMaximaSuppression(blobF, cornerF);
    // printf("nms %f sec\n",omp_get_wtime() - start); 
    // start = omp_get_wtime();  
    computeDescriptors(gradX, gradY, keypts);
    // printf("computing descriptors %f sec\n",omp_get_wtime() - start);  

    return keypts;
}

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
        Buckets.push_back(Bucket(features_per_bucket));
      }
    }

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
      buckets_nums_height_idx = current_features.points[i].y/bucket_size;
      buckets_nums_width_idx = current_features.points[i].x/bucket_size;
      buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
      Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);

    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
         buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
         Buckets[buckets_idx].get_features(current_features);
      }
    }

    std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}
