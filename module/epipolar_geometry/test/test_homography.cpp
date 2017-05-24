#include <iostream>
#include <vector>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#ifdef USE_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#else
#include <opencv2/features2d/features2d.hpp>
#endif

#include "homography.hpp"

void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::DMatch>& matches);

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: ./test_fundamental first_image second_image" << std::endl;
        return -1;
    }

    cv::Mat image0 = cv::imread(argv[1]);
    cv::Mat image1 = cv::imread(argv[2]);

    if(image0.empty() || image1.empty())
    {
        std::cout << "Error in open image: " << argv[1] << " " << argv[2] << std:: endl;
        return -1;
    }

    cv::Mat gray0, gray1;
    cvtColor(image0, gray0, cv::COLOR_RGB2GRAY);
    cvtColor(image1, gray1, cv::COLOR_RGB2GRAY);

    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    std::vector<cv::DMatch> matches;
    getGoodMatches(gray0, gray1, keypoints0, keypoints1, matches);

    if(matches.size() < 8)
    {
        std::cout << "No enough match points! Please ajust the parameters in keypoint matching or change the input images" << std::endl;
        return -1;
    }

    int point_count = matches.size();
    std::vector<cv::Point2f> points0(point_count);
    std::vector<cv::KeyPoint> good_keypoints0, good_keypoints1;
    std::vector<cv::Point2f> points1(point_count);
    for (int i = 0; i < point_count; i++)
    {
        good_keypoints0.push_back(keypoints0[matches[i].queryIdx]);
        good_keypoints1.push_back(keypoints1[matches[i].trainIdx]);
        points0[i] = keypoints0[matches[i].queryIdx].pt;
        points1[i] = keypoints1[matches[i].trainIdx].pt;
    }

    //! by OpenCV
    double cv_start_time = (double)cv::getTickCount();
    cv::Mat cv_H = cv::findHomography(points0, points1, 0);
    double cv_time = ((double)cv::getTickCount() - cv_start_time) / cv::getTickFrequency();
    if (cv_H.empty())
    {
        std::cout << "Error in finding fundamental matrix by openCV!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund fundamental matrix:\n" << cv_H << std::endl;
    }

    //! by vk
    double vk_start_time = (double)cv::getTickCount();
    cv::Mat vk_F = vk::findHomographyMat(points0, points1, vk::HM_DLT, 1.0, -1);
    double vk_time = ((double)cv::getTickCount() - vk_start_time) / cv::getTickFrequency();
    if (vk_F.empty())
    {
        std::cout << "Error in finding fundamental matrix by visionkit!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund fundamental matrix:\n" << vk_F << std::endl;
    }

    double cv_error = 0, vk_error = 0;
    int cv_count = 0;
    int vk_count = 0;
    std::cout << "Total points:" << points0.size() << std::endl;
    std::cout << "CV Time: " << cv_time << " Error:" << cv_error << " Inliers:" << cv_count << std::endl;
    std::cout << "VK Time: " << vk_time << " Error:" << vk_error << " Inliers:" << vk_count << std::endl;


    //! Draw matches
    cv::Mat img_matches;
    drawMatches(image0, keypoints0, image1, keypoints1, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1));

    cv::imshow("Good Matches", img_matches);

    cv::waitKey(0);
    return 0;
}


void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::DMatch>& matches)
{
#ifdef USE_XFEATURES2D
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(1000);
#else
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
#endif

    kps0.clear();
    kps1.clear();
    cv::Mat descriptors0, descriptors1;
    detector->detectAndCompute(src, cv::Mat(), kps0, descriptors0);
    detector->detectAndCompute(dest, cv::Mat(), kps1, descriptors1);

#ifdef USE_XFEATURES2D
    cv::FlannBasedMatcher matcher;
#else
    cv::BFMatcher matcher(cv::NORM_HAMMING);
#endif

    std::vector<cv::DMatch> temp_matches;
    matcher.match(descriptors0, descriptors1, temp_matches);

    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptors0.rows; i++)
    {
        double dist = temp_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::cout << "-- Max dist :" << max_dist
        << "-- Min dist :" << min_dist << std::endl;

    matches.clear();
    for (int i = 0; i < descriptors0.rows; i++)
    {
        if (temp_matches[i].distance < 3 * min_dist)
        {
            matches.push_back(temp_matches[i]);
        }
    }

    std::sort(matches.begin(), matches.end());
}