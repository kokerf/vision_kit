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

#include "base.hpp"
#include "homography.hpp"

void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::DMatch>& matches);

int drawHomographyMatches(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Mat& img_match, const cv::Mat& homography,
    const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, double& error);

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
    cv::Mat mask;
    cv::Mat cv_H = cv::findHomography(points0, points1, cv::RANSAC, 5.991/2);// threshold = (5.991/2)^2 = 8.97
    double cv_time = ((double)cv::getTickCount() - cv_start_time) / cv::getTickFrequency();
    if (cv_H.empty())
    {
        std::cout << "Error in finding homography matrix by openCV!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund homography matrix:\n" << cv_H << std::endl;
    }

    //! by vk
    double vk_start_time = (double)cv::getTickCount();
    cv::Mat vk_H = vk::findHomographyMat(points0, points1, vk::HM_RANSAC, 1.224);// threshold = 5.991*1.224^2=8.98
    double vk_time = ((double)cv::getTickCount() - vk_start_time) / cv::getTickFrequency();
    if (vk_H.empty())
    {
        std::cout << "Error in finding homography matrix by visionkit!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund homography matrix:\n" << vk_H << std::endl;
    }

    cv::Mat cv_img_matches;
    cv::Mat vk_img_matches;
    double cv_error = 0, vk_error = 0;
    int cv_count = drawHomographyMatches(image0, image1, cv_img_matches, cv_H, points0, points1, cv_error);
    int vk_count = drawHomographyMatches(image0, image1, vk_img_matches, vk_H, points0, points1, vk_error);
    std::cout << "Total points:" << points0.size() << std::endl;
    std::cout << "CV Time: " << cv_time << " Error:" << cv_error << " Inliers:" << cv_count << std::endl;
    std::cout << "VK Time: " << vk_time << " Error:" << vk_error << " Inliers:" << vk_count << std::endl;


    //! Draw matches
    cv::Mat img_matches;
    drawMatches(image0, keypoints0, image1, keypoints1, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1));

    cv::imshow("Good Matches", img_matches);
    cv::imshow("CV Homography Matches", cv_img_matches);
    cv::imshow("VK Homography Matches", vk_img_matches);

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
        if (temp_matches[i].distance < 5 * min_dist)
        {
            matches.push_back(temp_matches[i]);
        }
    }

    std::sort(matches.begin(), matches.end());
}

int drawHomographyMatches(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Mat& img_match, const cv::Mat& homography,
    const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, double& error)
{
    const int N = pts_prev.size();
    if(N != pts_next.size())
        return -1;

    const int cols = img_prev.cols;
    img_match = cv::Mat(img_prev.rows, img_prev.cols*2, img_prev.type());
    img_prev.copyTo(img_match.colRange(0, cols));
    img_next.copyTo(img_match.colRange(cols, 2*cols));
    cv::Point2f offset(cols, 0);

    cv::Mat H = homography.clone();
    if(H.type() != CV_32FC1)
    {
        H.convertTo(H, CV_32FC1);
    }
    const cv::Mat Hinv = H.inv();
    const float threshold = 5.991;//! sigma = 1
    int count = 0;
    error = 0;

    for(int n = 0; n < N; ++n)
    {
        const cv::Point2f& p1 = pts_prev[n];
        const cv::Point2f& p2 = pts_next[n];

        const double error1 = vk::transferError(p1, p2, H.ptr<float>(0));
        const double error2 = vk::transferError(p2, p1, Hinv.ptr<float>(0));

        const double transfer_error = error1 + error2;
        if(transfer_error < threshold)
        {
            count++;
            error+=transfer_error;
        }
        else
            continue;

        cv::Scalar color(255.0*rand()/RAND_MAX, 255.0*rand()/RAND_MAX, 255.0*rand()/RAND_MAX);
        cv::circle(img_match, p1, 3, color, 1, cv::LINE_AA);
        cv::circle(img_match, p2+offset, 3, color, 1, cv::LINE_AA);
        cv::line(img_match, p1, p2+offset, color);
    }
    if(count != 0)
        error /= count;

    return count;
}