#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <stdlib.h>

#include "fundamental.hpp"

void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1, std::vector<cv::DMatch>& matches);

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: ./find_F_matrix first_image second_image" << std::endl;
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
    cv::Mat img_matches;
    drawMatches(image0, keypoints0, image1, keypoints1,
        matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if(matches.size() < 8)
    {
        std::cout << "No enough match points! Please ajust the parameters in keypoint matching or change the input images" << std::endl;
        return -1;
    }
    std::sort(matches.begin(), matches.end());

    //! Estimation of fundamental matrix using the 8POINT algorithm
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


    //! 8POINT algorithm by OpenCV
    cv::Mat fundamental_matrix0 = findFundamentalMat(points0, points1, cv::FM_8POINT, 3, 0.99);
    if (fundamental_matrix0.empty())
    {
        std::cout << "Error in finding fundamental matrix by 8POINT!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund fundamental matrix:\n" << fundamental_matrix0 << std::endl;
    }

    //! 8POINT algorithm by vk
    cv::Mat fundamental_matrix1 = vk::findFundamentalMat(points0, points1, vk::FM_8POINT);
    if (fundamental_matrix1.empty())
    {
        std::cout << "Error in finding fundamental matrix by 8POINT!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund fundamental matrix:\n" << fundamental_matrix1 << std::endl;
    }

    //! Draw epipolar lines
    cv::Mat img_epipolar0 = image0.clone();
    cv::Mat img_epipolar1 = image1.clone();
    int step = points0.size() / 15 + 1;
    for(int i = 0; i < points0.size(); i+= step)
    {
        //! In second image
        cv::Mat point0 = (cv::Mat_<float>(3, 1) << points0[i].x, points0[i].y, 1);
        cv::Mat epipolar_line = fundamental_matrix1 * point0;

        float a = epipolar_line.at<float>(0, 0);
        float b = epipolar_line.at<float>(1, 0);
        float c = epipolar_line.at<float>(2, 0);
        if(fabs(b) < FLT_EPSILON)
            continue;

        cv::Point2f start(0, 0), end(image1.cols, 0);
        start.y = -(a*start.x + c) / b;
        end.y = -(a*end.x + c) / b;

        cv::Scalar color(255*rand()/RAND_MAX, 255*rand()/RAND_MAX, 255*rand()/RAND_MAX);
        cv::circle(img_epipolar1, points1[i], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar1, start, end, color);

        //! In first image
        cv::Mat point1 = (cv::Mat_<float>(3, 1) << points1[i].x, points1[i].y, 1);
        epipolar_line = fundamental_matrix1.t() * point1;

        a = epipolar_line.at<float>(0, 0);
        b = epipolar_line.at<float>(1, 0);
        c = epipolar_line.at<float>(2, 0);
        if(fabs(b) < FLT_EPSILON)
            continue;

        start.y = -(a*start.x + c) / b;
        end.y = -(a*end.x + c) / b;

        cv::circle(img_epipolar0, points0[i], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar0, start, end, color);
    }

    cv::Mat img_epipolar(img_epipolar0.rows, img_epipolar0.cols*2, img_epipolar0.type());
    img_epipolar0.copyTo(img_epipolar.colRange(0, img_epipolar0.cols));
    img_epipolar1.copyTo(img_epipolar.colRange(img_epipolar0.cols, img_epipolar.cols));

    cv::imshow("Good Mathes", img_matches);
    cv::imshow("Epipolar Lines", img_epipolar);
    cv::imwrite("matches.png", img_matches);
    cv::imwrite("img_epipolar.png", img_epipolar);
    cv::waitKey(0);

    return 0;
}
void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1, std::vector<cv::DMatch>& matches)
{
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(500);

    kps0.clear();
    kps1.clear();
    cv::Mat descriptors0, descriptors1;
    detector->detectAndCompute(src, cv::Mat(), kps0, descriptors0);
    detector->detectAndCompute(dest, cv::Mat(), kps1, descriptors1);

    cv::FlannBasedMatcher matcher;
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
}