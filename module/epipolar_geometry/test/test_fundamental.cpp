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

#include "fundamental.hpp"

void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::DMatch>& matches);
int drawEpipolarLines(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Mat& img_epipolar, const cv::Mat& fundamental,
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
        std::cout << "No enough match points! Please adjust the parameters in keypoint matching or change the input images" << std::endl;
        return -1;
    }

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

    //! by OpenCV
    double cv_start_time = (double)cv::getTickCount();
    cv::Mat cv_F = findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99);
    double cv_time = ((double)cv::getTickCount() - cv_start_time) / cv::getTickFrequency();
    if (cv_F.empty())
    {
        std::cout << "Error in finding fundamental matrix by openCV!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Fund fundamental matrix:\n" << cv_F << std::endl;
    }

    //! by vk
    double vk_start_time = (double)cv::getTickCount();
    cv::Mat vk_F = vk::findFundamentalMat(points0, points1, vk::FM_RANSAC, 1.0, 1000);
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

    //! Draw epipolar lines
    cv::Mat cv_img_epipolar;
    cv::Mat vk_img_epipolar;
    double cv_error = 0, vk_error = 0;
    int cv_count = drawEpipolarLines(image0, image1, cv_img_epipolar, cv_F, points0, points1, cv_error);
    int vk_count = drawEpipolarLines(image0, image1, vk_img_epipolar, vk_F, points0, points1, vk_error);
    std::cout << "Total points:" << points0.size() << std::endl;
    std::cout << "CV Time: " << cv_time << " Error:" << cv_error << " Inliers:" << cv_count << std::endl;
    std::cout << "VK Time: " << vk_time << " Error:" << vk_error << " Inliers:" << vk_count << std::endl;

    //! Draw matches
    cv::Mat img_matches;
    drawMatches(image0, keypoints0, image1, keypoints1, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1));

    cv::imshow("Good Matches", img_matches);
    cv::imshow("Epipolar Lines of cv", cv_img_epipolar);
    cv::imshow("Epipolar Lines of vk", vk_img_epipolar);

    //cv::imwrite("matches.png", img_matches);
    //cv::imwrite("img_epipolar.png", img_epipolar);
    cv::waitKey(0);

    return 0;
}

void getGoodMatches(const cv::Mat& src, const cv::Mat& dest, std::vector<cv::KeyPoint>& kps0, std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::DMatch>& matches)
{
#ifdef USE_XFEATURES2D
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(500);
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

int drawEpipolarLines(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Mat& img_epipolar, const cv::Mat& fundamental,
    const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, double& error)
{
    const int N = pts_prev.size();
    if(N != pts_next.size())
        return -1;

    cv::Mat img_epipolar1 = img_prev.clone();
    cv::Mat img_epipolar2 = img_next.clone();

    cv::Mat MF = fundamental.clone();
    if(MF.type() != CV_32FC1)
    {
        MF.convertTo(MF, CV_32FC1);
    }
    const float* F = MF.ptr<float>(0);
    const int cols = img_prev.cols;
    const float threshold = 3.841;//! sigma = 1
    int count = 0;
    error = 0;
    for(int n = 0; n < N; ++n)
    {
        //! point X1 = (u1, v1, 1)^T in first image
        //! poInt X2 = (u2, v2, 1)^T in second image
        const double u1 = pts_prev[n].x;
        const double v1 = pts_prev[n].y;
        const double u2 = pts_next[n].x;
        const double v2 = pts_next[n].y;

        //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
        //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
        const double a2 = F[0]*u1 + F[1]*v1 + F[2];
        const double b2 = F[3]*u1 + F[4]*v1 + F[5];
        const double c2 = F[6]*u1 + F[7]*v1 + F[8];

        const double a1 = F[0]*u2 + F[3]*v2 + F[6];
        const double b1 = F[1]*u2 + F[4]*v2 + F[7];
        const double c1 = F[2]*u2 + F[5]*v2 + F[8];

        const double dist2 = a2*u2 + b2*v2 + c2;
        const double square_dist2 = dist2*dist2 / (a2*a2 + b2*b2);

        const double dist1 = a1*u1 + b1*v1 + c1;
        const double square_dist1 = dist1*dist1 / (a1*a1 + b1*b1);

        if (square_dist1 < threshold && square_dist2 < threshold)
        {
            count++;
            error += square_dist1 + square_dist2;
        }
        else
            continue;

        if(fabs(b1) < FLT_EPSILON || fabs(b2) < FLT_EPSILON)
            continue;

        //! points of the epipolar line within the image
        cv::Point2f start1(0, 0), end1(cols, 0), start2(0, 0), end2(cols, 0);
        start2.y = -(a2*start2.x + c2) / b2;
        end2.y = -(a2*end2.x + c2) / b2;
        start1.y = -(a1*start1.x + c1) / b1;
        end1.y = -(a1*end1.x + c1) / b1;

        //! draw lines and points in each image
        cv::Scalar color(255.0*rand()/RAND_MAX, 255.0*rand()/RAND_MAX, 255.0*rand()/RAND_MAX);

        cv::circle(img_epipolar1, pts_prev[n], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar1, start1, end1, color);

        cv::circle(img_epipolar2, pts_next[n], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar2, start2, end2, color);
    }
    if(count != 0)
        error /= count;

    //! copy the two images into one image
    img_epipolar = cv::Mat(img_epipolar1.rows, img_epipolar1.cols*2, img_epipolar1.type());
    img_epipolar1.copyTo(img_epipolar.colRange(0, cols));
    img_epipolar2.copyTo(img_epipolar.colRange(cols, 2*cols));

    return count;
}