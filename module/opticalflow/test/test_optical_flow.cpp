#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>
#include <base.hpp>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    if(argc!=3)
    {
        cout << "Usage: ./opticalflow image1 image2" <<endl;
        return -1;
    }

    //! Load image
    Mat image0 = cv::imread(argv[1]);
    Mat image1 = cv::imread(argv[2]);
    if(image0.empty())
    {
        cout << "Can not open image1: " << argv[1] << endl;
        return -1;
    }
    if(image1.empty())
    {
        cout << "Can not open image2: " << argv[2] << endl;
        return -1;
    }

    Mat gray0, gray1;
    cvtColor(image0, gray0, COLOR_BGR2GRAY);
    cvtColor(image1, gray1, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Ptr<ORB> detector = ORB::create(500);
    detector->detect(gray0, keypoints, Mat());

    vector<cv::Point2f> points_prev;
    if(!keypoints.empty())
    {
        for(vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
        {
            points_prev.push_back(it->pt);
        }
    }

    vector<cv::Point2f> points_next;
    vector<float> errors;

   vk::OpticalFlow::computePyrLK(gray0, gray1, points_prev, points_next, errors, cv::Size(21, 21), 3, 40, 0.001);

    int n_levels = 4;
    //! compute Pyramid images
    std::vector<cv::Mat> pyramid_prev, pyramid_next;
    vk::computePyramid(gray0, pyramid_prev, 2, n_levels -1);
    vk::computePyramid(gray1, pyramid_next, 2, n_levels -1);
    cv::Mat Sharr_dx = (cv::Mat_<float>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
    cv::Mat Sharr_dy = (cv::Mat_<float>(3, 3) << -3, -10, -3, 0, 0, 0, 3, 10, 3);
    std::vector<cv::Mat> pyramid_gradient_x(n_levels), pyramid_gradient_y(n_levels);
    for (int i = 0; i < n_levels; ++i)
    {
        vk::conv_32f(pyramid_next[i], pyramid_gradient_x[i], Sharr_dx, 32);
        vk::conv_32f(pyramid_next[i], pyramid_gradient_y[i], Sharr_dy, 32);
    }

    cv::Point2f p,q;
    int l = 3;
    p.x = keypoints[1].pt.x / (1 << l);
    p.y = keypoints[1].pt.y / (1 << l);
    cv:; Size size(21, 21);

    vk::align2D(pyramid_prev[l], pyramid_next[l], pyramid_gradient_x[l], pyramid_gradient_y[l],
        size, p, q);

    cv::Mat keypoint_image;
    drawKeypoints(image0, keypoints, keypoint_image);
    imshow("image", image0);
    waitKey(0);
}