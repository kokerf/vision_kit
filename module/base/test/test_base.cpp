#include <iostream>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "base.hpp"
#include "util.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./test_base image1 image2" <<endl;
        return -1;
    }

    //! Load image
    cv::Mat src_img = cv::imread(argv[1]);
    cv::Mat dst_img = cv::imread(argv[2]);
    if(src_img.empty())
    {
        cout << "Can not open image: " << argv[1] << endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);

    //!
    //! Testing Pyramidal
    //!
    float scale = 2;
    uint16_t level = 3;
    cout << "=================================" << endl
         << "--- Testing Pyramidal Images     " << endl
         << "--- Scale: " << scale << endl
         << "--- Level: " << level << endl
         << "=================================" << endl;

    vector<cv::Mat> images;
    level = vk::computePyramid(src_img, images, scale, level);
    uint16_t  height = images[0].rows;
    uint16_t  width = 0;
    for(uint16_t i = 0; i < level+1; i++)
    {
        width += images[i].cols;
    }

    cv::Mat pyramid(height, width, src_img.type());
    uint16_t cols = 0;
    for(uint16_t i = 0; i < level+1; i++)
    {
        images[i].copyTo(pyramid.rowRange(0, images[i].rows).colRange(cols, cols+images[i].cols));
        cols += images[i].cols;
    }

    cv::imshow("pyramid", pyramid);
    cv::waitKey(0);
    cv::destroyWindow("pyramid");

    //!
    //! Testing convolution
    //!
    cout << endl
        << "=================================" << endl
        << "--- Testing  convolution          " << endl
        << "=================================" << endl;
    cv::Mat kernel1 = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernel2 = (cv::Mat_<float>(3,3)<< -1,-2,-1, 0, 0, 0, 1, 2, 1);
    cv::Mat kernel3 = (cv::Mat_<float>(3,3)<< 1, 1, 1, 1, 1, 1, 1, 1, 1);
    cv::Mat convolution1, convolution2, convolution3;
    vk::conv_32f(gray, convolution1, kernel1, 8);
    vk::conv_32f(gray, convolution2, kernel2, 8);
    vk::conv_32f(gray, convolution3, kernel3, 9*256);
    cv::Mat convolution = cv::Mat(cv::Size(gray.cols*3, gray.rows), CV_32FC1);
    convolution1.copyTo(convolution.colRange(0, gray.cols));
    convolution2.copyTo(convolution.colRange(gray.cols, gray.cols*2));
    convolution3.copyTo(convolution.colRange(gray.cols*2, gray.cols*3));
    cv::imshow("convolution", convolution);
    cv::waitKey(0);
    cv::destroyWindow("convolution");

    //!
    //! Testing match
    //!
    cout << endl
        << "=================================" << endl
        << "--- Testing  match          " << endl
        << "=================================" << endl;
    vector<cv::Point2f> src_pt, dst_pt;
    cv::Mat src_gray, dst_gray;
    cv::cvtColor(src_img, src_gray, cv::COLOR_RGB2GRAY);
    cv::cvtColor(dst_img, dst_gray, cv::COLOR_RGB2GRAY);
    vk::getCorrespondPoints(src_gray, dst_gray, src_pt, dst_pt, 40);
    cv::Mat match;
    vk::drowMatchPoits(src_img, dst_img, src_pt, dst_pt, match);
    cv::imshow("match", match);
    cv::waitKey(0);
    cv::destroyWindow("match");

    return 0;
}