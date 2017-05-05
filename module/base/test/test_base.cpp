#include <iostream>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "base.hpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    if(argc!=2)
    {
        cout << "Usage: ./test_base image" <<endl;
        return -1;
    }

    //! Load image
    Mat origin = cv::imread(argv[1]);
    if(origin.empty())
    {
        cout << "Can not open image: " << argv[1] << endl;
        return -1;
    }
    cv::Mat gray;
    cvtColor(origin, gray, COLOR_BGR2GRAY);

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

    vector<Mat> images;
    vk::computePyramid(origin, images, scale, level);
    uint16_t  height = images[0].rows;
    uint16_t  width = 0;
    for(uint16_t i = 0; i < level+1; i++)
    {
        width += images[i].cols;
    }

    cv::Mat pyramid(height, width, origin.type());
    uint16_t cols = 0;
    for(uint16_t i = 0; i < level+1; i++)
    {
        images[i].copyTo(pyramid.rowRange(0, images[i].rows).colRange(cols, cols+images[i].cols));
        cols += images[i].cols;
    }

    imshow("pyramid", pyramid);
    waitKey(0);
    cv::destroyWindow("pyramid");

    //!
    //! Testing convolution
    //!
    cout << endl
        << "=================================" << endl
        << "--- Testing  convolution          " << endl
        << "=================================" << endl;
    cv::Mat kernel1 = (Mat_<float>(3,3)<< -1, 0, 1, -2, 0, 2, -1, 0, 1); kernel1/=8;
    cv::Mat kernel2 = (Mat_<float>(3,3)<< -1,-2,-1, 0, 0, 0, 1, 2, 1); kernel2/=8;
    cv::Mat kernel3 = (Mat_<float>(3,3)<< 1, 1, 1, 1, 1, 1, 1, 1, 1); kernel3/=9*256;
    cv::Mat convolution1, convolution2, convolution3;
    vk::conv(gray, convolution1, kernel1);
    vk::conv(gray, convolution2, kernel2);
    vk::conv(gray, convolution3, kernel3);
    cv::Mat convolution = cv::Mat(cv::Size(gray.cols*3, gray.rows), CV_32FC1);
    convolution1.copyTo(convolution.colRange(0, gray.cols));
    convolution2.copyTo(convolution.colRange(gray.cols, gray.cols*2));
    convolution3.copyTo(convolution.colRange(gray.cols*2, gray.cols*3));
    imshow("convolution", convolution);
    waitKey(0);
    cv::destroyWindow("convolution");

    return 0;
}