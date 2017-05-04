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

    float scale = 2;
    uint16_t level = 3;
    cout << "---------------------------------" << endl
         << "--- Creating Pyramidal Images    " << endl
         << "--- Scale: " << scale << endl
         << "--- Level: " << level << endl
         << "---------------------------------" << endl;

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

    imshow("pyr", pyramid);
    waitKey(0);
}