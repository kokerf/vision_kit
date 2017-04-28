#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char const *argv[])
{
	//! Load image
	Mat srcImage = cv::imread("../../../data/1.png");
	Mat destImage = cv::imread("../../data/2.png");

	imshow("image", srcImage);
}