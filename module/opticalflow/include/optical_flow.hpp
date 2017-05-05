#ifndef _OPTICAL_FLOW_HPP_
#define _OPTICAL_FLOW_HPP_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace vk{

class OpticalFlow
{
public:
    OpticalFlow();
    ~OpticalFlow();
    static void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Point2d& points_prev, cv::Point2d& points_next,
        std::vector<float>& errors, const cv::Size& size, const int level = 3, const int times = 40, const double eps = 0.001);

};//! OpticalFlow

}//! vk

#endif