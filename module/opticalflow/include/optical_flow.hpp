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
    static void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& points_prev, std::vector<cv::Point2f>& points_next,
        std::vector<float>& errors, const cv::Size& size, const int level = 3, const int times = 40, const float eps = 0.001);

};//! OpticalFlow


bool align2D(const cv::Mat& T, const cv::Mat& I, const cv::Mat& GTx, const cv::Mat& GTy,
    const cv::Size size, const cv::Point2f& p, cv::Point2f& q);
}//! vk

#endif