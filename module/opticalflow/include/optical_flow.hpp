#ifndef _OPTICAL_FLOW_HPP_
#define _OPTICAL_FLOW_HPP_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace vk{

void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_next,
        std::vector<float>& errors, const cv::Size& win_size, const int level = 3, const int times = 40, const float eps = 0.001);

class OpticalFlow
{
public:
    OpticalFlow(const cv::Size& win_size, const int level = 3, const int times = 40, const float eps = 0.001);
    ~OpticalFlow();

    void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, const std::vector<cv::Point2f>& pts_prev,
        std::vector<cv::Point2f>& pts_next, std::vector<float>& errors);

    void createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next);

    void trackPoint(const cv::Point2f& pt_prev, cv::Point2f& pt_next, const int max_level, float& error, bool& status);

    int getMaxLevel() {
        return max_level_;
    }

private:
    cv::Size win_size_;
    int max_level_;
    double criteria_;
    int max_iters_;
    int win_eara_;
    double EPS_S2_;

    std::vector<cv::Mat> pyr_prev_, pyr_next_;
    std::vector<cv::Mat> pyr_grad_x_, pyr_grad_y_;
};//! OpticalFlow


bool align2D(const cv::Mat& T, const cv::Mat& I, const cv::Mat& GTx, const cv::Mat& GTy,
    const cv::Size size, const cv::Point2f& p, cv::Point2f& q);
}//! vk

#endif