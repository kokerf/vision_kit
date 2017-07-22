#ifndef _OPTICAL_FLOW_HPP_
#define _OPTICAL_FLOW_HPP_

#include <vector>
#include <stdint.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define USE_INT
#define GET_TIME

#ifdef USE_INT
    #define deriv_type int32_t
    #define gray_type int32_t
    #define W_BITS 14
#else
    #define deriv_type float
    #define gray_type float
#endif

namespace vk{

void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_next,
        std::vector<uchar>& statuses, std::vector<float>& errors, const cv::Size& win_size = cv::Size(21,21), const int level = 3, const int times = 40, const float eps = 0.001);

void computePyrLKParallel(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_next,
        std::vector<uchar>& statuses, std::vector<float>& errors, const cv::Size& win_size = cv::Size(21, 21), const int level = 3, const int times = 40, const float eps = 0.001);

class OpticalFlow
{
public:
    OpticalFlow(const cv::Size& win_size, const int level = 3, const int times = 40, const float eps = 0.001);
    ~OpticalFlow();

    int computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, const std::vector<cv::Point2f>& pts_prev,
        std::vector<cv::Point2f>& pts_next, std::vector<float>& errors);

    void createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next);

    void calcGradientPyramid();

    void trackPoint(const cv::Point2f& pt_prev, cv::Point2f& pt_next, float& error, uchar& status);

    static int createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Mat>& pyr_prev, std::vector<cv::Mat>& pyr_next, const int level);

    static void calcGradient(cv::Mat& img, cv::Mat& grad_x, cv::Mat& grad_y);

    static void trackPoint(const std::vector<cv::Mat>& pyr_prev, const std::vector<cv::Mat>& pyr_next, const std::vector<cv::Mat>& pyr_grad_x, const std::vector<cv::Mat>& pyr_grad_y,
        const cv::Point2f& pt_prev, cv::Point2f& pt_next, float& error, uchar& status, const cv::Size& win_size, const double EPS_S2, const int max_iters);

private:

    cv::Size win_size_;
    int max_level_;
    int max_iters_;
    double criteria_;
    int win_eara_;
    double EPS_S2_;

    std::vector<cv::Mat> pyr_prev_, pyr_next_;
    std::vector<cv::Mat> pyr_grad_x_, pyr_grad_y_;
};//! OpticalFlow

class LKTrackerParallel : public cv::ParallelLoopBody
{
public:
    LKTrackerParallel(const std::vector<cv::Mat>& pyr_prev, const std::vector<cv::Mat>& pyr_next,
        const std::vector<cv::Mat>& pyr_grad_x, const std::vector<cv::Mat>& pyr_grad_y,
        const std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_next, std::vector<float>& error, std::vector<uchar>& status, cv::Size win_size, int max_level, double criteria, int times);

    void operator()(const cv::Range& range) const;

private:

    const std::vector<cv::Mat> *pyr_prev_iter_, *pyr_next_iter_;
    const std::vector<cv::Mat> *pyr_grad_x_iter_, *pyr_grad_y_iter_;

    const std::vector<cv::Point2f> *pts_prev_iter_;
    std::vector<cv::Point2f> *pts_next_iter_;
    std::vector<uchar> *status_;
    std::vector<float> *error_;

    cv::Size win_size_;
    int max_level_;
    int iter_times_;
    double EPS_S2_;
};

}//! vk

#endif