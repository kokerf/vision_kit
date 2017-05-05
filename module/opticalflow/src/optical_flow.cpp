#include <stdint.h>
#include <vector>
#include <assert.h>

#include "optical_flow.hpp"
#include "base.hpp"

namespace vk{

void OpticalFlow::computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, cv::Point2d& points_prev, cv::Point2d& points_next,
    std::vector<float> errors, const cv::Size& size, const int level = 3, const int times = 40, const double eps = 0.001)
{
    assert(size.height>0 && size.width>0);
    assert(level>=0 && times>0 && eps>0);

    //! compute Pyramid images
    std::vector<cv::Mat> pyramid_prev, pyramid_next;
    computePyramid(img_prev, pyramid_prev, 2, level);
    computePyramid(img_next, pyramid_next, 2, level);

    uint16_t n = points_prev.size();
}

}//! vk