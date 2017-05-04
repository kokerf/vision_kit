#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <assert.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "base.hpp"

namespace vk{

bool computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level)
{
    assert(scale_factor > 1.0);
    assert(!image.empty());

    image_pyramid.resize(level+1);

    image_pyramid[0] = image.clone();
    for(int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i-1].cols/scale_factor), round(image_pyramid[i-1].rows/scale_factor));

        cv::resize(image_pyramid[i-1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return true;
}

}//! vk