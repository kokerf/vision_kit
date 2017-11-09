#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace vk
{

/**
 * 
 */
void getCorrespondPoints(const cv::Mat &src, const cv::Mat &dst, std::vector<cv::Point2f> &pt_src, std::vector<cv::Point2f> &pt_dst, const size_t num, const float px_err = 0.5);

void drowMatchPoits(const cv::Mat &src, const cv::Mat &dst, std::vector<cv::Point2f> &pt_src, std::vector<cv::Point2f> &pt_dst, cv::Mat &match);

}

#endif