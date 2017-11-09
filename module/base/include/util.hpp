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

void drawEpipolarLines(const cv::Mat &img_prev, const cv::Mat &img_next, const std::vector<cv::Point2f> &pts_prev, const std::vector<cv::Point2f> &pts_next, const cv::Mat &fundamental, cv::Mat &img_epipolar);

}

#endif