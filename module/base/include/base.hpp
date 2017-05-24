#ifndef _BASE_HPP_
#define _BASE_HPP_

#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdint.h>

#include <opencv2/core/core.hpp>

#include "math.hpp"

namespace vk{


/**
 * [ComputePyramid computing pyramid images]
 * @param  image         [input image]
 * @param  image_pyramid [output pyramid images]
 * @param  scale_factor  [pyramid decimation ratio, greater than 1]
 * @param  level         [0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used]
 * @param  min_size      [the minimum size of the image in top level of pyramid]
 * @return               [return the max level]
 */
int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid,
    const float scale_factor = 1.2f, const uint16_t level = 3, const cv::Size min_size = cv::Size(40, 40));

/**
 * [conv_32f convolute images with 3x3 kernel]
 * @param src    [input image, type should be CV_8UC1]
 * @param dest   [output image, type is CV_32FC1]
 * @param kernel [3x3 kernel, type is CV_32FC1]
 */
void conv_32f(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div);

/**
 * [conv_16S description]
 * @param src    [input image, type should be CV_8UC1]
 * @param dest   [output image, type is CV_16SC1]
 * @param kernel [3x3 kernel, type is CV_16SC1]
 * @param div    [divide after convolution]
 */
void conv_16s(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div);

/**
 * [makeBorders make border for image by copying margin]
 * @param src      [input image]
 * @param dest     [output image]
 * @param col_side [added cols for each side]
 * @param row_side [added rows for eache side]
 */
void makeBorders(const cv::Mat& src, cv::Mat& dest, const int col_side = 1, const int row_side = 1);

/**
 * [align2D to align a pitch to another image]
 * @param  T        [templet image]
 * @param  I        [destination image]
 * @param  GTx      [gradient of templet in x]
 * @param  GTy      [gradient of templet in x]
 * @param  size     [pitch size]
 * @param  p        [centre of the pitch in templet]
 * @param  q        [centre of the pitch in destination]
 * @param  EPS      [Threshold value for termination criteria]
 * @param  MAX_ITER [Maximum iteration count]
 * @return          [return ture if found]
 */
bool align2D(const cv::Mat& T, const cv::Mat& I, const cv::Mat& GTx, const cv::Mat& GTy,
    const cv::Size size, const cv::Point2f& p, cv::Point2f& q, const float EPS = 1E-5f, const int MAX_ITER = 100);

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
/**
 * [interpolateMat_32f bilinear interpolation of float image]
 * @param  mat [input image for interpolation, type of CV_32FC1]
 * @param  u   [pixel location in cols]
 * @param  v   [pixel location in rows]
 * @return     [the value of bilinear interpolation]
 */
inline float interpolateMat_32f(const cv::Mat& mat, float u, float v);

/**
 * [interpolateMat_8u bilinear interpolation of uchar image]
 * @param  mat [input image for interpolation, type of CV_8UC1]
 * @param  u   [pixel location in cols]
 * @param  v   [pixel location in rows]
 * @return     [the value of bilinear interpolation]
 */
inline float interpolateMat_8u(const cv::Mat& mat, float u, float v);

/**
 * [Normalize normalize points by isotropic scaling]
 * @param points      [points input for mormalizing]
 * @param points_norm [normalized points]
 * @param T           [Transform matrix of normalizing]
 */
void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T);

}//! vk

#endif