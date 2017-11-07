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
 * @param row_side [added rows for each side]
 */
void makeBorders(const cv::Mat& src, cv::Mat& dest, const int col_side = 1, const int row_side = 1);

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
/**
 * [interpolateMat_32f bilinear interpolation of float image]
 * @param  mat [input image for interpolation, type of CV_32FC1]
 * @param  u   [pixel location in cols]
 * @param  v   [pixel location in rows]
 * @return     [the value of bilinear interpolation]
 */
inline float interpolateMat_32f(const cv::Mat& mat, const float u, const float v)
{
    assert(mat.type() == CV_32F);
    float x = floor(u);
    float y = floor(v);
    float subpix_x = u - x;
    float subpix_y = v - y;
    float wx0 = 1.0 - subpix_x;
    float wx1 = subpix_x;
    float wy0 = 1.0 - subpix_y;
    float wy1 = subpix_y;

    float val00 = mat.at<float>(y, x);
    float val10 = mat.at<float>(y, x + 1);
    float val01 = mat.at<float>(y + 1, x);
    float val11 = mat.at<float>(y + 1, x + 1);
    return (wx0*wy0)*val00 + (wx1*wy0)*val10 + (wx0*wy1)*val01 + (wx1*wy1)*val11;
}

/**
 * [interpolateMat_8u bilinear interpolation of uchar image]
 * @param  mat [input image for interpolation, type of CV_8UC1]
 * @param  u   [pixel location in cols]
 * @param  v   [pixel location in rows]
 * @return     [the value of bilinear interpolation]
 */
inline float interpolateMat_8u(const cv::Mat& mat, const float u, const float v)
{
    assert(mat.type() == CV_8UC1);
    int x = floor(u);
    int y = floor(v);
    float subpix_x = u - x;
    float subpix_y = v - y;

    float w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
    float w01 = (1.0f - subpix_x)*subpix_y;
    float w10 = subpix_x*(1.0f - subpix_y);
    float w11 = 1.0f - w00 - w01 - w10;

    //! addr(Mij) = M.data + M.step[0]*i + M.step[1]*j
    const int stride = mat.step.p[0];
    unsigned char* ptr = mat.data + y*stride + x;
    return w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride + 1];
}

/**
 * [Normalize normalize points by isotropic scaling]
 * @param points      [points input for normalizing]
 * @param points_norm [normalized points]
 * @param T           [Transform matrix of normalizing]
 */
void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T);

/**
 * [transferError compute transfer error]
 * @param  p1  [point in the first image]
 * @param  p2  [point in the second image]
 * @param  H12 [3*3 arrary, homograph matrix from the first image to the second image]
 * @return     [transfer error]
 */
inline float transferError(const cv::Point2f& p1, const cv::Point2f& p2, const float* H12)
{
    const float u1 = p1.x;
    const float v1 = p1.y;
    const float u2 = p2.x;
    const float v2 = p2.y;

    const float w1in2 = H12[6] * u1 + H12[7] * v1 + H12[8];
    const float u1in2 = (H12[0] * u1 + H12[1] * v1 + H12[2]) / w1in2;
    const float v1in2 = (H12[3] * u1 + H12[4] * v1 + H12[5]) / w1in2;

    return (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);
}

}//! vk

#endif