#include <vector>
#include <stdint.h>
#include <assert.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "base.hpp"

namespace vk {

bool computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level)
{
    assert(scale_factor > 1.0);
    assert(!image.empty());

    image_pyramid.resize(level + 1);

    image_pyramid[0] = image.clone();
    for (int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i - 1].cols / scale_factor), round(image_pyramid[i - 1].rows / scale_factor));

        cv::resize(image_pyramid[i - 1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return true;
}

void conv_32f(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div)
{
    assert(!src.empty());
    assert(src.type() == CV_8UC1);
    assert(kernel.type() == CV_32FC1);
    assert(kernel.cols == 3 && kernel.rows == 3);

    //! copy borders for image
    cv::Mat img_extend;
    makeBorders(src, img_extend, 1, 1);

    //! calculate the dest image
    float *kernel_data = (float*) kernel.data;

    dest = cv::Mat::zeros(src.size(), CV_32FC1);
    int u,v;
    const uint16_t src_cols = src.cols;
    const uint16_t src_rows = src.rows;
    for(int ir = 0; ir < src_rows; ++ir)
    {
        v = ir + 1;
        float* dest_ptr = dest.ptr<float>(ir);
        uint8_t* extd_ptr = img_extend.ptr<uint8_t>(v);
        for(int ic = 0; ic < src_cols; ++ic)
        {
            u = ic + 1;

            dest_ptr[ic] = kernel_data[0] * extd_ptr[u - 1 - src_cols]
                        + kernel_data[1] * extd_ptr[u - src_cols]
                        + kernel_data[2] * extd_ptr[u + 1 - src_cols]
                        + kernel_data[3] * extd_ptr[u - 1]
                        + kernel_data[4] * extd_ptr[u]
                        + kernel_data[5] * extd_ptr[u + 1]
                        + kernel_data[6] * extd_ptr[u - 1 + src_cols]
                        + kernel_data[7] * extd_ptr[u + src_cols]
                        + kernel_data[8] * extd_ptr[u + 1 + src_cols];

            dest_ptr[ic] /= div;
        }
    }
}

void conv_16s(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div)
{
    assert(!src.empty());
    assert(src.type() == CV_8UC1);
    assert(kernel.type() == CV_16SC1);
    assert(kernel.cols == 3 && kernel.rows == 3);

    //! copy borders for image
    cv::Mat img_extend;
    makeBorders(src, img_extend, 1, 1);

    //! calculate the dest image
    int16_t *kernel_data = (int16_t*) kernel.data;

    dest = cv::Mat::zeros(src.size(), CV_16SC1);
    int u,v;
    const uint16_t src_cols = src.cols;
    const uint16_t src_rows = src.rows;
    for(int ir = 0; ir < src_rows; ++ir)
    {
        v = ir + 1;
        int16_t* dest_ptr = dest.ptr<int16_t>(ir);
        uint8_t* extd_ptr = img_extend.ptr<uint8_t>(v);
        for(int ic = 0; ic < src_cols; ++ic)
        {
            u = ic + 1;

            dest_ptr[ic] = kernel_data[0] * extd_ptr[u - 1 - src_cols]
                        + kernel_data[1] * extd_ptr[u - src_cols]
                        + kernel_data[2] * extd_ptr[u + 1 - src_cols]
                        + kernel_data[3] * extd_ptr[u - 1]
                        + kernel_data[4] * extd_ptr[u]
                        + kernel_data[5] * extd_ptr[u + 1]
                        + kernel_data[6] * extd_ptr[u - 1 + src_cols]
                        + kernel_data[7] * extd_ptr[u + src_cols]
                        + kernel_data[8] * extd_ptr[u + 1 + src_cols];

            dest_ptr[ic] /= div;
        }
    }
}

void makeBorders(const cv::Mat& src, cv::Mat& dest, const int col_side, const int row_side)
{
    assert(!src.empty());
    assert(col_side > 0 && row_side > 0);

    const uint16_t src_cols = src.cols;
    const uint16_t src_rows = src.rows;

    cv::Mat border = cv::Mat::zeros(cv::Size(src_cols + row_side *2, src_rows + col_side*2), CV_8UC1);
    src.copyTo(border.rowRange(col_side, col_side + src_rows).colRange(row_side, row_side + src_cols));
    for(int ir = 0; ir < col_side; ++ir)
    {
        src.row(0).copyTo(border.row(ir).colRange(row_side, row_side + src_cols));
        int ir_inv = border.rows - ir - 1;
        src.row(src_rows - 1).copyTo(border.row(ir_inv).colRange(row_side, row_side + src_cols));
    }

    for(int ic = 0; ic < row_side; ++ic)
    {
        border.col(row_side).copyTo(border.col(ic));
        int ic_inv = border.cols - ic - 1;
        border.col(row_side + src_cols - 1).copyTo(border.col(ic_inv));
    }

    dest = border.clone();
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
float interpolateMat_32f(const cv::Mat& mat, float u, float v)
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

int16_t interpolateMat_16s(const cv::Mat& mat, float u, float v)
{
    assert(mat.type() == CV_16SC1);
    int16_t x = floor(u);
    int16_t y = floor(v);
    float a = u - x;
    float b = v - y;
    const int16_t W_BITS = 14;//! for a and b is smaller than 1, int is 16bit
    int16_t w00 = roundl((1.f - a)*(1.f - b)*(1 << W_BITS));
    int16_t w01 = roundl(a*(1.f - b)*(1 << W_BITS));
    int16_t w10 = roundl((1.f - a)*b*(1 << W_BITS));
    int16_t w11 = (1 << W_BITS) -w00 - w01 - w10;

    int16_t val00 = mat.at<int16_t>(y, x);
    int16_t val10 = mat.at<int16_t>(y, x + 1);
    int16_t val01 = mat.at<int16_t>(y + 1, x);
    int16_t val11 = mat.at<int16_t>(y + 1, x + 1);

    return VK_DESCALE(w00*val00 + w01*val10 + w10*val01 + w11*val11, W_BITS);
}

float interpolateMat_8u(const cv::Mat& mat, float u, float v)
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
    return w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride + 1] + 0.5;//! add 0.5 to round off!
}
// float interpolateMat_8u(const cv::Mat& mat, float u, float v)
// {
//     assert(mat.type() == CV_8UC1);
//     int16_t x = floor(u);
//     int16_t y = floor(v);
//     float a = u - x;
//     float b = v - y;
//     const int16_t W_BITS = 14;//! for a and b is smaller than 1, int is 16bit
//     int16_t w00 = roundl((1.f - a)*(1.f - b)*(1 << W_BITS));
//     int16_t w01 = roundl(a*(1.f - b)*(1 << W_BITS));
//     int16_t w10 = roundl((1.f - a)*b*(1 << W_BITS));
//     int16_t w11 = (1 << W_BITS) -w00 - w01 - w10;

//     uint8_t val00 = mat.at<uint8_t>(y, x);
//     uint8_t val10 = mat.at<uint8_t>(y, x + 1);
//     uint8_t val01 = mat.at<uint8_t>(y + 1, x);
//     uint8_t val11 = mat.at<uint8_t>(y + 1, x + 1);

//     return (w00*val00 + w01*val10 + w10*val01 + w11*val11)* 1.0/(1 << 14);
// }

}//! vk