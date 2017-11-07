#include <vector>
#include <stdint.h>
#include <assert.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "base.hpp"

namespace vk {

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size)
{
    assert(scale_factor > 1.0);
    assert(!image.empty());

    image_pyramid.resize(level + 1);

    image_pyramid[0] = image.clone();
    for(int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i - 1].cols / scale_factor), round(image_pyramid[i - 1].rows / scale_factor));

        if(size.height < min_size.height || size.width < min_size.width)
        {
            return level-1;
        }

        cv::resize(image_pyramid[i - 1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return level;
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

void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T)
{
    const int N = points.size();
    if(N == 0)
        return;

    points_norm.resize(N);

    cv::Point2f mean(0,0);
    for(int i = 0; i < N; ++i)
    {
        mean += points[i];
    }
    mean = mean/N;

    cv::Point2f mean_dev(0,0);

    for(int i = 0; i < N; ++i)
    {
        points_norm[i] = points[i] - mean;

        mean_dev.x += fabs(points_norm[i].x);
        mean_dev.y += fabs(points_norm[i].y);
    }
    mean_dev /= N;

    const float scale_x = 1.0/mean_dev.x;
    const float scale_y = 1.0/mean_dev.y;

    for(int i=0; i<N; i++)
    {
        points_norm[i].x *= scale_x;
        points_norm[i].y *= scale_y;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = scale_x;
    T.at<float>(1,1) = scale_y;
    T.at<float>(0,2) = -mean.x*scale_x;
    T.at<float>(1,2) = -mean.y*scale_y;
}

}//! vk