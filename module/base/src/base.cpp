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
    for (int i = 1; i <= level; ++i)
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

bool align2D(const cv::Mat& T, const cv::Mat& I, const cv::Mat& GTx, const cv::Mat& GTy,
    const cv::Size size, const cv::Point2f& p, cv::Point2f& q, const float EPS, const int MAX_ITER)
{
    const int cols = size.width;
    const int rows = size.height;

    cv::Point2f start, end;
    start.x = p.x - floor(cols/2);
    start.y = p.y - floor(rows/2);
    end.x = start.x + cols;
    end.y = start.y + cols;

    if(start.x < 0 || start.x > T.cols || end.y < 0 || end.y > T.rows)
        return false;

    q = p;// -cv::Point2f(4, 4);

    cv::Mat dxy;
    cv::Mat warpT = cv::Mat::zeros(cols, rows, CV_32FC1);
    cv::Mat warpGx = cv::Mat::zeros(cols, rows, CV_32FC1);
    cv::Mat warpGy = cv::Mat::zeros(cols, rows, CV_32FC1);
    cv::Mat H = cv::Mat::zeros(2,2,CV_32FC1);
    for(int y = 0; y < rows; ++y)
    {
        float* pw = warpT.ptr<float>(y);
        float* px = warpGx.ptr<float>(y);
        float* py = warpGy.ptr<float>(y);
        for(int x = 0; x < cols; ++x, pw++, px++, py++)
        {
            (*pw) = (float)interpolateMat_8u(T, start.x+x, start.y+y);
            (*px) = interpolateMat_32f(GTx, start.x+x, start.y+y);
            (*py) = interpolateMat_32f(GTy, start.x+x, start.y+y);

            dxy = (cv::Mat_<float>(1, 2) << (*px), (*py));
            H += dxy.t() * dxy;
        }
    }
    cv::Mat invH = H.inv();

    int iter = 0;
    cv::Mat warpI = cv::Mat(rows, cols, CV_32FC1);
    cv::Mat next_win;
    cv::Mat prev_win;
    cv::Mat error;
    while(iter++ < MAX_ITER)
    {
        cv::Mat Jres = cv::Mat::zeros(2, 1, CV_32FC1);
        cv::Mat dq = cv::Mat::zeros(2, 1, CV_32FC1);
        cv::Point2f qstart(q.x-floor(cols/2), q.y-floor(rows/2));
        if(qstart.x < 0 || qstart.y < 0 || qstart.x+ cols > I.cols || qstart.y+rows > I.rows)
            return false;

        float mean_error=0;
        for(int y = 0; y < rows; ++y)
        {
            float* pw = warpT.ptr<float>(y);
            float* px = warpGx.ptr<float>(y);
            float* py = warpGy.ptr<float>(y);
            for(int x = 0; x < cols; ++x, pw++, px++, py++)
            {

                float qw = interpolateMat_8u(I, qstart.x+x, qstart.y+y);
                float diff = *pw - qw;
                mean_error += diff*diff;

                warpI.at<float>(y, x) = qw;
                dxy = (cv::Mat_<float>(1, 2) << (*px), (*py));
                Jres += diff* dxy.t();
            }
        }
        mean_error /= rows*cols;
        dq = invH * Jres;
        q.x += dq.at<float>(0, 0);
        q.y += dq.at<float>(1, 0);

        warpT.convertTo(prev_win, CV_8UC1);
        warpI.convertTo(next_win, CV_8UC1);
        cv::Mat error = cv::Mat(next_win.size(), CV_8SC1);
        error = prev_win - next_win;// next_win - prev_win;
    }

    return true;
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
inline float interpolateMat_32f(const cv::Mat& mat, float u, float v)
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

inline float interpolateMat_8u(const cv::Mat& mat, float u, float v)
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