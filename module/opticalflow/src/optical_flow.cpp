#include <iostream>
#include <stdint.h>
#include <vector>
#include <assert.h>

#include "optical_flow.hpp"
#include "base.hpp"

namespace vk{

OpticalFlow::OpticalFlow(const cv::Size& win_size, const int level, const int times, const float eps):
    win_size_(win_size), max_level_(level), max_iters_(times), criteria_(eps)
{
    assert(win_size_.height > 0 && win_size_.width > 0);
    assert(max_level_ >= 0);
    assert(max_iters_ > 0);
    assert(criteria_ > 0);
    win_eara_ = win_size_.height * win_size_.width;
    EPS_S2_ = criteria_*criteria_;
}

OpticalFlow::~OpticalFlow()
{
    pyr_prev_.clear();
    pyr_next_.clear();
    pyr_grad_x_.clear();
    pyr_grad_y_.clear();
}

void OpticalFlow::createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next)
{
    assert(img_prev.type() == CV_8UC1);
    assert(img_next.type() == CV_8UC1);

    const int n_levels = max_level_ + 1;//! levels contian 0 - level
    //! compute Pyramid images
    computePyramid(img_prev, pyr_prev_, 2, max_level_);
    computePyramid(img_next, pyr_next_, 2, max_level_);

    cv::Mat Sharr_dx = (cv::Mat_<float>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
    cv::Mat Sharr_dy = (cv::Mat_<float>(3, 3) << -3,-10, -3, 0, 0, 0, 3, 10, 3);
    pyr_grad_x_.resize(n_levels), pyr_grad_y_.resize(n_levels);
    for(int i = 0; i < n_levels; ++i)
    {
        conv_32f(pyr_next_[i], pyr_grad_x_[i], Sharr_dx, 32);
        conv_32f(pyr_next_[i], pyr_grad_y_[i], Sharr_dy, 32);
    }
}

void OpticalFlow::trackPoint(const cv::Point2f& pt_prev, cv::Point2f& pt_next, const int max_level, float& error, bool& status)
{
    const int half_win_height = win_size_.height/2;
    const int half_win_width = win_size_.width/2;

    cv::Point2f q = pt_prev / (1 << max_level + 1);
    for(int l = max_level; l >= 0; l--)
    {
        status = true;
        q *= 2;
        //! point location in l-lewel
        cv::Point2f p = pt_prev / (1 << l);
        cv::Point2f win_start(p.x - half_win_width, p.y - half_win_height);
        cv::Point2f win_end(win_start.x + win_size_.width, win_start.y + win_size_.height);


        // float a = p[0] - ip[0];
        // float b = p[1] - ip[1];
        //! cite from OpenCV
        // const int16_t W_BITS = 14;//! for a and b is smaller than 1, int is 16bit
        // //const float FLT_SCALE = 1.f/(1 << 20);
        // int16_t w00 = roundl((1.f - a)*(1.f - b)*(1 << W_BITS));
        // int16_t w01 = roundl(a*(1.f - b)*(1 << W_BITS));
        // int16_t w10 = roundl((1.f - a)*b*(1 << W_BITS));
        // int16_t w11 = (1 << W_BITS) -w00 - w01 - w10;

        const cv::Mat &pyr_prev = pyr_prev_[l];
        const cv::Mat &pyr_next = pyr_next_[l];
        cv::Mat grad_x = pyr_grad_x_[l];
        cv::Mat grad_y = pyr_grad_y_[l];

        cv::Mat prev_w = cv::Mat::zeros(win_size_, CV_32FC1);
        cv::Mat grad_wx = cv::Mat::zeros(win_size_, CV_32FC1);
        cv::Mat grad_wy = cv::Mat::zeros(win_size_, CV_32FC1);

        const int pyr_cols = pyr_prev.cols;
        const int pyr_rows = pyr_prev.rows;
        if(win_start.x < 0 || win_start.y < 0 || win_end.x > pyr_cols || win_end.y > pyr_rows)
        {
            status = false;
            continue;
        }

        //! get spatial gradient matrix
        float* pIpw = prev_w.ptr<float>(0);
        float* pGwx = grad_wx.ptr<float>(0);
        float* pGwy = grad_wy.ptr<float>(0);
        double G00 = 0, G01 = 0, G11 = 0;
        for(int iwy = 0; iwy < win_size_.height; ++iwy)
        {
            //const float* pI  = &pyr_prev.ptr<float>(iwy + wy_start)[wx_start];
            //float* pGx = &pyr_grad_x.ptr<float>(iwy + wy_start)[wx_start];
            //float* pGy = &pyr_grad_y.ptr<float>(iwy + wy_start)[wx_start];

            for(int iwx = 0; iwx < win_size_.width; ++iwx, pGwx++, pGwy++, pIpw++)
            {
                //float im = (w00*pI[iwx]+w01*pI[iwx+1]+w10*pI[iwx+pyr_cols]+w11*pI[iwx+1+pyr_cols])*1.0/(1<<W_BITS);
                float im = interpolateMat_8u(pyr_prev, win_start.x+iwx, win_start.y+iwy);
                float dx = interpolateMat_32f(grad_x, win_start.x +iwx, win_start.y +iwy);
                float dy = interpolateMat_32f(grad_y, win_start.x +iwx, win_start.y +iwy);
                // float dx = VK_DESCALE(w00*pGx[0] + w01*pGx[1]
                //     + w10*pGx[pyr_cols] + w11*pGx[pyr_cols+1], W_BITS);
                // float dy = VK_DESCALE(w00*pGy[0] + w01*pGy[1]
                //     + w10*pGy[pyr_cols] + w11*pGy[pyr_cols+1], W_BITS);

                (*pIpw) = im;
                (*pGwx) = dx;
                (*pGwy) = dy;

                G00 += dx*dx;
                G01 += dx*dy;
                G11 += dy*dy;
            }
        }

        double det = G00*G11 - G01*G01;
        if(abs(det) < VK_EPS)
        {
            status = false;
            std::cerr << " The gradient matrix is irreversible !!!" << std::endl;
            break;
        }

        //! iteration
        cv::Point2f delta;
        cv::Point2f v(0, 0);
        int it = 0;
        cv::Mat diff_img = cv::Mat::zeros(win_size_, CV_32FC1);
        while(it++ < max_iters_)
        {
            if(q.x < half_win_width || q.x > pyr_cols - half_win_width ||
                q.y < half_win_height || q.y > pyr_rows - half_win_height)
            {
                if(l == 0) {status = false;}
                break;
            }

            cv::Mat next_win = pyr_next(cv::Rect(q.x - half_win_width, q.y - half_win_height, win_size_.width, win_size_.height));
            cv::Mat p_win;
            prev_w.convertTo(p_win, CV_8UC1);

            //! get mismatch vector
            pIpw = prev_w.ptr<float>(0);
            pGwx = grad_wx.ptr<float>(0);
            pGwy = grad_wy.ptr<float>(0);

            cv::Point2f b(0,0);
            win_start = cv::Point2f(q.x - half_win_width, q.y - half_win_height);
            win_end = cv::Point2f(win_start.x + win_size_.width, win_start.y + win_size_.height);
            error = 0;
            for(int iwy = 0; iwy < win_size_.height; ++iwy)
            {
                for(int iwx = 0; iwx < win_size_.width; ++iwx, pGwx++, pGwy++, pIpw++)
                {
                    float Ip = interpolateMat_8u(pyr_next, win_start.x+iwx, win_start.y+iwy);
                    float diff = (*pIpw) - Ip;
                    float dx = (*pGwx);
                    float dy = (*pGwy);

                    diff_img.at<float>(iwy, iwx) = diff;

                    b.x += diff * dx;
                    b.y += diff * dy;

                    error += diff*diff;
                }
            }
            error /= win_eara_;

            delta.x = (G11 * b.x - G01 * b.y)/det;
            delta.y = (-G01 * b.x + G00 * b.y)/det;

            //! iteration termination
            if(delta.x*delta.x + delta.y*delta.y < EPS_S2_)
            {
                break;
            }

            if(abs(delta.x) < 0.001 && abs(delta.y) < 0.001)
            {
                break;
            }

            q += delta;
        }//! iteration
        //! update
        pt_next = p;
    }//! levels
}

void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& points_prev, std::vector<cv::Point2f>& points_next,
    std::vector<float>& errors, const cv::Size& win_size, const int level, const int times, const float eps)
{
    OpticalFlow optical_flow(win_size, level, times, eps);

    optical_flow.createPyramid(img_prev, img_next);

    //! each points in img_prev to find a corresponding location in img_next
    points_next.resize(points_prev.size());
    errors.resize(points_prev.size(), -1);
    for(std::vector<cv::Point2f>::iterator ipt = points_prev.begin(); ipt != points_prev.end(); ++ipt)
    {
        cv::Point2f pt_next;
        float error = 0;
        bool status = false;
        optical_flow.trackPoint(*ipt, pt_next, level, error, status);

        if(status)
        {
            points_next[ipt - points_prev.begin()] = pt_next;
            errors[ipt - points_prev.begin()] = error;
        }
        else
        {
            points_next[ipt - points_prev.begin()] = cv::Point2f();
            errors[ipt - points_prev.begin()] = -1;
        }

    }//! iterator of points

}

bool align2D(const cv::Mat& T, const cv::Mat& I, const cv::Mat& GTx, const cv::Mat& GTy,
    const cv::Size size, const cv::Point2f& p, cv::Point2f& q)
{
    const float EPS = 1E-5f; // Threshold value for termination criteria.
    const int MAX_ITER = 100;  // Maximum iteration count.

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
            break;


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

}//! vk