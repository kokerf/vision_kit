#include <iostream>
#include <stdint.h>
#include <vector>
#include <assert.h>

#include "optical_flow.hpp"
#include "base.hpp"

namespace vk{

void OpticalFlow::computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& points_prev, std::vector<cv::Point2f>& points_next,
    std::vector<float>& errors, const cv::Size& size, const int level, const int times, const float eps)
{
    assert(img_prev.type() == CV_8UC1);
    assert(img_next.type() == CV_8UC1);
    assert(size.height>0 && size.width>0);
    assert(level>=0 && times>0 && eps>0);

    int n_levels = level + 1;//! levels contian 0 - level
    double EPS_S2 = eps * eps;

    //! compute Pyramid images
    std::vector<cv::Mat> pyramid_prev, pyramid_next;
    computePyramid(img_prev, pyramid_prev, 2, level);
    computePyramid(img_next, pyramid_next, 2, level);

    cv::Mat Sharr_dx = (cv::Mat_<float>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
    cv::Mat Sharr_dy = (cv::Mat_<float>(3, 3) << -3,-10, -3, 0, 0, 0, 3, 10, 3);
    std::vector<cv::Mat> pyramid_gradient_x(n_levels), pyramid_gradient_y(n_levels);
    for(int i = 0; i < n_levels; ++i)
    {
        conv_32f(pyramid_next[i], pyramid_gradient_x[i], Sharr_dx, 32);
        conv_32f(pyramid_next[i], pyramid_gradient_y[i], Sharr_dy, 32);
    }

    //! each points in img_prev to find a corresponding location in img_next
    const int wx = size.width;
    const int wy = size.height;
    points_next.resize(points_prev.size());
    errors.resize(points_prev.size(), -1);
    for(std::vector<cv::Point2f>::iterator ipt = points_prev.begin()+1; ipt != points_prev.end(); ++ipt)
    {
        //! point location in 0-level
        const float x = ipt->x;
        const float y = ipt->y;

        float g[2] = {0}; //! guess relative location of point in next image
        float error = 0;
        for(int l = level; l >= 0; l--)
        {
            //! point location in l-lewel
            float p[2] = {x/(1 << level), y/(1 << level)};
            int ip[2] = {floor(p[0]), floor(p[1])};
            const float wx_start = p[0] - floor(wx / 2);
            const float wy_start = p[1] - floor(wy / 2);
            const float wx_end = wx_start + wx;
            const float wy_end = wy_start + wy;

            // float a = p[0] - ip[0];
            // float b = p[1] - ip[1];
            //! cite from OpenCV
            // const int16_t W_BITS = 14;//! for a and b is smaller than 1, int is 16bit
            // //const float FLT_SCALE = 1.f/(1 << 20);
            // int16_t w00 = roundl((1.f - a)*(1.f - b)*(1 << W_BITS));
            // int16_t w01 = roundl(a*(1.f - b)*(1 << W_BITS));
            // int16_t w10 = roundl((1.f - a)*b*(1 << W_BITS));
            // int16_t w11 = (1 << W_BITS) -w00 - w01 - w10;

            const cv::Mat &pyr_prev = pyramid_prev[l];
            const cv::Mat &pyr_next = pyramid_next[l];
            cv::Mat pyr_grad_x = pyramid_gradient_x[l];
            cv::Mat pyr_grad_y = pyramid_gradient_y[l];

            cv::Mat prev_w = cv::Mat::zeros(wy, wx, CV_32FC1);
            cv::Mat grad_wx = cv::Mat::zeros(wy, wx, CV_32FC1);
            cv::Mat grad_wy = cv::Mat::zeros(wy, wx, CV_32FC1);

            const int pyr_cols = pyr_prev.cols;
            const int pyr_rows = pyr_prev.rows;
            if(wx_start < 0 ||  wx_end > pyr_cols || wy_start < 0 || wy_end > pyr_rows)
                break;

            //! get spatial gradient matrix
            float* pIpw = prev_w.ptr<float>(0);
            float* pGwx = grad_wx.ptr<float>(0);
            float* pGwy = grad_wy.ptr<float>(0);
            double G00 = 0, G01 = 0, G11 = 0;
            for(int iwy = 0; iwy < wy; ++iwy)
            {
                //const float* pI  = &pyr_prev.ptr<float>(iwy + wy_start)[wx_start];
                //float* pGx = &pyr_grad_x.ptr<float>(iwy + wy_start)[wx_start];
                //float* pGy = &pyr_grad_y.ptr<float>(iwy + wy_start)[wx_start];

                for(int iwx = 0; iwx < wx; ++iwx, pGwx++, pGwy++, pIpw++)
                {
                    //float im = (w00*pI[iwx]+w01*pI[iwx+1]+w10*pI[iwx+pyr_cols]+w11*pI[iwx+1+pyr_cols])*1.0/(1<<W_BITS);
                    float im = interpolateMat_8u(pyr_prev, wx_start+iwx, wy_start+iwy);
                    float dx = interpolateMat_32f(pyr_grad_x, wx_start+iwx, wy_start+iwy);
                    float dy = interpolateMat_32f(pyr_grad_y, wx_start+iwx, wy_start+iwy);
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
            if(abs(det) < 1e-20)
            {
                std::cerr << " The  gradient matrix is irreversible !!!" << std::endl;
                break;
            }
            // double iG00 = 0, iG01 = 0, iG11 = 0;
            // iG00 = G11/det;
            // iG01 = -G01/det;
            // iG11 = G00/det;

            //! iteration
            float v[2] = {0, 0};
            float q[2] = {0, 0};
            float delta[2] = { 0, 0 };
            q[0] = p[0] + g[0];
            q[1] = p[1] + g[1];
            for(int t = 0; t < times; ++t)
            {

                if(q[0] < wx/2 || q[0] > pyr_cols-wx/2 ||
                   q[1] < wy/2 || q[1] > pyr_rows-wy/2)
                    break;

                cv::Mat next_win = pyr_next(cv::Rect(q[0] - wx / 2, q[1] - wx / 2, wx, wy));
                cv::Mat p_win;
                prev_w.convertTo(p_win, CV_8UC1);

                //! get mismatch vector
                pIpw = prev_w.ptr<float>(0);
                pGwx = grad_wx.ptr<float>(0);
                pGwy = grad_wy.ptr<float>(0);
                float mismatch[2] = {0};
                float qx = q[0] - wx/2;
                float qy = q[1] - wy/2;
                cv::Mat Jres = cv::Mat::zeros(2, 1, CV_32FC1);
                for(int iwy = 0; iwy < wy; ++iwy)
                {
                    for(int iwx = 0; iwx < wx; ++iwx, pGwx++, pGwy++, pIpw++)
                    {
                        float Iq = interpolateMat_8u(pyr_next, qx+iwx, qy+iwy);
                        float diff = (*pIpw) - Iq;
                        float dx = (*pGwx);
                        float dy = (*pGwy);
                        mismatch[0] += diff * dx;
                        mismatch[1] += diff * dy;

                        error += diff*diff;
                    }
                }
                error /= wx*wy;
                // delta[0] = iG00 * mismatch[0] + iG01 * mismatch[1];
                // delta[1] = iG01 * mismatch[0] + iG11 * mismatch[1];
                delta[0] = (G11 * mismatch[0] - G01 * mismatch[1])/det;
                delta[1] = (-G01 * mismatch[0] + G00 * mismatch[1])/det;

                //! iteration termination
                if(delta[0]*delta[0] + delta[1]*delta[1] < EPS_S2)
                {
                    break;
                }

                if(abs(delta[0]) < 0.001 && abs(delta[1]) < 0.001)
                {
                    break;
                }

                v[0] += delta[0];
                v[1] += delta[1];
                q[0] = p[0] + v[0];
                q[1] = p[1] + v[1];
            }//! iteration

            //! update guess location
            g[0] += v[0];
            g[1] += v[1];
            g[0] *= 2;
            g[1] *= 2;

        }//! levels

        //! get location in next image
        //points_next[ipt - points_prev.begin()].x = x + g[0];
        //points_next[ipt - points_prev.begin()].y = y + g[1];
        //errors[ipt - points_prev.begin()] = error;

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