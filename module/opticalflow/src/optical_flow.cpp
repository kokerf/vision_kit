#include <iostream>
#include <stdint.h>
#include <vector>
#include <assert.h>
#include <time.h>

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

void OpticalFlow::calcGradient()
{
    const int n_levels = pyr_next_.size();//! levels contian 0 - level

    //! use sharr
    deriv_type Sharr_dx[9] = { -3, 0, 3, -10, 0, 10, -3, 0, 3};
    deriv_type Sharr_dy[9] = { -3, -10, -3, 0, 0, 0, 3, 10, 3};

    pyr_grad_x_.resize(n_levels), pyr_grad_y_.resize(n_levels);
    for(int i = 0; i < n_levels; ++i)
    {
        cv::Mat& src = pyr_next_[i];
        cv::Mat& grad_x = pyr_grad_x_[i];
        cv::Mat& grad_y = pyr_grad_y_[i];
        //! copy borders for image
        cv::Mat img_extend;
        makeBorders(src, img_extend, 1, 1);

        grad_x = cv::Mat::zeros(src.size(), cv::DataType<deriv_type>::type);
        grad_y = cv::Mat::zeros(src.size(), cv::DataType<deriv_type>::type);

        int u,v;
        const uint16_t src_cols = src.cols;
        const uint16_t src_rows = src.rows;
        for(int ir = 0; ir < src_rows; ++ir)
        {
            v = ir + 1;
            uint8_t* extd_ptr = img_extend.ptr<uint8_t>(v);
            deriv_type* gx_ptr = grad_x.ptr<deriv_type>(ir);
            deriv_type* gy_ptr = grad_y.ptr<deriv_type>(ir);
            for(int ic = 0; ic < src_cols; ++ic)
            {
                u = ic + 1;

                gx_ptr[ic] = Sharr_dx[0] * extd_ptr[u - 1 - src_cols]
                            + Sharr_dx[1] * extd_ptr[u - src_cols]
                            + Sharr_dx[2] * extd_ptr[u + 1 - src_cols]
                            + Sharr_dx[3] * extd_ptr[u - 1]
                            + Sharr_dx[4] * extd_ptr[u]
                            + Sharr_dx[5] * extd_ptr[u + 1]
                            + Sharr_dx[6] * extd_ptr[u - 1 + src_cols]
                            + Sharr_dx[7] * extd_ptr[u + src_cols]
                            + Sharr_dx[8] * extd_ptr[u + 1 + src_cols];

                gy_ptr[ic] = Sharr_dy[0] * extd_ptr[u - 1 - src_cols]
                            + Sharr_dy[1] * extd_ptr[u - src_cols]
                            + Sharr_dy[2] * extd_ptr[u + 1 - src_cols]
                            + Sharr_dy[3] * extd_ptr[u - 1]
                            + Sharr_dy[4] * extd_ptr[u]
                            + Sharr_dy[5] * extd_ptr[u + 1]
                            + Sharr_dy[6] * extd_ptr[u - 1 + src_cols]
                            + Sharr_dy[7] * extd_ptr[u + src_cols]
                            + Sharr_dy[8] * extd_ptr[u + 1 + src_cols];

                gx_ptr[ic] /= 32;
                gy_ptr[ic] /= 32;
            }
        }
    }
}

void OpticalFlow::createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next)
{
    assert(img_prev.type() == CV_8UC1);
    assert(img_next.type() == CV_8UC1);

    //! compute Pyramid images
    computePyramid(img_prev, pyr_prev_, 2, max_level_);
    computePyramid(img_next, pyr_next_, 2, max_level_);

    //! calculate gradient for each level
    calcGradient();
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


        int x = floor(p.x);
        int y = floor(p.y);
        float subpix_x = p.x - x;
        float subpix_y = p.y - y;
        //! cite from OpenCV
        const int16_t W_BITS = 14;//! for a and b is smaller than 1, int is 16bit
        //const float FLT_SCALE = 1.f/(1 << 20);
        int16_t iw00 = roundl((1.f - subpix_x)*(1.f - subpix_y)*(1 << W_BITS));
        int16_t iw01 = roundl(subpix_x*(1.f - subpix_y)*(1 << W_BITS));
        int16_t iw10 = roundl((1.f - subpix_x)*subpix_y*(1 << W_BITS));
        int16_t iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        float w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
        float w01 = (1.0f - subpix_x)*subpix_y;
        float w10 = subpix_x*(1.0f - subpix_y);
        float w11 = 1.0f - w00 - w01 - w10;

        const cv::Mat &pyr_prev = pyr_prev_[l];
        const cv::Mat &pyr_next = pyr_next_[l];
        const cv::Mat& grad_x = pyr_grad_x_[l];
        const cv::Mat& grad_y = pyr_grad_y_[l];

        const int pyr_cols = pyr_prev.cols;
        const int pyr_rows = pyr_prev.rows;
        if(win_start.x < 0 || win_start.y < 0 || win_end.x > pyr_cols || win_end.y > pyr_rows)
        {
            status = false;
            continue;
        }

        cv::Mat prev_w = cv::Mat::zeros(win_size_, CV_32FC1);
        cv::Mat grad_wx = cv::Mat::zeros(win_size_, cv::DataType<deriv_type>::type);
        cv::Mat grad_wy = cv::Mat::zeros(win_size_, cv::DataType<deriv_type>::type);
        float* pTwi = prev_w.ptr<float>(0);
        deriv_type* pGwx = grad_wx.ptr<deriv_type>(0);
        deriv_type* pGwy = grad_wy.ptr<deriv_type>(0);
        //! get spatial gradient matrix
        double G00 = 0, G01 = 0, G11 = 0;
        int x_start = floor(win_start.x);
        int y_start = floor(win_start.y);
        for(int yi = 0; yi < win_size_.height; ++yi)
        {
            const uint8_t* pTw = &pyr_prev.ptr<uint8_t>(yi+y_start)[x_start];
            const deriv_type* pGx = &grad_x.ptr<deriv_type>(yi+y_start)[x_start];
            const deriv_type* pGy = &grad_y.ptr<deriv_type>(yi+y_start)[x_start];

            for(int xi = 0; xi < win_size_.width; ++xi, pGwx++, pGwy++, pTwi++)
            {
                float Ti = (w00*pTw[xi] + w01*pTw[xi+pyr_cols] + w10*pTw[xi+1] + w11*pTw[xi+pyr_cols+1]);
                deriv_type dx = VK_DESCALE(iw00*pGx[xi] + iw01*pGx[xi+pyr_cols] + iw10*pGx[xi+1] + iw11*pGx[xi+pyr_cols+1], W_BITS);
                deriv_type dy = VK_DESCALE(iw00*pGy[xi] + iw01*pGy[xi+pyr_cols] + iw10*pGy[xi+1] + iw11*pGy[xi+pyr_cols+1], W_BITS);
                //float Ti = interpolateMat_8u(pyr_prev, win_start.x+xi, win_start.y+yi);
                //float dx = interpolateMat_32f(grad_x, win_start.x +xi, win_start.y +yi);
                //float dy = interpolateMat_32f(grad_y, win_start.x +xi, win_start.y +yi);

                //! store in Mats of win_size_
                (*pTwi) = Ti;
                (*pGwx) = dx;
                (*pGwy) = dy;

                //! gradient matrix(Hession)
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
            win_start = cv::Point2f(q.x - half_win_width, q.y - half_win_height);
            win_end = cv::Point2f(win_start.x + win_size_.width, win_start.y + win_size_.height);
            if(win_start.x < 0 || win_start.y < 0 || win_end.x > pyr_cols || win_end.y > pyr_rows)
            {
                if(l == 0) {status = false;}
                break;
            }

            int x = floor(q.x);
            int y = floor(q.y);
            float subpix_x = q.x - x;
            float subpix_y = q.y - y;

            float w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
            float w01 = (1.0f - subpix_x)*subpix_y;
            float w10 = subpix_x*(1.0f - subpix_y);
            float w11 = 1.0f - w00 - w01 - w10;

            cv::Mat next_win = pyr_next(cv::Rect(q.x - half_win_width, q.y - half_win_height, win_size_.width, win_size_.height));
            cv::Mat p_win;
            prev_w.convertTo(p_win, CV_8UC1);

            //! get mismatch vector
            pTwi = prev_w.ptr<float>(0);
            pGwx = grad_wx.ptr<deriv_type>(0);
            pGwy = grad_wy.ptr<deriv_type>(0);

            cv::Point2f b(0,0);
            x_start = floor(win_start.x);
            y_start = floor(win_start.y);
            error = 0;
            for(int yi = 0; yi < win_size_.height; ++yi)
            {
                const uint8_t* pIw = &pyr_next.ptr<uint8_t>(yi+y_start)[x_start];
                for(int xi = 0; xi < win_size_.width; ++xi, pGwx++, pGwy++, pTwi++)
                {
                    float Ii = w00*pIw[xi] + w01*pIw[xi+pyr_cols] + w10*pIw[xi+1] + w11*pIw[xi+pyr_cols+1];
                    //Ii = interpolateMat_8u(pyr_next, win_start.x+xi, win_start.y+yi);
                    float diff = (*pTwi) - Ii;
                    deriv_type dx = (*pGwx);
                    deriv_type dy = (*pGwy);

                    diff_img.at<float>(yi, xi) = diff;

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
    clock_t t0 = clock();
    optical_flow.createPyramid(img_prev, img_next);
    clock_t t1 = clock();
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
    clock_t t2 = clock();
    std::cout << "time: " << (float)(t1-t0)/CLOCKS_PER_SEC
     <<" " << (float)(t2-t1)/CLOCKS_PER_SEC <<std::endl;

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