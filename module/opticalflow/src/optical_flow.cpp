#include <iostream>
#include <vector>
#include <cmath>
#include <stdint.h>
#include <assert.h>
#include <omp.h>
#ifndef _OPENMP
#pragma warning("OpenMP not supported")
#endif

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

#ifdef GET_TIME
    for(int i = 0; i < 5; i++)
    {
        getTimes[i] = 0;
        nTimes[i] = 0;
    }
#endif

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
        cv::Mat& src = pyr_prev_[i];
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
#ifndef USE_INT
                gx_ptr[ic] /= 32.0;
                gy_ptr[ic] /= 32.0;
#endif
            }
        }
    }
}

void OpticalFlow::createPyramid(const cv::Mat& img_prev, const cv::Mat& img_next)
{
#ifdef GET_TIME
    double start_time0 = (double)cv::getTickCount();
#endif

    assert(img_prev.type() == CV_8UC1);
    assert(img_next.type() == CV_8UC1);

    //! compute Pyramid images
    int max_level0 = computePyramid(img_prev, pyr_prev_, 2, max_level_);
    int max_level1 = computePyramid(img_next, pyr_next_, 2, max_level_);

    max_level_ = VK_MIN(max_level0, max_level1);
    pyr_prev_.resize(max_level_+1);
    pyr_next_.resize(max_level_+1);

#ifdef GET_TIME
    double start_time1 = (double)cv::getTickCount();
#endif

    //! calculate gradient for each level
    calcGradient();

#ifdef GET_TIME
    getTimes[0] += (double)(start_time1-start_time0)/cv::getTickFrequency();
    nTimes[0]++;
    getTimes[1] += (double)(cv::getTickCount()-start_time0)/cv::getTickFrequency();
    nTimes[1]++;
#endif
}

void OpticalFlow::trackPoint(const cv::Point2f& pt_prev, cv::Point2f& pt_next, const int max_level, float& error, uchar& status)
{
    const int half_win_height = win_size_.height/2;
    const int half_win_width = win_size_.width/2;

    cv::Point2f q = pt_prev / (1 << (max_level + 1));
    for(int l = max_level; l >= 0; l--)
    {
#ifdef GET_TIME
        double start_time = (double)cv::getTickCount();
#endif
        status = 1;
        q *= 2;

        //! get images in l-level
        const cv::Mat& pyr_prev = pyr_prev_[l];
        const cv::Mat& pyr_next = pyr_next_[l];
        const cv::Mat& grad_x = pyr_grad_x_[l];
        const cv::Mat& grad_y = pyr_grad_y_[l];
        const int pyr_cols = pyr_prev.cols;
        const int pyr_rows = pyr_prev.rows;

        //! point location in l-lewel
        cv::Point2f p = pt_prev / (1 << l);
        cv::Point2f win_start(p.x - half_win_width, p.y - half_win_height);
        cv::Point2f win_end(win_start.x + win_size_.width, win_start.y + win_size_.height);

        //! check boundary of p's patch
        if(win_start.x < 0 || win_start.y < 0 || win_end.x > pyr_cols || win_end.y > pyr_rows)
        {
            if(l == 0)
            {
                status = 0;
            }
            continue;
        }

        //! bilinear interpolation
        int x = floor(p.x);
        int y = floor(p.y);
        float subpix_x = p.x - x;
        float subpix_y = p.y - y;
        float w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
        float w01 = (1.0f - subpix_x)*subpix_y;
        float w10 = subpix_x*(1.0f - subpix_y);
        //float w11 = 1.0f - w00 - w01 - w10;

#ifdef USE_INT
        //! cite from OpenCV
        const float FLT_SCALE = 1.0 * (1 << W_BITS);
        int32_t iw00 = roundl(w00*(1 << W_BITS));
        int32_t iw01 = roundl(w01*(1 << W_BITS));
        int32_t iw10 = roundl(w10*(1 << W_BITS));
        int32_t iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
#endif

        //! create spatial gradient matrix and previous patch of win_size_
        cv::Mat prev_w = cv::Mat::zeros(win_size_, cv::DataType<gray_type>::type);
        cv::Mat grad_xw = cv::Mat::zeros(win_size_, cv::DataType<deriv_type>::type);
        cv::Mat grad_yw = cv::Mat::zeros(win_size_, cv::DataType<deriv_type>::type);
        gray_type* pTwi = prev_w.ptr<gray_type>(0);
        deriv_type* pGxw = grad_xw.ptr<deriv_type>(0);
        deriv_type* pGyw = grad_yw.ptr<deriv_type>(0);

        //! get spatial gradient matrix of win_size_
        float G00 = 0, G01 = 0, G11 = 0;
        int x_start = floor(win_start.x);
        int y_start = floor(win_start.y);
        for(int yi = 0; yi < win_size_.height; ++yi)
        {
            const uint8_t* pT = &pyr_prev.ptr<uint8_t>(yi+y_start)[x_start];
            const deriv_type* pGx = &grad_x.ptr<deriv_type>(yi+y_start)[x_start];
            const deriv_type* pGy = &grad_y.ptr<deriv_type>(yi+y_start)[x_start];

            for(int xi = 0; xi < win_size_.width; ++xi, pGxw++, pGyw++, pTwi++)
            {
#ifdef USE_INT
                gray_type Ti = iw00*pT[xi] + iw01*pT[xi+pyr_cols] + iw10*pT[xi+1] + iw11*pT[xi+pyr_cols+1];
                deriv_type dx = iw00*pGx[xi] + iw01*pGx[xi+pyr_cols] + iw10*pGx[xi+1] + iw11*pGx[xi+pyr_cols+1];
                deriv_type dy = iw00*pGy[xi] + iw01*pGy[xi+pyr_cols] + iw10*pGy[xi+1] + iw11*pGy[xi+pyr_cols+1];
#else
                gray_type Ti = (w00*pT[xi] + w01*pT[xi+pyr_cols] + w10*pT[xi+1] + w11*pT[xi+pyr_cols+1]);
                deriv_type dx = w00*pGx[xi] + w01*pGx[xi+pyr_cols] + w10*pGx[xi+1] + w11*pGx[xi+pyr_cols+1];
                deriv_type dy = w00*pGy[xi] + w01*pGy[xi+pyr_cols] + w10*pGy[xi+1] + w11*pGy[xi+pyr_cols+1];
#endif
                //! store in Mats of win_size_
                (*pTwi) = Ti;
                (*pGxw) = dx;
                (*pGyw) = dy;

                //! gradient matrix(Hession)
                G00 += 1.0*dx*dx;
                G01 += 1.0*dx*dy;
                G11 += 1.0*dy*dy;
            }
        }

        float det = G00*G11 - G01*G01;
        if(fabs(det) < VK_EPS)
        {
            status = 0;
            std::cerr << " The gradient matrix is irreversible !!!" << std::endl;
            break;
        }

#ifdef GET_TIME
        getTimes[2] += ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
        nTimes[2]++;
#endif
        //! iteration
        cv::Point2f delta;
        int it = 0;
        while(it++ < max_iters_)
        {

#ifdef GET_TIME
            start_time = (double)cv::getTickCount();
#endif
            win_start = cv::Point2f(q.x - half_win_width, q.y - half_win_height);
            win_end = cv::Point2f(win_start.x + win_size_.width, win_start.y + win_size_.height);
            //! check boundary of q's patch
            if(win_start.x < 0 || win_start.y < 0 || win_end.x > pyr_cols || win_end.y > pyr_rows)
            {
                if(l == 0)
                {
                    status = 0;
                }
                break;
            }

            //! bilinear interpolation
            x = floor(q.x);
            y = floor(q.y);
            subpix_x = q.x - x;
            subpix_y = q.y - y;

            w00 = (1.0f - subpix_x)*(1.0f - subpix_y);
            w01 = (1.0f - subpix_x)*subpix_y;
            w10 = subpix_x*(1.0f - subpix_y);
            //w11 = 1.0f - w00 - w01 - w10;
#ifdef USE_INT
            iw00 = roundl(w00*(1 << W_BITS));
            iw01 = roundl(w01*(1 << W_BITS));
            iw10 = roundl(w10*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
#endif
            //! get mismatch vector
            pTwi = prev_w.ptr<gray_type>(0);
            pGxw = grad_xw.ptr<deriv_type>(0);
            pGyw = grad_yw.ptr<deriv_type>(0);

            cv::Point2f b(0,0);
            x_start = floor(win_start.x);
            y_start = floor(win_start.y);
            error = 0;
            for(int yi = 0; yi < win_size_.height; ++yi)
            {
                const uint8_t* pIw = &pyr_next.ptr<uint8_t>(yi+y_start)[x_start];
                for(int xi = 0; xi < win_size_.width; ++xi, pGxw++, pGyw++, pTwi++)
                {
#ifdef USE_INT
                    gray_type Ii = iw00*pIw[xi] + iw01*pIw[xi+pyr_cols] + iw10*pIw[xi+1] + iw11*pIw[xi+pyr_cols+1];
#else
                    gray_type Ii = w00*pIw[xi] + w01*pIw[xi+pyr_cols] + w10*pIw[xi+1] + w11*pIw[xi+pyr_cols+1];
#endif
                    float diff = (*pTwi) - Ii;
                    deriv_type dx = (*pGxw);
                    deriv_type dy = (*pGyw);

                    b.x += diff * dx;
                    b.y += diff * dy;

                    error += fabs(diff);
                }
            }

#ifdef USE_INT
            error /= win_eara_ * FLT_SCALE;
            delta.x = 32*(G11 * b.x - G01 * b.y)/det;
            delta.y = 32*(-G01 * b.x + G00 * b.y)/det;
#else
            error /= win_eara_;
            delta.x = (G11 * b.x - G01 * b.y)/det;
            delta.y = (-G01 * b.x + G00 * b.y)/det;
#endif

            //! if not in 0-level, 0.01 is small enough
            if(l>0 && fabs(delta.x) < 0.01 && fabs(delta.y) < 0.01)
            {
                break;
            }
            //! iteration termination
            if(delta.x*delta.x + delta.y*delta.y < EPS_S2_)
            {
                break;
            }

            //! update p
            q += delta;

#ifdef GET_TIME
            getTimes[3] += ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
            nTimes[3]++;
#endif
        }//! end of iteration

        pt_next = q;

    }//! end of levels
}

void computePyrLK(const cv::Mat& img_prev, const cv::Mat& img_next, std::vector<cv::Point2f>& points_prev, std::vector<cv::Point2f>& points_next,
    std::vector<uchar>& statuses, std::vector<float>& errors, const cv::Size& win_size, const int level, const int times, const float eps)
{
#ifdef GET_TIME
    double start_time = (double)cv::getTickCount();
#endif

    const int total_points = points_prev.size();
    if(total_points < 1)
        return;

    OpticalFlow optical_flow(win_size, level, times, eps);

    optical_flow.createPyramid(img_prev, img_next);

    //! each points in img_prev to find a corresponding location in img_next
    points_next.resize(total_points);
    statuses.resize(total_points, false);
    errors.resize(total_points, -1);

#pragma omp parallel for
    for(int i = 0; i < total_points; ++i)
    {
        cv::Point2f pt_next;
        float& error = errors[i];
        uchar status = false;
        optical_flow.trackPoint(points_prev[i], pt_next, level, error, status);

        statuses[i] = status;
        if(status==1)
        {
            points_next[i] = pt_next;
        }
        else
        {
            points_next[i] = cv::Point2f();
        }

    }//! iterator of points

#ifdef GET_TIME
    optical_flow.getTimes[4] += ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
    optical_flow.nTimes[4]++;
    std::cout << "================="
    << "\n Total time: "     << optical_flow.getTimes[4]/optical_flow.nTimes[4]
    << "\n create Pyramid: " << optical_flow.getTimes[0]
                      << " " << optical_flow.getTimes[0]/optical_flow.nTimes[0]
    << "\n calc  gradient: " << optical_flow.getTimes[1]
                      << " " << optical_flow.getTimes[1]/optical_flow.nTimes[1]
    << "\n prev   precess: " << optical_flow.getTimes[2]
                             << " " << optical_flow.getTimes[2]/optical_flow.nTimes[2]
    << "\n      iteration: " << optical_flow.getTimes[3]
                             << " " << optical_flow.getTimes[3]/optical_flow.nTimes[3]
    << "\n iter per point: " << optical_flow.nTimes[3]/total_points << " " << optical_flow.nTimes[3]/total_points/(level+1)
    << "\n=================" << std::endl;
#endif

}

}//! vk