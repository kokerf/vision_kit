#include <vector>
#include <cmath>
#include <assert.h>
#include <opencv2/core/core.hpp>

#include "fundamental.hpp"

namespace vk{

cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type)
{
    Fundamental fundamental(pts_prev, pts_next, type);

    return fundamental.slove();
}

Fundamental::Fundamental(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type):
    run_type(type)
{
    assert(pts_prev.size() == pts_next.size());
    Normalize(pts_prev, pts_prev_norm_, T1_);
    Normalize(pts_next, pts_next_norm_, T2_);
}

Fundamental::~Fundamental()
{
    pts_prev_norm_.clear();
    pts_next_norm_.clear();
    T1_.release(), T2_.release();
}

void Fundamental::Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T)
{
    const int N = points.size();
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

cv::Mat Fundamental::slove()
{

    cv::Mat F_norm;
    switch(run_type)
    {
    //case vk::FM_7POINT: run7points();
    case vk::FM_8POINT: F_norm = run8points(); break;
    default: break;
    }

    cv::Mat F = T2_.t()*F_norm*T1_;
    float F22 = F.at<float>(2, 2);
    if (fabs(F22) > FLT_EPSILON)
        F /= F22;

    return F;
}

cv::Mat Fundamental::run8points()
{
    const int N = pts_prev_norm_.size();
    assert(N >= 8);

    cv::Mat A(N, 9, CV_32F);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev_norm_[i].x;
        const float v1 = pts_prev_norm_[i].y;
        const float u2 = pts_next_norm_[i].x;
        const float v2 = pts_next_norm_[i].y;
        float* ai = A.ptr<float>(i);

        ai[0] = u2*u1;
        ai[1] = u2*v1;
        ai[2] = u2;
        ai[3] = v2*u1;
        ai[4] = v2*v1;
        ai[5] = v2;
        ai[6] = u1;
        ai[7] = v1;
        ai[8] = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2) = 0;

    return u*cv::Mat::diag(w)*vt;;
}


}//! vk