#include <vector>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "base.hpp"
#include "fundamental.hpp"

namespace vk{

cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type, float sigma, int iterations)
{
    Fundamental fundamental(pts_prev, pts_next, type, sigma, iterations);

    return fundamental.slove();
}

Fundamental::Fundamental(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type, float sigma, int iterations):
    pts_prev_(pts_prev), pts_next_(pts_next), run_type_(type), sigma2_(sigma*sigma), iterations_(iterations)
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

cv::Mat Fundamental::getInliers()
{
    if(run_type_ == FM_RANSAC)
        return inliners_.clone();

    return cv::Mat();
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

    cv::Mat F;
    switch(run_type_)
    {
    //case vk::FM_7POINT: run7points();
    case vk::FM_8POINT: F = run8points(pts_prev_norm_, pts_next_norm_); break;
    case vk::FM_RANSAC: F = runRANSAC(pts_prev_norm_, pts_next_norm_, inliners_); break;
    default: break;
    }

    return F;
}

cv::Mat Fundamental::run8points(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next)
{
    const int N = pts_prev.size();
    assert(N >= 8);

    cv::Mat A(N, 9, CV_32F);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev[i].x;
        const float v1 = pts_prev[i].y;
        const float u2 = pts_next[i].x;
        const float v2 = pts_next[i].y;
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

    cv::eigen(A.t()*A, w, vt);
    //cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2) = 0;

    cv::Mat F_norm = u*cv::Mat::diag(w)*vt;

    cv::Mat F = T2_.t()*F_norm*T1_;
    float F22 = F.at<float>(2, 2);
    if(fabs(F22) > FLT_EPSILON)
        F /= F22;

    return F;
}

cv::Mat Fundamental::runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& inliners)
{
    const int N = pts_prev.size();
    assert(N >= 8);

    const double threshold = 3.841*sigma2_;
    cv::Mat F_out;

    bool adaptive = true;
    int niters = 1000;
    if(iterations_ != -1)
    {
        adaptive = false;
        niters = VK_MIN(niters, iterations_);
        niters = VK_MIN(niters, 20);
    }

    std::vector<int> total_points;
    for(int i = 0; i < N; ++i)
    {
        total_points.push_back(i);
    }

    std::vector<cv::Point2f> pt0(8);
    std::vector<cv::Point2f> pt1(8);
    int max_inliners = 0;
    char* inliners_arr;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<int> points = total_points;
        for(int i = 0; i < 8; ++i)
        {
            int randi = vk::Rand(0, points.size()-1);
            pt0[i] = pts_prev_norm_[points[randi]];
            pt1[i] = pts_next_norm_[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        cv::Mat F_temp = run8points(pt0, pt1);
        float *F = F_temp.ptr<float>(0);

        int inliers_count = 0;
        cv::Mat inliners_temp = cv::Mat::zeros(N, 1, CV_8UC1);
        inliners_arr = inliners_temp.ptr<char>(0);
        for(int n = 0; n < N; ++n)
        {
            //! point X1 = (u1, v1, 1)^T in first image
            //! poInt X2 = (u2, v2, 1)^T in second image
            const double u1 = pts_prev_[n].x;
            const double v1 = pts_prev_[n].y;
            const double u2 = pts_next_[n].x;
            const double v2 = pts_next_[n].y;

            //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
            const double a2 = F[0]*u1 + F[1]*v1 + F[2];
            const double b2 = F[3]*u1 + F[4]*v1 + F[5];
            const double c2 = F[6]*u1 + F[7]*v1 + F[8];
            //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
            const double a1 = F[0]*u2 + F[3]*v2 + F[6];
            const double b1 = F[1]*u2 + F[4]*v2 + F[7];
            const double c1 = F[2]*u2 + F[5]*v2 + F[8];

            //! distance from point to line: d^2 = |ax+by+c|^2/(a^2+b^2)
            //! X2 to L2 in second image
            const double dist2 = a2*u2 + b2*v2 + c2;
            const double square_dist2 = dist2*dist2/(a2*a2 + b2*b2);
            //! X1 to L1 in first image
            const double dist1 = a1*u1 + b1*v1 + c1;
            const double square_dist1 = dist1*dist1/(a1*a1 + b1*b1);

            const double error = VK_MAX(square_dist1, square_dist2);

            if(error < threshold)
            {
                inliners_arr[n] = 1;
                inliers_count++;
                F_out = F_temp;
            }

        }

        if(inliers_count > max_inliners)
        {
            max_inliners = inliers_count;
            inliners = inliners_temp.clone();

            if (adaptive)
            {
                double ratio = VK_MAX(inliers_count*1.0 / N, 0.5);
                niters = -2.0 / log(1 - pow(ratio, 8));
            }
        }

    }//! iterations

    pt0.clear();
    pt1.clear();
    inliners_arr = inliners.ptr<char>(0);
    for(int n = 0; n < N; ++n)
    {
        if(inliners_arr[n] != 1)
        {
            continue;
        }

        pt0.push_back(pts_prev_norm_[n]);
        pt1.push_back(pts_next_norm_[n]);
    }

    return run8points(pt0, pt1);
}


}//! vk