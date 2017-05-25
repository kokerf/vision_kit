#include <vector>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include "base.hpp"
#include "fundamental.hpp"

namespace vk{

cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    FundamentalType type, float sigma, int max_iterations)
{
    Fundamental fundamental(pts_prev, pts_next, type, sigma, max_iterations);

    return fundamental.slove();
}

Fundamental::Fundamental(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    FundamentalType type, float sigma, int max_iterations):
    pts_prev_(pts_prev), pts_next_(pts_next), run_type_(type), sigma2_(sigma*sigma), max_iterations_(max_iterations)
{
    assert(pts_prev.size() == pts_next.size());
    Normalize(pts_prev, pts_prev_norm_, T1_);
    Normalize(pts_next, pts_next_norm_, T2_);
}

Fundamental::~Fundamental()
{
    pts_prev_.clear();
    pts_next_.clear();
    pts_prev_norm_.clear();
    pts_next_norm_.clear();

    inliners_.release();
    T1_.release(), T2_.release();
    F_.release();
}

cv::Mat Fundamental::getInliers()
{
    if(run_type_ == FM_RANSAC)
        return inliners_.clone();

    return cv::Mat();
}

cv::Mat Fundamental::slove()
{
    switch(run_type_)
    {
    //case vk::FM_7POINT: run7points();
    case vk::FM_8POINT: F_ = run8points(pts_prev_norm_, pts_next_norm_, T1_, T2_); break;
    case vk::FM_RANSAC: F_ = runRANSAC(pts_prev_norm_, pts_next_norm_, T1_, T2_, inliners_); break;
    default: break;
    }

    return F_;
}

cv::Mat Fundamental::run8points(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2)
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

    cv::Mat F = T2.t()*F_norm*T1;
    float F22 = F.at<float>(2, 2);
    if(fabs(F22) > FLT_EPSILON)
        F /= F22;

    return F;
}

cv::Mat Fundamental::runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2, cv::Mat& inliners)
{
    const int N = pts_prev.size();
    const int modelPoints = 8;
    assert(N >= modelPoints);

    const double threshold = 3.841*sigma2_;
    const int max_iters = VK_MIN(VK_MAX(max_iterations_, 1), 2000);

    std::vector<int> total_points;
    for(int i = 0; i < N; ++i)
    {
        total_points.push_back(i);
    }

    std::vector<cv::Point2f> pt1(modelPoints);
    std::vector<cv::Point2f> pt2(modelPoints);
    char* inliners_arr;
    int max_inliners = 0;
    int niters = max_iters;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<int> points = total_points;
        for(int i = 0; i < modelPoints; ++i)
        {
            int randi = vk::Rand(0, points.size()-1);
            pt1[i] = pts_prev_norm_[points[randi]];
            pt2[i] = pts_next_norm_[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        cv::Mat F_temp = run8points(pt1, pt2, T1, T2);

        int inliers_count = 0;
        cv::Mat inliners_temp = cv::Mat::zeros(N, 1, CV_8UC1);
        inliners_arr = inliners_temp.ptr<char>(0);
        for(int n = 0; n < N; ++n)
        {
            float error1, error2;
            computeErrors(pts_prev_[n], pts_next_[n], F_temp.ptr<float>(0), error1, error2);

            const float error = VK_MAX(error1, error2);

            if(error < threshold)
            {
                inliners_arr[n] = 1;
                inliers_count++;
                //F_out = F_temp;
            }

        }

        if(inliers_count > max_inliners)
        {
            max_inliners = inliers_count;
            inliners = inliners_temp.clone();

           if(inliers_count < N)
           {
               //! N = log(1-p)/log(1-omega^s)
               //! p = 99%
               //! number of set: s = 8
               //! omega = inlier points / total points
               const double num = log(1-0.99);
               const double omega = inliers_count*1.0 / N;
               const double denom = log(1 - pow(omega, modelPoints));

               niters = (denom >=0 || -num >= max_iters*(-denom)) ? max_iters : round(num / denom);
           }
           else
               break;
        }

    }//! iterations

    pt1.clear();
    pt2.clear();
    inliners_arr = inliners.ptr<char>(0);
    for(int n = 0; n < N; ++n)
    {
        if(inliners_arr[n] != 1)
        {
            continue;
        }

        pt1.push_back(pts_prev_norm_[n]);
        pt2.push_back(pts_next_norm_[n]);
    }

    return run8points(pt1, pt2, T1, T2);
}

inline void Fundamental::computeErrors(const cv::Point2f& p1, const cv::Point2f& p2, const float* F, float& err1, float& err2)
{
    //! point X1 = (u1, v1, 1)^T in first image
    //! poInt X2 = (u2, v2, 1)^T in second image
    const float u1 = p1.x;
    const float v1 = p1.y;
    const float u2 = p2.x;
    const float v2 = p2.y;

    //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
    const float a2 = F[0]*u1 + F[1]*v1 + F[2];
    const float b2 = F[3]*u1 + F[4]*v1 + F[5];
    const float c2 = F[6]*u1 + F[7]*v1 + F[8];
    //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
    const float a1 = F[0]*u2 + F[3]*v2 + F[6];
    const float b1 = F[1]*u2 + F[4]*v2 + F[7];
    const float c1 = F[2]*u2 + F[5]*v2 + F[8];

    //! distance from point to line: d^2 = |ax+by+c|^2/(a^2+b^2)
    //! X2 to L2 in second image
    const float dist2 = a2*u2 + b2*v2 + c2;
    const float square_dist2 = dist2*dist2/(a2*a2 + b2*b2);
    //! X1 to L1 in first image
    const float dist1 = a1*u1 + b1*v1 + c1;
    const float square_dist1 = dist1*dist1/(a1*a1 + b1*b1);

    err1 = square_dist1;
    err2 = square_dist2;
}


}//! vk