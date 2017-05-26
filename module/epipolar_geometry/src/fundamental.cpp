#include <iostream>
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
    run_type_(type), pts_prev_(pts_prev), pts_next_(pts_next), sigma2_(sigma*sigma), max_iterations_(max_iterations)
{
    assert(pts_prev.size() == pts_next.size());
}

Fundamental::~Fundamental()
{
    pts_prev_.clear();
    pts_next_.clear();

    inliners_.release();
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
    int modules = 0;
    switch(run_type_)
    {
    case vk::FM_7POINT: modules = run7point(pts_prev_, pts_next_, F_); break;
    case vk::FM_8POINT: modules = run8point(pts_prev_, pts_next_, F_); break;
    case vk::FM_RANSAC: modules = runRANSAC(pts_prev_, pts_next_, F_, inliners_); break;
    default: break;
    }

    return F_;
}

int Fundamental::run8point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F)
{
    const int N = pts_prev.size();
    assert(N >= 8);

    std::vector<cv::Point2f> pts_prev_norm;
    std::vector<cv::Point2f> pts_next_norm;
    cv::Mat T1, T2;
    Normalize(pts_prev, pts_prev_norm, T1);
    Normalize(pts_next, pts_next_norm, T2);

    cv::Mat A(N, 9, CV_32F);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev_norm[i].x;
        const float v1 = pts_prev_norm[i].y;
        const float u2 = pts_next_norm[i].x;
        const float v2 = pts_next_norm[i].y;
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

    F = T2.t()*F_norm*T1;
    float F22 = F.at<float>(2, 2);
    if(fabs(F22) > FLT_EPSILON)
        F /= F22;

    return 1;
}

int Fundamental::run7point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F)
{
    const int N = pts_prev.size();
    assert(N == 7);

    float a[7*9], c[4], r[3];
    cv::Mat A(7, 9, CV_32F, a);
    cv::Mat coeffs(1, 4, CV_32F, c);
    cv::Mat roots(1, 3, CV_32F, r);
    float* a_ptr = &a[0];
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev[i].x;
        const float v1 = pts_prev[i].y;
        const float u2 = pts_next[i].x;
        const float v2 = pts_next[i].y;

        a_ptr[0] = u2*u1;
        a_ptr[1] = u2*v1;
        a_ptr[2] = u2;
        a_ptr[3] = v2*u1;
        a_ptr[4] = v2*v1;
        a_ptr[5] = v2;
        a_ptr[6] = u1;
        a_ptr[7] = v1;
        a_ptr[8] = 1;

        a_ptr+=9;
    }

    cv::Mat u,w,vt;

    cv::eigen(A.t()*A, w, vt);

    //cv::Mat W, U, Vt;
    //cv::SVDecomp(A, W, U, Vt, cv::SVD::MODIFY_A + cv::SVD::FULL_UV);

    //! F = alpha*F1 + (1-alpha)*F2 = alpha*(F1-F2) + F2
    //! let det(F)=0
    float* f1 = vt.ptr<float>(7);
    float* f2 = vt.ptr<float>(8);

    for(int i = 0; i < 9; i++)
        f1[i] -= f2[i];

    //! Matlab can help you ^_^
    //! det(F) = c0*alpha^3 + c1*alpha^2 + c2*alpha + c3 = 0
    const double M00 = f1[4]*f1[8] - f1[5]*f1[7];
    const double M01 = f1[3]*f1[8] - f1[5]*f1[6];
    const double M02 = f1[3]*f1[7] - f1[4]*f1[6];
    const double M10 = f1[1]*f1[8] - f1[2]*f1[7];
    const double M11 = f1[0]*f1[8] - f1[2]*f1[6];
    const double M12 = f1[0]*f1[7] - f1[1]*f1[6];
    const double M20 = f1[1]*f1[5] - f1[2]*f1[4];
    const double M21 = f1[0]*f1[5] - f1[2]*f1[3];
    const double M22 = f1[0]*f1[4] - f1[1]*f1[3];

    const double N00 = f2[4]*f2[8] - f2[5]*f2[7];
    const double N01 = f2[3]*f2[8] - f2[5]*f2[6];
    const double N02 = f2[3]*f2[7] - f2[4]*f2[6];
    const double N10 = f2[1]*f2[8] - f2[2]*f2[7];
    const double N11 = f2[0]*f2[8] - f2[2]*f2[6];
    const double N12 = f2[0]*f2[7] - f2[1]*f2[6];
    const double N20 = f2[1]*f2[5] - f2[2]*f2[4];
    const double N21 = f2[0]*f2[5] - f2[2]*f2[3];
    const double N22 = f2[0]*f2[4] - f2[1]*f2[3];

    c[0] = f1[0]*M00 - f1[1]*M01 + f1[2]*M02;

    c[1] = f2[0]*M00 - f2[1]*M01 + f2[2]*M02
         - f2[3]*M10 + f2[4]*M11 - f2[5]*M12
         + f2[6]*M11 - f2[7]*M21 + f2[8]*M22;

    c[2] = f1[0]*N00 - f1[1]*N01 + f1[2]*N02
         - f1[3]*N10 + f1[4]*N11 - f1[5]*N12
         + f1[6]*N20 - f1[7]*N21 + f1[8]*N22;

    c[3] = f2[0]*N00 - f2[1]*N01 + f2[2]*N01;

    //! solve the cubic equation, can be 1 or 3 solutions
    //! if there are 3 solution, where 3D points and camera centres lie on a ruled quadric referred to as a critical surface(from MVG)
    int n = solveCubic(coeffs, roots);

    if(n < 1 || n > 3)
    {
        std::cout << "Error in solution of 7-point algrithom" << std::endl;
        return 0;
    }

    F = cv::Mat(3*n, 3, CV_32F);
    float* f = F.ptr<float>(0);
    for(int k = 0; k < n; ++k, f+=9)
    {
        float alpha = r[k];
        float F22 = f1[8]*alpha + f2[8];
        float mu = 1.0;

        if(fabs(F22) > FLT_EPSILON)
        {
            mu = 1./F22;
            alpha *= mu;
            f[8] = 1;
        }
        else
        {
            f[8] = 0;
        }

        for(int i = 0; i < 8; i++)
            f[i] = f1[i]*alpha + f2[i]*mu;
    }

    return n;
}

int Fundamental::runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F, cv::Mat& inliners)
{
    const int N = pts_prev.size();
    const int modelPoints = 7;
    assert(N >= modelPoints);

    const double threshold = 3.841*sigma2_;
    const int max_iters = VK_MIN(VK_MAX(max_iterations_, 1), 2000);

    std::vector<int> total_points;
    for(int i = 0; i < N; ++i)
    {
        total_points.push_back(i);
    }

    std::vector<cv::Point2f> pts1(modelPoints);
    std::vector<cv::Point2f> pts2(modelPoints);
    std::vector<cv::Point2f> pts1_norm;
    std::vector<cv::Point2f> pts2_norm;
    cv::Mat T1, T2, F_temp;
    char* inliners_arr;
    int max_inliners = 0;
    int niters = max_iters;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<int> points = total_points;
        for(int i = 0; i < modelPoints; ++i)
        {
            int randi = vk::Rand(0, points.size()-1);
            pts1[i] = pts_prev[points[randi]];
            pts2[i] = pts_next[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        //int models = run8point(pts1, pts2, F_temp);
        Normalize(pts1, pts1_norm, T1);
        Normalize(pts2, pts2_norm, T2);
        cv::Mat F_norm;
        int models = run7point(pts1_norm, pts2_norm, F_norm);

        for(int k = 0; k < models; k++)
        {
            F_temp = T2.t()*F_norm.rowRange(k*3, k*3 + 3)*T1;

            int inliers_count = 0;
            cv::Mat inliners_temp = cv::Mat::zeros(N, 1, CV_8UC1);
            inliners_arr = inliners_temp.ptr<char>(0);
            for (int n = 0; n < N; ++n)
            {
                float error1, error2;
                computeErrors(pts_prev[n], pts_next[n], F_temp.ptr<float>(0), error1, error2);

                const float error = VK_MAX(error1, error2);

                if (error < threshold)
                {
                    inliners_arr[n] = 1;
                    inliers_count++;
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
                    const double num = log(1 - 0.99);
                    const double omega = inliers_count*1.0 / N;
                    const double denom = log(1 - pow(omega, modelPoints));

                    niters = (denom >= 0 || -num >= max_iters*(-denom)) ? max_iters : round(num / denom);
                }
                else
                    break;
            }
        }//! models

    }//! iterations

    pts1.clear();
    pts2.clear();
    inliners_arr = inliners.ptr<char>(0);
    for(int n = 0; n < N; ++n)
    {
        if(inliners_arr[n] != 1)
        {
            continue;
        }

        pts1.push_back(pts_prev[n]);
        pts2.push_back(pts_next[n]);
    }

    return run8point(pts1, pts2, F);
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