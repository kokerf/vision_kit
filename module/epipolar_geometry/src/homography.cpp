#include <vector>
#include <opencv2/core/core.hpp>

#include "base.hpp"
#include "homography.hpp"

namespace vk {

cv::Mat findHomographyMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    HomographyType type, float sigma, int max_iterations)
{
    Homography homography(pts_prev, pts_next, type, sigma, max_iterations);

    return homography.slove();
}

Homography::Homography(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    HomographyType type, float sigma, int max_iterations) :
    run_type_(type), pts_prev_(pts_prev), pts_next_(pts_next), sigma2_(sigma*sigma), max_iterations_(max_iterations)
{
    assert(pts_prev.size() == pts_next.size());
    Normalize(pts_prev, pts_prev_norm_, T1_);
    Normalize(pts_next, pts_next_norm_, T2_);
}

Homography::~Homography()
{
    pts_prev_.clear();
    pts_next_.clear();
    pts_prev_norm_.clear();
    pts_next_norm_.clear();

    inliners_.release();
    T1_.release(), T2_.release();
    H_.release();
}

cv::Mat Homography::slove()
{
    switch(run_type_)
    {
    case HM_DLT: H_ = runDLT(pts_prev_norm_, pts_next_norm_, T1_, T2_); break;
    case HM_RANSAC: H_ = runRANSAC(pts_prev_norm_, pts_next_norm_, T1_, T2_, inliners_); break;
    default: break;
    }

    return H_;
}

cv::Mat Homography::runDLT(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2)
{
    const int N = pts_prev.size();
    assert(N >= 4);

    cv::Mat A(2*N, 9, CV_32F);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev[i].x;
        const float v1 = pts_prev[i].y;
        const float u2 = pts_next[i].x;
        const float v2 = pts_next[i].y;
        float* ai = A.ptr<float>(i*2);

        ai[0] = 0.0;
        ai[1] = 0.0;
        ai[2] = 0.0;
        ai[3] = -u1;
        ai[4] = -v1;
        ai[5] = -1;
        ai[6] = v2*u1;
        ai[7] = v2*v1;
        ai[8] = v2;

        ai[0+9] = u1;
        ai[1+9] = v1;
        ai[2+9] = 1;
        ai[3+9] = 0.0;
        ai[4+9] = 0.0;
        ai[5+9] = 0.0;
        ai[6+9] = -u2*u1;
        ai[7+9] = -u2*v1;
        ai[8+9] = -u2;
    }

    cv::Mat u,w,vt;

    cv::eigen(A.t()*A, w, vt);

    cv::Mat H_norm = vt.row(8).reshape(0, 3);

    cv::Mat H = T2.inv()*H_norm*T1;
    float H22 = H.at<float>(2, 2);
    if(fabs(H22) > FLT_EPSILON)
        H /= H22;

    return H;
}

cv::Mat Homography::runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2, cv::Mat& inliners)
{
    const int N = pts_prev.size();
    const int modelPoints = 4;
    assert(N >= modelPoints);

    const double threshold = 5.991*sigma2_;
    const int max_iters = VK_MIN(VK_MAX(max_iterations_, 1), 2000);

    std::vector<cv::Point2f> pt1(modelPoints);
    std::vector<cv::Point2f> pt2(modelPoints);
    char* inliners_arr;
    int max_inliners = 0;
    int niters = max_iters;
    int* select_points = new int[N];
    for(int iter = 0; iter < niters; iter++)
    {
        sampleNpoints(0, N-1, modelPoints, select_points);
        for(int i = 0; i < modelPoints; ++i)
        {
            pt1[i] = pts_prev_norm_[select_points[i]];
            pt2[i] = pts_next_norm_[select_points[i]];
        }

        cv::Mat H_temp = runDLT(pt1, pt2, T1, T2);
        cv::Mat H_temp_inv = H_temp.inv();

        int inliers_count = 0;
        cv::Mat inliners_temp = cv::Mat::zeros(N, 1, CV_8UC1);
        inliners_arr = inliners_temp.ptr<char>(0);
        for(int n = 0; n < N; ++n)
        {
            float error1 = transferError(pts_prev_[n], pts_next_[n], H_temp.ptr<float>(0));
            float error2 = transferError(pts_next_[n], pts_prev_[n], H_temp_inv.ptr<float>(0));

            const float error = error1+error2;

            if(error < threshold)
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
                const double num = log(1-0.99);
                const double omega = inliers_count*1.0 / N;
                const double denom = log(1 - pow(omega, modelPoints));

                niters = (denom >=0 || -num >= max_iters*(-denom)) ? max_iters : round(num / denom);
            }
            else
                break;
        }

    }//! iterations
    delete[] select_points;

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

    return runDLT(pt1, pt2, T1, T2);
}

}//! vk