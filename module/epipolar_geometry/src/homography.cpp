#include <vector>
#include <opencv2/core/core.hpp>

#include "base.hpp"
#include "homography.hpp"

namespace vk {

cv::Mat findHomographyMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    HomographyType type, float sigma, int iterations)
{
    Homography homography(pts_prev, pts_next, type, sigma, iterations);

    return homography.slove();
}

Homography::Homography(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    HomographyType type, float sigma, int iterations) :
    pts_prev_(pts_prev), pts_next_(pts_next), run_type_(type), sigma2_(sigma*sigma), iterations_(iterations)
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
    case HM_RANSAC:; break;
    default: break;
    }

    return H_;
}

cv::Mat Homography::runDLT(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2)
{
    const int N = pts_prev.size();
    assert(N >= 8);

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

}//! vk