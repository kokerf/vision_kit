#ifndef _HOMOGRAPHY_HPP_
#define _HOMOGRAPHY_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace vk{

enum HomographyType {
    HM_DLT = 1,     //! use all points
    HM_RANSAC = 2,  //! use RANSAC
};

cv::Mat findHomographyMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    HomographyType type, float sigma=1, int max_iterations=2000);

class Homography
{
public:
    Homography(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
        HomographyType type=HM_RANSAC, float sigma=1, int max_iterations=2000);
    ~Homography();

    cv::Mat slove();

    cv::Mat runDLT(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2);

    cv::Mat runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& T1, cv::Mat& T2, cv::Mat& inliners);

private:
    const HomographyType run_type_;
    std::vector<cv::Point2f> pts_prev_;
    std::vector<cv::Point2f> pts_next_;
    std::vector<cv::Point2f> pts_prev_norm_;
    std::vector<cv::Point2f> pts_next_norm_;

    cv::Mat inliners_;
    cv::Mat T1_, T2_;
    cv::Mat H_;
    float sigma2_;
    int max_iterations_;
};

}//! vk

#endif