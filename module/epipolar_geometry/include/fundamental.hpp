#ifndef _FUNDAMENTAL_HPP_
#define _FUNDAMENTAL_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace vk{

enum FundamentalType {
    FM_7POINT = 1,
    FM_8POINT = 2,
};

cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type=FM_8POINT);

class Fundamental
{
public:
    Fundamental(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, FundamentalType type);
    ~Fundamental();

    void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T);
    cv::Mat slove();
    cv::Mat run8points();

private:
    const FundamentalType run_type;
    std::vector<cv::Point2f> pts_prev_norm_;
    std::vector<cv::Point2f> pts_next_norm_;
    cv::Mat T1_, T2_;
};


}//! vk

#endif