#ifndef _FUNDAMENTAL_HPP_
#define _FUNDAMENTAL_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace vk{

enum FundamentalType {
    FM_7POINT = 1,
    FM_8POINT = 2,
    FM_RANSAC = 4,
};
/**
 * [findFundamentalMat description]
 * @param  pts_prev       [points in the first image]
 * @param  pts_next       [points in the second image]
 * @param  type           [type of algrithm]
 * @param  sigma          [sigma of Gaussian distribution of points]
 * @param  max_iterations [max iteration times fot RANSAC]
 * @return                [3*3 fundamental matrix]
 */
cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
    FundamentalType type=FM_8POINT, float sigma=1, int max_iterations=2000);

/**
 * Fundametal matrix to find
 */
class Fundamental
{
public:
    Fundamental(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next,
        FundamentalType type=FM_RANSAC, float sigma=1, int max_iterations=2000);

    ~Fundamental();

/**
 * [slove find the fundamental matrix]
 * @return [3*3 fundamental matrix]
 */
    cv::Mat slove();

/**
 * [getInliers get inliers by Fundamental maxtix]
 * @return [inliers Mat, is inlier if set to 1]
 */
    cv::Mat getInliers();

/**
 * [computeErrors get the distance from point to its epipolar line]
 * @param p1   [point in the first image]
 * @param p2   [point in the second image]
 * @param F    [3*3 array, fundamental matrix of the two images]
 * @param err1 [distance of p1 to its epipolar line]
 * @param err2 [distance of p2 to its epipolar line]
 */
    static inline void computeErrors(const cv::Point2f& p1, const cv::Point2f& p2, const float* F, float& err1, float& err2);

private:

/**
 * [run8points 8-point algrothm]
 * @param  pts_prev [normalized points in the first image]
 * @param  pts_next [normalized points in the second image]
 * @param  F        [3*3 fundamental matrix]
 * @return          [1 if ok]
 */
    int run8point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F);

/**
* [run7point 7-point algrothm]
* @param  pts_prev [normalized points in the first image]
* @param  pts_next [normalized points in the second image]
* @param  F        [3*3n fundamental matrix, n=1 or 3]
* @return          [n solution, 1 or 3]
*/
    int run7point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F);
/**
 * [runRANSAC RANSCA for 8-points algrothm]
 * @param  pts_prev [normalized points in the first image]
 * @param  pts_next [normalized points in the second image]
 * @param  F        [3*3 fundamental matrix]
 * @param  inliners [output inliners]
 * @return          [1 if ok]
 */
    int runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F, cv::Mat& inliners);

private:
    const FundamentalType run_type_;
    std::vector<cv::Point2f> pts_prev_;
    std::vector<cv::Point2f> pts_next_;

    cv::Mat inliners_;
    cv::Mat F_;
    float sigma2_;
    int max_iterations_;
};


}//! vk

#endif