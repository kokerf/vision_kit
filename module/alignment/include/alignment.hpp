#ifndef _ALIGNMENT_HPP_
#define _ALIGNMENT_HPP_

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>

#include <Eigen/Core>

namespace vk{

/* =================================         Fuction           ================================= */

/**
* [align2D to align a pitch to another image]
* @param  cur_img         [destination image]
* @param  ref_patch       [align path of templet image]
* @param  ref_patch_gx    [gradient of ref_patch in x]
* @param  ref_patch_gy    [gradient of ref_patch in y]
* @param  cur_px_estimate [centre of the patch found in cur_img]
* @param  MAX_ITER        [Maximum iteration count]
* @param  EPS             [Threshold value for termination criteria]
* @return                 [return true if found]
*/
bool align2D(const cv::Mat& cur_img, const cv::Mat& ref_patch, const cv::Mat& ref_patch_gx, const cv::Mat& ref_patch_gy,
    cv::Point2f& cur_px_estimate, const int MAX_ITER = 30, const float EPS = 1E-2f);

/**
* [getPatch get patch from image]
* @param  src_img         [source image]
* @param  dst_img         [patch from source image]
* @param  centre          [centre of the patch in source image]
* @param  size            [patch size to extract]
* @param  affine          [warp of patch, based on centre of patch]
* @return                 [void]
*/
void getPatch(const cv::Mat &src_img, cv::Mat &dst_img, const cv::Point2f &centre, const int size = 8, const cv::Mat &affine = cv::Mat::eye(2, 2, CV_32FC1));


/* =================================          Class           ================================= */

/**
*   Class Align, the base Class
*/
class Align
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Align(const cv::Mat& ref_patch, const cv::Mat& ref_patch_gx, const cv::Mat& ref_patch_gy, std::vector<std::pair<int, int> > &partern);

    bool run(const cv::Mat& cur_img, Eigen::VectorXd &estimate, const size_t MAX_ITER = 30, const double EPS = 1E-2f);

    void printInfo();

protected:
    //! virtual function, which should be override in child class
    virtual void perCompute() = 0;

    virtual const bool computeResiduals(const cv::Mat &cur_img, double &mean_error) = 0;

    virtual const bool update(double &step) = 0;

protected:
    const size_t N;
    Eigen::Vector2d offset_;
    Eigen::VectorXd ref_patch_;
    Eigen::VectorXd ref_gradx_;
    Eigen::VectorXd ref_grady_;
    std::vector<std::pair<int, int> > partern_;

    Eigen::MatrixXd H_;                //! S*S
    Eigen::MatrixXd Hinv_;             //! S*S
    Eigen::MatrixXd Jac_;              //! N*S
    Eigen::RowVectorXd Jres_;          //! 1*S

    Eigen::VectorXd estimate_;         //! S*1

    //! struct to save print information
    struct InfoMsg {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        size_t id;
        long long t_start;
        long long t_end;
        double error;
        double step;
        bool converged;
        Eigen::VectorXd estimate;
    };
    std::vector<InfoMsg> out_info_;
};

/**
*   Class Align2D
*   Module: pixel 2D drift
*   Algorithm: Inverse Compositional
*/
class Align2D : public Align
{
public:
    using Align::Align;

protected:
    void perCompute();

    const bool computeResiduals(const cv::Mat& cur_img, double &error);

    const bool update(double &step);
};

/**
*   Class Align2DI
*   Module: pixel 2D drift with bias(illumination or exposure differences)
*   Algorithm: Inverse Compositional
*/
class Align2DI : public Align
{
public:
    using Align::Align;

protected:
    void perCompute();

    const bool computeResiduals(const cv::Mat& cur_img, double &error);

    const bool update(double &step);
};

/**
*   Class AlignESM2DI
*   Module: pixel 2D drift with bias(illumination or exposure differences)
*   Algorithm: Efficient Second-order Minimization
*/
class AlignESM2DI : public Align2DI
{
public:
    using Align2DI::Align2DI;

protected:
    void perCompute();

    const bool computeResiduals(const cv::Mat& cur_img, double &error);

    const bool update(double &step);
protected:

    Eigen::VectorXd Res_;
};

/**
*   Class Align1DI
*   Module: pixel drift in a direction with bias(illumination or exposure differences)
*   Algorithm: Inverse Compositional
*/
class Align1DI : public Align2DI
{
public:
    using Align2DI::Align2DI;

    bool run(const cv::Mat& cur_img, Eigen::VectorXd &estimate, const Eigen::Vector2d &pixel, const Eigen::Vector2d &direction, const size_t MAX_ITER = 30, const double EPS = 1E-2f);

protected:
    void perCompute();

    const bool computeResiduals(const cv::Mat& cur_img, double &error);

    const bool update(double &step);

protected:
    Eigen::Vector2d Pixel_;
    Eigen::Vector2d Dir_;

    Eigen::Matrix2d H_;
    Eigen::VectorXd Jac_;
    Eigen::RowVector2d Jres_;
};

}//! vk

#endif