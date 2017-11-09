#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include "util.hpp"

namespace vk{

void getCorrespondPoints(const cv::Mat &src, const cv::Mat &dst, std::vector<cv::Point2f> &pt_src, std::vector<cv::Point2f> &pt_dst, const size_t num, const float px_err)
{
    assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);

    pt_src.reserve(num);
    pt_dst.reserve(num);

    std::vector<cv::Point2f> pt1;
    std::vector<cv::Point2f> pt2;
    std::vector<float> errors;
    std::vector<uchar> status;

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.001);

    cv::goodFeaturesToTrack(src, pt1, num, 0.01, 5, cv::Mat());
    cv::cornerSubPix(src, pt1, cv::Size(21, 21), cv::Size(-1, -1), criteria);
    cv::calcOpticalFlowPyrLK(src, dst, pt1, pt2, status, errors, cv::Size(21, 21), 3, criteria);
    
    std::vector<cv::Point2f> pt1_temp, pt2_temp;
    for(size_t i = 0; i < status.size(); i++)
    {
        if(status[i])
        {
            pt1_temp.push_back(pt1[i]);
            pt2_temp.push_back(pt2[i]);
        }
    }

    pt1 = pt1_temp;
    pt2 = pt2_temp;
    cv::calcOpticalFlowPyrLK(dst, src, pt2, pt1, status, errors, cv::Size(21, 21), 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    const float error = px_err*px_err;
    for(size_t i = 0; i < status.size(); i++)
    {
        if(status[i])
        {
            cv::Point2f diff = pt1[i] - pt1_temp[i];
            if(diff.x*diff.x + diff.y*diff.y < error)
            {
                pt_src.push_back(pt1[i]);
                pt_dst.push_back(pt2[i]);
            }
        }
    }

    if(pt_src.size() > num)
    {
        pt_src.resize(num);
        pt_dst.resize(num);
    }

}

void drowMatchPoits(const cv::Mat &src, const cv::Mat &dst, std::vector<cv::Point2f> &pt_src, std::vector<cv::Point2f> &pt_dst, cv::Mat &match)
{
    assert(src.size() == dst.size() && src.type() == dst.type());
    assert(pt_src.size() == pt_dst.size());

    match = cv::Mat(src.rows, src.cols * 2, src.type());
    src.copyTo(match.colRange(0, src.cols));
    dst.copyTo(match.colRange(src.cols, src.cols * 2));
    if(match.channels() != 3)
    {
        cv::cvtColor(match, match, cv::COLOR_GRAY2RGB);
    }

    cv::Point2f offset(src.cols, 0);
    for(size_t i = 0; i < pt_src.size(); i++)
    {
        cv::RNG rng(i);
        const cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(match, pt_src[i], 5, color);
        cv::circle(match, pt_dst[i] + offset, 5, color);
        cv::line(match, pt_src[i], pt_dst[i] + offset, color);
    }
}

void drawEpipolarLines(const cv::Mat &img_prev, const cv::Mat &img_next, const std::vector<cv::Point2f> &pts_prev, const std::vector<cv::Point2f> &pts_next,
                      const cv::Mat &fundamental, cv::Mat &img_epipolar)
{
    const int N = pts_prev.size();
    assert(N == pts_next.size());

    cv::Mat img_epipolar1 = img_prev.clone();
    cv::Mat img_epipolar2 = img_next.clone();
    if(img_epipolar1.channels() != 3)
        cv::cvtColor(img_epipolar1, img_epipolar1, cv::COLOR_GRAY2RGB);
    if(img_epipolar2.channels() != 3)
        cv::cvtColor(img_epipolar2, img_epipolar2, cv::COLOR_GRAY2RGB);

    cv::Mat MF = fundamental.clone();
    if (MF.type() != CV_32FC1)
    {
        MF.convertTo(MF, CV_32FC1);
    }
    const float *F = MF.ptr<float>(0);
    const int cols = img_prev.cols;

    for (int n = 0; n < N; ++n)
    {
        //! point X1 = (u1, v1, 1)^T in first image
        //! poInt X2 = (u2, v2, 1)^T in second image
        const double u1 = pts_prev[n].x;
        const double v1 = pts_prev[n].y;
        const double u2 = pts_next[n].x;
        const double v2 = pts_next[n].y;

        //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
        //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
        const double a2 = F[0] * u1 + F[1] * v1 + F[2];
        const double b2 = F[3] * u1 + F[4] * v1 + F[5];
        const double c2 = F[6] * u1 + F[7] * v1 + F[8];

        const double a1 = F[0] * u2 + F[3] * v2 + F[6];
        const double b1 = F[1] * u2 + F[4] * v2 + F[7];
        const double c1 = F[2] * u2 + F[5] * v2 + F[8];

        //! points of the epipolar line within the image
        cv::Point2f start1(0, 0), end1(cols, 0), start2(0, 0), end2(cols, 0);
        start2.y = -(a2 * start2.x + c2) / b2;
        end2.y = -(a2 * end2.x + c2) / b2;
        start1.y = -(a1 * start1.x + c1) / b1;
        end1.y = -(a1 * end1.x + c1) / b1;

        //! draw lines and points in each image
        cv::Scalar color(255.0 * rand() / RAND_MAX, 255.0 * rand() / RAND_MAX, 255.0 * rand() / RAND_MAX);

        cv::circle(img_epipolar1, pts_prev[n], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar1, start1, end1, color);

        cv::circle(img_epipolar2, pts_next[n], 3, color, 1, cv::LINE_AA);
        cv::line(img_epipolar2, start2, end2, color);
    }

    //! copy the two images into one image
    img_epipolar = cv::Mat(img_epipolar1.rows, img_epipolar1.cols * 2, img_epipolar1.type());
    img_epipolar1.copyTo(img_epipolar.colRange(0, cols));
    img_epipolar2.copyTo(img_epipolar.colRange(cols, 2 * cols));
}

}