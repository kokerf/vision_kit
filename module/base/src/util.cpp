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

}