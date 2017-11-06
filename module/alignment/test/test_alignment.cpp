#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "alignment.hpp"
#include "base.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    if (argc != 3 )
    {
        cout << "Usage: ./test_alignment ref_img cur_img" << endl;
        return -1;
    }

    //! Load images
    cv::Mat ref_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat cur_img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if(ref_img.empty() || cur_img.empty())
    {
        cout << "Can not open image: " << argv[1] << " or " << argv[2] << endl;
        return -1;
    }

    cv::Ptr<cv::ORB> detector = cv::ORB::create(100);
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    vector<cv::KeyPoint> keypoints0, keypoints1;
    cv::Mat descriptors0, descriptors1;
    std::vector<cv::DMatch> matches;

    detector->detectAndCompute(ref_img, cv::Mat(), keypoints0, descriptors0);
    detector->detectAndCompute(cur_img, cv::Mat(), keypoints1, descriptors1);
    matcher.match(descriptors0, descriptors1, matches);
    sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b){ return a.distance < b.distance;});
    matches.resize(10);

    cv::Mat match_img;
    cv::drawMatches(ref_img, keypoints0, cur_img, keypoints1, matches, match_img);
    cv::imshow("match", match_img);
    cv::waitKey(0);

    
    cv::Point2f ref_pt, cur_pt;
    int idx = 0;
    const int win_size = 100;
    const int half_win_size = win_size / 2;
    do {
        ref_pt = keypoints0[matches[idx].queryIdx].pt;
        cur_pt = keypoints1[matches[idx].trainIdx].pt;
        idx++;
    }
    while(ref_pt.x <= half_win_size || ref_pt.y <= half_win_size || ref_pt.x >= ref_img.cols - half_win_size || ref_pt.y > ref_img.rows - half_win_size ||
        cur_pt.x <= half_win_size || cur_pt.y <= half_win_size || cur_pt.x >= ref_img.cols - half_win_size || cur_pt.y > ref_img.rows - half_win_size);

    cv::Mat ref_patch = ref_img.colRange(floor(ref_pt.x) - half_win_size, floor(ref_pt.x) + half_win_size).rowRange(floor(ref_pt.y) - half_win_size, floor(ref_pt.y) + half_win_size).clone();
    cv::Mat cur_patch = cur_img.colRange(floor(cur_pt.x) - half_win_size, floor(cur_pt.x) + half_win_size).rowRange(floor(cur_pt.y) - half_win_size, floor(cur_pt.y) + half_win_size).clone();
    cv::Point2f ref_patch_pt = cv::Point2f(half_win_size + ref_pt.x - floor(ref_pt.x), half_win_size + ref_pt.y - floor(ref_pt.y));
    cv::Point2f cur_patch_pt = cv::Point2f(half_win_size + cur_pt.x - floor(cur_pt.x), half_win_size + cur_pt.y - floor(cur_pt.y));

    cv::cvtColor(ref_patch, ref_patch, cv::COLOR_GRAY2RGB);
    cv::cvtColor(cur_patch, cur_patch, cv::COLOR_GRAY2RGB);
    cv::resize(ref_patch, ref_patch, ref_patch.size() * 4);
    cv::resize(cur_patch, cur_patch, cur_patch.size() * 4);
    cv::circle(ref_patch, ref_patch_pt*4, 5*4, cv::Scalar(255, 0, 0));
    cv::circle(cur_patch, cur_patch_pt*4, 5*4, cv::Scalar(255, 0, 0));
    cv::imshow("ref_patch", ref_patch);
    cv::imshow("cur_patch", cur_patch);
    cv::waitKey(0);

    const int pattern_size = 10;

    cv::Mat ref_template;
    cv::Mat ref_template_gradx, ref_template_grady;
    vk::Alignment::getPatch(ref_img, ref_template, ref_pt, pattern_size);
    cv::Sobel(ref_template, ref_template_gradx, CV_32FC1, 1, 0);
    cv::Sobel(ref_template, ref_template_grady, CV_32FC1, 0, 1);

    ref_template.adjustROI(-1, -1, -1, -1);
    ref_template_gradx.adjustROI(-1, -1, -1, -1);
    ref_template_grady.adjustROI(-1, -1, -1, -1);
    cv::Point2f cur_estimate = cur_pt + cv::Point2f(2, -2);
    bool succeed = vk::Alignment::align2D(cur_img, ref_template, ref_template_gradx, ref_template_grady, cur_estimate, 60);

    std::cout << "Succeed: " << succeed << " , Corner match: [" << cur_pt.x << ", " << cur_pt.y << "], Aligen match: [" << cur_estimate.x << ", " << cur_estimate.y << "]" << std::endl;

    return 0;
}