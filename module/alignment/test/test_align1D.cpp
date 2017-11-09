#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "alignment.hpp"
#include "util.hpp"

#include <opencv2/core/eigen.hpp>

using namespace std;

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: ./test_alignment ref_img cur_img" << endl;
        return -1;
    }

    //! Load images
    cv::Mat ref_img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat cur_img = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    if (ref_img.empty() || cur_img.empty())
    {
        cout << "Can not open image: " << argv[1] << " or " << argv[2] << endl;
        return -1;
    }

    cv::Mat ref_gray, cur_gray;
    cv::Mat match_img;
    cv::Mat epipolar_img;
    if(ref_img.channels() != 1)
        cv::cvtColor(ref_img, ref_gray, cv::COLOR_RGB2GRAY);
    if(cur_img.channels() != 1)
        cv::cvtColor(cur_img, cur_gray, cv::COLOR_RGB2GRAY);

    std::vector<cv::Point2f> ref_pts, cur_pts;
    vk::getCorrespondPoints(ref_gray, cur_gray, ref_pts, cur_pts, 200);
    vk::drowMatchPoits(ref_img, cur_img, ref_pts, cur_pts, match_img);
    cv::imshow("match", match_img);
    cv::waitKey(0);

    cv::Mat F = cv::findFundamentalMat(ref_pts, cur_pts, cv::FM_RANSAC, 3, 0.99);
    cv::Mat F_32f;
    F.convertTo(F_32f, CV_32FC1);
    vk::drawEpipolarLines(ref_img, cur_img, ref_pts, cur_pts, F_32f, epipolar_img);
    cv::imshow("epipolar line", epipolar_img);
    cv::waitKey(0);

    //! Select one reference point
    cv::Point2f ref_pt = ref_pts[0];
    cv::Point2f cur_pt = cur_pts[0];
    const int win_size = 100;
    const int half_win_size = win_size / 2;
    size_t idx = 0;
    do {
        ref_pt = ref_pts[idx];
        cur_pt = cur_pts[idx];
        if(idx < ref_pts.size()){
            idx++;
        }
        else{
            std::cerr << "Error: There is no point satisfied!!!" << std::endl;
            return 1;
        }
    } while(ref_pt.x <= half_win_size || ref_pt.y <= half_win_size || ref_pt.x >= ref_img.cols - half_win_size || ref_pt.y > ref_img.rows - half_win_size ||
        cur_pt.x <= half_win_size || cur_pt.y <= half_win_size || cur_pt.x >= ref_img.cols - half_win_size || cur_pt.y > ref_img.rows - half_win_size);

    //! Get epiplor line in current image
    Eigen::MatrixXd eF;
    cv::cv2eigen(F, eF);
    Eigen::Vector3d px_ref{ ref_pt.x, ref_pt.y, 1 };
    Eigen::Vector3d px_cur{ cur_pt.x, cur_pt.y, 1 };
    Eigen::Vector3d epiplor_line = eF * px_ref;
    Eigen::Vector2d Ox{ 0, -epiplor_line[2] / epiplor_line[1] };
    Eigen::Vector2d dir = px_cur.head(2) - Ox;
    dir.normalize();

    //! Get reference template
    const int pattern_size = 10;

    cv::Mat ref_template;
    cv::Mat ref_template_gradx, ref_template_grady;
    vk::getPatch(ref_gray, ref_template, ref_pt, pattern_size);
    cv::Sobel(ref_template, ref_template_gradx, CV_32FC1, 1, 0);
    cv::Sobel(ref_template, ref_template_grady, CV_32FC1, 0, 1);

    ref_template.adjustROI(-1, -1, -1, -1);
    ref_template_gradx.adjustROI(-1, -1, -1, -1);
    ref_template_grady.adjustROI(-1, -1, -1, -1);

    //! Partern
    std::vector<std::pair<int, int> > partern(64);
    for(size_t i = 0; i < 8; i++)
    {
        for(size_t j = 0; j < 8; j++)
        {
            partern[i * 8 + j].first = i;
            partern[i * 8 + j].second = j;
        }
    }
    const double offset = 2;
    Eigen::Vector2d px_cur_start = px_cur.head(2) + offset * dir;
    //! ===== Align2DI =====
    vk::Align2DI align1(ref_template, ref_template_gradx, ref_template_grady, partern);
    Eigen::VectorXd estimate1 = Eigen::Vector3d{ px_cur_start[0], px_cur_start[1], 0 };
    align1.run(cur_gray, estimate1, 60);
    align1.printInfo();
    std::cout << "True point: " << px_cur.head(2).transpose() << ", estimate:" << estimate1.head(2).transpose() << std::endl;

    //! ===== Align1DI =====
    vk::Align1DI align2(ref_template, ref_template_gradx, ref_template_grady, partern);
    Eigen::VectorXd estimate2 = Eigen::Vector2d{ 0, 0 };
    align2.run(cur_gray, estimate2, px_cur_start, dir);
    align2.printInfo();
    std::cout << "True point: " << px_cur.head(2).transpose() << ", estimate:" << (px_cur_start + estimate2[0] * dir).transpose() << std::endl;

    const size_t size = 5;
    const size_t half_size = size / 2;
    partern.resize(size);
    //! attention!!! this partern is just for test
    for(size_t i = -half_size; i <= half_size; i++)
    {
        partern[i].first = dir[0] * half_size + ref_template.cols;
        partern[i].second = dir[1] * half_size + ref_template.rows;
    }
    vk::Align1DI align3(ref_template, ref_template_gradx, ref_template_grady, partern);
    Eigen::VectorXd estimate3 = Eigen::Vector2d{ 0, 0 };
    align3.run(cur_gray, estimate3, px_cur_start, dir);
    align3.printInfo();
    std::cout << "True point: " << px_cur.head(2).transpose() << ", estimate:" << (px_cur_start + estimate3[0] * dir).transpose() << std::endl;

    cv::waitKey(0);

    //! Show find patch
    cv::Point2f cur_pt1(estimate1[0], estimate1[1]);
    cv::Point2f cur_pt2(px_cur_start[0] + estimate2[0] * dir[0],  px_cur_start[1] + estimate2[0] * dir[1]);
    cv::Mat ref_patch = ref_img.colRange(floor(ref_pt.x) - half_win_size, floor(ref_pt.x) + half_win_size).rowRange(floor(ref_pt.y) - half_win_size, floor(ref_pt.y) + half_win_size).clone();
    cv::Mat cur_patch1 = cur_img.colRange(floor(cur_pt1.x) - half_win_size, floor(cur_pt1.x) + half_win_size).rowRange(floor(cur_pt1.y) - half_win_size, floor(cur_pt1.y) + half_win_size).clone();
    cv::Mat cur_patch2 = cur_img.colRange(floor(cur_pt2.x) - half_win_size, floor(cur_pt2.x) + half_win_size).rowRange(floor(cur_pt2.y) - half_win_size, floor(cur_pt2.y) + half_win_size).clone();
    cv::Point2f ref_patch_pt = cv::Point2f(half_win_size + ref_pt.x - floor(ref_pt.x), half_win_size + ref_pt.y - floor(ref_pt.y));
    cv::Point2f cur_patch_pt1 = cv::Point2f(half_win_size + cur_pt1.x - floor(cur_pt1.x), half_win_size + cur_pt1.y - floor(cur_pt1.y));
    cv::Point2f cur_patch_pt2 = cv::Point2f(half_win_size + cur_pt2.x - floor(cur_pt2.x), half_win_size + cur_pt2.y - floor(cur_pt2.y));

    cv::resize(ref_patch, ref_patch, ref_patch.size() * 4);
    cv::resize(cur_patch1, cur_patch1, cur_patch1.size() * 4);
    cv::resize(cur_patch2, cur_patch2, cur_patch2.size() * 4);
    cv::circle(ref_patch, ref_patch_pt * 4, 5 * 4, cv::Scalar(255, 0, 0));
    cv::circle(cur_patch1, cur_patch_pt1 * 4, 5 * 4, cv::Scalar(255, 0, 0));
    cv::circle(cur_patch2, cur_patch_pt2 * 4, 5 * 4, cv::Scalar(255, 0, 0));
    cv::imshow("ref_patch", ref_patch);
    cv::imshow("cur_patch1", cur_patch1);
    cv::imshow("cur_patch2", cur_patch2);
    cv::waitKey(0);

    return 0;
}