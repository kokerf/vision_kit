#include <iostream>
#include <vector>
//#include <utility>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>

#include "base.hpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    if(argc!=3)
    {
        cout << "Usage: ./test_opticalflow image1 image2" <<endl;
        return -1;
    }

    //! Load image
    Mat image0 = cv::imread(argv[1]);
    Mat image1 = cv::imread(argv[2]);
    if(image0.empty() || image1.empty())
    {
        cout << "Can not open image: " << argv[1] <<" " << argv[2] << endl;
        return -1;
    }

    //! convert to gray
    Mat gray0, gray1;
    cvtColor(image0, gray0, COLOR_BGR2GRAY);
    cvtColor(image1, gray1, COLOR_BGR2GRAY);

    //! get corners in first image
    vector<KeyPoint> keypoints;
    Ptr<ORB> detector = ORB::create(100);
    detector->detect(gray0, keypoints, Mat());

    vector<KeyPoint> keypoints0, keypoints1, keypoints2, keypoints3;
    vector<DMatch> matches0, matches1;
    vector<cv::Point2f> points_prev, points_next0, points_next1;
    vector<float> errors0, errors1;
    vector<unsigned char> status0, status1;

    if(!keypoints.empty())
    {
        for(vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
        {
            points_prev.push_back(it->pt);
        }
    }

    //! ===========================
    //!  vk::computePyrLK
    //! ===========================
    double cv_start_time = (double)cv::getTickCount();
    vk::computePyrLK(gray0, gray1, points_prev, points_next0, status0, errors0, cv::Size(21, 21), 3, 40, 0.001);
    double cv_time = ((double)cv::getTickCount() - cv_start_time) / cv::getTickFrequency();
    cout << "vk::computePyrLK: " << cv_time <<endl;

    //! check status and get good matchs
    float avg_error0 = 0;
    for(int i = 0; i < points_prev.size(); i++)
    {
       if(!status0[i])
           continue;
       if(points_next0[i].x < 0 || points_next0[i].x > image0.cols || points_next0[i].y < 0 || points_next0[i].y > image0.rows)
           continue;

       int new_i = static_cast<int>(matches0.size());
       cv::KeyPoint kp0, kp1;
       kp0.pt = points_prev[i];
       kp1.pt = points_next0[i];
       keypoints0.push_back(kp0);
       keypoints1.push_back(kp1);
       matches0.push_back(DMatch(new_i, new_i, 0));
       avg_error0 += errors0[i];
    }
    avg_error0 /= keypoints0.size();
    cout << "Total tracking points:" << matches0.size() << ", mean errors:" << avg_error0 << endl;

    //! ============================
    //!  cv::calcOpticalFlowPyrLK
    //! ============================
    double vk_start_time = (double)cv::getTickCount();
    cv::calcOpticalFlowPyrLK(
        gray0, gray1,
        points_prev, points_next1,
        status1, errors1,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.001)
        );
    double vk_time = ((double)cv::getTickCount() - vk_start_time) / cv::getTickFrequency();
    cout << "cv::calcOpticalFlowPyrLK: " << vk_time <<endl;

    //! check status and get good matchs
    float avg_error1 = 0;
    for(int i = 0; i < points_prev.size(); i++)
    {
        if(!status1[i])
            continue;
        if(points_next1[i].x < 0 || points_next1[i].x > image0.cols || points_next1[i].y < 0 || points_next1[i].y > image0.rows)
            continue;

        int new_i = static_cast<int>(matches1.size());
        cv::KeyPoint kp0,kp1;
        kp0.pt = points_prev[i];
        kp1.pt = points_next1[i];
        keypoints2.push_back(kp0);
        keypoints3.push_back(kp1);
        matches1.push_back(DMatch(new_i, new_i, 0));
        avg_error1 += errors1[i];
    }
    avg_error1 /= keypoints2.size();
    cout << "Total tracking points:" << matches1.size() << ", mean errors:" << avg_error1 << endl;

    //! calculate difference
    vector< pair<float,float> > diff;
    float diff_x = 0;
    float diff_y = 0;
    int diff_n = 0;
    for (int i = 0; i < points_prev.size(); i++)
    {
        if(status0[i] != 1 || status1[i] != 1)
        {
            diff.push_back(make_pair(0,0));
            continue;
        }

        float dx = points_next0[i].x - points_next1[i].x;
        float dy = points_next0[i].y - points_next1[i].y;
        diff_x += fabs(dx);
        diff_y += fabs(dy);
        diff_n++;
        diff.push_back(make_pair(dx,dy));
    }
    diff_x /= diff_n;
    diff_y /= diff_n;
    cout << "avrage difference in x and y:" << diff_x << " " << diff_y << endl;

    //! draw images
    Mat keypoint_image, match_image0, match_image1;
    drawKeypoints(image0, keypoints, keypoint_image);
    cv::drawMatches(keypoint_image, keypoints0, image0, keypoints1, matches0, match_image0, cv::Scalar(255, 0, 0), cv::Scalar(200, 0, 10));
    cv::drawMatches(keypoint_image, keypoints2, image0, keypoints3, matches1, match_image1, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
    cv::imshow("vk", match_image0);
    cv::imshow("cv", match_image1);
    waitKey(0);
}