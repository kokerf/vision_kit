#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>
#include <base.hpp>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    if(argc!=3)
    {
        cout << "Usage: ./opticalflow image1 image2" <<endl;
        return -1;
    }

    //! Load image
    Mat image0 = cv::imread(argv[1]);
    Mat image1 = cv::imread(argv[2]);
    if(image0.empty())
    {
        cout << "Can not open image1: " << argv[1] << endl;
        return -1;
    }
    if(image1.empty())
    {
        cout << "Can not open image2: " << argv[2] << endl;
        return -1;
    }

    Mat gray0, gray1;
    cvtColor(image0, gray0, COLOR_BGR2GRAY);
    cvtColor(image1, gray1, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Ptr<ORB> detector = ORB::create(100);
    detector->detect(gray0, keypoints, Mat());

    vector<cv::Point2f> points_prev;
    if(!keypoints.empty())
    {
        for(vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
        {
            points_prev.push_back(it->pt);
        }
    }

    vector<cv::Point2f> points_next;
    vector<float> errors;

    clock_t start_time = clock();
    vk::computePyrLK(gray0, gray1, points_prev, points_next, errors, cv::Size(21, 21), 3, 40, 0.001);
    clock_t end_time = clock();
    cout << "vk::computePyrLK: " << (float)(end_time - start_time) / CLOCKS_PER_SEC <<endl;

    vector<KeyPoint> keypoints0, keypoints1;
    vector<DMatch> matches;
    if (!points_prev.empty())
    {
       for(int i = 0; i < points_prev.size();i++)
       {
           if(errors[i] == -1)
               continue;

           int new_i = static_cast<int>(matches.size());
           cv::KeyPoint kp0, kp1;
           kp0.pt = points_prev[i];
           kp1.pt = points_next[i];
           keypoints0.push_back(kp0);
           keypoints1.push_back(kp1);
           matches.push_back(DMatch(new_i, new_i, 0));
       }
    }
    cout << "Total tracking points:" << matches.size() << endl;

    Mat keypoint_image, match_image;
    drawKeypoints(image0, keypoints, keypoint_image);
    cv::drawMatches(keypoint_image, keypoints0, image0, keypoints1, matches, match_image, cv::Scalar(255, 0, 0), cv::Scalar(200, 0, 10));
    cv::imshow("match", match_image);
    imshow("image", image0);
    //waitKey(0);


    std::vector<unsigned char> status;
    std::vector<float> error;
    std::vector<cv::Point2f>  currPts_temp;
    //! tracking by LK opticalflow
    start_time = clock();
    cv::calcOpticalFlowPyrLK(
        image0, image1,
        points_prev, currPts_temp,
        status, error,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.001)
        );
    end_time = clock();
    cout << "cv::calcOpticalFlowPyrLK: " << (float)(end_time - start_time) / CLOCKS_PER_SEC <<endl;

    //! check status and get good matchs
    vector<KeyPoint> keypoints2, keypoints3;
    matches.clear();
    int status_num = status.size();
    for (int i = 0; i < status_num; i++)
    {
        if (status[i] == 1)
        {
            if (currPts_temp[i].x < 0 || currPts_temp[i].x > image0.cols || currPts_temp[i].y < 0 || currPts_temp[i].y > image0.rows)
                continue;

            int new_i = static_cast<int>(matches.size());
            cv::KeyPoint kp0,kp1;
            kp0.pt = points_prev[i];
            kp1.pt = currPts_temp[i];
            keypoints2.push_back(kp0);
            keypoints3.push_back(kp1);
            matches.push_back(DMatch(new_i, new_i, 0));
        }
    }


    vector<pair<float,float>> diff;
    for (int i = 0; i < points_prev.size(); i++)
    {
        float dx = points_next[i].x - currPts_temp[i].x;
        float dy = points_next[i].y - currPts_temp[i].y;
        diff.push_back(make_pair(dx,dy));
    }



    cout << "Total tracking points:" << matches.size() << endl;
    Mat match_image1;
    cv::drawMatches(keypoint_image, keypoints2, image0, keypoints3, matches, match_image1, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
    cv::imshow("match_opencv", match_image1);
    waitKey(0);
}