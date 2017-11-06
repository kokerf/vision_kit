#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include "alignment.hpp"

namespace vk{


bool Alignment::align2D(const cv::Mat& cur_img, const cv::Mat& ref_patch, const cv::Mat& ref_patch_gx, const cv::Mat& ref_patch_gy, 
    cv::Point2f& cur_px_estimate, const int MAX_ITER, const float EPS)
{
    const int cols = ref_patch.cols;
    const int rows = ref_patch.rows;
    const float min_update_squared = EPS*EPS;
    bool converged = false;

    cv::Mat H = cv::Mat::zeros(3,3,CV_32FC1);
    for(int y = 0; y < rows; ++y)
    {
        const float* gx_ref = ref_patch_gx.ptr<float>(y);
        const float* gy_ref = ref_patch_gy.ptr<float>(y);
        for(int x = 0; x < cols; ++x)
        {
            cv::Mat J = (cv::Mat_<float>(1, 3) << *(gx_ref+x), *(gy_ref+x), 1.0f);
            H += J.t() * J;
        }
    }
    cv::Mat invH = H.inv();

    //! temp parameters
    float u = cur_px_estimate.x;
    float v = cur_px_estimate.y;

    const int cur_step = cur_img.step.p[0];
    float intensity_diff = 0;

#ifdef _OUTPUT_MESSAGES_
    cv::Mat warpI = cv::Mat(rows, cols, CV_32FC1);
    std::cout << "== Alignment::align2D, input point: [" << std::setiosflags(std::ios::fixed) << std::setprecision(3) << u << ", " << v << "], start iteration:\n";
#endif

    int iter = 0;
    while(iter++ < MAX_ITER)
    {
#ifdef _OUTPUT_MESSAGES_
        std::cout << "* iter: " << std::setw(3) << iter;
#endif
        cv::Mat Jres = cv::Mat::zeros(3, 1, CV_32FC1);
        cv::Mat dp = cv::Mat::zeros(3, 1, CV_32FC1);

        cv::Point2f cur_patch_LT(u - floor(cols / 2), v - floor(rows / 2));
        if(cur_patch_LT.x < 0 || cur_patch_LT.y < 0 || cur_patch_LT.x+ cols > cur_img.cols || cur_patch_LT.y+rows > cur_img.rows)
            break;

        // compute interpolation weights
        const int u_r = floor(u);
        const int v_r = floor(v);
        const float subpix_x = u - u_r;
        const float subpix_y = v - v_r;
        const float wTL = (1.0 - subpix_x)*(1.0 - subpix_y);
        const float wTR = subpix_x * (1.0 - subpix_y);
        const float wBL = (1.0 - subpix_x)*subpix_y;
        const float wBR = subpix_x * subpix_y;

        float mean_error = 0;
        for(int y = 0; y < rows; ++y)
        {
            const float* i_ref = ref_patch.ptr<float>(y);
            const float* gx_ref = ref_patch_gx.ptr<float>(y);
            const float* gy_ref = ref_patch_gy.ptr<float>(y);
            const uint8_t* i_cur = (uint8_t*)(cur_img.ptr<uint8_t>(y + v_r - rows / 2) + u_r - cols / 2);
            for(int x = 0; x < cols; ++x)
            {
                const float cur_intensity = wTL*i_cur[x] + wTR*i_cur[x + 1] + wBL*i_cur[cur_step + x] + wBR*i_cur[cur_step + x + 1];
                const float residual = cur_intensity - i_ref[x] + intensity_diff;

#ifdef _OUTPUT_MESSAGES_
                mean_error += residual*residual;
                warpI.at<float>(y, x) = cur_intensity;
#endif

                cv::Mat J = (cv::Mat_<float>(1, 3) << *(gx_ref + x), *(gy_ref + x), 1.0f);
                Jres += residual* J.t();
            }
        }

        dp = invH * Jres;
        u -= dp.at<float>(0, 0);
        v -= dp.at<float>(1, 0);
        intensity_diff -= dp.at<float>(2, 0);

#ifdef _OUTPUT_MESSAGES_
        mean_error /= rows*cols;
        std::cout << ", point: [" << u << ", " << v << "], intensity_diff: " << std::setw(7) << intensity_diff << ", mean_error: " << mean_error << "\n";
#endif

        if(dp.dot(dp) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    cur_px_estimate = cv::Point2f(u, v);

#ifdef _OUTPUT_MESSAGES_
    std::cout << "== converged: " << converged << ", point: [" << u << ", " << v << "]" << std::endl;
#endif

    return converged;
}


void Alignment::getPatch(const cv::Mat &src_img, cv::Mat &dst_img, const cv::Point2f &centre, const int size, const cv::Mat &affine)
{
    assert(src_img.type() == CV_8UC1);
    assert(affine.size() == cv::Size(2,2) && affine.type() == CV_32FC1);
    const float half_size = size * 0.5;
    const int src_step = src_img.step.p[0];

    //! get affine parameters
    const float xWarp0 = affine.at<float>(0, 0);
    const float xWarp1 = affine.at<float>(0, 1);
    const float yWarp0 = affine.at<float>(1, 0);
    const float yWarp1 = affine.at<float>(1, 1);

    dst_img = cv::Mat(size, size, CV_32FC1);
    for(size_t y = 0; y < size; y++)
    {
        float *dst = dst_img.ptr<float>(y);
        for(size_t x = 0; x < size; x++)
        {
            //! get affine pixel
            const cv::Point2f pt(x - half_size, y - half_size);
            const float u = xWarp0 * pt.x + xWarp1 * pt.y + centre.x;
            const float v = yWarp0 * pt.x + yWarp1 * pt.y + centre.y;

            //! check border
            if(u < 0 || v < 0 || u >= src_img.cols - 1 || v >= src_img.rows - 1)
                dst[x] = 0;
            
            //! compute interpolation weights
            const int u_r = floor(u);
            const int v_r = floor(v);
            const float subpix_x = u - u_r;
            const float subpix_y = v - v_r;
            const float wTL = (1.0 - subpix_x)*(1.0 - subpix_y);
            const float wTR = subpix_x * (1.0 - subpix_y);
            const float wBL = (1.0 - subpix_x)*subpix_y;
            const float wBR = subpix_x * subpix_y;
        
            const uint8_t *src = src_img.ptr<uint8_t>(v_r) + u_r;
            dst[x] = wTL*src[0] + wTR*src[1] + wBL*src[src_step] + wBR*src[src_step + 1];
        }
    }

}

}