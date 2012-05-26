#include <frame_image_proc/matcher.h>

namespace frame_image_proc {

    void StereoMatcher::operator()(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp, int dtype) {
        if(use_gpu) {
            do_gpu_stereo(left, right, disp, dtype);
        } else {
            cpu_stereo_(left, right, disp, dtype);
        }
    }

    void StereoMatcher::do_gpu_stereo(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp, int dtype) {
        cv::gpu::GpuMat gpu_left, gpu_right, gpu_disparity;
        cv::Mat tmpDisparity;

        gpu_left.upload(left);
        gpu_right.upload(right);
        gpu_stereo_(gpu_left, gpu_right, gpu_disparity);

        gpu_disparity.download(tmpDisparity);
        tmpDisparity.convertTo(disp, dtype, 16.0, 0);
    }

    void StereoMatcher::determineValidWindow(int width, int height, int &left, int &top, int &right, int &bottom)
    {
        int border, ndisp, mindisp;
        if(use_gpu) {
            border = gpu_stereo_.winSize;
            ndisp = gpu_stereo_.ndisp;
            mindisp = 0;

        } else {
            cv::Ptr<CvStereoBMState> params = cpu_stereo_.state;

            border   = params->SADWindowSize / 2;
            ndisp = params->numberOfDisparities;
            mindisp = params->minDisparity;
        }

        int wtf = (mindisp >= 0) ? border + mindisp: std::max(border, -mindisp);
        left   = ndisp + mindisp + border - 1;
        right  = width - 1 - wtf;
        top    = border;
        bottom = height - 1 - border;
    }

}
