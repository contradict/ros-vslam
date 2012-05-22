#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "frame_common/frame.h"

namespace frame_common
{
  class FrameProcGpu : public FrameProc
  {
  public:
    /// Create a FrameProc object using ORB feature detector/descriptor
    FrameProcGpu(void);

    /// \brief Set up stereo frame, assumes frame has camera parameters already set.
    /// \param frame The frame to be processed.
    /// \param img   Left camera image.
    /// \param imgr  Right camera image.
    /// \param nfrac Fractional disparity. If above 0, then imgr is a 16-bit fractional disparity
    ///              image instead, with <nfrac> counts per pixel disparity
    /// \param setPointCloud.  True if point cloud is to be set up from disparities
    void setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, int nfrac = 0, 
                        bool setPointCloud=false);


    /// \brief Set up stereo frame, assumes frame has camera parameters already set.
    /// \param frame The frame to be processed.
    /// \param img   Left camera image.
    /// \param imgr  Right camera image.
    /// \param nfrac Fractional disparity. If above 0, then imgr is a disparity
    ///              image instead.
    /// \param mask  ROI for left image
    /// \param setPointCloud.  True if point cloud is to be set up from disparities
    void setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, const cv::Mat &left_mask, int nfrac = 0,
                        bool setPointCloud=false);


  private:
    cv::Ptr<cv::gpu::StereoBM_GPU> StereoBM_;

    cv::gpu::GpuMat gpu_img_left, gpu_img_right;
    cv::gpu::GpuMat gpu_disparity;
  };
}
